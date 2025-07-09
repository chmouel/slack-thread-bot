import datetime
import logging
import os
import re
import sys
from typing import Dict

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes", "on")
DEBUG_FILE = os.environ.get("DEBUG_FILE", "")
if DEBUG_FILE:
    if os.path.exists(DEBUG_FILE):
        DEBUG_MODE = True
if DEBUG_MODE and not DEBUG_FILE:
    DEBUG_FILE = "/tmp/slack-bot-debug.log"

LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DEBUG_FILE) if DEBUG_MODE else logging.NullHandler(),
    ],
)

logger = logging.getLogger(__name__)


def clean_llm_response(text: str) -> str:
    """Remove unwanted tags from LLM response."""
    if not text:
        return ""

    text = text.strip()

    # 1. Remove <think>...</think> block from the beginning
    if text.startswith("<think>"):
        logger.debug("Original LLM response starts with <think> tag.")
        end_think_pos = text.find("</think>")
        if end_think_pos != -1:
            text = text[end_think_pos + len("</think>") :].lstrip()
            logger.debug("Cleaned LLM response after removing <think> tag.")

    # 2. Remove triple backticks if they wrap the entire content
    if text.startswith("```") and text.endswith("```"):
        logger.debug("Response is wrapped in triple backticks.")
        lines = text.splitlines()
        if len(lines) > 1:
            # Remove first line (e.g., ```json) and last line (```)
            text = "\n".join(lines[1:-1])
            logger.debug("Removed wrapping triple backticks.")
        else:
            # It's just ```...``` on one line, so remove them.
            text = text[3:-3]

    return text.strip()


class SlackThreadBot:
    """A Slack bot that copies thread conversations and formats them for easy sharing."""

    KEYWORD = "!copyt"  # Command to trigger thread copying
    GENSTORY_KEYWORD = "!genstory"  # Command to trigger Jira story generation
    CHECKDUPLI_KEYWORD = "!checkdupli"  # Command to trigger duplicate Jira search

    def __init__(self, bot_token: str, app_token: str):
        self.bot_token = bot_token
        self.app_token = app_token
        self.user_cache: Dict[str, str] = {}  # Cache for user display names

        # Initialize Slack app with debug logging
        self.app = App(token=bot_token, logger=logger if DEBUG_MODE else None)

        # Register event handlers
        self._register_handlers()

        logger.info("SlackThreadBot initialized")

    def _register_handlers(self):
        """Register all Slack event handlers"""

        @self.app.message(self.KEYWORD)
        def copy_thread_handler(message, client):
            self.handle_copy_thread(message, client)

        @self.app.message(self.CHECKDUPLI_KEYWORD)
        def checkdupli_handler(message, client):
            self.handle_checkdupli(message, client)

        @self.app.message(self.GENSTORY_KEYWORD)
        def genstory_handler(message, client):
            self.handle_genstory(message, client)

        @self.app.event("message")
        def handle_message_events(body):
            if DEBUG_MODE:
                logger.debug("Received message event: %s", body)

    def _send_dm(self, client, user_id: str, text: str):
        """Send a direct message to a user."""
        if not user_id:
            logger.warning("Cannot send DM, user_id is missing.")
            return
        try:
            dm_response = client.conversations_open(users=user_id)
            dm_channel = dm_response["channel"]["id"]
            client.chat_postMessage(channel=dm_channel, text=text)
            logger.info("Sent DM to user %s", user_id)
        except Exception as e:
            logger.error("Failed to send DM to user %s: %s", user_id, e)

    def _update_reaction(
        self, client, channel, ts, emoji_name, add=True
    ):
        """Add or remove a reaction from a message."""
        try:
            if add:
                client.reactions_add(channel=channel, name=emoji_name, timestamp=ts)
            else:
                client.reactions_remove(channel=channel, name=emoji_name, timestamp=ts)
        except Exception as e:
            logger.warning(
                "Failed to %s reaction '%s': %s",
                "add" if add else "remove",
                emoji_name,
                e,
            )

    def handle_copy_thread(self, message, client):  # pylint: disable=too-many-positional-arguments
        """Handle the !copyt command to copy thread conversations"""
        user = message.get("user")
        if user:
            display_name = self.get_user_display_name(client, user)
        else:
            display_name = "Unknown"
        logger.info(
            "Received %s command from user %s (ID: %s)",
            self.KEYWORD,
            display_name,
            user,
        )

        # Only process if the command is sent within a thread
        if not message.get("thread_ts"):
            logger.info("Command not sent in a thread, ignoring")
            return

        channel = message["channel"]
        ts = message["ts"]
        self._update_reaction(client, channel, ts, "hourglass_flowing_sand")

        try:
            thread_ts = message.get("thread_ts")
            # Get all replies in the thread
            result = client.conversations_replies(channel=channel, ts=thread_ts)

            if DEBUG_MODE:
                logger.debug(
                    "Retrieved %d messages from thread", len(result.get("messages", []))
                )

            messages = result.get("messages", [])
            if not messages:
                logger.warning("No messages found in thread")
                self._send_dm(client, user, "❌ No messages found in this thread.")
                self._update_reaction(client, channel, ts, "x")
                return

            # Filter out the command message itself
            filtered_messages = []
            for msg in messages:
                text = msg.get("text", "")
                # Skip messages that contain only the keyword or start with the keyword
                if text.strip() == self.KEYWORD or text.strip().startswith(
                    f"{self.KEYWORD} "
                ):
                    if DEBUG_MODE:
                        logger.debug("Filtering out command message: %s", text)
                    continue
                filtered_messages.append(msg)

            if not filtered_messages:
                logger.warning("No messages found in thread after filtering command")
                self._send_dm(
                    client,
                    user,
                    "❌ No conversation found in this thread (only command message).",
                )
                self._update_reaction(client, channel, ts, "x")
                return

            # Format messages into prompt
            prompt = self.format_thread_as_prompt(filtered_messages, client)
            if DEBUG_MODE:
                logger.debug("Formatted prompt length: %d characters", len(prompt))

            # Send as snippet to direct message
            user_id = message.get("user")
            if user_id:
                display_name = self.get_user_display_name(client, user_id)
                try:
                    # Open a DM channel with the user
                    dm_response = client.conversations_open(users=user_id)
                    dm_channel = dm_response["channel"]["id"]

                    # Create filename with timestamp for uniqueness
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"thread_conversation_{timestamp}.txt"

                    # Send as a snippet/file upload
                    client.files_upload_v2(
                        channel=dm_channel,
                        content=prompt,
                        filename=filename,
                        title="Thread Conversation",
                        snippet_type="text",
                        initial_comment="📋 Here's your thread conversation as a snippet - click to expand and copy all text!",
                    )

                    self._update_reaction(client, channel, ts, "white_check_mark")

                    logger.info(
                        "Successfully sent thread snippet to user %s via DM",
                        display_name,
                    )

                except Exception as dm_error:
                    logger.warning(
                        "Failed to send snippet to user %s id: %s: %s",
                        display_name,
                        user_id,
                        dm_error,
                    )
                    self._update_reaction(client, channel, ts, "x")
            else:
                logger.warning("No user ID found in message")
                self._update_reaction(client, channel, ts, "x")

        except Exception as e:
            error_msg = f"Error processing thread: {str(e)}"
            logger.error(error_msg, exc_info=DEBUG_MODE)
            self._send_dm(client, user, f"❌ {error_msg}")
            self._update_reaction(client, channel, ts, "x")
            if DEBUG_MODE:
                raise  # Re-raise in debug mode for full traceback
        finally:
            self._update_reaction(
                client, channel, ts, "hourglass_flowing_sand", add=False
            )

    def get_user_display_name(self, client, user_id: str) -> str:
        """Get the display name for a user ID with caching"""
        if user_id == "Unknown" or not user_id:
            return "Unknown User"

        # Check cache first
        if user_id in self.user_cache:
            if DEBUG_MODE:
                logger.debug(
                    "Using cached display name for user %s: %s",
                    user_id,
                    self.user_cache[user_id],
                )
            return self.user_cache[user_id]

        try:
            # Try to get user info from Slack API
            result = client.users_info(user=user_id)
            user_info = result.get("user", {})

            # Try display name first, then real name, then fallback to username
            display_name = (
                user_info.get("profile", {}).get("display_name")
                or user_info.get("profile", {}).get("real_name")
                or user_info.get("name")
                or f"User {user_id}"
            )

            # Cache the result
            self.user_cache[user_id] = display_name

            if DEBUG_MODE:
                logger.debug(
                    "Resolved and cached user %s to display name: %s",
                    user_id,
                    display_name,
                )

            return display_name

        except Exception as e:
            if DEBUG_MODE:
                logger.debug("Failed to get user info for %s: %s", user_id, e)

            # Cache the fallback too to avoid repeated API calls
            fallback_name = f"User {user_id}"
            self.user_cache[user_id] = fallback_name
            return fallback_name

    def format_thread_as_prompt(self, messages, client) -> str:
        """Format thread messages into a readable prompt"""
        logger.debug("Formatting %d messages into prompt", len(messages))
        formatted = []

        for i, msg in enumerate(messages):
            user_id = msg.get("user", "Unknown")
            text = msg.get("text", "")

            # Get the actual display name for the user (with caching)
            display_name = self.get_user_display_name(client, user_id)

            if DEBUG_MODE:
                logger.debug(
                    "Processing message %d: user_id=%s, display_name=%s, text_length=%d",
                    i + 1,
                    user_id,
                    display_name,
                    len(text),
                )

            # Clean up Slack formatting
            text = self.clean_slack_text(text)
            formatted.append(f"{display_name}: {text}")

        result = "\n\n".join(formatted)
        logger.debug(
            "Formatted prompt completed, total length: %d characters", len(result)
        )
        return result

    def handle_checkdupli(self, message, client):  # pylint: disable=too-many-positional-arguments
        """Handle the !checkdupli command to search for duplicate Jira issues"""
        user = message.get("user")
        if user:
            display_name = self.get_user_display_name(client, user)
        else:
            display_name = "Unknown"
        logger.info(
            "Received %s command from user %s (ID: %s)",
            self.CHECKDUPLI_KEYWORD,
            display_name,
            user,
        )

        if not message.get("thread_ts"):
            logger.info("Command not sent in a thread, ignoring")
            return

        channel = message["channel"]
        ts = message["ts"]
        self._update_reaction(client, channel, ts, "hourglass_flowing_sand")

        try:
            thread_ts = message.get("thread_ts")
            # Get all replies in the thread
            result = client.conversations_replies(channel=channel, ts=thread_ts)
            messages = result.get("messages", [])
            if not messages:
                logger.warning("No messages found in thread")
                self._send_dm(client, user, "❌ No messages found in this thread.")
                self._update_reaction(client, channel, ts, "x")
                return

            # Filter out the command message itself
            filtered_messages = []
            for msg in messages:
                text = msg.get("text", "")
                if text.strip() == self.CHECKDUPLI_KEYWORD or text.strip().startswith(
                    f"{self.CHECKDUPLI_KEYWORD} "
                ):
                    if DEBUG_MODE:
                        logger.debug(
                            "Filtering out checkdupli command message: %s", text
                        )
                    continue
                filtered_messages.append(msg)

            if not filtered_messages:
                logger.warning("No messages found in thread after filtering command")
                self._send_dm(
                    client,
                    user,
                    "❌ No conversation found in this thread (only command message).",
                )
                self._update_reaction(client, channel, ts, "x")
                return

            # Format messages into prompt for LLM
            thread_text = self.format_thread_as_prompt(filtered_messages, client)
            jira_prompt = (
                "Given the following Slack thread conversation, extract the most important keywords and search in Jira for potentially duplicate issues. "
                "List the top 5 potential duplicate Jira issues (with their keys and summaries) that might match the topic of this thread. "
                "If you cannot access Jira, just list the keywords you would use for the search.\n\n"
                f"Thread:\n{thread_text}\n\nPotential duplicate Jira issues:"
            )
            llm_response = self.call_llm(jira_prompt)
            if not llm_response:
                self._send_dm(
                    client,
                    user,
                    "❌ Failed to check for duplicate Jira issues from thread.",
                )
                self._update_reaction(client, channel, ts, "x")
                return

            # Post the result directly in the thread
            self._send_dm(
                client,
                user,
                f"🔎 Potential duplicate Jira issues (via LLM):\n{llm_response}",
            )
            self._update_reaction(client, channel, ts, "white_check_mark")
            logger.info(
                "Posted potential duplicate Jira issues for thread %s", thread_ts
            )

        except Exception as e:
            error_msg = f"Error checking for duplicate Jira issues: {str(e)}"
            logger.error(error_msg, exc_info=DEBUG_MODE)
            self._send_dm(client, user, f"❌ {error_msg}")
            self._update_reaction(client, channel, ts, "x")
            if DEBUG_MODE:
                raise  # Re-raise in debug mode for full traceback
        finally:
            self._update_reaction(
                client, channel, ts, "hourglass_flowing_sand", add=False
            )

    def handle_genstory(self, message, client):  # pylint: disable=too-many-positional-arguments
        """Handle the !genstory command to generate a Jira story from thread"""
        user = message.get("user")
        if user:
            display_name = self.get_user_display_name(client, user)
        else:
            display_name = "Unknown"
        logger.info(
            "Received %s command from user %s (ID: %s)",
            self.GENSTORY_KEYWORD,
            display_name,
            user,
        )

        if not message.get("thread_ts"):
            logger.info("Command not sent in a thread, ignoring")
            return

        channel = message["channel"]
        ts = message["ts"]
        self._update_reaction(client, channel, ts, "hourglass_flowing_sand")

        try:
            thread_ts = message.get("thread_ts")
            # Get all replies in the thread
            result = client.conversations_replies(channel=channel, ts=thread_ts)
            messages = result.get("messages", [])
            if not messages:
                logger.warning("No messages found in thread")
                self._send_dm(client, user, "❌ No messages found in this thread.")
                self._update_reaction(client, channel, ts, "x")
                return

            # Filter out the command message itself
            filtered_messages = []
            for msg in messages:
                text = msg.get("text", "")
                if text.strip() == self.GENSTORY_KEYWORD or text.strip().startswith(
                    f"{self.GENSTORY_KEYWORD} "
                ):
                    if DEBUG_MODE:
                        logger.debug("Filtering out genstory command message: %s", text)
                    continue
                filtered_messages.append(msg)

            if not filtered_messages:
                logger.warning("No messages found in thread after filtering command")
                self._send_dm(
                    client,
                    user,
                    "❌ No conversation found in this thread (only command message).",
                )
                self._update_reaction(client, channel, ts, "x")
                return

            # Format messages into prompt for LLM
            thread_text = self.format_thread_as_prompt(filtered_messages, client)
            jira_prompt = (
                "Given the following Slack thread conversation, generate a Jira story in Jira format. "
                "Include a summary, description, and acceptance criteria if possible.\n\n"
                f"Thread:\n{thread_text}\n\nJira Story:"
            )
            promptfile = os.path.dirname(__file__) + "/jira.prompt"
            if os.path.exists(promptfile):
                with open(promptfile, "r", encoding="utf8") as f:
                    jira_prompt = f.read().strip()
                    jira_prompt += "Thread:\n```\n" + thread_text + "\n```\n"

            # Call LLM (either via MCP or direct HTTP)
            llm_response = self.call_llm(jira_prompt)
            if not llm_response:
                self._send_dm(
                    client, user, "❌ Failed to generate Jira story from thread."
                )
                self._update_reaction(client, channel, ts, "x")
                return

            # Send the generated Jira story as a snippet to the user via DM
            user_id = message.get("user")
            if user_id:
                display_name = self.get_user_display_name(client, user_id)
                try:
                    dm_response = client.conversations_open(users=user_id)
                    dm_channel = dm_response["channel"]["id"]
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"jira_story_{timestamp}.txt"
                    client.files_upload_v2(
                        channel=dm_channel,
                        content=llm_response,
                        filename=filename,
                        title="Generated Jira Story",
                        snippet_type="text",
                        initial_comment="📝 Here's your generated Jira story from the thread!",
                    )
                    self._update_reaction(client, channel, ts, "white_check_mark")
                    logger.info(
                        "Successfully sent Jira story to user %s via DM",
                        display_name,
                    )
                except Exception as dm_error:
                    logger.warning(
                        "Failed to send Jira story to user %s id: %s: %s",
                        display_name,
                        user_id,
                        dm_error,
                    )
                    self._update_reaction(client, channel, ts, "x")
            else:
                logger.warning("No user ID found in message")
                self._update_reaction(client, channel, ts, "x")

        except Exception as e:
            error_msg = f"Error generating Jira story: {str(e)}"
            logger.error(error_msg, exc_info=DEBUG_MODE)
            self._send_dm(client, user, f"❌ {error_msg}")
            self._update_reaction(client, channel, ts, "x")
            if DEBUG_MODE:
                raise  # Re-raise in debug mode for full traceback
        finally:
            self._update_reaction(
                client, channel, ts, "hourglass_flowing_sand", add=False
            )

    def call_llm(self, prompt: str) -> str | None:
        """Call LLM via direct HTTP endpoint"""
        return self.call_llm_direct(prompt)

    def call_llm_direct(self, prompt: str) -> str | None:
        """Call OpenAI-compatible LLM endpoint to generate Jira story (original method)"""
        api_url = os.environ.get(
            "OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"
        )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            return None
        model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        logger.info("Calling LLM endpoint: %s with model %s", api_url, model)
        logger.info(
            "Using prompt for LLM query:\n%s\n\n", prompt[:1000]
        )  # Log first 1000 chars

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that writes Jira stories.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 800,
            "temperature": 0.2,
        }
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"]
            return clean_llm_response(raw_response)
        except Exception as e:
            logger.error("Failed to call LLM endpoint: %s", e)
            return None

    def clean_slack_text(self, text: str) -> str:
        """Remove Slack-specific formatting from text"""
        if DEBUG_MODE:
            logger.debug("Cleaning text: %s...", text[:100])

        original_text = text
        text = re.sub(r"<@[UW][A-Z0-9]+>", "[User]", text)  # User mentions
        text = re.sub(r"<#[C][A-Z0-9]+\|([^>]+)>", r"#\1", text)  # Channel mentions
        text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2", text)  # Links with text
        text = re.sub(r"<([^>]+)>", r"\1", text)  # Plain links

        if DEBUG_MODE and text != original_text:
            logger.debug("Text cleaned: '%s' -> '%s'", original_text, text)

        return text

    def validate_environment(self) -> bool:
        """Validate required environment variables"""
        required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"]
        missing_vars = []

        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(
                "Missing required environment variables: %s", ", ".join(missing_vars)
            )
            return False

        logger.info("Environment validation passed")
        return True

    def start(self):
        """Start the Slack bot"""
        logger.info("Starting Slack Thread Bot...")

        if DEBUG_MODE:
            logger.info("🐛 DEBUG MODE ENABLED")
            logger.debug("Environment variables:")
            for key in ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "DEBUG"]:
                value = os.environ.get(key, "Not set")
                if "TOKEN" in key and value != "Not set":
                    value = f"{value[:10]}..." if len(value) > 10 else value
                logger.debug("  %s: %s", key, value)

        if not self.validate_environment():
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)

        try:
            # Test the bot token first
            try:
                test_response = self.app.client.auth_test()
                logger.info(
                    "✅ Bot authenticated as: %s", test_response.get("user", "Unknown")
                )
                logger.info("✅ Team: %s", test_response.get("team", "Unknown"))
            except Exception as e:
                logger.error("❌ Bot token authentication failed: %s", e)
                raise

            handler = SocketModeHandler(self.app, self.app_token)
            logger.info("Socket mode handler created successfully")
            logger.info("🚀 Bot is starting...")
            logger.info(
                "📝 Send '%s' in any Slack thread to copy conversation!", self.KEYWORD
            )
            logger.info(
                "🤖 Send '%s' in any Slack thread for LLM story generation!",
                self.GENSTORY_KEYWORD,
            )

            logger.info("🌐 Using direct HTTP for LLM calls")

            logger.info("🐛 All messages will be logged in debug mode")
            logger.info("💾 User display names will be cached to improve performance")

            handler.start()

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error("Failed to start bot: %s", str(e), exc_info=DEBUG_MODE)
            sys.exit(1)


def main():
    """Main entry point"""
    bot_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")

    if not bot_token or not app_token:
        logger.error(
            "Missing required environment variables: SLACK_BOT_TOKEN and/or SLACK_APP_TOKEN"
        )
        sys.exit(1)

    bot = SlackThreadBot(bot_token, app_token)
    bot.start()


if __name__ == "__main__":
    main()
