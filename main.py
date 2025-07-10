import datetime
import logging
import os
import re
import subprocess
import sys
from typing import Dict

import google.generativeai as genai
import requests
from airporttime import AirportTime
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
DEFAULT_TIMEZONES = (
    "Bangalore/Asia/Kolkata,Paris/Europe/Paris,Boston/America/New_York,Tel Aviv/Asia/Jerusalem",
)
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
    ACTIONS_KEYWORD = "!actions"  # Command to trigger action item extraction
    TZ_KEYWORD = "!tz"  # Command to trigger timezone conversion

    def __init__(self, bot_token: str, app_token: str):
        self.bot_token = bot_token
        self.app_token = app_token
        self.user_cache: Dict[str, str] = {}  # Cache for user display names
        self.user_tz_cache: Dict[str, str] = {}  # Cache for user timezones
        self.team_url: str | None = None
        self._date_command: str | None = None

        # Initialize Slack app with debug logging
        self.app = App(token=bot_token, logger=logger if DEBUG_MODE else None)

        # Register event handlers
        self._register_handlers()

        logger.info("SlackThreadBot initialized")

    def _register_handlers(self):
        """Register all Slack event handlers"""

        @self.app.event("app_home_opened")
        def handle_app_home_opened_events(client, event, logger):
            """Handle the app_home_opened event to display a help interface."""
            user_id = event["user"]
            logger.info(f"App home opened by user {user_id}")
            try:
                help_blocks = self._get_help_blocks(user_id)
                client.views_publish(
                    user_id=user_id,
                    view={
                        "type": "home",
                        "blocks": help_blocks,
                    },
                )
                logger.info(f"Successfully published App Home view for user {user_id}")
            except Exception as e:
                logger.error(f"Error publishing App Home view for user {user_id}: {e}")

        @self.app.action("show_tz_examples")
        def handle_show_tz_examples(ack, body, client, logger):
            """Handle the 'show_tz_examples' button click and open a modal."""
            ack()
            try:
                logger.info(
                    "Opening !tz examples modal for user %s", body["user"]["id"]
                )
                client.views_open(
                    trigger_id=body["trigger_id"],
                    view={
                        "type": "modal",
                        "title": {
                            "type": "plain_text",
                            "text": f"{self.TZ_KEYWORD} Examples",
                        },
                        "close": {"type": "plain_text", "text": "Close"},
                        "blocks": [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"The `{self.TZ_KEYWORD}` command is highly flexible. Here are some examples to get you started:",
                                },
                            },
                            {"type": "divider"},
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": "*Basic Conversions:*\n"
                                    "• `!tz 10h00`\n"
                                    "• `!tz 10:30 tomorrow`\n"
                                    "• `!tz 5pm next monday`\n"
                                    "• `!tz now` (or just `!tz`)",
                                },
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": "*Using Airport Codes (IATA):*\n"
                                    "• `!tz 10h00 BLR,CDG`\n"
                                    "• `!tz tomorrow 10:00 JFK`\n"
                                    "• `!tz 2pm SFO`",
                                },
                            },
                            {
                                "type": "context",
                                "elements": [
                                    {
                                        "type": "mrkdwn",
                                        "text": "You can use multiple airport codes separated by spaces or commas.",
                                    }
                                ],
                            },
                        ],
                    },
                )
            except Exception as e:
                logger.error("Failed to open modal: %s", e)

        @self.app.message(re.compile(f"^{re.escape(self.KEYWORD)}"))
        def copy_thread_handler(message, client):
            self.handle_copy_thread(message, client)

        @self.app.message(re.compile(f"^{re.escape(self.GENSTORY_KEYWORD)}"))
        def genstory_handler(message, client):
            self.handle_genstory(message, client)

        @self.app.message(re.compile(f"^{re.escape(self.ACTIONS_KEYWORD)}"))
        def actions_handler(message, client):
            self.handle_actions(message, client)

        @self.app.message(re.compile(f"^{re.escape(self.TZ_KEYWORD)}"))
        def tz_handler(message, client):
            self.handle_tz(message, client)

        @self.app.event("message")
        def handle_message_events(body):
            if DEBUG_MODE:
                logger.debug("Received message event: %s", body)

    def _send_error_with_help(self, client, user_id: str, error_message: str):
        """Send an error message to a user, along with detailed help."""
        if not user_id:
            logger.warning("Cannot send error DM, user_id is missing.")
            return
        try:
            dm_response = client.conversations_open(users=user_id)
            dm_channel = dm_response["channel"]["id"]

            error_block = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"❌ {error_message}",
                },
            }

            help_blocks = self._get_help_blocks()

            client.chat_postMessage(
                channel=dm_channel,
                blocks=[error_block] + help_blocks,
                text=f"Error: {error_message}",
            )
            logger.info("Sent error DM with help to user %s", user_id)
        except Exception as e:
            logger.error("Failed to send error DM to user %s: %s", user_id, e)

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

    def _update_reaction(self, client, channel, ts, emoji_name, add=True):
        # pylint: disable=too-many-nested-blocks
        try:
            if add:
                client.reactions_add(channel=channel, name=emoji_name, timestamp=ts)
            else:
                # Remove only if the bot user added the reaction
                user_id = client.auth_test()["user_id"]
                reactions_info = client.reactions_get(channel=channel, timestamp=ts)
                message = reactions_info.get("message", {})
                reactions = message.get("reactions", [])
                for reaction in reactions:
                    if reaction.get("name") == emoji_name:
                        for user in reaction.get("users", []):
                            if user == user_id:
                                client.reactions_remove(
                                    channel=channel, name=emoji_name, timestamp=ts
                                )
                                break
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

            # Add thread URL to the top of the prompt
            thread_link = self._get_thread_link(channel, thread_ts)
            if thread_link:
                logger.info("Generated thread link: %s", thread_link)
                prompt = f"Source Thread: {thread_link}\n{prompt}"

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
                        initial_comment=(
                            f"📋 Here’s your conversation from this <{thread_link}|thread> as a snippet. Click to expand and copy all text!"
                            if thread_link
                            else "📋 Here’s your conversation as a snippet. Click to expand and copy all text!"
                        ),
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

    def _get_thread_link(self, channel_id: str, thread_ts: str) -> str | None:
        """Construct the permanent URL for a Slack thread."""
        if not self.team_url:
            logger.warning("Team URL not available, cannot generate thread link.")
            return None
        # Format timestamp by removing the dot
        formatted_ts = thread_ts.replace(".", "")
        return f"{self.team_url}archives/{channel_id}/p{formatted_ts}"

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

    def get_user_timezone(self, client, user_id: str) -> str | None:
        """Get the timezone for a user ID with caching."""
        if not user_id or user_id == "Unknown":
            return None

        if user_id in self.user_tz_cache:
            logger.debug(
                "Using cached timezone for user %s: %s",
                user_id,
                self.user_tz_cache[user_id],
            )
            return self.user_tz_cache[user_id]

        try:
            result = client.users_info(user=user_id)
            user_info = result.get("user", {})
            timezone = user_info.get("tz")

            if timezone:
                self.user_tz_cache[user_id] = timezone
                logger.debug(
                    "Resolved and cached user %s timezone: %s", user_id, timezone
                )
                return timezone

        except Exception as e:
            logger.debug("Failed to get timezone for user %s: %s", user_id, e)

        return None

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

            # Format messages into prompt for LLM
            thread_text = self.format_thread_as_prompt(messages, client)
            thread_link = self._get_thread_link(channel, thread_ts)
            if thread_link:
                logger.info("Generated thread link: %s", thread_link)

            jira_prompt = (
                "Given the following Slack thread conversation, generate a Jira story in Jira format. "
                "Include a summary, description, and acceptance criteria if possible.\n\n"
            )

            jira_prompt += f"Thread:\n{thread_text}\n\nJira Story:"

            promptfile = os.path.dirname(__file__) + "/jira.prompt"
            if os.path.exists(promptfile):
                with open(promptfile, "r", encoding="utf8") as f:
                    jira_prompt = f.read().strip()
                    jira_prompt += "\n\nThread:\n```\n" + thread_text + "\n```\n"

            # Call LLM (either via MCP or direct HTTP)
            llm_response = self.call_llm(jira_prompt)
            if not llm_response:
                self._send_dm(
                    client, user, "❌ Failed to generate Jira story from thread."
                )
                self._update_reaction(client, channel, ts, "x")
                return

            # Cache the story if the directory is set
            cache_dir = os.environ.get("CACHE_STORY_DIR")
            if cache_dir:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    story_filename = os.path.join(cache_dir, f"story-{timestamp}.md")
                    with open(story_filename, "w", encoding="utf-8") as f:
                        f.write(llm_response)
                    logger.info("User story cached to %s", story_filename)
                except Exception as e:
                    logger.error("Failed to cache user story: %s", e)

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
                        initial_comment=(
                            f"📝 Here's your generated Jira story from this <{thread_link}|thread>!"
                            if thread_link
                            else "📝 Here's your generated Jira story from the thread!"
                        ),
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

    def handle_actions(self, message, client):  # pylint: disable=too-many-positional-arguments
        """Handle the !actions command to extract action items from a thread."""
        user = message.get("user")
        if user:
            display_name = self.get_user_display_name(client, user)
        else:
            display_name = "Unknown"
        logger.info(
            "Received %s command from user %s (ID: %s)",
            self.ACTIONS_KEYWORD,
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

            # Format messages into prompt for LLM
            thread_text = self.format_thread_as_prompt(messages, client)

            promptfile = os.path.dirname(__file__) + "/actions.prompt"
            if os.path.exists(promptfile):
                with open(promptfile, "r", encoding="utf8") as f:
                    actions_prompt = f.read().strip()
                    actions_prompt = actions_prompt.replace(
                        "{thread_text}", thread_text
                    )

            # Call LLM
            llm_response = self.call_llm(actions_prompt)
            if not llm_response:
                self._send_dm(
                    client, user, "❌ Failed to extract action items from thread."
                )
                self._update_reaction(client, channel, ts, "x")
                return

            # Format the response as blocks
            blocks = self._format_actions_as_blocks(llm_response)

            # Post the response to the thread
            client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, blocks=blocks, text=llm_response
            )
            self._update_reaction(client, channel, ts, "white_check_mark")
            logger.info(
                "Successfully sent action items to user %s in channel %s",
                display_name,
                channel,
            )

        except Exception as e:
            error_msg = f"Error extracting action items: {str(e)}"
            logger.error(error_msg, exc_info=DEBUG_MODE)
            self._send_dm(client, user, f"❌ {error_msg}")
            self._update_reaction(client, channel, ts, "x")
            if DEBUG_MODE:
                raise  # Re-raise in debug mode for full traceback
        finally:
            self._update_reaction(
                client, channel, ts, "hourglass_flowing_sand", add=False
            )

    def handle_tz(self, message, client):
        """Handle the !tz command to convert timezones."""
        user_id = message.get("user")
        display_name = self.get_user_display_name(client, user_id)
        logger.info(
            "Received %s command from user %s (ID: %s)",
            self.TZ_KEYWORD,
            display_name,
            user_id,
        )

        channel = message["channel"]
        ts = message["ts"]
        thread_ts = message.get("thread_ts", ts)

        command_text = message.get("text", "").strip()
        args_str = command_text[len(self.TZ_KEYWORD) :].strip()

        time_str = "now"
        airport_codes = []

        if args_str:
            # Regex to find all 3-letter words at the end of the string, separated by spaces or commas
            match = re.search(
                r"((?:\s|,|(?<=,)\s*)([A-Z]{3}))+$", args_str, re.IGNORECASE
            )
            if match:
                codes_str = match.group(0).strip()
                # Clean and validate codes
                potential_codes = [c for c in re.split(r"[\s,]+", codes_str) if c]
                if all(
                    re.fullmatch(r"[A-Z]{3}", code, re.IGNORECASE)
                    for code in potential_codes
                ):
                    airport_codes = [c.upper() for c in potential_codes]
                    time_str = args_str[: match.start()].strip() or "now"
                else:
                    time_str = args_str
            else:
                time_str = args_str

        # Convert 10h00 to 10:00 for compatibility
        time_str = re.sub(r"(\d+)h(\d+)", r"\1:\2", time_str)

        self._update_reaction(client, channel, ts, "hourglass_flowing_sand")

        try:
            # Get the user's timezone, falling back to environment or UTC
            user_tz = self.get_user_timezone(client, user_id)
            base_tz = user_tz or os.environ.get("BATZ_BASE_TZ", "UTC")

            timezones = {}
            invalid_codes = []
            if airport_codes:
                for code in airport_codes:
                    try:
                        airport_time = AirportTime(code)
                        timezones[code] = getattr(airport_time.airport, "timezone")
                    except Exception:
                        logger.warning("Invalid airport code: %s", code)
                        invalid_codes.append(code)

                if invalid_codes:
                    self._send_dm(
                        client,
                        user_id,
                        f"❌ Invalid airport code(s): {', '.join(invalid_codes)}. Please use valid IATA airport codes.",
                    )

                if not timezones:
                    self._update_reaction(client, channel, ts, "x")
                    return
            else:
                timezones_str = os.environ.get(
                    "BATZ_TIMEZONES",
                    DEFAULT_TIMEZONES,
                )
                timezones = dict(
                    item.split("/", 1) for item in timezones_str.split(",")
                )

            # Get emoji mappings from environment
            emojis_str = os.environ.get("BATZ_EMOJIS", "")
            emojis = (
                dict(item.split("/", 1) for item in emojis_str.split(","))
                if emojis_str
                else {}
            )

            results = []
            for name, tz in timezones.items():
                converted_time, error = self._convert_time_with_date(
                    time_str, base_tz, tz
                )
                if error:
                    raise ValueError(f"Failed to convert time for {tz}: {error}")
                results.append({"name": name, "time": converted_time, "tz": tz})

            # Format and send the response
            response_text = self._format_tz_results(results, emojis)
            client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text=response_text
            )
            self._update_reaction(client, channel, ts, "white_check_mark", add=False)
            logger.info(
                "Successfully sent timezone conversion to user %s in channel %s",
                display_name,
                channel,
            )

        except Exception as e:
            error_msg = f"Error converting timezones: {e}"
            logger.error(error_msg, exc_info=DEBUG_MODE)
            self._send_dm(client, user_id, f"❌ {error_msg}")
            self._update_reaction(client, channel, ts, "x")
        finally:
            self._update_reaction(
                client, channel, ts, "hourglass_flowing_sand", add=False
            )

    def _convert_time_with_date(self, time_str, base_tz, target_tz):
        """Convert time using GNU date command, detecting gdate if available."""
        # Try to find GNU date command
        date_cmd = "date"
        date_format = "+%a %Y-%m-%d %H:%M:%S %Z"

        try:
            if time_str and time_str.strip():
                sanitized_time = self._sanitize_time_input(time_str.strip())
                if not sanitized_time:
                    return None, "Invalid time format provided"

                command = [
                    date_cmd,
                    f'--date=TZ="{base_tz}" {sanitized_time}',
                    date_format,
                ]
                env = {"TZ": target_tz}
            else:
                # Just show current time in target timezone
                # Equivalent to: TZ="${target_tz}" date "+${date_format}"
                command = [date_cmd, date_format]
                env = {"TZ": target_tz}

            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            return process.stdout.strip(), None
        except FileNotFoundError:
            return (
                None,
                f"`{date_cmd}` not found. Please install GNU date (e.g., `brew install coreutils` on macOS).",
            )
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip()
            return None, f"Error executing `{date_cmd}`: {error_message}"

    def _sanitize_time_input(self, time_str: str) -> str | None:
        """Sanitize time input to prevent shell injection."""
        if not time_str:
            return None

        # Allow patterns like: "now", "10:30", "2024-01-15 10:30", "yesterday", "today", "tomorrow"
        # "1 hour ago", "2 days ago", "+1 hour", "-30 minutes", etc.
        allowed_pattern = r"^[a-zA-Z0-9\s:.\-+/]+$"

        if not re.match(allowed_pattern, time_str):
            logger.warning("Potentially unsafe time input blocked: %s", time_str)
            return None

        # Additional validation for common time formats
        dangerous_chars = [
            ";",
            "&",
            "|",
            "`",
            "$",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            ">",
            "<",
            "\\",
            '"',
            "'",
        ]
        if any(char in time_str for char in dangerous_chars):
            logger.warning("Dangerous characters detected in time input: %s", time_str)
            return None

        # Limit length to prevent buffer overflow attempts
        if len(time_str) > 100:
            logger.warning("Time input too long: %s", time_str)
            return None

        return time_str

    def _get_help_blocks(self, user_id: str | None = None) -> list:
        """Generate a detailed help message with all commands."""
        welcome_text = (
            f"👋 Hi <@{user_id}>! I'm your friendly Slack Thread Bot."
            if user_id
            else "Slack Thread Bot Help"
        )
        return [
            {
                "type": "image",
                "image_url": "https://storage.googleapis.com/chmouel-public-images/slack-thread-bot/slack-thread-bot.png",
                "alt_text": "A friendly robot banner",
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{welcome_text}\nI'm here to help you work with Slack threads more efficiently. Here are the commands you can use:",
                },
            },
            {"type": "divider"},
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📋 Core Commands",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*`{self.KEYWORD}`*\nCopies the entire thread and sends it to you in a direct message as a text snippet.",
                },
                "accessory": {
                    "type": "image",
                    "image_url": "https://storage.googleapis.com/chmouel-public-images/slack-thread-bot/copy.png",
                    "alt_text": "Copy icon",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*`{self.GENSTORY_KEYWORD}`*\nUses AI to generate a Jira story from the thread and DMs it to you.",
                },
                "accessory": {
                    "type": "image",
                    "image_url": "https://storage.googleapis.com/chmouel-public-images/slack-thread-bot/ai.png",
                    "alt_text": "AI icon",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*`{self.ACTIONS_KEYWORD}`*\nUses AI to extract a concise list of action items and posts them as a reply in the thread.",
                },
                "accessory": {
                    "type": "image",
                    "image_url": "https://storage.googleapis.com/chmouel-public-images/slack-thread-bot/checklist.png",
                    "alt_text": "Checklist icon",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*`{self.TZ_KEYWORD}`*\nConverts a time to different timezones.",
                },
                "accessory": {
                    "type": "image",
                    "image_url": "https://storage.googleapis.com/chmouel-public-images/slack-thread-bot/timezone.png",
                    "alt_text": "Timezone icon",
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Show Timezone Examples",
                            "emoji": True,
                        },
                        "action_id": "show_tz_examples",
                    }
                ],
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "For more details, check out the <https://github.com/chmouel/slack-thread-bot|source code on GitHub>.",
                    }
                ],
            },
        ]

    def _format_actions_as_blocks(self, llm_response: str) -> list:
        """Format the action items into Slack Block Kit."""
        if not llm_response or llm_response.strip().lower() == "no action items found.":
            return [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "No action items found.",
                    },
                }
            ]

        action_items = [
            item.strip() for item in llm_response.split("\n") if item.strip()
        ]
        blocks = []
        for item in action_items:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• {item}",
                    },
                }
            )
        return blocks

    def _format_tz_results(self, results, emojis):
        """Format the timezone conversion results for Slack."""
        lines = []
        for res in results:
            emoji = emojis.get(res["name"], "")
            emoji_space = " " if emoji else ""
            lines.append(f"• {emoji}{emoji_space}*{res['name']}*: {res['time']}")
        return "\n".join(lines)

    def call_llm(self, prompt: str) -> str | None:
        """Call the configured LLM provider."""
        llm_provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        if llm_provider == "gemini":
            return self.call_gemini(prompt)
        if llm_provider == "openai":
            return self.call_openai(prompt)

        logger.error("Invalid LLM_PROVIDER configured: %s", llm_provider)
        return None

    def call_openai(self, prompt: str) -> str | None:
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
        if DEBUG_MODE:
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

    def call_gemini(self, prompt: str) -> str | None:
        """Call the Gemini API."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            return None
        genai.configure(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        logger.info("Calling Gemini with model %s", model_name)
        try:
            response = model.generate_content(prompt)
            if DEBUG_MODE:
                logger.info("Raw Gemini response: %s", response.text)
            return response.text
        except Exception as e:
            logger.error("Failed to call Gemini API: %s", e)
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
        llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()

        if llm_provider == "openai":
            required_vars.append("OPENAI_API_KEY")
        elif llm_provider == "gemini":
            required_vars.append("GEMINI_API_KEY")
        else:
            logger.error("Invalid LLM_PROVIDER: %s", llm_provider)
            return False

        missing_vars = [v for v in required_vars if not os.environ.get(v)]
        if missing_vars:
            logger.error(
                "Missing required environment variables: %s", ", ".join(missing_vars)
            )
            return False

        logger.info("Environment validation passed for %s provider", llm_provider)
        return True

    def start(self):
        """Start the Slack bot"""
        logger.info("Starting Slack Thread Bot...")

        if DEBUG_MODE:
            logger.info("🐛 DEBUG MODE ENABLED")
            logger.debug("Environment variables:")
            for key in ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "DEBUG", "LLM_PROVIDER"]:
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
                self.team_url = test_response.get("url")
                logger.info(
                    "✅ Bot authenticated as: %s", test_response.get("user", "Unknown")
                )
                logger.info("✅ Team: %s", test_response.get("team", "Unknown"))
                logger.info("✅ Team URL: %s", self.team_url)
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
            logger.info(
                "✅ Send '%s' in any Slack thread to extract action items!",
                self.ACTIONS_KEYWORD,
            )
            logger.info(
                "🕒 Send '%s' in any Slack thread to convert timezones!",
                self.TZ_KEYWORD,
            )

            llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()
            logger.info("🌐 Using %s for LLM calls", llm_provider)

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
