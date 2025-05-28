import logging
import os
import sys
from typing import Dict, Optional
import re

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Configure logging
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes", "on")
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("slack-bot.log") if DEBUG_MODE else logging.NullHandler(),
    ],
)

logger = logging.getLogger(__name__)


class SlackThreadBot:
    """A Slack bot that copies thread conversations and formats them for easy sharing."""

    KEYWORD = "!copyt"  # Command to trigger thread copying

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
        def copy_thread_handler(message, say, client):
            self.handle_copy_thread(message, say, client)

        @self.app.event("message")
        def handle_message_events(body, logger):
            if DEBUG_MODE:
                logger.debug(f"Received message event: {body}")

    def handle_copy_thread(self, message, say, client):
        """Handle the !copyt command to copy thread conversations"""
        logger.info(
            f"Received {self.KEYWORD} command from user {message.get('user', 'Unknown')}"
        )

        try:
            thread_ts = message.get("thread_ts") or message["ts"]
            channel = message["channel"]

            if DEBUG_MODE:
                logger.debug(
                    f"Processing thread with ts: {thread_ts} in channel: {channel}"
                )
                logger.debug(f"Original message: {message}")

            # Get all replies in the thread
            result = client.conversations_replies(channel=channel, ts=thread_ts)

            if DEBUG_MODE:
                logger.debug(
                    f"Retrieved {len(result.get('messages', []))} messages from thread"
                )

            messages = result.get("messages", [])
            if not messages:
                logger.warning("No messages found in thread")
                say("❌ No messages found in this thread.")
                return

            # Format messages into prompt
            prompt = self.format_thread_as_prompt(messages, client)

            if DEBUG_MODE:
                logger.debug(f"Formatted prompt length: {len(prompt)} characters")

            # Send formatted prompt back
            response = f"```\n{prompt}\n```\nCopy the text above ☝️"

            if DEBUG_MODE:
                logger.debug(f"Sending response: {response[:100]}...")

            # Reply in the same thread if the command was sent in a thread
            if message.get("thread_ts"):
                say(response, thread_ts=message.get("thread_ts"))
            else:
                # If the command was not in a thread, create a thread by replying to the original message
                say(response, thread_ts=message["ts"])

            logger.info("Successfully processed thread copy command")

        except Exception as e:
            error_msg = f"Error processing thread: {str(e)}"
            logger.error(error_msg, exc_info=DEBUG_MODE)
            say(f"❌ {error_msg}")
            if DEBUG_MODE:
                raise  # Re-raise in debug mode for full traceback

    def get_user_display_name(self, client, user_id: str) -> str:
        """Get the display name for a user ID with caching"""
        if user_id == "Unknown" or not user_id:
            return "Unknown User"

        # Check cache first
        if user_id in self.user_cache:
            if DEBUG_MODE:
                logger.debug(
                    f"Using cached display name for user {user_id}: {self.user_cache[user_id]}"
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
                    f"Resolved and cached user {user_id} to display name: {display_name}"
                )

            return display_name

        except Exception as e:
            if DEBUG_MODE:
                logger.debug(f"Failed to get user info for {user_id}: {e}")

            # Cache the fallback too to avoid repeated API calls
            fallback_name = f"User {user_id}"
            self.user_cache[user_id] = fallback_name
            return fallback_name

    def format_thread_as_prompt(self, messages, client) -> str:
        """Format thread messages into a readable prompt"""
        logger.debug(f"Formatting {len(messages)} messages into prompt")
        formatted = []

        for i, msg in enumerate(messages):
            user_id = msg.get("user", "Unknown")
            text = msg.get("text", "")

            # Get the actual display name for the user (with caching)
            display_name = self.get_user_display_name(client, user_id)

            if DEBUG_MODE:
                logger.debug(
                    f"Processing message {i + 1}: user_id={user_id}, display_name={display_name}, text_length={len(text)}"
                )

            # Clean up Slack formatting
            text = self.clean_slack_text(text)
            formatted.append(f"{display_name}: {text}")

        result = "\n\n".join(formatted)
        logger.debug(
            f"Formatted prompt completed, total length: {len(result)} characters"
        )
        return result

    def clean_slack_text(self, text: str) -> str:
        """Remove Slack-specific formatting from text"""
        if DEBUG_MODE:
            logger.debug(f"Cleaning text: {text[:100]}...")

        original_text = text
        text = re.sub(r"<@[UW][A-Z0-9]+>", "[User]", text)  # User mentions
        text = re.sub(r"<#[C][A-Z0-9]+\|([^>]+)>", r"#\1", text)  # Channel mentions
        text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2", text)  # Links with text
        text = re.sub(r"<([^>]+)>", r"\1", text)  # Plain links

        if DEBUG_MODE and text != original_text:
            logger.debug(f"Text cleaned: '{original_text}' -> '{text}'")

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
                f"Missing required environment variables: {', '.join(missing_vars)}"
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
                logger.debug(f"  {key}: {value}")

        if not self.validate_environment():
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)

        try:
            # Test the bot token first
            try:
                test_response = self.app.client.auth_test()
                logger.info(
                    f"✅ Bot authenticated as: {test_response.get('user', 'Unknown')}"
                )
                logger.info(f"✅ Team: {test_response.get('team', 'Unknown')}")
            except Exception as e:
                logger.error(f"❌ Bot token authentication failed: {e}")
                raise

            handler = SocketModeHandler(self.app, self.app_token)
            logger.info("Socket mode handler created successfully")
            logger.info("🚀 Bot is starting...")
            logger.info(f"📝 Send '{self.KEYWORD}' in any Slack channel to test!")
            logger.info("🐛 All messages will be logged in debug mode")
            logger.info(f"💾 User display names will be cached to improve performance")
            handler.start()

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}", exc_info=DEBUG_MODE)
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
