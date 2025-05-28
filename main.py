import logging
import os
import sys

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

KEYWORD = "!copyt"  # Command to trigger thread copying
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

# Initialize Slack app with debug logging
app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"), logger=logger if DEBUG_MODE else None
)


# # Debug handler to catch ALL messages (only in debug mode)
# @app.message(".*")
# def debug_all_messages(message, say, client):
#     print(
#         f"📨 Received message: {message.get('text', 'No text')} from user {message.get('user', 'Unknown')}"
#     )
#     print(f"📨 Full message object: {message}")


@app.message(KEYWORD)
def copy_thread_handler(message, say, client):
    logger.info(
        f"Received !copy-thread command from user {message.get('user', 'Unknown')}"
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
        prompt = format_thread_as_prompt(messages)

        if DEBUG_MODE:
            logger.debug(f"Formatted prompt length: {len(prompt)} characters")

        # Send formatted prompt back
        response = f"```\n{prompt}\n```\nCopy the text above ☝️"

        if DEBUG_MODE:
            logger.debug(f"Sending response: {response[:100]}...")

        say(response)
        logger.info("Successfully processed !copy-thread command")

    except Exception as e:
        error_msg = f"Error processing thread: {str(e)}"
        logger.error(error_msg, exc_info=DEBUG_MODE)
        say(f"❌ {error_msg}")
        if DEBUG_MODE:
            raise  # Re-raise in debug mode for full traceback


@app.event("message")
def handle_message_events(body, logger):
    if DEBUG_MODE:
        logger.debug(f"Received message event: {body}")


def format_thread_as_prompt(messages):
    logger.debug(f"Formatting {len(messages)} messages into prompt")
    formatted = []

    for i, msg in enumerate(messages):
        user = msg.get("user", "Unknown")
        text = msg.get("text", "")

        if DEBUG_MODE:
            logger.debug(
                f"Processing message {i + 1}: user={user}, text_length={len(text)}"
            )

        # Clean up Slack formatting
        text = clean_slack_text(text)
        formatted.append(f"User {user}: {text}")

    result = "\n\n".join(formatted)
    logger.debug(f"Formatted prompt completed, total length: {len(result)} characters")
    return result


def clean_slack_text(text):
    # Remove Slack-specific formatting
    import re

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


def validate_environment():
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


if __name__ == "__main__":
    logger.info("Starting Slack Thread Bot...")

    if DEBUG_MODE:
        logger.info("🐛 DEBUG MODE ENABLED")
        logger.debug("Environment variables:")
        for key in ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "DEBUG"]:
            value = os.environ.get(key, "Not set")
            if "TOKEN" in key and value != "Not set":
                value = f"{value[:10]}..." if len(value) > 10 else value
            logger.debug(f"  {key}: {value}")

    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        sys.exit(1)

    try:
        # Test the bot token first
        try:
            test_response = app.client.auth_test()
            logger.info(
                f"✅ Bot authenticated as: {test_response.get('user', 'Unknown')}"
            )
            logger.info(f"✅ Team: {test_response.get('team', 'Unknown')}")
        except Exception as e:
            logger.error(f"❌ Bot token authentication failed: {e}")
            raise

        handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
        logger.info("Socket mode handler created successfully")
        logger.info("🚀 Bot is starting...")
        logger.info(f"📝 Send '{KEYWORD}' in any Slack channel to test!")
        logger.info("🐛 All messages will be logged in debug mode")
        handler.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}", exc_info=DEBUG_MODE)
        sys.exit(1)
