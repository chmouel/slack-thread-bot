# Slack Thread Bot

A Slack bot that copies thread conversations and formats them for easy sharing.

## Features

- Copy entire thread conversations with the `!copy-thread` command
- Clean formatting that removes Slack-specific markup
- Debug mode for troubleshooting
- Comprehensive logging

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your actual Slack tokens
   ```

3. Run the bot:

   ```bash
   python main.py
   ```

## Environment Variables

- `SLACK_BOT_TOKEN`: Your Slack bot token (starts with `xoxb-`)
- `SLACK_APP_TOKEN`: Your Slack app token (starts with `xapp-`)  
- `DEBUG`: Enable debug mode (`true`, `1`, `yes`, `on` for enabled)

## Debug Mode

Enable debug mode by setting the `DEBUG` environment variable:

```bash
export DEBUG=true
python main.py
```

Or run with debug mode temporarily:

```bash
DEBUG=true python main.py
```

Debug mode provides:

- Detailed logging to console and `slack-bot.log` file
- Step-by-step processing information
- Environment variable validation
- Enhanced error messages with stack traces

## Usage

1. Invite the bot to your Slack channel
2. In any message thread, type `!copy-thread`
3. The bot will format and return the entire thread conversation
4. Copy the formatted text for sharing elsewhere

## Troubleshooting

If the bot doesn't respond:

1. Enable debug mode to see detailed logs
2. Check that environment variables are set correctly
3. Verify the bot has proper permissions in your Slack workspace
4. Check the console output for error messages
