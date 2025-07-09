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
- `LLM_PROVIDER`: The desired LLM provider. Can be `openai` (default) or `gemini`.
- `OPENAI_API_KEY`: Your OpenAI API key (if using `openai`).
- `OPENAI_MODEL`: The OpenAI model to use (e.g., `gpt-3.5-turbo`).
- `GEMINI_API_KEY`: Your Gemini API key (if using `gemini`).
- `GEMINI_MODEL`: The Gemini model to use (e.g., `gemini-pro`).

## LLM Integration

This bot can integrate with either OpenAI or Gemini to generate Jira stories from Slack threads using the `!genstory` command.

To configure the LLM provider, set the `LLM_PROVIDER` environment variable to either `openai` or `gemini`.

### OpenAI

If `LLM_PROVIDER` is set to `openai` (or is unset), you must provide:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_MODEL`: (Optional) The model to use (defaults to `gpt-3.5-turbo`).

### Gemini

If `LLM_PROVIDER` is set to `gemini`, you must provide:

- `GEMINI_API_KEY`: Your Google AI Studio API key.
- `GEMINI_MODEL`: (Optional) The model to use (defaults to `gemini-pro`).

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
