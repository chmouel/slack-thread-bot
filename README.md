# Slack Thread Bot

A Slack bot that copies thread conversations, generates Jira stories, and formats them for easy sharing.

## ✨ Features

- Copy entire thread conversations with the `!copyt` command.
- Generate a Jira story from a thread with the `!genstory` command using OpenAI or Gemini.
- Includes a permanent link to the original Slack thread in the output.
- Cleans formatting by removing Slack-specific markup.
- Debug mode for easy troubleshooting.
- Comprehensive logging.

## 🚀 Setup & Running

1. **Clone the repository:**

    ```bash
    git clone https://github.com/chmouel/slack-thread-bot.git
    cd slack-thread-bot
    ```

2. **Install dependencies:**
    This project uses `uv` for package management.

    ```bash
    uv sync
    ```

3. **Configure environment variables:**
    Copy the example `.env` file and fill in your Slack and LLM provider tokens.

    ```bash
    cp .env.example .env
    ```

    Then edit `.env` with your actual tokens. See the "Environment Variables" section for details.

4. **Run the bot:**

    ```bash
    uv run main.py
    ```

    The bot will connect to Slack and be ready for commands!

## ⚙️ Environment Variables

- `SLACK_BOT_TOKEN`: Your Slack bot token (starts with `xoxb-`).
- `SLACK_APP_TOKEN`: Your Slack app token (starts with `xapp-`).
- `DEBUG`: Enable debug mode (`true`, `1`, `yes`, `on` for enabled).
- `LLM_PROVIDER`: The desired LLM provider. Can be `openai` or `gemini` (default).
- `OPENAI_API_KEY`: Your OpenAI API key (if using `openai`).
- `OPENAI_MODEL`: The OpenAI model to use (e.g., `gpt-3.5-turbo`).
- `GEMINI_API_KEY`: Your Gemini API key (if using `gemini`).
- `GEMINI_MODEL`: The Gemini model to use (defaults to `gemini-2.5-flash`).
- `CACHE_STORY_DIR`: (Optional) If set, the generated user story will be cached to this directory.
- `BATZ_BASE_TZ`: (Optional) The base timezone for the `!tz` command. Defaults to `UTC`.
- `BATZ_TIMEZONES`: (Optional) A comma-separated list of timezones for the `!tz` command, in `Name/Area/City` format.
- `BATZ_EMOJIS`: (Optional) A comma-separated list of city-to-emoji mappings for the `!tz` command, in `Name/emoji` format.

## 🤖 LLM Integration

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

## 🛠️ Usage

1. Invite the bot to your Slack channel.
2. In any message thread, use one of the following commands:
    - `!copyt`: The bot will format and send you a direct message with the entire thread conversation, including a link to the original thread.
    - `!genstory`: The bot will use an LLM (OpenAI or Gemini) to generate a Jira story from the thread conversation and send it to you as a snippet in a direct message. The story will also include a link to the original thread.
    - `!tz <time>`: The bot will convert the specified time across a pre-configured set of timezones and post the results in the thread. For example: `!tz 10:30am tomorrow`.

## 🔐 Slack App Permissions

For the bot to function correctly, your Slack App needs the following permissions (scopes):

- **`channels:history`**: View messages in public channels the bot is in.
- **`groups:history`**: View messages in private channels the bot is in.
- **`chat:write`**: Send messages as the bot.
- **`files:write`**: Upload files as snippets for DMs.
- **`im:write`**: Start direct messages with users.
- **`reactions:write`**: Add emoji reactions to messages.
- **`users:read`**: View user information to resolve display names.

## 🐛 Debug Mode

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

- Detailed logging to console and `/tmp/slack-bot-debug.log` file.
- Step-by-step processing information.
- Environment variable validation.
- Enhanced error messages with stack traces.

## 🔍 Troubleshooting

If the bot doesn't respond:

1. Enable debug mode to see detailed logs.
2. Check that all required environment variables are set correctly.
3. Verify the bot has the correct permissions in your Slack workspace (see above).
4. Check the console output for error messages.

