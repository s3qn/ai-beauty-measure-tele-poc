# ai-beauty-measure-tele-poc

A basic Telegram bot proof-of-concept built with
[python-telegram-bot](https://docs.python-telegram-bot.org/).

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Get a bot token from [@BotFather](https://t.me/BotFather) on Telegram.

3. Copy the example env file and add your token:

   ```bash
   cp .env.example .env
   # edit .env and set TELEGRAM_BOT_TOKEN
   ```

## Run

```bash
python bot.py
```

The bot responds to `/start`, `/help`, and echoes any other text message.

## Environment variables

| Variable             | Required | Description                                  |
| -------------------- | -------- | -------------------------------------------- |
| `TELEGRAM_BOT_TOKEN` | yes      | Token from @BotFather                        |
| `LOG_LEVEL`          | no       | Logging level (default `INFO`)               |

> `.env` is git-ignored — never commit your real token. Use `.env.example` as the template.
