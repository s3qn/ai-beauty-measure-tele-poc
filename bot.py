"""Entrypoint shim — the bot lives in the `app` package (CLAUDE.md §3).

Run with:
    python bot.py
Requires TELEGRAM_BOT_TOKEN in the environment or a .env file.
"""

from app.bot import main

if __name__ == "__main__":
    main()
