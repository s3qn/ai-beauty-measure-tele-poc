"""Basic Telegram bot (POC).

Run with:
    python bot.py

Requires TELEGRAM_BOT_TOKEN to be set in the environment or a .env file.
"""

import logging
import os


from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Greet the user on /start."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hi {user.first_name if user else 'there'}! I'm alive. "
        "Send me a message and I'll echo it back, or try /help."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List available commands on /help."""
    await update.message.reply_text(
        "Available commands:\n"
        "/start - say hello\n"
        "/help - show this message\n\n"
        "Any other text I'll echo back."
    )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo any non-command text message."""
    await update.message.reply_text(update.message.text)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. Copy .env.example to .env and add your token."
        )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    logger.info("Bot starting (long polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
