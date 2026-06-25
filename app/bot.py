"""Telegram handlers — wiring only (CLAUDE.md §3). No business logic here.

Each inbound message: hand image bytes to ``Pipeline.process`` and format the reply.
All decisions (validation, detection, measurement) live in the testable modules.
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Config
from .pipeline import Pipeline, sweep_temp_dir
from .schema import MeasurementResult, PipelineError, guidance_for

logger = logging.getLogger("beauty_bot")


def format_success(result: MeasurementResult, *, used_latest: bool = False) -> str:
    lines = ["Here are your face measurements:"]
    if used_latest:
        lines.insert(0, "Used your most recent photo.")
    for name, m in result.measurements.items():
        lines.append(f"• {name}: {m.value} ({m.unit})")
    lines.append("")
    lines.append(f"schema_version: {result.schema_version}")
    lines.append(f"correlation_id: {result.correlation_id}")
    return "\n".join(lines)


def format_error(err: PipelineError) -> str:
    return f"{guidance_for(err.code)}\n\ncorrelation_id: {err.correlation_id}"


def _reply_for(outcome, *, used_latest: bool = False) -> str:
    if isinstance(outcome, MeasurementResult):
        return format_success(outcome, used_latest=used_latest)
    return format_error(outcome)


async def _process_bytes(context, image_bytes, chat_id):
    """Run the (sync, CPU-bound) pipeline off the event loop."""
    pipeline: Pipeline = context.application.bot_data["pipeline"]
    return await asyncio.get_running_loop().run_in_executor(
        None, pipeline.process, image_bytes, chat_id
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a clear, front-facing selfie and I'll return geometric face "
        "measurements. I don't store your photo."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send a single, front-facing selfie (JPG or PNG). I measure facial geometry "
        "and reply with numbers only — no image is stored.\n\n"
        "/start - intro\n/help - this message"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    # message.photo is sizes of ONE image; take the highest resolution.
    photo = message.photo[-1]
    used_latest = message.media_group_id is not None  # album => tell the user
    file = await context.bot.get_file(photo.file_id)
    image_bytes = bytes(await file.download_as_bytearray())

    outcome = await _process_bytes(context, image_bytes, message.chat_id)
    await message.reply_text(_reply_for(outcome, used_latest=used_latest))


async def handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    doc = message.document
    if not (doc and doc.mime_type and doc.mime_type.startswith("image/")):
        await handle_non_image(update, context)
        return
    file = await context.bot.get_file(doc.file_id)
    image_bytes = bytes(await file.download_as_bytearray())
    outcome = await _process_bytes(context, image_bytes, message.chat_id)
    await message.reply_text(_reply_for(outcome))


async def handle_non_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # No usable image (text, sticker, non-image document, etc.) — NOT_AN_IMAGE (§6/§7).
    from .schema import ErrorCode

    await update.message.reply_text(guidance_for(ErrorCode.NOT_AN_IMAGE))


def build_application(cfg: Config) -> Application:
    sweep_temp_dir(cfg)
    app = Application.builder().token(cfg.telegram_bot_token).build()
    app.bot_data["pipeline"] = Pipeline(cfg)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image_document))
    # Everything else that isn't a command => ask for a photo.
    app.add_handler(
        MessageHandler(~filters.COMMAND & ~filters.PHOTO & ~filters.Document.IMAGE, handle_non_image)
    )
    return app


def main() -> None:
    cfg = Config.load()
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=cfg.log_level,
    )
    app = build_application(cfg)
    logger.info("Bot starting (long polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
