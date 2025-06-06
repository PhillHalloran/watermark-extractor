import logging
import logging.handlers
import os
from datetime import datetime


class Logger:
    """
    Singleton or module‐scoped logger. Writes to a daily rotating file `app.log`.
    """

    def __init__(self, log_dir: str = "logs", level: int = logging.INFO) -> None:
        """
        - Ensure `log_dir` exists (os.makedirs if needed).
        - Create a logger named "WatermarkApp" with level=level.
        - Attach a TimedRotatingFileHandler that rotates at midnight,
          keeps 7 days of backups.
        - Log format: "[YYYY-MM-DD HH:MM:SS] [LEVEL] message"
        """
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            raise OSError(f"Cannot create log directory: {e}")

        self.logger = logging.getLogger("WatermarkApp")
        self.logger.setLevel(level)

        # Only add handlers if none exist, to prevent duplicate handlers
        if not self.logger.handlers:
            log_path = os.path.join(log_dir, "app.log")
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=log_path,
                when="midnight",
                interval=1,
                backupCount=7,
                encoding="utf-8",
            )
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        """
        Log an INFO-level message.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        Log a WARNING-level message.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Log an ERROR-level message.
        """
        self.logger.error(message)


class AppError(Exception):
    """
    Custom exception for user-facing errors (e.g., download fails, OCR not installed).
    Contains a `user_message` which is safe to show in a dialog.
    """

    def __init__(self, user_message: str, internal_message: str = None) -> None:  # type: ignore
        """
        - user_message: text to display to end-user.
        - internal_message: optional detailed message to log.
        """
        super().__init__(user_message)
        self.user_message = user_message
        self.internal_message = internal_message


def show_error_dialog(user_message: str) -> None:
    """
    Trigger a modal popup (in GUI) showing `user_message`.
    Implementation detail is GUI-framework dependent (e.g., Tkinter, PyQt).
    """
    # Placeholder implementation:
    # Replace with framework-specific dialog (Tkinter, PyQt, etc.)
    print(f"[ERROR DIALOG] {user_message}")
