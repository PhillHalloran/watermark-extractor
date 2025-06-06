import os
import logging
import pytest
from datetime import datetime

from src.logger import Logger, AppError, show_error_dialog


def test_logger_creates_log_dir(tmp_path):
    # Choose a new directory under tmp_path
    log_dir = tmp_path / "logs"
    # Ensure it doesn't exist yet
    assert not log_dir.exists()

    # Initialize Logger; should create the directory
    logger = Logger(log_dir=str(log_dir), level=logging.DEBUG)
    assert log_dir.exists() and log_dir.is_dir()

    # Clean up handlers to avoid interference with other tests
    for handler in logger.logger.handlers:
        logger.logger.removeHandler(handler)


def test_logger_writes_messages(tmp_path):
    log_dir = tmp_path / "logs"
    logger = Logger(log_dir=str(log_dir), level=logging.DEBUG)

    # Write one message at each level
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Locate the log file
    log_file_path = log_dir / "app.log"
    assert log_file_path.exists()

    # Read log contents
    content = log_file_path.read_text().splitlines()
    # Expect at least three lines (one per message)
    assert any("[INFO] This is an info message" in line for line in content)
    assert any(
        "[WARNING] This is a warning message" in line for line in content)
    assert any("[ERROR] This is an error message" in line for line in content)

    # Verify timestamp and format roughly match "[YYYY-MM-DD HH:MM:SS] [LEVEL]"
    sample_line = next(line for line in content if "[INFO]" in line)
    # Extract everything before the first " [" which should be the timestamp
    timestamp_part = sample_line.split(" [")[0]
    # Parse the timestamp
    datetime.strptime(timestamp_part, "%Y-%m-%d %H:%M:%S")

    # Clean up handlers
    for handler in logger.logger.handlers:
        logger.logger.removeHandler(handler)


def test_logger_invalid_log_dir(tmp_path):
    # Create a file where a directory is expected
    fake_dir = tmp_path / "not_a_dir"
    fake_dir.write_text("I am a file, not a directory")
    # Attempting to use the same path as a directory should raise OSError
    with pytest.raises(OSError) as excinfo:
        Logger(log_dir=str(fake_dir))
    assert "Cannot create log directory" in str(excinfo.value)


def test_app_error_attributes():
    user_msg = "User-friendly error"
    internal_msg = "Detailed internal error"
    err = AppError(user_message=user_msg, internal_message=internal_msg)

    # The exception message (str) should be the user_message
    assert str(err) == user_msg
    # Attributes should be stored correctly
    assert err.user_message == user_msg
    assert err.internal_message == internal_msg


def test_show_error_dialog_prints_message(capsys):
    user_msg = "Something went wrong"
    show_error_dialog(user_msg)
    captured = capsys.readouterr()
    assert "[ERROR DIALOG] Something went wrong" in captured.out
