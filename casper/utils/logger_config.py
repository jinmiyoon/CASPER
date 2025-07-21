import logging
import os
import time

# Get the absolute path to the project root
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
start_time = time.strftime("%Y-%m-%d_%H-%M-%S")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up a logger that logs messages to both console and a log file.

    Parameters
    ----------
    name : str
        The name of the logger (typically __name__).
    level : int
        The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if this logger is already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    log_file_path = os.path.join(LOG_DIR, f"casper_log_{start_time}.log")

    # Console and file handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path)
    c_handler.setLevel(level)
    f_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
