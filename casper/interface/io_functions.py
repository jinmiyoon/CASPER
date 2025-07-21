import os

from utils.logger_config import setup_logger

logger = setup_logger(__name__)


def span_window(char: str = "-") -> None:
    print(char * os.get_terminal_size().columns)
    return


def print_greeting():
    span_window("#")
    logger.info("        CASPER")
    logger.info("Authors: Devin D. Whitten and Jinmi Yoon")
    logger.info("Please direct questions to: jinmi.yoon@gmail.com and devin.d.whitten@gmail.com")
    span_window("#")
