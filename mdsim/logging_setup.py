'''
    Logging setup for the mdsim package.
'''

import logging
from typing import Optional

_LOGGING_INITIALIZED = False


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging once for the whole package.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        None
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    _LOGGING_INITIALIZED = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger for the given name.

    Args:
        name: Name of the logger (usually __name__ of the module)
        level: Optional logging level to set for this logger

    Returns:
        logging.Logger: Configured logger instance
    """
    setup_logging()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
