import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> None:
    """Configure logging for the application.
    
    This sets up logging to both console and optionally a file, with
    appropriate formatting and log levels.
    
    Args:
        level: Logging level (e.g., "INFO", "DEBUG")
        log_file: Optional path to write logs to a file
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Set logging levels for some verbose libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)