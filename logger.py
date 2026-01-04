"""
Logger Module for Sign Language Detection

This module provides logging functionality for debugging and monitoring.

Author: Blazehue
Date: January 2026
"""

import logging
import os
from datetime import datetime


def setup_logger(name='isight', level=logging.INFO, log_dir='logs'):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        level: Logging level (default: INFO)
        log_dir (str): Directory for log files
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    log_filename = os.path.join(
        log_dir,
        f'{name}_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
