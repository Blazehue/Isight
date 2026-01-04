"""
ISight - Sign Language Detection System

A comprehensive, production-ready sign language detection system built with 
Google's MediaPipe Hands for maximum accuracy (>95% target).

Author: Blazehue
License: MIT
"""

from __version__ import (
    __version__,
    __author__,
    __license__,
    __url__,
    __description__
)

__all__ = [
    'feature_extraction',
    'data_collector',
    'model_trainer',
    'detector',
    'evaluator',
    'visualization',
    'sign_language_detector',
]

# Version info
print(f"ISight v{__version__} - {__description__}")
print(f"Author: {__author__}")
print(f"License: {__license__}")
