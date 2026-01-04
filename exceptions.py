"""
Exception Classes for Sign Language Detection

Custom exception classes for better error handling.

Author: Blazehue
Date: January 2026
"""


class ISightException(Exception):
    """Base exception class for ISight errors."""
    pass


class ModelNotFoundError(ISightException):
    """Raised when a model file cannot be found."""
    pass


class InvalidLandmarksError(ISightException):
    """Raised when hand landmarks are invalid or missing."""
    pass


class DataCollectionError(ISightException):
    """Raised when data collection fails."""
    pass


class TrainingDataError(ISightException):
    """Raised when training data is invalid or insufficient."""
    pass


class ModelTrainingError(ISightException):
    """Raised when model training fails."""
    pass


class CameraError(ISightException):
    """Raised when camera initialization or access fails."""
    pass
