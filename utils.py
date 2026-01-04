"""
Utility Functions for Sign Language Detection

This module contains helper functions and utilities used across the project.

Author: Blazehue
Date: January 2026
"""

import numpy as np
import cv2


def validate_landmarks(hand_landmarks):
    """
    Validate that hand landmarks are properly detected.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
    
    Returns:
        bool: True if landmarks are valid, False otherwise
    """
    if hand_landmarks is None:
        return False
    
    if not hasattr(hand_landmarks, 'landmark'):
        return False
    
    if len(hand_landmarks.landmark) != 21:
        return False
    
    return True


def calculate_fps(start_time, end_time):
    """
    Calculate FPS from start and end times.
    
    Args:
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
    
    Returns:
        float: Frames per second
    """
    elapsed = end_time - start_time
    if elapsed > 0:
        return 1.0 / elapsed
    return 0.0


def format_confidence(confidence):
    """
    Format confidence score as percentage string.
    
    Args:
        confidence (float): Confidence score (0-1)
    
    Returns:
        str: Formatted percentage (e.g., "95.5%")
    """
    return f"{confidence * 100:.1f}%"
