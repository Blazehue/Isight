"""
Feature Extraction Module for Sign Language Detection

This module extracts comprehensive features from MediaPipe hand landmarks 
to achieve maximum accuracy in gesture recognition.

The feature vector includes:
- 3D landmark positions (63 features)
- Finger states (5 features)
- Joint angles (10 features)
- Landmark distances (12 features)
- Palm orientation (3 features)
- Bounding box features (4 features)

Total: ~97 robust features per frame

Author: Blazehue
Date: January 2026
"""

import numpy as np
import mediapipe as mp


def extract_features(hand_landmarks):
    """
    Extract robust features from MediaPipe landmarks.
    
    This is the main feature extraction pipeline that combines multiple
    feature types to create a comprehensive representation of hand gestures.
    
    Returns a comprehensive feature vector for gesture recognition.
    
    Features include:
    - Normalized 3D landmark positions (63 features)
    - Finger states (5 features)
    - Joint angles (10 features)
    - Landmark distances (12 features)
    - Palm orientation (3 features)
    - Bounding box features (4 features)
    
    Total: ~97 features
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object containing 21 3D points
    
    Returns:
        numpy.ndarray: Feature vector of length ~97
    """
    features = []
    
    # 1. Normalized landmark positions (x, y, z for all 21 points = 63 features)
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    
    # 2. Finger states (extended/bent for each finger = 5 features)
    finger_states = calculate_finger_states(hand_landmarks)
    features.extend(finger_states)
    
    # 3. Angles between joints (important for gesture differentiation)
    angles = calculate_joint_angles(hand_landmarks)
    features.extend(angles)
    
    # 4. Distances between key landmarks
    distances = calculate_landmark_distances(hand_landmarks)
    features.extend(distances)
    
    # 5. Palm orientation (normal vector)
    palm_orientation = calculate_palm_orientation(hand_landmarks)
    features.extend(palm_orientation)
    
    # 6. Hand bounding box features (normalized)
    bbox_features = calculate_bounding_box_features(hand_landmarks)
    features.extend(bbox_features)
    
    return np.array(features)


def calculate_finger_states(landmarks):
    """
    Determine if each finger is extended or bent.
    
    This function analyzes the position of fingertips relative to their joints
    to determine finger states. Critical for ASL recognition as many signs 
    differ by which fingers are extended.
    
    Args:
        landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: Binary states [thumb, index, middle, ring, pinky] where 1=extended, 0=bent
    """
    fingers = []
    
    # Thumb (special case - check x-axis for extended)
    # Thumb is extended if tip is farther from palm center than MCP joint
    thumb_tip = landmarks.landmark[4]
    thumb_mcp = landmarks.landmark[2]
    thumb_cmc = landmarks.landmark[1]
    wrist = landmarks.landmark[0]
    
    # Calculate if thumb is extended based on distance from wrist
    thumb_tip_dist = np.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
    thumb_mcp_dist = np.sqrt((thumb_mcp.x - wrist.x)**2 + (thumb_mcp.y - wrist.y)**2)
    thumb_extended = thumb_tip_dist > thumb_mcp_dist
    fingers.append(1 if thumb_extended else 0)
    
    # Other fingers (check y-axis - tip vs PIP joint)
    # Finger is extended if tip is above (lower y value) than PIP joint
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    finger_pips = [6, 10, 14, 18]
    
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = landmarks.landmark[tip_idx]
        pip = landmarks.landmark[pip_idx]
        # Extended if tip is above PIP (lower y coordinate)
        extended = tip.y < pip.y
        fingers.append(1 if extended else 0)
    
    return fingers


def calculate_joint_angles(landmarks):
    """
    Calculate angles at each joint to help differentiate similar gestures.
    
    This function computes the angle at each finger joint, providing geometric
    information that helps distinguish between gestures with similar hand positions
    but different finger configurations.
    
    Args:
        landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: Angles in degrees for 10 major joints (2 per finger)
    """
    angles = []
    
    # Define joint triplets (point1, joint, point2)
    joint_triplets = [
        # Thumb
        (1, 2, 3), (2, 3, 4),
        # Index finger
        (5, 6, 7), (6, 7, 8),
        # Middle finger
        (9, 10, 11), (10, 11, 12),
        # Ring finger
        (13, 14, 15), (14, 15, 16),
        # Pinky finger
        (17, 18, 19), (18, 19, 20)
    ]
    
    for p1_idx, joint_idx, p2_idx in joint_triplets:
        p1 = landmarks.landmark[p1_idx]
        joint = landmarks.landmark[joint_idx]
        p2 = landmarks.landmark[p2_idx]
        
        angle = calculate_angle_3points(
            [p1.x, p1.y, p1.z],
            [joint.x, joint.y, joint.z],
            [p2.x, p2.y, p2.z]
        )
        angles.append(angle)
    
    return angles


def calculate_angle_3points(p1, p2, p3):
    """
    Calculate angle at point p2 formed by the line segments p1-p2-p3.
    
    Uses vector mathematics to compute the angle between two line segments
    sharing a common point.
    
    Args:
        p1 (list): First point coordinates [x, y, z]
        p2 (list): Middle point (vertex) coordinates [x, y, z]
        p3 (list): Third point coordinates [x, y, z]
    
    Returns:
        float: Angle at p2 in degrees (0-180)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    # Clamp to [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def calculate_landmark_distances(landmarks):
    """
    Calculate Euclidean distances between important landmark pairs.
    
    Computes distances between key points on the hand to help recognize
    specific hand shapes and finger configurations.
    
    Args:
        landmarks: MediaPipe hand landmarks object
    
    Returns:
        list: Distances between 12 important landmark pairs
    """
    distances = []
    
    # Key distance pairs for ASL recognition
    important_pairs = [
        (4, 8),   # Thumb tip to index tip
        (4, 12),  # Thumb tip to middle tip
        (4, 16),  # Thumb tip to ring tip
        (4, 20),  # Thumb tip to pinky tip
        (8, 12),  # Index tip to middle tip
        (12, 16), # Middle tip to ring tip
        (16, 20), # Ring tip to pinky tip
        (0, 4),   # Wrist to thumb tip
        (0, 8),   # Wrist to index tip
        (0, 12),  # Wrist to middle tip
        (0, 16),  # Wrist to ring tip
        (0, 20),  # Wrist to pinky tip
    ]
    
    for p1_idx, p2_idx in important_pairs:
        p1 = landmarks.landmark[p1_idx]
        p2 = landmarks.landmark[p2_idx]
        distance = np.sqrt(
            (p1.x - p2.x)**2 + 
            (p1.y - p2.y)**2 + 
            (p1.z - p2.z)**2
        )
        distances.append(distance)
    
    return distances


def calculate_palm_orientation(landmarks):
    """
    Calculate palm normal vector - crucial for detecting hand rotation.
    Uses cross product of two palm vectors to get the normal.
    """
    # Use three points to define palm plane
    wrist = np.array([
        landmarks.landmark[0].x, 
        landmarks.landmark[0].y, 
        landmarks.landmark[0].z
    ])
    index_mcp = np.array([
        landmarks.landmark[5].x, 
        landmarks.landmark[5].y, 
        landmarks.landmark[5].z
    ])
    pinky_mcp = np.array([
        landmarks.landmark[17].x, 
        landmarks.landmark[17].y, 
        landmarks.landmark[17].z
    ])
    
    # Calculate normal vector using cross product
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)
    
    # Normalize
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude > 1e-6:
        normal = normal / normal_magnitude
    else:
        normal = np.array([0, 0, 1])  # Default normal
    
    return normal.tolist()


def calculate_bounding_box_features(landmarks):
    """
    Calculate bounding box features of the hand.
    Returns: [width, height, aspect_ratio, area]
    """
    # Get all x and y coordinates
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    
    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    aspect_ratio = width / (height + 1e-6)
    area = width * height
    
    return [width, height, aspect_ratio, area]


def get_feature_names():
    """
    Returns list of feature names for debugging and analysis.
    """
    feature_names = []
    
    # Landmark positions
    landmark_names = [
        'wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
        'index_mcp', 'index_pip', 'index_dip', 'index_tip',
        'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
        'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
        'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
    ]
    
    for name in landmark_names:
        feature_names.extend([f'{name}_x', f'{name}_y', f'{name}_z'])
    
    # Finger states
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for name in finger_names:
        feature_names.append(f'{name}_extended')
    
    # Joint angles
    angle_names = [
        'thumb_cmc_angle', 'thumb_mcp_angle',
        'index_mcp_angle', 'index_pip_angle',
        'middle_mcp_angle', 'middle_pip_angle',
        'ring_mcp_angle', 'ring_pip_angle',
        'pinky_mcp_angle', 'pinky_pip_angle'
    ]
    feature_names.extend(angle_names)
    
    # Distances
    distance_names = [
        'thumb_index_dist', 'thumb_middle_dist', 'thumb_ring_dist', 'thumb_pinky_dist',
        'index_middle_dist', 'middle_ring_dist', 'ring_pinky_dist',
        'wrist_thumb_dist', 'wrist_index_dist', 'wrist_middle_dist', 
        'wrist_ring_dist', 'wrist_pinky_dist'
    ]
    feature_names.extend(distance_names)
    
    # Palm orientation
    feature_names.extend(['palm_normal_x', 'palm_normal_y', 'palm_normal_z'])
    
    # Bounding box
    feature_names.extend(['bbox_width', 'bbox_height', 'bbox_aspect_ratio', 'bbox_area'])
    
    return feature_names
