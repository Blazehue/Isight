"""
Visualization Module for Sign Language Detection

This module provides comprehensive visualization and debugging tools for
hand landmark detection and gesture recognition.

Features:
- Hand landmark visualization
- Bounding box display
- Finger state indicators
- Prediction confidence display
- Performance metrics overlay

Author: Blazehue
Date: January 2026
"""

import cv2
import numpy as np
import mediapipe as mp
from feature_extraction import calculate_finger_states


class HandVisualizer:
    """
    Comprehensive hand visualization for debugging and display.
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Custom drawing specs for better visibility
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=3
        )
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=2
        )
    
    def draw_landmarks(self, image, hand_landmarks, draw_connections=True):
        """
        Draw hand landmarks on image.
        
        Args:
            image: BGR image
            hand_landmarks: MediaPipe hand landmarks
            draw_connections: Whether to draw connections between landmarks
        """
        if draw_connections:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.landmark_drawing_spec,
                self.connection_drawing_spec
            )
        else:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                None,
                self.landmark_drawing_spec,
                None
            )
        
        return image
    
    def draw_landmarks_detailed(self, image, hand_landmarks, 
                               show_indices=False, show_axes=False):
        """
        Draw detailed hand landmarks with optional indices and axes.
        
        Args:
            image: BGR image
            hand_landmarks: MediaPipe hand landmarks
            show_indices: Show landmark indices
            show_axes: Show 3D axes at wrist
        """
        h, w, _ = image.shape
        
        # Draw basic landmarks
        self.draw_landmarks(image, hand_landmarks, draw_connections=True)
        
        # Draw landmark indices
        if show_indices:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Draw index number
                cv2.putText(
                    image, str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 0), 1
                )
        
        # Draw 3D axes at wrist
        if show_axes:
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            
            # X-axis (red)
            cv2.arrowedLine(image, (wrist_x, wrist_y), 
                          (wrist_x + 50, wrist_y), (0, 0, 255), 2)
            cv2.putText(image, 'X', (wrist_x + 55, wrist_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Y-axis (green)
            cv2.arrowedLine(image, (wrist_x, wrist_y),
                          (wrist_x, wrist_y + 50), (0, 255, 0), 2)
            cv2.putText(image, 'Y', (wrist_x + 5, wrist_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return image
    
    def draw_finger_states(self, image, hand_landmarks, x_offset=10, y_offset=30):
        """
        Draw finger states (extended/bent) on image.
        
        Args:
            image: BGR image
            hand_landmarks: MediaPipe hand landmarks
            x_offset: X position for text
            y_offset: Starting Y position for text
        """
        finger_states = calculate_finger_states(hand_landmarks)
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        
        for i, (name, state) in enumerate(zip(finger_names, finger_states)):
            color = (0, 255, 0) if state == 1 else (0, 0, 255)
            status = "Extended" if state == 1 else "Bent"
            text = f"{name}: {status}"
            
            y_pos = y_offset + i * 25
            
            # Draw background rectangle for better readability
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                image,
                (x_offset - 2, y_pos - text_size[1] - 2),
                (x_offset + text_size[0] + 2, y_pos + 2),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                image, text,
                (x_offset, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
        
        return image
    
    def draw_bounding_box(self, image, hand_landmarks, color=(0, 255, 255), thickness=2):
        """
        Draw bounding box around hand.
        
        Args:
            image: BGR image
            hand_landmarks: MediaPipe hand landmarks
            color: Box color
            thickness: Line thickness
        """
        h, w, _ = image.shape
        
        # Get all landmark coordinates
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
        
        # Calculate bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Draw box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Draw dimensions
        box_width = x_max - x_min
        box_height = y_max - y_min
        cv2.putText(
            image, f"{box_width}x{box_height}px",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, color, 1
        )
        
        return image
    
    def draw_prediction(self, image, prediction, confidence, 
                       x_pos=None, y_pos=None, is_stable=False):
        """
        Draw prediction label with confidence.
        
        Args:
            image: BGR image
            prediction: Predicted gesture name
            confidence: Prediction confidence (0-1)
            x_pos: X position (default: bottom-left)
            y_pos: Y position (default: bottom)
            is_stable: Whether prediction is stable over time
        """
        h, w, _ = image.shape
        
        if x_pos is None:
            x_pos = 20
        if y_pos is None:
            y_pos = h - 80
        
        if prediction is None:
            text = "No Detection"
            color = (0, 0, 255)
        else:
            text = f"Sign: {prediction}"
            # Color based on confidence and stability
            if is_stable:
                color = (0, 255, 0)  # Green for stable
            elif confidence > 0.85:
                color = (0, 255, 255)  # Yellow for high confidence
            else:
                color = (0, 165, 255)  # Orange for lower confidence
        
        # Draw background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        cv2.rectangle(
            image,
            (x_pos - 5, y_pos - text_size[1] - 10),
            (x_pos + text_size[0] + 5, y_pos + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            image, text,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, color, 3
        )
        
        # Draw confidence
        if prediction is not None:
            conf_text = f"Confidence: {confidence:.1%}"
            conf_y = y_pos + 35
            
            # Confidence background
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(
                image,
                (x_pos - 5, conf_y - conf_size[1] - 5),
                (x_pos + conf_size[0] + 5, conf_y + 5),
                (0, 0, 0),
                -1
            )
            
            # Confidence text
            cv2.putText(
                image, conf_text,
                (x_pos, conf_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2
            )
            
            # Stability indicator
            if is_stable:
                cv2.putText(
                    image, "STABLE",
                    (x_pos + conf_size[0] + 20, conf_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )
        
        return image
    
    def draw_fps(self, image, fps, x_pos=None, y_pos=None):
        """
        Draw FPS counter.
        
        Args:
            image: BGR image
            fps: Frames per second
            x_pos: X position (default: top-right)
            y_pos: Y position (default: top)
        """
        h, w, _ = image.shape
        
        if x_pos is None:
            x_pos = w - 150
        if y_pos is None:
            y_pos = 30
        
        text = f"FPS: {fps:.1f}"
        color = (0, 255, 0) if fps >= 50 else (0, 165, 255) if fps >= 30 else (0, 0, 255)
        
        # Background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            image,
            (x_pos - 5, y_pos - text_size[1] - 5),
            (x_pos + text_size[0] + 5, y_pos + 5),
            (0, 0, 0),
            -1
        )
        
        # Text
        cv2.putText(
            image, text,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )
        
        return image
    
    def draw_top_predictions(self, image, top_predictions, x_pos=10, y_pos=150):
        """
        Draw top-k predictions with probabilities.
        Useful for debugging.
        
        Args:
            image: BGR image
            top_predictions: List of (gesture, probability) tuples
            x_pos: X position
            y_pos: Starting Y position
        """
        cv2.putText(
            image, "Top Predictions:",
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1
        )
        
        for i, (gesture, prob) in enumerate(top_predictions):
            y = y_pos + 25 + i * 25
            text = f"{i+1}. {gesture}: {prob:.1%}"
            
            # Color based on rank
            if i == 0:
                color = (0, 255, 0)
            elif i == 1:
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)
            
            # Draw probability bar
            bar_width = int(prob * 200)
            cv2.rectangle(
                image,
                (x_pos, y - 15),
                (x_pos + bar_width, y - 5),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                image, text,
                (x_pos, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
        
        return image
    
    def draw_detection_status(self, image, detector_status):
        """
        Draw comprehensive detector status information.
        
        Args:
            image: BGR image
            detector_status: Status dict from detector.get_status()
        """
        h, w, _ = image.shape
        
        # Create status panel at top
        panel_height = 100
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw status info
        y_pos = 25
        x_pos = 10
        
        # Current prediction
        if detector_status['stable_prediction']:
            status_text = f"Current: {detector_status['stable_prediction']}"
            status_color = (0, 255, 0)
        elif detector_status['last_prediction']:
            status_text = f"Detecting: {detector_status['last_prediction']}"
            status_color = (0, 255, 255)
        else:
            status_text = "Waiting for gesture..."
            status_color = (200, 200, 200)
        
        cv2.putText(image, status_text, (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Buffer status
        buffer_text = f"Buffer: {detector_status['buffer_size']}/5"
        cv2.putText(image, buffer_text, (x_pos, y_pos + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Confidence bar
        if detector_status['stable_confidence'] > 0:
            conf = detector_status['stable_confidence']
            bar_y = y_pos + 50
            bar_width = int(conf * 300)
            cv2.rectangle(image, (x_pos, bar_y), (x_pos + 300, bar_y + 20),
                         (50, 50, 50), -1)
            cv2.rectangle(image, (x_pos, bar_y), (x_pos + bar_width, bar_y + 20),
                         (0, 255, 0), -1)
            cv2.putText(image, f"{conf:.0%}", (x_pos + bar_width + 10, bar_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS
        self.draw_fps(image, detector_status['fps'])
        
        return image
    
    def create_comparison_view(self, original, processed, labels=None):
        """
        Create side-by-side comparison view.
        
        Args:
            original: Original image
            processed: Processed image with overlays
            labels: Optional labels for each image
        
        Returns:
            Combined image
        """
        # Resize if needed
        h1, w1 = original.shape[:2]
        h2, w2 = processed.shape[:2]
        
        if h1 != h2:
            target_h = min(h1, h2)
            original = cv2.resize(original, (int(w1 * target_h / h1), target_h))
            processed = cv2.resize(processed, (int(w2 * target_h / h2), target_h))
        
        # Combine horizontally
        combined = np.hstack([original, processed])
        
        # Add labels
        if labels:
            h, w = combined.shape[:2]
            mid_w = w // 2
            
            if len(labels) >= 1:
                cv2.putText(combined, labels[0], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if len(labels) >= 2:
                cv2.putText(combined, labels[1], (mid_w + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined


def draw_complete_visualization(image, hand_landmarks, prediction=None, 
                                confidence=None, is_stable=False, 
                                detector_status=None, show_detailed=True):
    """
    Draw complete visualization with all information.
    
    Args:
        image: BGR image
        hand_landmarks: MediaPipe hand landmarks
        prediction: Predicted gesture
        confidence: Prediction confidence
        is_stable: Whether prediction is stable
        detector_status: Detector status dict
        show_detailed: Whether to show detailed info (finger states, etc.)
    
    Returns:
        Annotated image
    """
    visualizer = HandVisualizer()
    
    # Draw hand landmarks
    image = visualizer.draw_landmarks(image, hand_landmarks, draw_connections=True)
    
    # Draw bounding box
    image = visualizer.draw_bounding_box(image, hand_landmarks)
    
    # Draw prediction
    if prediction is not None or confidence is not None:
        image = visualizer.draw_prediction(image, prediction, confidence, is_stable=is_stable)
    
    # Draw detailed info
    if show_detailed:
        image = visualizer.draw_finger_states(image, hand_landmarks)
    
    # Draw detector status
    if detector_status:
        image = visualizer.draw_detection_status(image, detector_status)
    
    return image
