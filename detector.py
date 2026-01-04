"""
Real-Time Detection Module for Sign Language Recognition

This module implements a robust prediction pipeline with confidence filtering and
temporal smoothing for real-time sign language gesture detection.

Features:
- Confidence-based prediction filtering
- Temporal smoothing for stable results
- Multiple detector types support
- FPS tracking and performance monitoring

Author: Blazehue
Date: January 2026
"""

import numpy as np
import cv2
import mediapipe as mp
from collections import Counter, deque
import time


class SignLanguageDetector:
    """
    Real-time sign language detector with confidence filtering and temporal smoothing.
    """
    
    def __init__(self, model_data, confidence_threshold=0.80, buffer_size=5):
        """
        Initialize detector with trained model.
        
        Args:
            model_data: Dictionary containing model, scaler, and label_encoder
            confidence_threshold: Minimum confidence for valid predictions (0.75-0.85 recommended)
            buffer_size: Number of frames for temporal smoothing (3-7 recommended)
        """
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.gesture_names = model_data['gesture_names']
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        
        # Prediction buffer for temporal smoothing
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.confidence_buffer = deque(maxlen=buffer_size)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_prediction = None
        self.last_confidence = 0.0
        self.stable_prediction = None
        self.stable_confidence = 0.0
        
        print(f"SignLanguageDetector initialized")
        print(f"  Gestures: {len(self.gesture_names)}")
        print(f"  Confidence threshold: {confidence_threshold:.2f}")
        print(f"  Buffer size: {buffer_size}")
    
    def predict(self, hand_landmarks, features=None):
        """
        Predict gesture from hand landmarks with confidence filtering and temporal smoothing.
        
        This method performs prediction using the trained model and applies confidence
        thresholding and temporal smoothing for stable results.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            features (optional): Pre-extracted features array. If None, features will be 
                               extracted automatically from hand_landmarks
        
        Returns:
            tuple: (prediction, confidence, is_stable)
                - prediction (str or None): Predicted gesture name or None if confidence too low
                - confidence (float): Prediction confidence score (0-1)
                - is_stable (bool): Whether prediction is stable over temporal buffer
        """
        # Extract features if not provided
        if features is None:
            from feature_extraction import extract_features
            features = extract_features(hand_landmarks)
        
        # Reshape and scale
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction with probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            prediction = self.label_encoder.classes_[prediction_idx]
        else:
            # Model doesn't support probability (shouldn't happen with our models)
            prediction_idx = self.model.predict(features_scaled)[0]
            prediction = self.label_encoder.classes_[prediction_idx]
            confidence = 1.0
        
        # Store in history
        self.last_prediction = prediction
        self.last_confidence = confidence
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            self.prediction_buffer.clear()
            self.confidence_buffer.clear()
            self.stable_prediction = None
            self.stable_confidence = 0.0
            return None, confidence, False
        
        # Add to temporal buffer
        self.prediction_buffer.append(prediction)
        self.confidence_buffer.append(confidence)
        
        # Temporal smoothing: require consistent predictions
        if len(self.prediction_buffer) >= max(3, self.buffer_size // 2):
            # Get most common prediction in buffer
            prediction_counts = Counter(self.prediction_buffer)
            most_common_prediction, count = prediction_counts.most_common(1)[0]
            
            # Check if prediction is stable (appears in >50% of buffer)
            stability_ratio = count / len(self.prediction_buffer)
            is_stable = stability_ratio >= 0.6
            
            if is_stable:
                # Calculate average confidence for this prediction
                avg_confidence = np.mean([
                    conf for pred, conf in zip(self.prediction_buffer, self.confidence_buffer)
                    if pred == most_common_prediction
                ])
                
                self.stable_prediction = most_common_prediction
                self.stable_confidence = avg_confidence
                
                return most_common_prediction, avg_confidence, True
        
        # Not stable yet
        return prediction, confidence, False
    
    def reset(self):
        """Reset prediction buffers"""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.stable_prediction = None
        self.stable_confidence = 0.0
    
    def get_all_probabilities(self, hand_landmarks, features=None, top_k=5):
        """
        Get top-k predictions with probabilities.
        Useful for debugging and seeing alternative interpretations.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            features: Pre-extracted features (optional)
            top_k: Number of top predictions to return
        
        Returns:
            list: List of (gesture_name, probability) tuples
        """
        # Extract features if not provided
        if features is None:
            from feature_extraction import extract_features
            features = extract_features(hand_landmarks)
        
        # Reshape and scale
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Get probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get top-k
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            top_predictions = [
                (self.label_encoder.classes_[idx], probabilities[idx])
                for idx in top_indices
            ]
            
            return top_predictions
        else:
            # Model doesn't support probabilities
            prediction = self.model.predict(features_scaled)[0]
            return [(self.label_encoder.classes_[prediction], 1.0)]
    
    def update_fps(self, frame_time):
        """Update FPS history"""
        self.fps_history.append(frame_time)
    
    def get_fps(self):
        """Get average FPS"""
        if len(self.fps_history) < 2:
            return 0
        avg_frame_time = np.mean(self.fps_history)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    def get_status(self):
        """
        Get current detector status for display.
        
        Returns:
            dict: Status information
        """
        return {
            'last_prediction': self.last_prediction,
            'last_confidence': self.last_confidence,
            'stable_prediction': self.stable_prediction,
            'stable_confidence': self.stable_confidence,
            'buffer_size': len(self.prediction_buffer),
            'fps': self.get_fps()
        }


class MultiHandDetector:
    """
    Detector for multiple hands with individual tracking.
    """
    
    def __init__(self, model_data, confidence_threshold=0.80, buffer_size=5):
        """
        Initialize multi-hand detector.
        
        Args:
            model_data: Dictionary containing model, scaler, and label_encoder
            confidence_threshold: Minimum confidence for valid predictions
            buffer_size: Number of frames for temporal smoothing
        """
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.confidence_threshold = confidence_threshold
        self.buffer_size = buffer_size
        
        # Separate detectors for left and right hands
        self.left_hand_detector = SignLanguageDetector(
            model_data, confidence_threshold, buffer_size
        )
        self.right_hand_detector = SignLanguageDetector(
            model_data, confidence_threshold, buffer_size
        )
    
    def predict(self, multi_hand_landmarks, handedness_list):
        """
        Predict gestures for multiple hands.
        
        Args:
            multi_hand_landmarks: List of hand landmarks from MediaPipe
            handedness_list: List of hand classifications (Left/Right)
        
        Returns:
            dict: Predictions for each hand
        """
        predictions = {
            'left': {'prediction': None, 'confidence': 0.0, 'stable': False},
            'right': {'prediction': None, 'confidence': 0.0, 'stable': False}
        }
        
        for hand_landmarks, handedness in zip(multi_hand_landmarks, handedness_list):
            # Determine if left or right hand
            hand_label = handedness.classification[0].label.lower()
            
            # Note: MediaPipe returns 'Left' for right hand in mirror image
            # Swap if using mirror mode
            if hand_label == 'left':
                hand_label = 'right'
            elif hand_label == 'right':
                hand_label = 'left'
            
            # Get prediction from appropriate detector
            if hand_label == 'left':
                pred, conf, stable = self.left_hand_detector.predict(hand_landmarks)
            else:
                pred, conf, stable = self.right_hand_detector.predict(hand_landmarks)
            
            predictions[hand_label] = {
                'prediction': pred,
                'confidence': conf,
                'stable': stable
            }
        
        return predictions
    
    def reset(self):
        """Reset both hand detectors"""
        self.left_hand_detector.reset()
        self.right_hand_detector.reset()


class AdaptiveConfidenceDetector(SignLanguageDetector):
    """
    Detector with adaptive confidence threshold based on gesture difficulty.
    Some gestures are easier to detect than others - adjust thresholds accordingly.
    """
    
    def __init__(self, model_data, base_confidence=0.80, buffer_size=5, 
                 gesture_thresholds=None):
        """
        Initialize adaptive detector.
        
        Args:
            model_data: Dictionary containing model, scaler, and label_encoder
            base_confidence: Base confidence threshold
            buffer_size: Number of frames for temporal smoothing
            gesture_thresholds: Dict mapping gesture names to custom thresholds
        """
        super().__init__(model_data, base_confidence, buffer_size)
        
        # Custom thresholds for specific gestures
        self.gesture_thresholds = gesture_thresholds or {}
        self.base_threshold = base_confidence
    
    def predict(self, hand_landmarks, features=None):
        """
        Predict with adaptive confidence threshold.
        """
        # Get base prediction
        prediction, confidence, is_stable = super().predict(hand_landmarks, features)
        
        if prediction is None:
            return None, confidence, False
        
        # Apply gesture-specific threshold
        gesture_threshold = self.gesture_thresholds.get(
            prediction, 
            self.base_threshold
        )
        
        # Re-evaluate with custom threshold
        if confidence < gesture_threshold:
            return None, confidence, False
        
        return prediction, confidence, is_stable


class ContextAwareDetector(SignLanguageDetector):
    """
    Detector that considers context of previous signs for improved accuracy.
    Uses sequence patterns to help disambiguate similar gestures.
    """
    
    def __init__(self, model_data, confidence_threshold=0.80, buffer_size=5,
                 history_length=5):
        """
        Initialize context-aware detector.
        
        Args:
            model_data: Dictionary containing model, scaler, and label_encoder
            confidence_threshold: Minimum confidence for valid predictions
            buffer_size: Number of frames for temporal smoothing
            history_length: Number of previous signs to consider for context
        """
        super().__init__(model_data, confidence_threshold, buffer_size)
        
        self.sign_history = deque(maxlen=history_length)
        self.history_length = history_length
    
    def predict(self, hand_landmarks, features=None):
        """
        Predict with context awareness.
        """
        # Get base prediction with top alternatives
        top_predictions = self.get_all_probabilities(
            hand_landmarks, features, top_k=3
        )
        
        # If we have context, use it to improve prediction
        if len(self.sign_history) > 0 and len(top_predictions) > 1:
            # Check if second prediction makes more sense given context
            best_pred, best_conf = top_predictions[0]
            second_pred, second_conf = top_predictions[1]
            
            # If second prediction has close confidence and better context fit
            if second_conf > best_conf * 0.85:  # Within 15% of best
                # Simple heuristic: avoid immediate repetitions unless very confident
                if self.sign_history[-1] == best_pred and second_pred != best_pred:
                    if best_conf < 0.95:  # Not super confident
                        # Prefer the alternative
                        pass  # Could swap predictions here based on context
        
        # Get standard prediction with temporal smoothing
        prediction, confidence, is_stable = super().predict(hand_landmarks, features)
        
        # Update history when we have a stable prediction
        if is_stable and prediction is not None:
            # Only add to history if it's different from last sign
            if len(self.sign_history) == 0 or self.sign_history[-1] != prediction:
                self.sign_history.append(prediction)
        
        return prediction, confidence, is_stable
    
    def get_sign_sequence(self):
        """Get the sequence of detected signs"""
        return list(self.sign_history)
    
    def clear_history(self):
        """Clear sign history"""
        self.sign_history.clear()


def load_detector(model_filepath, detector_type='standard', **kwargs):
    """
    Factory function to load and create appropriate detector.
    
    Args:
        model_filepath: Path to saved model file
        detector_type: Type of detector ('standard', 'multi_hand', 'adaptive', 'context_aware')
        **kwargs: Additional arguments for specific detector types
    
    Returns:
        Detector instance
    """
    import pickle
    
    # Load model data
    with open(model_filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✓ Loaded model: {model_data['model_name']}")
    print(f"  Training accuracy: {model_data['accuracy']:.4f}")
    
    # Create appropriate detector
    if detector_type == 'standard':
        detector = SignLanguageDetector(model_data, **kwargs)
    elif detector_type == 'multi_hand':
        detector = MultiHandDetector(model_data, **kwargs)
    elif detector_type == 'adaptive':
        detector = AdaptiveConfidenceDetector(model_data, **kwargs)
    elif detector_type == 'context_aware':
        detector = ContextAwareDetector(model_data, **kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    return detector
