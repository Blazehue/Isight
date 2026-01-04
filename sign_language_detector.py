"""
Main Application - High-Accuracy Sign Language Detector

Complete pipeline for real-time sign language detection using MediaPipe.
This is the main entry point for running the detection system.

Features:
- Real-time webcam detection
- Multiple detector types
- Configurable confidence thresholds
- Visual feedback and statistics

Author: Blazehue
Date: January 2026
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import argparse
import os
import time
from detector import load_detector
from visualization import draw_complete_visualization, HandVisualizer
from feature_extraction import extract_features


class SignLanguageApp:
    """
    Main application for real-time sign language detection.
    """
    
    def __init__(self, model_path, confidence_threshold=0.80, buffer_size=5,
                 detector_type='standard', show_detailed=True):
        """
        Initialize the application.
        
        Args:
            model_path: Path to trained model file
            confidence_threshold: Minimum confidence for detections (0.75-0.85)
            buffer_size: Temporal smoothing buffer size (3-7)
            detector_type: Type of detector ('standard', 'context_aware', etc.)
            show_detailed: Show detailed visualization
        """
        print(f"\n{'='*60}")
        print("INITIALIZING SIGN LANGUAGE DETECTOR")
        print(f"{'='*60}")
        
        # Load detector
        self.detector = load_detector(
            model_path,
            detector_type=detector_type,
            confidence_threshold=confidence_threshold,
            buffer_size=buffer_size
        )
        
        # Initialize MediaPipe Hands with optimal settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1  # Full model for accuracy
        )
        
        # Visualization
        self.visualizer = HandVisualizer()
        self.show_detailed = show_detailed
        
        # Performance tracking
        self.frame_times = []
        self.fps = 0
        
        # Settings
        self.confidence_threshold = confidence_threshold
        self.paused = False
        self.show_fps = True
        self.show_landmarks = True
        self.show_bbox = True
        self.show_finger_states = True
        self.show_top_predictions = False
        
        print(f"\n✓ Application initialized successfully")
        print(f"  Detector type: {detector_type}")
        print(f"  Confidence threshold: {confidence_threshold:.2f}")
        print(f"  Buffer size: {buffer_size}")
    
    def run(self, camera_id=0, width=1280, height=720, target_fps=60):
        """
        Run the main application loop.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            target_fps: Target FPS
        """
        print(f"\n{'='*60}")
        print("STARTING SIGN LANGUAGE DETECTOR")
        print(f"{'='*60}")
        print("\nControls:")
        print("  SPACE: Pause/Resume")
        print("  'f': Toggle FPS display")
        print("  'l': Toggle landmark display")
        print("  'b': Toggle bounding box")
        print("  's': Toggle finger states")
        print("  't': Toggle top predictions (debug)")
        print("  'r': Reset detector buffer")
        print("  'q' or ESC: Quit")
        print("\nPress any key to start...")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Wait for user
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "Press any key to start...", 
                       (50, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Sign Language Detector', frame)
            cv2.waitKey(0)
        
        print("\n✓ Camera initialized")
        print("✓ Detection started!\n")
        
        # Main loop
        frame_count = 0
        detection_history = []
        
        while True:
            frame_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            if not self.paused:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(rgb_frame)
                
                # Detect and visualize
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get prediction
                        prediction, confidence, is_stable = self.detector.predict(hand_landmarks)
                        
                        # Record detection
                        if is_stable and prediction:
                            detection_history.append({
                                'time': time.time(),
                                'gesture': prediction,
                                'confidence': confidence
                            })
                        
                        # Visualize
                        frame = self._visualize(
                            frame, hand_landmarks, prediction, 
                            confidence, is_stable
                        )
                else:
                    # No hand detected
                    self._draw_no_hand_message(frame)
            else:
                # Paused
                cv2.putText(frame, "PAUSED", 
                          (frame.shape[1]//2 - 100, frame.shape[0]//2),
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            
            # Update FPS
            frame_end = time.time()
            frame_time = frame_end - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            self.fps = 1.0 / np.mean(self.frame_times)
            
            # Draw FPS
            if self.show_fps:
                self.visualizer.draw_fps(frame, self.fps)
            
            # Draw detection history
            self._draw_detection_history(frame, detection_history)
            
            # Display
            cv2.imshow('Sign Language Detector', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord(' '):  # Space
                self.paused = not self.paused
                if self.paused:
                    print("⏸ Paused")
                else:
                    print("▶ Resumed")
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('b'):
                self.show_bbox = not self.show_bbox
            elif key == ord('s'):
                self.show_finger_states = not self.show_finger_states
            elif key == ord('t'):
                self.show_top_predictions = not self.show_top_predictions
                print(f"Top predictions: {'ON' if self.show_top_predictions else 'OFF'}")
            elif key == ord('r'):
                self.detector.reset()
                print("🔄 Detector buffer reset")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {self.fps:.1f}")
        print(f"Unique detections: {len(set(d['gesture'] for d in detection_history))}")
        print(f"Total stable detections: {len(detection_history)}")
        
        # Show most common detections
        if detection_history:
            from collections import Counter
            gesture_counts = Counter(d['gesture'] for d in detection_history)
            print(f"\nMost common gestures:")
            for gesture, count in gesture_counts.most_common(5):
                print(f"  {gesture}: {count} times")
    
    def _visualize(self, frame, hand_landmarks, prediction, confidence, is_stable):
        """Visualize detection results on frame"""
        # Draw landmarks
        if self.show_landmarks:
            frame = self.visualizer.draw_landmarks(frame, hand_landmarks, draw_connections=True)
        
        # Draw bounding box
        if self.show_bbox:
            frame = self.visualizer.draw_bounding_box(frame, hand_landmarks)
        
        # Draw finger states
        if self.show_finger_states:
            frame = self.visualizer.draw_finger_states(frame, hand_landmarks)
        
        # Draw prediction
        frame = self.visualizer.draw_prediction(
            frame, prediction, confidence, is_stable=is_stable
        )
        
        # Draw top predictions (debug mode)
        if self.show_top_predictions and prediction:
            top_preds = self.detector.get_all_probabilities(hand_landmarks, top_k=5)
            frame = self.visualizer.draw_top_predictions(frame, top_preds)
        
        return frame
    
    def _draw_no_hand_message(self, frame):
        """Draw message when no hand is detected"""
        h, w = frame.shape[:2]
        message = "Show your hand to the camera"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        
        x = (w - text_size[0]) // 2
        y = h - 100
        
        cv2.putText(frame, message, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    
    def _draw_detection_history(self, frame, history, max_display=5):
        """Draw recent detection history"""
        if not history:
            return
        
        # Get recent unique detections
        recent = []
        seen = set()
        for detection in reversed(history[-20:]):
            gesture = detection['gesture']
            if gesture not in seen:
                recent.append(detection)
                seen.add(gesture)
            if len(recent) >= max_display:
                break
        
        if not recent:
            return
        
        # Draw history panel
        h, w = frame.shape[:2]
        x_start = w - 250
        y_start = 100
        
        cv2.putText(frame, "Recent Detections:", 
                   (x_start, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, detection in enumerate(recent):
            y = y_start + 30 + i * 30
            text = f"{detection['gesture']} ({detection['confidence']:.0%})"
            cv2.putText(frame, text, (x_start, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='High-Accuracy Sign Language Detector with MediaPipe'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        help='Path to trained model file (.pkl)'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.80,
        help='Confidence threshold (0.75-0.85 recommended, default: 0.80)'
    )
    
    parser.add_argument(
        '--buffer', '-b',
        type=int,
        default=5,
        help='Temporal smoothing buffer size (3-7 recommended, default: 5)'
    )
    
    parser.add_argument(
        '--detector',
        type=str,
        choices=['standard', 'context_aware', 'adaptive'],
        default='standard',
        help='Detector type (default: standard)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Frame width (default: 1280)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Frame height (default: 720)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='Target FPS (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Get model path
    if args.model:
        model_path = args.model
    else:
        # Try to find model in current directory
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'model' in f.lower()]
        
        if pkl_files:
            print("Available model files:")
            for i, f in enumerate(pkl_files, 1):
                print(f"  {i}. {f}")
            
            choice = input(f"\nSelect model (1-{len(pkl_files)}, or enter path): ").strip()
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(pkl_files):
                    model_path = pkl_files[idx]
                else:
                    model_path = choice
            except ValueError:
                model_path = choice
        else:
            model_path = input("Enter path to model file: ").strip()
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("\nPlease train a model first using:")
        print("  python data_collector.py  # Collect training data")
        print("  python model_trainer.py   # Train the model")
        return
    
    # Create and run app
    app = SignLanguageApp(
        model_path=model_path,
        confidence_threshold=args.confidence,
        buffer_size=args.buffer,
        detector_type=args.detector
    )
    
    app.run(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        target_fps=args.fps
    )


if __name__ == "__main__":
    main()
