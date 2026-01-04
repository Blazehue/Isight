"""
Data Collection Module for Sign Language Detection

This module provides an interactive interface for collecting high-quality training samples
with variation in position, angle, and distance for robust gesture recognition.

Author: Blazehue
Date: January 2026
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from datetime import datetime
from feature_extraction import extract_features


class DataCollector:
    """
    Collect diverse training samples for each gesture with proper variation.
    """
    
    def __init__(self, save_dir='training_data'):
        """
        Initialize the DataCollector with specified save directory.
        
        Args:
            save_dir (str): Directory to save training data (default: 'training_data')
        """
        self.data = []
        self.labels = []
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize MediaPipe with optimal settings for data collection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        print(f"DataCollector initialized. Data will be saved to: {save_dir}")
    
    def collect_samples(self, gesture_name, num_samples=150, show_instructions=True):
        """
        Collect multiple samples of each gesture.
        
        Guidelines for quality data:
        - Different hand positions (center, left, right, top, bottom)
        - Different angles (palm facing camera, angled, side view)
        - Different distances from camera (50cm, 1m, 2m)
        - Multiple users if possible for better generalization
        
        Args:
            gesture_name: Name of the gesture/sign to collect
            num_samples: Number of samples to collect (default: 150)
            show_instructions: Whether to show collection instructions
        """
        print(f"\n{'='*60}")
        print(f"COLLECTING DATA FOR GESTURE: {gesture_name.upper()}")
        print(f"{'='*60}")
        
        if show_instructions:
            print("\nINSTRUCTIONS FOR HIGH-QUALITY DATA COLLECTION:")
            print("1. Vary hand position: Move hand around the frame")
            print("2. Vary hand distance: Move closer and farther from camera")
            print("3. Vary hand angle: Rotate hand slightly in different directions")
            print("4. Keep the gesture consistent while varying position/angle")
            print("5. Press SPACE to manually capture a sample")
            print("6. Press 'q' to finish early")
            print("\nPress any key to start collecting...")
            
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        if show_instructions:
            # Wait for user to be ready
            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.putText(frame, "Press any key to start...", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Data Collection", frame)
                    if cv2.waitKey(1) != -1:
                        break
        
        count = 0
        frame_skip = 0
        auto_capture_interval = 10  # Capture every 10 frames automatically
        
        # Track positions to ensure diversity
        position_counts = {
            'center': 0, 'left': 0, 'right': 0, 'top': 0, 'bottom': 0
        }
        
        print(f"\nCollecting {num_samples} samples...")
        print("Show the gesture and move it around!")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Draw collection UI
            self._draw_collection_ui(frame, gesture_name, count, num_samples, position_counts)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
                    
                    # Auto-capture or manual capture
                    key = cv2.waitKey(1) & 0xFF
                    should_capture = False
                    
                    if key == ord(' '):  # Manual capture with spacebar
                        should_capture = True
                    elif frame_skip >= auto_capture_interval:  # Auto capture
                        should_capture = True
                        frame_skip = 0
                    
                    if should_capture:
                        # Extract features
                        features = extract_features(hand_landmarks)
                        
                        # Determine position
                        position = self._determine_hand_position(hand_landmarks)
                        position_counts[position] += 1
                        
                        # Store data
                        self.data.append(features)
                        self.labels.append(gesture_name)
                        count += 1
                        
                        # Flash green to indicate capture
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                    (0, 255, 0), 10)
                    
                    frame_skip += 1
            else:
                # No hand detected
                cv2.putText(frame, "NO HAND DETECTED!", 
                          (50, frame.shape[0] - 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Data Collection", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if count >= 50:  # Minimum samples
                    print(f"\nEarly exit. Collected {count} samples.")
                    break
                else:
                    print(f"\nNeed at least 50 samples. Currently have {count}.")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Collected {count} samples for '{gesture_name}'")
        print(f"  Position distribution: {position_counts}")
        
        return count
    
    def _draw_collection_ui(self, frame, gesture_name, count, total, position_counts):
        """Draw UI elements for data collection"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Gesture name
        cv2.putText(frame, f"Gesture: {gesture_name.upper()}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Progress
        cv2.putText(frame, f"Samples: {count}/{total}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Progress bar
        progress = count / total
        bar_width = int((w - 40) * progress)
        cv2.rectangle(frame, (20, 90), (w - 20, 110), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 90), (20 + bar_width, 110), (0, 255, 0), -1)
        
        # Position guide (show where to move hand)
        guide_y = 150
        cv2.putText(frame, "Move hand to different positions:", 
                   (20, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        positions = ['Center', 'Left', 'Right', 'Top', 'Bottom']
        for i, pos in enumerate(positions):
            pos_lower = pos.lower()
            count_pos = position_counts.get(pos_lower, 0)
            color = (0, 255, 0) if count_pos > total // 10 else (0, 165, 255)
            cv2.putText(frame, f"{pos}: {count_pos}", 
                       (20, guide_y + 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Instructions at bottom
        cv2.putText(frame, "SPACE: Manual capture | Q: Quit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    def _determine_hand_position(self, hand_landmarks):
        """Determine which region of frame the hand is in"""
        # Get center of hand (average of all landmarks)
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Classify position
        if center_x < 0.33:
            if center_y < 0.33:
                return 'top'
            elif center_y > 0.66:
                return 'bottom'
            return 'left'
        elif center_x > 0.66:
            if center_y < 0.33:
                return 'top'
            elif center_y > 0.66:
                return 'bottom'
            return 'right'
        else:
            if center_y < 0.33:
                return 'top'
            elif center_y > 0.66:
                return 'bottom'
            return 'center'
    
    def save_data(self, filename=None):
        """Save collected data to file"""
        if not self.data:
            print("No data to save!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        
        data_dict = {
            'data': np.array(self.data),
            'labels': np.array(self.labels),
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(self.data),
            'num_gestures': len(set(self.labels))
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"\n✓ Data saved to: {filepath}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Unique gestures: {len(set(self.labels))}")
        
        return filepath
    
    def load_data(self, filepath):
        """Load previously collected data"""
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = list(data_dict['data'])
        self.labels = list(data_dict['labels'])
        
        print(f"✓ Loaded data from: {filepath}")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Unique gestures: {len(set(self.labels))}")
        
        return data_dict
    
    def get_statistics(self):
        """Get statistics about collected data"""
        if not self.data:
            print("No data collected yet.")
            return
        
        from collections import Counter
        label_counts = Counter(self.labels)
        
        print(f"\n{'='*60}")
        print("DATA COLLECTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data)}")
        print(f"Unique gestures: {len(label_counts)}")
        print(f"\nSamples per gesture:")
        for gesture, count in sorted(label_counts.items()):
            print(f"  {gesture}: {count}")
        print(f"{'='*60}\n")
    
    def collect_multiple_gestures(self, gesture_list, samples_per_gesture=150):
        """
        Collect data for multiple gestures in sequence.
        
        Args:
            gesture_list: List of gesture names to collect
            samples_per_gesture: Number of samples per gesture
        """
        print(f"\n{'='*60}")
        print(f"COLLECTING DATA FOR {len(gesture_list)} GESTURES")
        print(f"{'='*60}")
        print(f"Gestures: {', '.join(gesture_list)}")
        print(f"Samples per gesture: {samples_per_gesture}")
        print(f"Total samples to collect: {len(gesture_list) * samples_per_gesture}")
        
        for i, gesture in enumerate(gesture_list, 1):
            print(f"\n[{i}/{len(gesture_list)}] Next gesture: {gesture}")
            input("Press Enter when ready...")
            
            self.collect_samples(gesture, samples_per_gesture, show_instructions=(i == 1))
            
            # Short break between gestures
            if i < len(gesture_list):
                print("\nTake a 5-second break...")
                import time
                time.sleep(5)
        
        print(f"\n{'='*60}")
        print("ALL GESTURES COLLECTED!")
        print(f"{'='*60}")
        self.get_statistics()
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """Example usage of DataCollector"""
    collector = DataCollector()
    
    # ASL Alphabet (you can customize this list)
    asl_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # Or start with a smaller set for testing
    test_gestures = ['A', 'B', 'C', 'Hello', 'Thank You']
    
    print("Data Collection for Sign Language Detection")
    print("=" * 60)
    print("\nOptions:")
    print("1. Collect data for a single gesture")
    print("2. Collect data for multiple gestures")
    print("3. Collect full ASL alphabet")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        gesture_name = input("Enter gesture name: ").strip()
        num_samples = int(input("Number of samples (default 150): ").strip() or "150")
        collector.collect_samples(gesture_name, num_samples)
        collector.save_data()
    
    elif choice == '2':
        gestures_input = input("Enter gesture names (comma-separated): ").strip()
        gesture_list = [g.strip() for g in gestures_input.split(',')]
        num_samples = int(input("Samples per gesture (default 150): ").strip() or "150")
        collector.collect_multiple_gestures(gesture_list, num_samples)
        collector.save_data()
    
    elif choice == '3':
        num_samples = int(input("Samples per letter (default 150): ").strip() or "150")
        print("\nThis will collect data for all 26 ASL letters.")
        print(f"Total samples: {26 * num_samples}")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            collector.collect_multiple_gestures(asl_alphabet, num_samples)
            collector.save_data()
    
    collector.get_statistics()


if __name__ == "__main__":
    main()
