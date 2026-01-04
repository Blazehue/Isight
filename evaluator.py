"""
Evaluation Module for Sign Language Detection
Comprehensive accuracy testing, debugging, and performance analysis
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os


class AccuracyEvaluator:
    """
    Evaluate model accuracy across different conditions and scenarios.
    """
    
    def __init__(self, model_filepath):
        """
        Initialize evaluator with trained model.
        
        Args:
            model_filepath: Path to saved model file
        """
        # Load model
        with open(model_filepath, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.label_encoder = self.model_data['label_encoder']
        self.gesture_names = self.model_data['gesture_names']
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        print(f"AccuracyEvaluator initialized")
        print(f"  Model: {self.model_data['model_name']}")
        print(f"  Training accuracy: {self.model_data['accuracy']:.4f}")
        print(f"  Gestures: {len(self.gesture_names)}")
    
    def test_real_time_accuracy(self, gesture_name, num_samples=50, 
                               confidence_threshold=0.80):
        """
        Test real-time accuracy for a specific gesture.
        User performs the gesture and we measure detection accuracy.
        
        Args:
            gesture_name: Gesture to test
            num_samples: Number of samples to collect for testing
            confidence_threshold: Minimum confidence for valid detection
        
        Returns:
            dict: Test results
        """
        print(f"\n{'='*60}")
        print(f"TESTING REAL-TIME ACCURACY: {gesture_name.upper()}")
        print(f"{'='*60}")
        print(f"Samples to collect: {num_samples}")
        print(f"Confidence threshold: {confidence_threshold:.2f}")
        print("\nPerform the gesture consistently and hold it steady.")
        print("Press any key to start...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Wait for user
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "Press any key to start testing...", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Accuracy Test", frame)
                if cv2.waitKey(1) != -1:
                    break
        
        # Collect samples
        correct = 0
        incorrect = 0
        no_detection = 0
        low_confidence = 0
        predictions_list = []
        confidences_list = []
        
        count = 0
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract features and predict
                    from feature_extraction import extract_features
                    features = extract_features(hand_landmarks)
                    features = np.array(features).reshape(1, -1)
                    features_scaled = self.scaler.transform(features)
                    
                    # Predict
                    if hasattr(self.model, 'predict_proba'):
                        probs = self.model.predict_proba(features_scaled)[0]
                        pred_idx = np.argmax(probs)
                        confidence = probs[pred_idx]
                        prediction = self.label_encoder.classes_[pred_idx]
                    else:
                        pred_idx = self.model.predict(features_scaled)[0]
                        prediction = self.label_encoder.classes_[pred_idx]
                        confidence = 1.0
                    
                    # Record result
                    predictions_list.append(prediction)
                    confidences_list.append(confidence)
                    
                    if confidence < confidence_threshold:
                        low_confidence += 1
                        status_text = f"Low Confidence: {confidence:.2%}"
                        status_color = (0, 165, 255)
                    elif prediction == gesture_name:
                        correct += 1
                        status_text = f"CORRECT: {prediction} ({confidence:.2%})"
                        status_color = (0, 255, 0)
                    else:
                        incorrect += 1
                        status_text = f"WRONG: {prediction} (expected {gesture_name})"
                        status_color = (0, 0, 255)
                    
                    count += 1
                    
                    # Draw feedback
                    cv2.putText(frame, f"Sample {count}/{num_samples}", 
                              (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, status_text,
                              (20, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.8, status_color, 2)
                    
                    # Progress bar
                    progress = count / num_samples
                    bar_width = int((frame.shape[1] - 40) * progress)
                    cv2.rectangle(frame, (20, 60), (frame.shape[1] - 20, 80), (50, 50, 50), -1)
                    cv2.rectangle(frame, (20, 60), (20 + bar_width, 80), (0, 255, 0), -1)
            else:
                no_detection += 1
                cv2.putText(frame, "NO HAND DETECTED", 
                          (20, frame.shape[0] - 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Accuracy Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate metrics
        total_detections = correct + incorrect + low_confidence
        accuracy = correct / total_detections if total_detections > 0 else 0
        avg_confidence = np.mean(confidences_list) if confidences_list else 0
        
        # Find common misclassifications
        wrong_predictions = [p for p in predictions_list if p != gesture_name]
        common_mistakes = Counter(wrong_predictions).most_common(3)
        
        # Results
        results = {
            'gesture': gesture_name,
            'total_samples': count,
            'correct': correct,
            'incorrect': incorrect,
            'low_confidence': low_confidence,
            'no_detection': no_detection,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'all_predictions': predictions_list,
            'all_confidences': confidences_list,
            'common_mistakes': common_mistakes
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {gesture_name}")
        print(f"{'='*60}")
        print(f"Total samples: {count}")
        print(f"Correct predictions: {correct} ({correct/count*100:.1f}%)")
        print(f"Incorrect predictions: {incorrect} ({incorrect/count*100:.1f}%)")
        print(f"Low confidence: {low_confidence} ({low_confidence/count*100:.1f}%)")
        print(f"No detection: {no_detection}")
        print(f"\nAccuracy (valid detections): {accuracy:.2%}")
        print(f"Average confidence: {avg_confidence:.2%}")
        
        if common_mistakes:
            print(f"\nCommon misclassifications:")
            for mistake, count_mistakes in common_mistakes:
                print(f"  {mistake}: {count_mistakes} times")
        
        return results
    
    def test_all_gestures(self, samples_per_gesture=30, confidence_threshold=0.80):
        """
        Test accuracy for all gestures in the model.
        
        Args:
            samples_per_gesture: Number of samples to test per gesture
            confidence_threshold: Minimum confidence for valid detection
        
        Returns:
            dict: Complete test results
        """
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ACCURACY TEST")
        print(f"{'='*60}")
        print(f"Gestures to test: {len(self.gesture_names)}")
        print(f"Samples per gesture: {samples_per_gesture}")
        print(f"Total samples: {len(self.gesture_names) * samples_per_gesture}")
        
        all_results = {}
        
        for i, gesture in enumerate(self.gesture_names, 1):
            print(f"\n[{i}/{len(self.gesture_names)}] Next gesture: {gesture}")
            input("Press Enter when ready...")
            
            results = self.test_real_time_accuracy(
                gesture, samples_per_gesture, confidence_threshold
            )
            all_results[gesture] = results
            
            # Short break
            if i < len(self.gesture_names):
                print("\nTake a 3-second break...")
                time.sleep(3)
        
        # Overall statistics
        self._print_overall_statistics(all_results)
        self._plot_overall_results(all_results)
        
        return all_results
    
    def _print_overall_statistics(self, all_results):
        """Print overall statistics across all gestures"""
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS")
        print(f"{'='*60}")
        
        total_samples = sum(r['total_samples'] for r in all_results.values())
        total_correct = sum(r['correct'] for r in all_results.values())
        total_incorrect = sum(r['incorrect'] for r in all_results.values())
        total_low_conf = sum(r['low_confidence'] for r in all_results.values())
        
        overall_accuracy = total_correct / (total_correct + total_incorrect + total_low_conf)
        
        print(f"Total samples: {total_samples}")
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        print(f"  Correct: {total_correct}")
        print(f"  Incorrect: {total_incorrect}")
        print(f"  Low confidence: {total_low_conf}")
        
        # Per-gesture accuracy
        print(f"\nPer-gesture accuracy:")
        gesture_accuracies = [(g, r['accuracy']) for g, r in all_results.items()]
        gesture_accuracies.sort(key=lambda x: x[1])
        
        for gesture, accuracy in gesture_accuracies:
            status = "✓" if accuracy >= 0.95 else "⚠" if accuracy >= 0.90 else "✗"
            print(f"  {status} {gesture:20s}: {accuracy:.2%}")
        
        # Find problematic gestures
        low_acc = [g for g, acc in gesture_accuracies if acc < 0.90]
        if low_acc:
            print(f"\n⚠ Gestures needing improvement (<90% accuracy):")
            for gesture in low_acc:
                results = all_results[gesture]
                print(f"  {gesture}: {results['accuracy']:.2%}")
                if results['common_mistakes']:
                    print(f"    Often confused with: {results['common_mistakes'][0][0]}")
    
    def _plot_overall_results(self, all_results):
        """Plot overall results"""
        gestures = list(all_results.keys())
        accuracies = [all_results[g]['accuracy'] for g in gestures]
        avg_confidences = [all_results[g]['avg_confidence'] for g in gestures]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Accuracy bar chart
        colors = ['green' if acc >= 0.95 else 'orange' if acc >= 0.90 else 'red' 
                 for acc in accuracies]
        ax1.bar(gestures, accuracies, color=colors, alpha=0.7)
        ax1.axhline(y=0.95, color='g', linestyle='--', label='95% target')
        ax1.axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
        ax1.set_xlabel('Gesture')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Real-Time Accuracy by Gesture')
        ax1.set_ylim([0, 1.0])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Average confidence
        ax2.bar(gestures, avg_confidences, color='blue', alpha=0.6)
        ax2.set_xlabel('Gesture')
        ax2.set_ylabel('Average Confidence')
        ax2.set_title('Average Prediction Confidence by Gesture')
        ax2.set_ylim([0, 1.0])
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('accuracy_evaluation_results.png', dpi=300)
        print(f"\n✓ Results saved to: accuracy_evaluation_results.png")
        plt.close()
    
    def analyze_confusion_pairs(self, test_results):
        """
        Analyze which gesture pairs are commonly confused.
        
        Args:
            test_results: Results from test_all_gestures()
        
        Returns:
            list: Most commonly confused pairs
        """
        confusion_pairs = defaultdict(int)
        
        for true_gesture, results in test_results.items():
            for pred_gesture in results['all_predictions']:
                if pred_gesture != true_gesture:
                    pair = tuple(sorted([true_gesture, pred_gesture]))
                    confusion_pairs[pair] += 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*60}")
        print("MOST COMMONLY CONFUSED GESTURE PAIRS")
        print(f"{'='*60}")
        
        for i, (pair, count) in enumerate(sorted_pairs[:10], 1):
            print(f"{i:2d}. {pair[0]} <-> {pair[1]}: {count} confusions")
        
        return sorted_pairs
    
    def test_under_conditions(self, gesture_name, conditions, samples_per_condition=20):
        """
        Test accuracy under different conditions (lighting, distance, angle, etc.)
        
        Args:
            gesture_name: Gesture to test
            conditions: List of condition names
            samples_per_condition: Samples to collect per condition
        
        Returns:
            dict: Results per condition
        """
        print(f"\n{'='*60}")
        print(f"CONDITION-BASED TESTING: {gesture_name}")
        print(f"{'='*60}")
        
        condition_results = {}
        
        for condition in conditions:
            print(f"\nCondition: {condition}")
            print("Set up the testing environment for this condition.")
            input("Press Enter when ready...")
            
            results = self.test_real_time_accuracy(
                gesture_name, samples_per_condition, confidence_threshold=0.80
            )
            condition_results[condition] = results
        
        # Compare results
        print(f"\n{'='*60}")
        print(f"CONDITION COMPARISON: {gesture_name}")
        print(f"{'='*60}")
        
        for condition, results in condition_results.items():
            print(f"{condition:20s}: {results['accuracy']:.2%} (avg conf: {results['avg_confidence']:.2%})")
        
        return condition_results
    
    def benchmark_fps(self, duration=30):
        """
        Benchmark FPS performance.
        
        Args:
            duration: Test duration in seconds
        
        Returns:
            dict: Performance metrics
        """
        print(f"\n{'='*60}")
        print(f"FPS BENCHMARK")
        print(f"{'='*60}")
        print(f"Duration: {duration} seconds")
        print("\nShow your hand to test processing speed...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        frame_times = []
        detection_times = []
        prediction_times = []
        total_frames = 0
        frames_with_hand = 0
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe detection
            detect_start = time.time()
            results = self.hands.process(rgb_frame)
            detect_end = time.time()
            detection_times.append(detect_end - detect_start)
            
            if results.multi_hand_landmarks:
                frames_with_hand += 1
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Feature extraction and prediction
                    pred_start = time.time()
                    from feature_extraction import extract_features
                    features = extract_features(hand_landmarks)
                    features = np.array(features).reshape(1, -1)
                    features_scaled = self.scaler.transform(features)
                    _ = self.model.predict(features_scaled)
                    pred_end = time.time()
                    prediction_times.append(pred_end - pred_start)
            
            frame_end = time.time()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            total_frames += 1
            
            # Display
            elapsed = time.time() - start_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {duration}s", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("FPS Benchmark", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate metrics
        avg_fps = 1.0 / np.mean(frame_times)
        min_fps = 1.0 / np.max(frame_times)
        max_fps = 1.0 / np.min(frame_times)
        avg_detection_time = np.mean(detection_times) * 1000  # ms
        avg_prediction_time = np.mean(prediction_times) * 1000 if prediction_times else 0
        
        results = {
            'total_frames': total_frames,
            'frames_with_hand': frames_with_hand,
            'duration': duration,
            'avg_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'avg_detection_time_ms': avg_detection_time,
            'avg_prediction_time_ms': avg_prediction_time
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("FPS BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Total frames: {total_frames}")
        print(f"Frames with hand detected: {frames_with_hand}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Min FPS: {min_fps:.1f}")
        print(f"Max FPS: {max_fps:.1f}")
        print(f"Average detection time: {avg_detection_time:.2f} ms")
        print(f"Average prediction time: {avg_prediction_time:.2f} ms")
        
        if avg_fps >= 55:
            print("\n✓ EXCELLENT: Meets 60 FPS target!")
        elif avg_fps >= 45:
            print("\n✓ GOOD: Close to 60 FPS target")
        elif avg_fps >= 30:
            print("\n⚠ ACCEPTABLE: Meets minimum 30 FPS")
        else:
            print("\n✗ POOR: Below 30 FPS - optimization needed")
        
        return results
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'hands'):
            self.hands.close()


def main():
    """Example usage"""
    import sys
    
    print("="*60)
    print("SIGN LANGUAGE ACCURACY EVALUATOR")
    print("="*60)
    
    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        model_file = input("\nEnter path to model file: ").strip()
    
    if not os.path.exists(model_file):
        print(f"Error: Model file not found: {model_file}")
        return
    
    evaluator = AccuracyEvaluator(model_file)
    
    print("\nEvaluation Options:")
    print("1. Test single gesture")
    print("2. Test all gestures")
    print("3. Test under different conditions")
    print("4. FPS benchmark")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        gesture = input("Enter gesture name: ").strip()
        num_samples = int(input("Number of samples (default 50): ").strip() or "50")
        evaluator.test_real_time_accuracy(gesture, num_samples)
    
    elif choice == '2':
        samples = int(input("Samples per gesture (default 30): ").strip() or "30")
        results = evaluator.test_all_gestures(samples)
        evaluator.analyze_confusion_pairs(results)
    
    elif choice == '3':
        gesture = input("Enter gesture name: ").strip()
        conditions = ['normal_lighting', 'low_lighting', 'bright_lighting', 
                     'close_distance', 'far_distance']
        evaluator.test_under_conditions(gesture, conditions)
    
    elif choice == '4':
        duration = int(input("Test duration in seconds (default 30): ").strip() or "30")
        evaluator.benchmark_fps(duration)


if __name__ == "__main__":
    main()
