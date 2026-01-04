# Examples

This directory contains example scripts and usage patterns for ISight.

## Basic Examples

### Example 1: Simple Detection

```python
from detector import load_detector
import cv2
import mediapipe as mp

# Load model
detector = load_detector('sign_language_model.pkl')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        prediction, confidence, is_stable = detector.predict(hand_landmarks)
        
        if prediction:
            print(f"Detected: {prediction} ({confidence:.2f})")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: Batch Processing

```python
from evaluator import AccuracyEvaluator

evaluator = AccuracyEvaluator('model.pkl')

gestures_to_test = ['A', 'B', 'C', 'D', 'E']
results = {}

for gesture in gestures_to_test:
    accuracy = evaluator.test_real_time_accuracy(gesture, num_samples=50)
    results[gesture] = accuracy
    print(f"{gesture}: {accuracy:.1f}% accuracy")
```

### Example 3: Custom Training

```python
from data_collector import DataCollector
from model_trainer import ModelTrainer

# Collect data
collector = DataCollector()
gestures = ['Hello', 'Thanks', 'Please']

for gesture in gestures:
    collector.collect_samples(gesture, num_samples=150)

collector.save_data('custom_data.pkl')

# Train model
trainer = ModelTrainer('custom_data.pkl')
trainer.prepare_data()
trainer.train_all_models(optimize_rf=True)
trainer.evaluate_model(show_plots=True)
trainer.save_model('custom_model.pkl')
```

## See Also

- [User Guide](../USER_GUIDE.md) for detailed tutorials
- [API Documentation](../API.md) for complete API reference
