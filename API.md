# ISight API Documentation

## Module Overview

### feature_extraction.py

#### `extract_features(hand_landmarks)`
Extracts comprehensive feature vector from MediaPipe hand landmarks.

**Parameters:**
- `hand_landmarks`: MediaPipe hand landmarks object

**Returns:**
- `numpy.ndarray`: Feature vector of length ~97

**Features Included:**
- 63 normalized 3D landmark positions
- 5 finger states (extended/bent)
- 10 joint angles
- 12 landmark distances
- 3 palm orientation values
- 4 bounding box features

---

### data_collector.py

#### `class DataCollector`

**Methods:**

##### `__init__(save_dir='training_data')`
Initialize data collector.

##### `collect_samples(gesture_name, num_samples=150, show_instructions=True)`
Collect training samples for a gesture.

**Parameters:**
- `gesture_name` (str): Name of the gesture
- `num_samples` (int): Number of samples to collect
- `show_instructions` (bool): Show collection instructions

##### `save_data(filename=None)`
Save collected data to file.

---

### model_trainer.py

#### `class ModelTrainer`

**Methods:**

##### `__init__(data_filepath=None)`
Initialize model trainer.

##### `prepare_data(test_size=0.2, random_state=42)`
Prepare data for training.

##### `train_random_forest(optimize=False)`
Train Random Forest classifier.

**Parameters:**
- `optimize` (bool): Perform hyperparameter optimization

**Returns:**
- `float`: Cross-validation accuracy

##### `train_all_models(optimize_rf=False)`
Train all available models and select the best.

##### `save_model(filename='sign_language_model.pkl')`
Save trained model to file.

---

### detector.py

#### `class SignLanguageDetector`

**Methods:**

##### `__init__(model_data, confidence_threshold=0.80, buffer_size=5)`
Initialize detector.

**Parameters:**
- `model_data` (dict): Model data including model, scaler, label_encoder
- `confidence_threshold` (float): Minimum confidence (0.75-0.85)
- `buffer_size` (int): Frames for temporal smoothing (3-7)

##### `predict(hand_landmarks, features=None)`
Predict gesture from hand landmarks.

**Parameters:**
- `hand_landmarks`: MediaPipe hand landmarks
- `features` (optional): Pre-extracted features

**Returns:**
- `tuple`: (prediction, confidence, is_stable)

##### `get_top_k_predictions(hand_landmarks, k=3)`
Get top k predictions with confidences.

**Parameters:**
- `hand_landmarks`: MediaPipe hand landmarks
- `k` (int): Number of top predictions

**Returns:**
- `list`: Top k predictions as (gesture, confidence) tuples

---

### evaluator.py

#### `class AccuracyEvaluator`

**Methods:**

##### `__init__(model_filepath)`
Initialize evaluator with trained model.

##### `test_real_time_accuracy(gesture_name, num_samples=30)`
Test accuracy for a specific gesture.

**Parameters:**
- `gesture_name` (str): Gesture to test
- `num_samples` (int): Number of test samples

**Returns:**
- `float`: Accuracy percentage

##### `benchmark_fps(duration=30)`
Benchmark FPS performance.

**Parameters:**
- `duration` (int): Test duration in seconds

**Returns:**
- `dict`: Performance statistics

---

### visualization.py

#### `class HandVisualizer`

**Methods:**

##### `__init__()`
Initialize hand visualizer.

##### `draw_landmarks(frame, hand_landmarks)`
Draw hand landmarks on frame.

##### `draw_bounding_box(frame, hand_landmarks)`
Draw bounding box around hand.

##### `draw_finger_states(frame, hand_landmarks, position)`
Draw finger state indicators.

---

## Usage Examples

### Basic Detection

```python
from detector import load_detector
import cv2

detector = load_detector('model.pkl')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Process frame...
    prediction, confidence, is_stable = detector.predict(hand_landmarks)
    print(f"Detected: {prediction} ({confidence:.2f})")
```

### Custom Training

```python
from model_trainer import ModelTrainer

trainer = ModelTrainer('data.pkl')
trainer.prepare_data()
trainer.train_random_forest(optimize=True)
trainer.save_model('my_model.pkl')
```

### Batch Evaluation

```python
from evaluator import AccuracyEvaluator

evaluator = AccuracyEvaluator('model.pkl')
gestures = ['A', 'B', 'C']
for gesture in gestures:
    acc = evaluator.test_real_time_accuracy(gesture, 50)
    print(f"{gesture}: {acc:.1f}%")
```
