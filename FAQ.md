# Frequently Asked Questions (FAQ)

## General Questions

### What is ISight?
ISight is a high-accuracy sign language detection system that uses Google's MediaPipe Hands to recognize hand gestures in real-time with >95% accuracy.

### What programming language is it written in?
ISight is written in Python 3.8+ and uses popular libraries like OpenCV, MediaPipe, and scikit-learn.

### Is this for ASL (American Sign Language)?
Yes, but it can be adapted for any sign language system. The core technology recognizes hand shapes and positions, which can be trained on any gesture set.

### Can it recognize full sentences?
Currently, ISight is optimized for single static gestures. Sentence recognition would require additional sequence modeling capabilities.

---

## Technical Questions

### How many gestures can it recognize?
There's no hard limit, but we recommend starting with 10-30 gestures for optimal accuracy. More gestures may require more training data and potentially lower individual gesture accuracy.

### What's the detection speed?
ISight runs at 55-60 FPS on modern hardware, providing real-time detection with minimal latency (~18ms per frame).

### What accuracy can I expect?
With proper training data (150+ samples per gesture), you can achieve:
- Overall accuracy: >95%
- Per-gesture accuracy: >90%
- False positive rate: <5%

### What hardware do I need?
- CPU: Any modern processor (Intel i5/AMD Ryzen 5 or better)
- RAM: 8GB minimum, 16GB recommended
- Camera: Any webcam (720p or higher recommended)
- GPU: Not required, but MediaPipe can use GPU acceleration if available

---

## Setup & Installation

### What dependencies are required?
- Python 3.8+
- opencv-python
- mediapipe
- numpy
- scikit-learn
- matplotlib
- seaborn

Install all with: `pip install -r requirements.txt`

### Why is MediaPipe not installing?
Try:
```bash
pip install mediapipe --no-cache-dir
```
If that fails, ensure you have the correct Python version (3.8-3.11).

### Can I use this on a Raspberry Pi?
MediaPipe has experimental support for Raspberry Pi 4, but performance may be limited. Expect lower FPS and may need to reduce model complexity.

---

## Data Collection

### How many samples should I collect per gesture?
- Minimum: 50 samples
- Recommended: 150 samples
- More is better if you vary position, distance, and angle

### Why is variation important?
Variation ensures the model generalizes well. Collect samples with:
- Different hand positions (center, left, right, top, bottom)
- Different distances (close, medium, far)
- Different angles (palm facing camera, angled, side view)

### Can multiple people contribute training data?
Yes! Data from multiple people improves generalization and makes the system work better for everyone.

### Should I collect data in different lighting?
Yes, collecting samples in various lighting conditions helps the model work robustly in different environments.

---

## Training

### Which model should I use?
- **Random Forest**: Best balance of accuracy and speed (recommended)
- **SVM**: Good accuracy, slightly slower
- **Neural Network**: May overfit with small datasets
- **Gradient Boosting**: High accuracy, slower training

### How long does training take?
- Random Forest: 5-10 seconds
- SVM: 30-60 seconds
- Neural Network: 60-90 seconds
- Gradient Boosting: 90-120 seconds

### What if my accuracy is low?
1. Collect more diverse training data
2. Ensure gestures are distinct
3. Check for mislabeled samples
4. Review confusion matrix to identify problem gestures

---

## Detection

### Why are my predictions unstable?
Increase the buffer_size for more temporal smoothing:
```python
detector = SignLanguageDetector(model_data, buffer_size=7)
```

### How do I adjust confidence threshold?
```python
detector = SignLanguageDetector(model_data, confidence_threshold=0.85)
```
- Lower (0.75): More detections, may include false positives
- Higher (0.85): Fewer detections, more certain

### Why isn't my hand being detected?
- Ensure good lighting
- Keep hand 30-100cm from camera
- Show palm clearly to camera
- Check MediaPipe settings (min_detection_confidence)

---

## Troubleshooting

### Camera not working
1. Check camera permissions
2. Try different camera index (0, 1, 2...)
3. Ensure camera isn't used by another app
4. Test with: `cap = cv2.VideoCapture(0)`

### Low FPS
1. Reduce camera resolution to 640x480
2. Use model_complexity=0
3. Reduce buffer_size to 3
4. Close other applications

### Getting confused gestures
1. Identify confusion pairs (e.g., A vs S)
2. Collect more data for those specific gestures
3. Use higher confidence threshold
4. Ensure gestures are visually distinct

---

## Advanced Usage

### Can I use two hands?
Yes, set `max_num_hands=2` in MediaPipe settings. You'll need to modify the feature extraction and model to handle dual-hand gestures.

### Can I add custom features?
Yes! Edit `feature_extraction.py` and add your custom features to the `extract_features()` function.

### Can I integrate this into my application?
Absolutely! ISight is designed to be modular. Import the `SignLanguageDetector` class and use it in your own Python applications.

### Is there an API?
Currently, ISight is a library/toolkit. You can build your own REST API around it using Flask or FastAPI.

---

## Contributing

### How can I contribute?
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome:
- Bug fixes
- Performance improvements
- Documentation updates
- New features
- Training datasets

### Can I use this commercially?
Yes! ISight is licensed under the MIT License, allowing commercial use. See [LICENSE](LICENSE) for details.

---

## Support

### Where can I get help?
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review [API.md](API.md) for code reference
3. Search existing GitHub issues
4. Open a new issue with details

### How do I report a bug?
Open a GitHub issue with:
- Error message
- Steps to reproduce
- System information (OS, Python version)
- Expected vs actual behavior
