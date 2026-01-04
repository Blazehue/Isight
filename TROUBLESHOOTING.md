# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Problem: ModuleNotFoundError for mediapipe
**Solution:**
```bash
pip install --upgrade mediapipe
# If that fails, try:
pip install mediapipe --no-cache-dir
```

#### Problem: OpenCV import error
**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

---

### Camera Issues

#### Problem: Camera not opening (cv2.VideoCapture fails)
**Solutions:**
1. Check camera permissions in system settings
2. Try different camera index:
   ```python
   cap = cv2.VideoCapture(1)  # Try 0, 1, 2...
   ```
3. Check if camera is used by another application

#### Problem: Low FPS / Laggy video
**Solutions:**
1. Reduce camera resolution:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```
2. Use model_complexity=0 for faster processing
3. Close other applications

---

### Detection Issues

#### Problem: No hand detected
**Solutions:**
1. Ensure good lighting
2. Lower min_detection_confidence:
   ```python
   hands = mp_hands.Hands(min_detection_confidence=0.5)
   ```
3. Keep hand at appropriate distance (30-100cm)
4. Show palm clearly to camera

#### Problem: Low prediction confidence
**Solutions:**
1. Collect more training data (150+ samples)
2. Ensure variety in training data
3. Check if gesture is consistent
4. Lower confidence threshold temporarily for testing

#### Problem: Wrong predictions / Confusing gestures
**Solutions:**
1. Identify confusion pairs (e.g., A vs S, M vs N)
2. Collect more data for confused gestures
3. Use higher confidence threshold for those gestures
4. Review confusion matrix to understand errors

---

### Training Issues

#### Problem: Low training accuracy (<90%)
**Solutions:**
1. Collect more diverse training data
2. Ensure gestures are distinct
3. Try different ML algorithms
4. Check for mislabeled data

#### Problem: Model file not saving
**Solutions:**
1. Check write permissions in directory
2. Specify absolute path:
   ```python
   trainer.save_model('C:/path/to/model.pkl')
   ```

#### Problem: Out of memory during training
**Solutions:**
1. Reduce number of samples
2. Use simpler model (Random Forest instead of Neural Network)
3. Close other applications

---

### Performance Issues

#### Problem: Slow FPS (<30)
**Solutions:**
1. Use model_complexity=0:
   ```python
   hands = mp_hands.Hands(model_complexity=0)
   ```
2. Reduce buffer_size:
   ```python
   detector = SignLanguageDetector(model_data, buffer_size=3)
   ```
3. Disable detailed visualization

#### Problem: High CPU usage
**Solutions:**
1. Limit frame rate processing
2. Use static_image_mode=False
3. Process every Nth frame only

---

### Data Collection Issues

#### Problem: Not enough variation in data
**Solutions:**
1. Move hand to different positions
2. Vary distance from camera
3. Change hand angles
4. Collect in different lighting

#### Problem: Hand landmarks not stable
**Solutions:**
1. Increase min_tracking_confidence:
   ```python
   hands = mp_hands.Hands(min_tracking_confidence=0.8)
   ```
2. Keep hand steady during collection
3. Improve lighting conditions

---

## Error Messages

### "Model file not found"
- Check that model.pkl exists in the correct directory
- Specify full path to model file

### "Invalid landmarks"
- Ensure hand is clearly visible
- Check MediaPipe detection is working
- Verify hand landmarks have 21 points

### "Insufficient training data"
- Collect at least 50 samples per gesture
- Aim for 100-150 samples for best results

---

## Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check feature extraction:

```python
from feature_extraction import extract_features
features = extract_features(hand_landmarks)
print(f"Feature count: {len(features)}")
print(f"Feature range: {min(features)} to {max(features)}")
```

---

## Getting Help

1. Check existing issues on GitHub
2. Review documentation in README.md
3. Enable debug logging for detailed error info
4. Open an issue with:
   - Error message
   - System information
   - Steps to reproduce

---

## Known Limitations

- Single hand tracking only (by default)
- Static gestures work better than dynamic signs
- Requires good lighting conditions
- May struggle with similar-looking gestures
- Not designed for full ASL sentence recognition
