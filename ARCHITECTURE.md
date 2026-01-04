# ISight - Sign Language Detector

## Project Structure

```
mediapipe/
├── data_collector.py       # Data collection module
├── detector.py             # Real-time detection engine
├── evaluator.py            # Model evaluation and testing
├── feature_extraction.py   # Feature engineering
├── model_trainer.py        # ML model training
├── sign_language_detector.py  # Main application
├── visualization.py        # Visualization tools
├── quick_start.py         # Quick start guide
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── logger.py              # Logging module
├── exceptions.py          # Custom exceptions
├── requirements.txt       # Dependencies
├── README.md             # Main documentation
├── USER_GUIDE.md         # User guide
├── CONTRIBUTING.md       # Contribution guidelines
├── CHANGELOG.md          # Version history
└── LICENSE               # MIT License
```

## Features

- MediaPipe-based hand landmark detection
- Multiple ML model support
- Real-time gesture recognition
- Comprehensive evaluation tools
- Easy data collection interface
- Detailed visualization

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Collect training data: `python data_collector.py`
3. Train models: `python model_trainer.py`
4. Run detector: `python sign_language_detector.py`

For detailed instructions, see [README.md](README.md).
