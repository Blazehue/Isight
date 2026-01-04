"""
Quick Start Script for Sign Language Detector

This script demonstrates the complete workflow from data collection to real-time detection,
providing step-by-step guidance for setting up the sign language detection system.

Author: Blazehue
Date: January 2026
"""

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  HIGH-ACCURACY SIGN LANGUAGE DETECTOR WITH MEDIAPIPE        ║
    ║  Quick Start Guide                                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n📋 COMPLETE WORKFLOW:\n")
    
    print("Step 1: Data Collection")
    print("=" * 60)
    print("Collect training samples for each gesture you want to recognize.")
    print("Recommended: 100-200 samples per gesture with variation in:")
    print("  - Position (center, left, right, top, bottom)")
    print("  - Distance (close, medium, far)")
    print("  - Angle (palm facing, angled, side view)")
    print("\nCommand:")
    print("  python data_collector.py")
    print("\nExample for single gesture:")
    print("  python -c \"from data_collector import DataCollector;")
    print("             dc = DataCollector();")
    print("             dc.collect_samples('Hello', 150);")
    print("             dc.save_data('hello_data.pkl')\"")
    
    print("\n" + "=" * 60)
    print("Step 2: Model Training")
    print("=" * 60)
    print("Train multiple ML models and automatically select the best one.")
    print("Models tested: Random Forest, SVM, Neural Network, Gradient Boosting")
    print("\nCommand:")
    print("  python model_trainer.py path/to/training_data.pkl")
    print("\nQuick training:")
    print("  python -c \"from model_trainer import ModelTrainer;")
    print("             mt = ModelTrainer('training_data.pkl');")
    print("             mt.prepare_data();")
    print("             mt.train_all_models();")
    print("             mt.evaluate_model();")
    print("             mt.save_model()\"")
    
    print("\n" + "=" * 60)
    print("Step 3: Real-Time Detection")
    print("=" * 60)
    print("Run the detector with your trained model.")
    print("\nBasic usage:")
    print("  python sign_language_detector.py --model model.pkl")
    print("\nWith custom settings:")
    print("  python sign_language_detector.py \\")
    print("      --model model.pkl \\")
    print("      --confidence 0.80 \\")
    print("      --buffer 5 \\")
    print("      --detector context_aware")
    
    print("\n" + "=" * 60)
    print("Step 4: Accuracy Evaluation")
    print("=" * 60)
    print("Test your model's accuracy comprehensively.")
    print("\nCommand:")
    print("  python evaluator.py model.pkl")
    print("\nQuick single gesture test:")
    print("  python -c \"from evaluator import AccuracyEvaluator;")
    print("             ev = AccuracyEvaluator('model.pkl');")
    print("             ev.test_real_time_accuracy('Hello', 50)\"")
    
    print("\n" + "=" * 60)
    print("\n💡 PRO TIPS FOR HIGH ACCURACY:\n")
    print("1. Data Quality Matters:")
    print("   - Vary position, distance, and angle during collection")
    print("   - Collect in different lighting conditions")
    print("   - Keep gestures consistent but context varied")
    
    print("\n2. Optimize Confidence Threshold:")
    print("   - Start with 0.80")
    print("   - Increase to 0.85 if too many false positives")
    print("   - Decrease to 0.75 if missing valid gestures")
    
    print("\n3. Handle Confusing Pairs:")
    print("   - ASL letters A & S are very similar")
    print("   - M & N differ only slightly")
    print("   - Collect extra data for these pairs")
    
    print("\n4. Temporal Smoothing:")
    print("   - Buffer size 5 = good stability")
    print("   - Smaller (3) = faster response, less stable")
    print("   - Larger (7) = more stable, slower response")
    
    print("\n5. Feature Engineering:")
    print("   - System extracts ~97 features per frame:")
    print("     * 63 landmark positions (21 points × 3D)")
    print("     * 5 finger states (extended/bent)")
    print("     * 10 joint angles")
    print("     * 12 landmark distances")
    print("     * 3 palm orientation")
    print("     * 4 bounding box features")
    
    print("\n" + "=" * 60)
    print("\n📊 TARGET PERFORMANCE METRICS:\n")
    print("✓ Overall Accuracy: >95%")
    print("✓ Per-Gesture Accuracy: >90%")
    print("✓ FPS: 55-60 with full features")
    print("✓ False Positive Rate: <5%")
    print("✓ Detection Confidence: >0.75")
    
    print("\n" + "=" * 60)
    print("\n🔧 KEYBOARD CONTROLS (Main App):\n")
    print("  SPACE     - Pause/Resume")
    print("  F         - Toggle FPS display")
    print("  L         - Toggle landmarks")
    print("  B         - Toggle bounding box")
    print("  S         - Toggle finger states")
    print("  T         - Toggle top predictions (debug)")
    print("  R         - Reset detector buffer")
    print("  Q or ESC  - Quit")
    
    print("\n" + "=" * 60)
    print("\n📁 PROJECT STRUCTURE:\n")
    print("  feature_extraction.py      - Extract 97 features from landmarks")
    print("  data_collector.py          - Interactive data collection")
    print("  model_trainer.py           - Train and compare models")
    print("  detector.py                - Real-time detection variants")
    print("  visualization.py           - Visualization tools")
    print("  evaluator.py               - Accuracy testing")
    print("  sign_language_detector.py  - Main application")
    
    print("\n" + "=" * 60)
    print("\n🚀 EXAMPLE: Full ASL Alphabet Detector\n")
    
    print("# Step 1: Collect data for all letters")
    print("from data_collector import DataCollector")
    print("collector = DataCollector()")
    print("asl_alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M',")
    print("                'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']")
    print("collector.collect_multiple_gestures(asl_alphabet, samples_per_gesture=150)")
    print("collector.save_data('asl_alphabet.pkl')")
    
    print("\n# Step 2: Train model")
    print("from model_trainer import ModelTrainer")
    print("trainer = ModelTrainer('asl_alphabet.pkl')")
    print("trainer.prepare_data()")
    print("trainer.train_all_models(optimize_rf=True)")
    print("trainer.evaluate_model(show_plots=True)")
    print("trainer.save_model('asl_detector.pkl')")
    
    print("\n# Step 3: Test accuracy")
    print("from evaluator import AccuracyEvaluator")
    print("evaluator = AccuracyEvaluator('asl_detector.pkl')")
    print("results = evaluator.test_all_gestures(samples_per_gesture=30)")
    print("evaluator.analyze_confusion_pairs(results)")
    
    print("\n# Step 4: Run detector")
    print("python sign_language_detector.py --model asl_detector.pkl")
    
    print("\n" + "=" * 60)
    print("\n❓ TROUBLESHOOTING:\n")
    print("Low Accuracy?")
    print("  1. Check confusion matrix to identify problem gestures")
    print("  2. Collect more varied data for those gestures")
    print("  3. Increase confidence threshold to reduce false positives")
    print("  4. Use context_aware detector for sequence-based improvement")
    
    print("\nLow FPS?")
    print("  1. Reduce camera resolution (1280x720 → 640x480)")
    print("  2. Use model_complexity=0 in MediaPipe (lite model)")
    print("  3. Increase frame skip interval")
    print("  4. Disable detailed visualization")
    
    print("\nHand Not Detected?")
    print("  1. Ensure good lighting")
    print("  2. Check camera is not blocked")
    print("  3. Lower min_detection_confidence (0.7 → 0.5)")
    print("  4. Move hand closer to camera")
    
    print("\n" + "=" * 60)
    print("\n✨ Ready to build your high-accuracy sign language detector!")
    print("Start with: python data_collector.py\n")


if __name__ == "__main__":
    main()
