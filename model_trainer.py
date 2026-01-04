"""
Model Training Module for Sign Language Detection
Trains multiple models and selects the best performer for maximum accuracy
"""

import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """
    Train and evaluate multiple models for sign language detection.
    Automatically selects the best performing model.
    """
    
    def __init__(self, data_filepath=None):
        self.data = None
        self.labels = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
        
        if data_filepath:
            self.load_data(data_filepath)
    
    def load_data(self, filepath):
        """Load training data from pickle file"""
        print(f"Loading data from: {filepath}")
        
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        self.data = data_dict['data']
        self.labels = data_dict['labels']
        
        print(f"✓ Loaded {len(self.data)} samples")
        print(f"✓ {len(set(self.labels))} unique gestures")
        
        # Check for class imbalance
        from collections import Counter
        label_counts = Counter(self.labels)
        min_samples = min(label_counts.values())
        max_samples = max(label_counts.values())
        
        if max_samples / min_samples > 2:
            print(f"⚠ Warning: Class imbalance detected!")
            print(f"  Min samples: {min_samples}, Max samples: {max_samples}")
            print(f"  Consider collecting more data for underrepresented gestures.")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for training: split, scale, encode labels
        """
        print("\nPreparing data...")
        
        # Check if we have enough classes
        unique_labels = len(set(self.labels))
        if unique_labels < 2:
            print(f"\n⚠ ERROR: Only {unique_labels} gesture found in training data!")
            print("You need at least 2 different gestures to train a classifier.")
            print("\nPlease collect data for multiple gestures:")
            print("  python data_collector.py")
            print("\nExample gestures to collect:")
            print("  - ASL letters: A, B, C, D, E, ...")
            print("  - Common signs: Hello, Thank You, Please, Sorry, ...")
            print("  - Numbers: 1, 2, 3, 4, 5, ...")
            raise ValueError(f"Need at least 2 gestures, but only found {unique_labels}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded  # Maintain class distribution
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Training samples: {len(self.X_train)}")
        print(f"✓ Testing samples: {len(self.X_test)}")
        print(f"✓ Features per sample: {self.X_train.shape[1]}")
    
    def train_random_forest(self, optimize=True):
        """
        Train Random Forest Classifier
        Fast training, good baseline accuracy
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            print("Training with default parameters...")
            model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        test_acc = model.score(self.X_test, self.y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        print(f"\n✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Testing Accuracy: {test_acc:.4f}")
        print(f"✓ Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['random_forest'] = model
        
        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc
            self.best_model = model
            self.best_model_name = 'random_forest'
        
        return model, test_acc
    
    def train_svm(self, optimize=False):
        """
        Train Support Vector Machine
        Excellent for high-dimensional data, slower training
        """
        print("\n" + "="*60)
        print("TRAINING SUPPORT VECTOR MACHINE")
        print("="*60)
        
        if optimize:
            print("Performing hyperparameter optimization (this may take a while)...")
            param_grid = {
                'C': [1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf']
            }
            
            svm = SVC(probability=True, random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
            print("Training with default parameters...")
            model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"\n✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Testing Accuracy: {test_acc:.4f}")
        
        self.models['svm'] = model
        
        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc
            self.best_model = model
            self.best_model_name = 'svm'
        
        return model, test_acc
    
    def train_neural_network(self, optimize=False):
        """
        Train Multi-Layer Perceptron Neural Network
        Can achieve highest accuracy with proper tuning
        """
        print("\n" + "="*60)
        print("TRAINING NEURAL NETWORK")
        print("="*60)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'hidden_layer_sizes': [(256, 128, 64), (512, 256, 128), (128, 64)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001],
                'learning_rate': ['constant', 'adaptive']
            }
            
            nn = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
            grid_search = GridSearchCV(
                nn, param_grid, cv=3, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False
            )
            print("Training with default parameters...")
            model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"\n✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Testing Accuracy: {test_acc:.4f}")
        print(f"✓ Training iterations: {model.n_iter_}")
        
        self.models['neural_network'] = model
        
        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc
            self.best_model = model
            self.best_model_name = 'neural_network'
        
        return model, test_acc
    
    def train_gradient_boosting(self):
        """
        Train Gradient Boosting Classifier
        Often achieves excellent accuracy
        """
        print("\n" + "="*60)
        print("TRAINING GRADIENT BOOSTING CLASSIFIER")
        print("="*60)
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=0
        )
        
        print("Training...")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        train_acc = model.score(self.X_train, self.y_train)
        test_acc = model.score(self.X_test, self.y_test)
        
        print(f"\n✓ Training Accuracy: {train_acc:.4f}")
        print(f"✓ Testing Accuracy: {test_acc:.4f}")
        
        self.models['gradient_boosting'] = model
        
        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc
            self.best_model = model
            self.best_model_name = 'gradient_boosting'
        
        return model, test_acc
    
    def train_all_models(self, optimize_rf=True, optimize_svm=False, optimize_nn=False):
        """
        Train all available models and compare performance
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        results = {}
        
        # Train each model
        print("\n1/4: Random Forest")
        _, acc_rf = self.train_random_forest(optimize=optimize_rf)
        results['Random Forest'] = acc_rf
        
        print("\n2/4: Support Vector Machine")
        _, acc_svm = self.train_svm(optimize=optimize_svm)
        results['SVM'] = acc_svm
        
        print("\n3/4: Neural Network")
        _, acc_nn = self.train_neural_network(optimize=optimize_nn)
        results['Neural Network'] = acc_nn
        
        print("\n4/4: Gradient Boosting")
        _, acc_gb = self.train_gradient_boosting()
        results['Gradient Boosting'] = acc_gb
        
        # Summary
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            marker = " ⭐ BEST" if model_name.lower().replace(' ', '_') == self.best_model_name else ""
            print(f"{model_name:25s}: {accuracy:.4f}{marker}")
        
        print(f"\n✓ Best model: {self.best_model_name.upper()}")
        print(f"✓ Best accuracy: {self.best_accuracy:.4f}")
        
        return results
    
    def evaluate_model(self, model=None, show_plots=True):
        """
        Comprehensive evaluation of model performance
        """
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model_name = "custom"
        
        print("\n" + "="*60)
        print(f"DETAILED EVALUATION: {model_name.upper()}")
        print("="*60)
        
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Per-class report
        print(f"\nPer-Gesture Performance:")
        class_names = self.label_encoder.classes_
        report = classification_report(
            self.y_test, y_pred, 
            target_names=class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        if show_plots:
            self._plot_confusion_matrix(cm, class_names, model_name)
            self._plot_per_class_accuracy(self.y_test, y_pred, class_names)
        
        # Identify problematic gestures
        per_class_acc = []
        for i, gesture in enumerate(class_names):
            mask = self.y_test == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == i).sum() / mask.sum()
                per_class_acc.append((gesture, class_acc))
        
        # Find low accuracy gestures
        low_acc_threshold = 0.90
        low_acc_gestures = [(g, a) for g, a in per_class_acc if a < low_acc_threshold]
        
        if low_acc_gestures:
            print(f"\n⚠ Gestures with accuracy < {low_acc_threshold:.0%}:")
            for gesture, acc in sorted(low_acc_gestures, key=lambda x: x[1]):
                print(f"  {gesture}: {acc:.2%} - Need more training data!")
        else:
            print(f"\n✓ All gestures have accuracy >= {low_acc_threshold:.0%}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc
        }
    
    def _plot_confusion_matrix(self, cm, class_names, model_name):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300)
        print(f"✓ Confusion matrix saved: confusion_matrix_{model_name}.png")
        plt.close()
    
    def _plot_per_class_accuracy(self, y_true, y_pred, class_names):
        """Plot per-class accuracy bar chart"""
        per_class_acc = []
        for i in range(len(class_names)):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).sum() / mask.sum()
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0)
        
        plt.figure(figsize=(14, 6))
        colors = ['red' if acc < 0.90 else 'green' for acc in per_class_acc]
        plt.bar(class_names, per_class_acc, color=colors, alpha=0.7)
        plt.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
        plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
        plt.xlabel('Gesture', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Per-Gesture Accuracy', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1.0])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('per_gesture_accuracy.png', dpi=300)
        print(f"✓ Per-gesture accuracy plot saved: per_gesture_accuracy.png")
        plt.close()
    
    def save_model(self, filepath=None, model=None):
        """Save trained model, scaler, and label encoder"""
        if model is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model_name = "custom"
        
        if filepath is None:
            filepath = f'sign_language_model_{model_name}.pkl'
        
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_name': model_name,
            'accuracy': self.best_accuracy,
            'feature_count': self.X_train.shape[1],
            'gesture_names': self.label_encoder.classes_.tolist()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to: {filepath}")
        print(f"  Model type: {model_name}")
        print(f"  Accuracy: {self.best_accuracy:.4f}")
        print(f"  Gestures: {len(self.label_encoder.classes_)}")
        
        return filepath
    
    @staticmethod
    def load_model(filepath):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✓ Model loaded from: {filepath}")
        print(f"  Model type: {model_data['model_name']}")
        print(f"  Training accuracy: {model_data['accuracy']:.4f}")
        print(f"  Gestures: {', '.join(model_data['gesture_names'])}")
        
        return model_data


def main():
    """Example usage of ModelTrainer"""
    import sys
    
    print("="*60)
    print("SIGN LANGUAGE MODEL TRAINER")
    print("="*60)
    
    # Check if data file is provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = input("\nEnter path to training data file: ").strip()
    
    if not os.path.exists(data_file):
        print(f"Error: File not found: {data_file}")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(data_file)
    
    # Prepare data
    trainer.prepare_data()
    
    # Training options
    print("\nTraining Options:")
    print("1. Train all models (quick)")
    print("2. Train all models (with optimization)")
    print("3. Train specific model")
    
    choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"
    
    if choice == '1':
        trainer.train_all_models(optimize_rf=False, optimize_svm=False, optimize_nn=False)
    elif choice == '2':
        print("\nWarning: Optimization will take significantly longer!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            trainer.train_all_models(optimize_rf=True, optimize_svm=True, optimize_nn=True)
    elif choice == '3':
        print("\nAvailable models:")
        print("1. Random Forest")
        print("2. Support Vector Machine")
        print("3. Neural Network")
        print("4. Gradient Boosting")
        model_choice = input("Select model (1-4): ").strip()
        
        if model_choice == '1':
            trainer.train_random_forest(optimize=True)
        elif model_choice == '2':
            trainer.train_svm(optimize=True)
        elif model_choice == '3':
            trainer.train_neural_network(optimize=True)
        elif model_choice == '4':
            trainer.train_gradient_boosting()
    
    # Evaluate best model
    trainer.evaluate_model(show_plots=True)
    
    # Save model
    save_choice = input("\nSave model? (y/n, default=y): ").strip().lower() or 'y'
    if save_choice == 'y':
        trainer.save_model()


if __name__ == "__main__":
    main()
