# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:41:07 2026

@author: fatemeh
"""

# src/train_lstm.py
# LSTM Model Training for IoT Predictive Maintenance

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_data():
    """Load the preprocessed data."""
    print("📂 Loading preprocessed data...")
    
    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    print(f"✅ X_train: {X_train.shape}")
    print(f"✅ X_val: {X_val.shape}")
    print(f"✅ X_test: {X_test.shape}")
    print(f"✅ y_train: {y_train.shape}")
    print(f"✅ y_val: {y_val.shape}")
    print(f"✅ y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_lstm_model(input_shape):
    """
    Build LSTM model for binary classification.
    input_shape: (timesteps, features)
    """
    print("\n🏗️ Building LSTM model...")
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the LSTM model with early stopping.
    """
    print("\nTraining LSTM model...")
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    """
    print("\n📊 Evaluating model on test data...")
    
    # Predict probabilities
    y_pred_proba = model.predict(X_test)
    
    # Convert to binary (threshold 0.5)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    
    # Confusion matrix
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return y_pred, y_pred_proba

def save_model(model, history):
    """Save the trained model and training history."""
    print("\n💾 Saving model...")
    
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras")
    np.save("models/training_history.npy", history.history)
    
    print("✅ Model saved to: models/lstm_model.keras")
    print("✅ Training history saved to: models/training_history.npy")

def run_training_pipeline():
    """Run the complete training pipeline."""
    print("="*60)
    print("🤖 LSTM MODEL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load data
    print("\n1. Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # 2. Build model
    print("\n2. Building LSTM model...")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # 3. Train model
    print("\n3. Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # 4. Evaluate model
    print("\n4. Evaluating model...")
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 5. Save model
    print("\n5. Saving model...")
    save_model(model, history)
    
    # 6. Save predictions to CSV
    print("\n6. Saving predictions to CSV...")
    
    # Create a DataFrame with predictions
    test_results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': y_pred_proba.flatten()
    })
    
    # Add test index (to match original rows)
    test_results['test_index'] = range(len(test_results))
    
    os.makedirs("output", exist_ok=True)
    test_results.to_csv("output/test_predictions.csv", index=False)
    print("✅ Predictions saved to: output/test_predictions.csv")
    
    print("\n" + "="*60)
    print("✅ LSTM TRAINING COMPLETE!")
    print("="*60)
    
    # Summary
    print(f"\n📊 Final Test Results:")
    print(f"   Accuracy: {np.mean(y_pred == y_test):.4f}")
    print(f"   Total Test Samples: {len(y_test)}")
    print(f"   Total Failures in Test: {y_test.sum()}")
    print(f"   Predicted Failures: {y_pred.sum()}")

if __name__ == "__main__":
    import pandas as pd
    run_training_pipeline()