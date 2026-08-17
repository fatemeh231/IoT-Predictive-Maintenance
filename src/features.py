# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:35:40 2026

@author: fatemeh
"""

# src/features.py
# Feature Engineering Script for IoT Predictive Maintenance

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_cleaned_data():
    """Load the cleaned data from Parquet."""
    df = pd.read_parquet("data/processed/cleaned_data.parquet")
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def create_time_features(df):
    """Create time-based features from timestamp."""
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    print("   ✅ Added time features: hour, day_of_week, day_of_month")
    return df

def create_rolling_features(df, window=10):
    """
    Create rolling statistics for sensor columns.
    window: number of previous readings to consider.
    """
    sensor_cols = ['vibration_rms', 'temperature_motor', 'current_phase_avg', 
                   'pressure_level', 'rpm']
    
    # Group by machine_id to keep each machine's history separate
    grouped = df.groupby('machine_id')
    
    for col in sensor_cols:
        # Rolling mean
        df[f'{col}_rolling_mean'] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Rolling std
        df[f'{col}_rolling_std'] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
    
    print(f"   ✅ Added rolling features with window={window}")
    return df

def create_lag_features(df, lags=[1, 2, 3]):
    """Create lag features for sensor columns."""
    sensor_cols = ['vibration_rms', 'temperature_motor', 'current_phase_avg', 
                   'pressure_level', 'rpm']
    
    grouped = df.groupby('machine_id')
    
    for col in sensor_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = grouped[col].shift(lag)
    
    print(f"   ✅ Added lag features: {lags}")
    return df

def prepare_sequences(df, feature_cols, target_col='failure_within_24h', seq_length=24):
    """
    Prepare data for LSTM by creating sequences of length `seq_length`.
    """
    print(f"🔄 Preparing sequences (seq_length={seq_length})...")
    
    # Fill NaN values from lag features
    df = df.fillna(0)
    
    # Get unique machines
    machines = df['machine_id'].unique()
    
    X_list = []
    y_list = []
    machine_ids = []
    timestamps = []
    
    for machine in machines:
        machine_data = df[df['machine_id'] == machine].sort_values('timestamp')
        
        # Get features and target
        X = machine_data[feature_cols].values
        y = machine_data[target_col].values
        time = machine_data['timestamp'].values
        m_id = machine_data['machine_id'].values
        
        # Create sequences
        for i in range(seq_length, len(X)):
            X_list.append(X[i-seq_length:i])
            y_list.append(y[i])
            machine_ids.append(m_id[i])
            timestamps.append(time[i])
    
    X_seq = np.array(X_list)
    y_seq = np.array(y_list)
    
    print(f"   ✅ Created {len(X_seq)} sequences")
    print(f"   ✅ X shape: {X_seq.shape}")
    print(f"   ✅ y shape: {y_seq.shape}")
    
    return X_seq, y_seq, machine_ids, timestamps

def split_and_scale(X, y, test_size=0.2, val_size=0.2):
    """Split data into train/validation/test and scale features."""
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)} sequences")
    print(f"   Validation: {len(X_val)} sequences")
    print(f"   Test: {len(X_test)} sequences")
    
    # Scale features (fit on train, transform on val and test)
    # Reshape for scaling: (samples, timesteps, features) → (samples * timesteps, features)
    original_shape = X_train.shape
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled.reshape(original_shape)
    
    # Transform validation and test
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled = scaler.transform(X_val_flat)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    print(f"\n📊 After Scaling:")
    print(f"   X_train: {X_train_scaled.shape}")
    print(f"   X_val: {X_val_scaled.shape}")
    print(f"   X_test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def run_feature_pipeline():
    """Run the complete feature engineering pipeline."""
    print("="*60)
    print("📊 FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # 1. Load cleaned data
    print("\n1. Loading cleaned data...")
    df = load_cleaned_data()
    
    # 2. Create time features
    print("\n2. Creating time features...")
    df = create_time_features(df)
    
    # 3. Create rolling features
    print("\n3. Creating rolling features...")
    df = create_rolling_features(df, window=10)
    
    # 4. Create lag features
    print("\n4. Creating lag features...")
    df = create_lag_features(df, lags=[1, 2, 3])
    
    # 5. Define feature columns (drop non-feature columns)
    drop_cols = ['timestamp', 'failure_type', 'failure_within_24h', 
                 'rul_hours', 'estimated_repair_cost']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    print(f"\n5. Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"   - {col}")
    
    # 6. Prepare sequences
    print("\n6. Preparing sequences for LSTM...")
    X, y, machine_ids, timestamps = prepare_sequences(
        df, feature_cols, target_col='failure_within_24h', seq_length=24
    )
    
    # 7. Split and scale
    print("\n7. Splitting and scaling data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(
        X, y, test_size=0.2, val_size=0.2
    )
    
    # 8. Save processed data
    print("\n8. Saving processed data...")
    import os
    os.makedirs("data/processed", exist_ok=True)
    
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/y_test.npy", y_test)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, r"C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\iot-predictive-maintenance\model\scaler.pkl")
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*60)
    print(f"\n📊 Final Shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_val: {y_val.shape}")
    print(f"   y_test: {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

if __name__ == "__main__":
    run_feature_pipeline()