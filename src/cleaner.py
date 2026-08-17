# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:28:40 2026

@author: fatemeh
"""

# src/cleaner.py
# Data Cleaning Script for IoT Predictive Maintenance

import pandas as pd
import numpy as np
import os

def load_and_clean():
    """
    Load the raw data and clean it.
    """
    # 1. Load data
    file_path = r"C:\Users\fatemeh\OneDrive\Desktop\codes_tutorial_uni\iot-predictive-maintenance\data\raw\predictive_maintenance_v3.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print("📂 Loading data...")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Convert timestamp to datetime
    print("🔄 Converting timestamp...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. Handle missing values
    print("🔄 Handling missing values...")
    numeric_cols = [
        'vibration_rms', 
        'temperature_motor', 
        'current_phase_avg', 
        'pressure_level', 
        'rpm'
    ]
    
    for col in numeric_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   ✅ {col}: filled {null_count} nulls with median {median_val:.2f}")
    
    # 4. Encode categorical columns
    print("🔄 Encoding categorical columns...")
    df['machine_type_encoded'] = df['machine_type'].astype('category').cat.codes
    df['operating_mode_encoded'] = df['operating_mode'].astype('category').cat.codes
    
    # 5. Drop original categorical columns
    df = df.drop(['machine_type', 'operating_mode'], axis=1)
    print(f"   ✅ Dropped original categorical columns")
    
    # 6. Sort by machine_id and timestamp (important for time-series)
    print("🔄 Sorting by machine_id and timestamp...")
    df = df.sort_values(['machine_id', 'timestamp']).reset_index(drop=True)
    
    # 7. Save cleaned data
    output_path = "data/processed/cleaned_data.parquet"
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")
    
    # 8. Show summary
    print("\n" + "="*60)
    print("cLEANED DATA SUMMARY")
    print("="*60)
    print(f"   Rows: {df.shape[0]}")
    print(f"   Columns: {df.shape[1]}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Machines: {df['machine_id'].nunique()}")
    print(f"   Failure rate: {df['failure_within_24h'].mean()*100:.2f}%")
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = load_and_clean()