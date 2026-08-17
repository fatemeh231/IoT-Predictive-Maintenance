# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:06:50 2026

@author: fatemeh
"""
# main.py
# Healthcare Claims Analyzer - Complete Pipeline

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cleaner import HealthcareCleaner
from src.model import ClaimDenialModel
from src.config import RAW_PATH, PROCESSED_PATH, MODEL_PATH

if __name__ == "__main__":
    print("="*60)
    print("🏥 HEALTHCARE CLAIMS ANALYZER")
    print("="*60)
    print()
    
    # ============================================
    # STEP 1: Clean Data (Skip if already done)
    # ============================================
    
    print("🔄 STEP 1: Data Cleaning")
    print("-"*40)
    
    # Check if master dataset already exists
    if os.path.exists(PROCESSED_PATH):
        print(f"✅ Master dataset already exists at: {PROCESSED_PATH}")
        print("   Loading existing dataset...")
        master_df = pd.read_parquet(PROCESSED_PATH)
        print(f"   Shape: {master_df.shape[0]} rows, {master_df.shape[1]} columns")
    else:
        cleaner = HealthcareCleaner(RAW_PATH)
        master_df = cleaner.run_pipeline()
        
        if master_df is None:
            print("❌ Pipeline failed. Check your data files.")
            exit()
        
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        master_df.to_parquet(PROCESSED_PATH, index=False)
        print(f"\n✅ Master dataset saved to: {PROCESSED_PATH}")
        print(f"   Shape: {master_df.shape[0]} rows, {master_df.shape[1]} columns")
    
    # ============================================
    # STEP 2: Machine Learning
    # ============================================
    
    print("\n" + "="*60)
    print("🔄 STEP 2: Machine Learning")
    print("="*60)
    
    # Load or train model
    model = ClaimDenialModel(use_smote=False)  # Set to True if imbalanced-learn is installed
    
    if os.path.exists(MODEL_PATH):
        print("📂 Loading existing model...")
        model.load_model(MODEL_PATH)
        print("   Model loaded successfully!")
    else:
        print("🔄 Training new model...")
        model.train(master_df)
        model.save_model(MODEL_PATH)
    
    # ============================================
    # STEP 3: Add Predictions to Dataset
    # ============================================
    
    print("\n" + "="*60)
    print("🔄 STEP 3: Adding Predictions")
    print("="*60)
    
    # Check if predictions already exist
    if 'DENIAL_PROBABILITY' in master_df.columns:
        print("✅ Predictions already exist in dataset.")
        print(f"   High-risk claims (>50%): {master_df[master_df['DENIAL_PROBABILITY'] > 0.5].shape[0]}")
    else:
        # Add predictions with threshold (lower = more recall)
        master_df = model.add_predictions(master_df, threshold=0.2)
        
        # Re-save with predictions
        master_df.to_parquet(PROCESSED_PATH, index=False)
        print(f"✅ Dataset updated with predictions and re-saved to: {PROCESSED_PATH}")
    
    # ============================================
    # STEP 4: Summary
    # ============================================
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    denial_rate = master_df['IS_DENIED'].mean() * 100
    print(f"\n✅ Total Claims: {len(master_df):,}")
    print(f"✅ Denied Claims: {master_df['IS_DENIED'].sum():,} ({denial_rate:.2f}%)")
    print(f"✅ Claim Types: {master_df['CLAIM_TYPE_NAME'].unique().tolist()}")
    
    if 'DENIAL_PROBABILITY' in master_df.columns:
        high_risk = master_df[master_df['DENIAL_PROBABILITY'] > 0.5].shape[0]
        print(f"\n✅ High-Risk Claims (>50%): {high_risk:,} ({high_risk/len(master_df)*100:.2f}%)")
    
    print("\n📌 Next Steps:")
    print("  1. Open Power BI and connect to:")
    print(f"     -> {PROCESSED_PATH}")
    print("  2. Build dashboard with:")
    print("     - Denial Rate by Claim Type")
    print("     - Denial Rate by Diagnosis")
    print("     - Denial Rate by Provider")
    print("     - Model Predictions (High-Risk Claims)")
    print(f"  3. Use the model to predict claim denials in real-time")
    print("\n" + "="*60)
    print("🎉 Healthcare Claims Analyzer Complete!")
    print("="*60)