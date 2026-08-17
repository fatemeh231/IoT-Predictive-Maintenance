# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:24:11 2026

@author: fatemeh
"""

import pandas as pd
import numpy as np
import os

# 1. LOAD THE DATA
file_path = "predictive_maintenance_v3.csv"

print("="*80)
print("📊 IoT PREDICTIVE MAINTENANCE - DATA EXPLORATION")
print("="*80)

# Check if file exists
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    print("Please make sure the file is in the correct location.")
    exit()

# Load the data
print(f"\n📂 Loading file: {file_path}")
df = pd.read_csv(file_path)
print(f"✅ Loaded successfully!")

# 2. BASIC INFORMATION
print("\n" + "="*80)
print("📋 BASIC INFORMATION")
print("="*80)

print(f"\n📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n📋 Data Types:")
print(df.dtypes)

print("\n📋 First 5 Rows:")
print(df.head())

print("\n📋 Last 5 Rows:")
print(df.tail())

# 3. NULL / MISSING VALUES
print("\n" + "="*80)
print("🔍 NULL / MISSING VALUES")
print("="*80)

null_counts = df.isnull().sum()
null_percent = (null_counts / len(df)) * 100

null_summary = pd.DataFrame({
    'Null Count': null_counts,
    'Null %': null_percent
}).sort_values('Null %', ascending=False)

print("\n📋 Null Values per Column:")
print(null_summary)

total_nulls = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"\n📊 Total Null Cells: {total_nulls} out of {total_cells} ({total_nulls/total_cells*100:.2f}%)")

# 4. UNIQUE VALUES (For Categorical Columns)
print("\n" + "="*80)
print("🔍 UNIQUE VALUES (Categorical Columns)")
print("="*80)

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    for col in categorical_cols:
        print(f"\n📋 {col}:")
        print(f"   Unique values: {df[col].nunique()}")
        print(f"   Sample: {df[col].unique()[:10]}")
else:
    print("⚠️ No categorical columns found.")

# ============================================
# 5. STATISTICAL SUMMARY (Numeric Columns)
# ============================================

print("\n" + "="*80)
print("📊 STATISTICAL SUMMARY (Numeric Columns)")
print("="*80)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if numeric_cols:
    print("\n📋 Summary Statistics:")
    print(df[numeric_cols].describe())
else:
    print("⚠️ No numeric columns found.")

# ============================================
# 6. CHECK FOR DUPLICATES
# ============================================

print("\n" + "="*80)
print("🔍 DUPLICATE ROWS")
print("="*80)

duplicate_count = df.duplicated().sum()
print(f"\n📋 Duplicate rows: {duplicate_count} ({duplicate_count/len(df)*100:.2f}%)")

# 7. TARGET VARIABLE ANALYSIS (if exists)
print("\n" + "="*80)
print("🎯 TARGET VARIABLE ANALYSIS")
print("="*80)

target_cols = ['failure_within_24h', 'rul_hours', 'failure_type', 'target']

for col in target_cols:
    if col in df.columns:
        print(f"\n📋 {col}:")
        if df[col].dtype in ['int64', 'float64']:
            print(f"   Min: {df[col].min()}")
            print(f"   Max: {df[col].max()}")
            print(f"   Mean: {df[col].mean():.2f}")
            print(f"   Std: {df[col].std():.2f}")
            if col == 'failure_within_24h':
                failure_count = df[col].sum()
                print(f"   Failures: {failure_count} ({failure_count/len(df)*100:.2f}%)")
        else:
            print(f"   Unique values: {df[col].unique()[:10]}")

# ============================================
# 8. SENSOR COLUMN ANALYSIS
# ============================================

print("\n" + "="*80)
print("📡 SENSOR COLUMN ANALYSIS")
print("="*80)

sensor_cols = [col for col in df.columns if 'sensor' in col.lower() or 'vibration' in col.lower() or 'temp' in col.lower()]

if sensor_cols:
    print(f"\n📋 Sensor Columns Found: {len(sensor_cols)}")
    for col in sensor_cols:
        print(f"\n   📊 {col}:")
        print(f"      Min: {df[col].min():.2f}")
        print(f"      Max: {df[col].max():.2f}")
        print(f"      Mean: {df[col].mean():.2f}")
        print(f"      Std: {df[col].std():.2f}")
else:
    print("⚠️ No sensor columns found. Check column names.")

# 9. MACHINE / ASSET ANALYSIS
print("\n" + "="*80)
print("🏭 MACHINE / ASSET ANALYSIS")
print("="*80)

machine_cols = [col for col in df.columns if 'machine' in col.lower() or 'asset' in col.lower() or 'device' in col.lower()]

if machine_cols:
    for col in machine_cols:
        print(f"\n📋 {col}:")
        print(f"   Unique machines: {df[col].nunique()}")
        print(f"   Sample: {df[col].unique()[:10]}")
        print(f"   Value counts:")
        print(df[col].value_counts().head(10))
else:
    print("⚠️ No machine/asset columns found.")

# 10. SUMMARY
print("\n" + "="*80)
print("📋 SUMMARY")
print("="*80)

print(f"""
✅ Dataset loaded successfully!
   - Total Rows: {df.shape[0]}
   - Total Columns: {df.shape[1]}
   - Missing Values: {total_nulls} ({total_nulls/total_cells*100:.2f}%)
   - Duplicate Rows: {duplicate_count} ({duplicate_count/len(df)*100:.2f}%)

📊 Key Findings:
   - Categorical Columns: {len(categorical_cols)}
   - Numeric Columns: {len(numeric_cols)}
   - Sensor Columns: {len(sensor_cols)}
""")

print("="*80)
print("🎉 Exploration Complete!")
print("="*80)