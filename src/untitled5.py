# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:49:28 2026

@author: fatemeh
"""

import pandas as pd
df = pd.read_csv("output/test_predictions.csv")
print(df.head())
print(df['predicted'].sum())  # Number of predicted failures