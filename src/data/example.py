# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:12:00 2023

@author: isaac
"""

# %%
import os
import pandas as pd
import process_pipeline

# %%
current_dir = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(current_dir))
data_raw_path = os.path.join(project_dir, 'data', 'raw')
data_processed_path = os.path.join(project_dir, 'data', 'processed')
train_path = os.path.join(data_raw_path, 'carInsurance_train.csv')

# %%
df = pd.read_csv(train_path)
df = process_pipeline.process_data(df)
