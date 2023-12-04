# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:44:43 2023

@author: isaac
"""

# %% Importing libraries
import os
import pandas as pd

# %% Pathing
current_dir = os.getcwd()
project_dir = os.path.dirname(os.path.dirname(current_dir))
data_raw_path = os.path.join(project_dir, 'data', 'raw')
data_processed_path = os.path.join(project_dir, 'data', 'processed')

# %%
def process_education(data):
    data['Education'] = data['Education'].fillna('No Education')
    return data

def process_communication(data):
    data['Communication'] = data['Communication'].fillna('No Communication')
    data['Communication'] = data['Communication'].apply(lambda data: 0 if data == 'No Communication' else 1)
    return data

def process_contact(data):
    data['CallStart'] = pd.to_datetime(data['CallStart'], format='%H:%M:%S')
    data['CallEnd'] = pd.to_datetime(data['CallEnd'], format='%H:%M:%S')

    # Calculate call duration and create a new column 'CallDuration'
    data['CallDuration'] = data['CallEnd'] - data['CallStart']
    data['CallDuration'] = data['CallDuration'].dt.total_seconds()
    return data

def process_outcome(data):
    data['Outcome'] = data['Outcome'].apply(lambda data: 1 if data == 'success' else 0)
    return data

def process_job(data):
    data['Job'] = data['Job'].fillna('unemployed')
    return data

def process_data(data):
    data = process_communication(data)
    data = process_education(data)
    data = process_contact(data)
    data = process_outcome(data)
    data = process_job(data)
    return data

def save_process(data):
    data = process_data(data)
    data.to_csv(data_processed_path, 'train_processed.csv')
    return data






