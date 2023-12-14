# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:44:43 2023

@author: isaac
"""

# %% Importing libraries
import os
import numpy as np
import pandas as pd
from datetime import datetime

# %% Pathing
CURRENT_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW_PATH = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_DIR, 'data', 'processed')

# %% Helper Function
# Define a function to categorize time
def categorize_time(total_seconds):
    if 6 * 60 * 60 <= total_seconds < 12 * 60 * 60:
        return 'Morning'
    elif 12 * 60 * 60 <= total_seconds < 18 * 60 * 60:
        return 'Afternoon'
    else:
        return 'Evening'


# %% Imputer Function

# Job
def imputerJob(data):
    data['Job'] = data['Job'].fillna('unemployed')
    return data

# Education
def imputerEducation(data):
    data['Education'] = data['Education'].fillna('No Education')
    return data

# Communication
def imputerCommunication(data):
    data['Communication'] = data['Communication'].fillna('No Communication')
    return data

# Outcome
def imputerOutcome(data):
    data['Outcome'] = data['Outcome'].fillna('failure')
    return data

# %% Feature Engineering

# Age 

# Job 
def featureJob(data):
    data['Job'] = data['Job'].astype('category')
    return data

# Marital 
def featureMarital(data):
    data['Marital'] = data['Marital'].astype('category')
    return data

# Eduaction 
def featureEducation(data):
    data['Education'] = data['Education'].astype('category')
    return data

# Default 

# Balance 
def featureBalance(data):
    data['logBalance'] = np.log(data['Balance'])
    data['sqrtBalance'] = np.where(data['Balance'] >= 0, np.sqrt(data['Balance']), 0)
    data['cbrtBalance'] = np.cbrt(data['Balance'])
    data['negativeBalance'] = data['Balance'].apply(lambda x: 1 if x < 0 else 0)

    # after log some of it has missing data, fill the nan with small_constants
    small_constant = 1e-10
    data['logBalance'].fillna(small_constant, inplace=True)

    # handling negative -inf
    data['logBalance'] = data['logBalance'].apply(lambda x: small_constant if x == -np.inf else x)
    return data

# HHInsurance # CarLoan

# Communication
def featureCommunication(data):
    # simplify category
    data['HasCommuncation'] = data['Communication'].apply(lambda data: 0 if data == 'No Communication' else 1)

    data['Communication'] = data['Communication'].astype('category')
    return data

# LastContactMonth
def featureLastContactMonth(data):
    # Assuming 'month' is a column in your DataFrame
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    data['LastContactMonth'] = data['LastContactMonth'].map(lambda x: month_order.index(x) + 1)  # Map strings to numeric values

    # Cyclic encoding
    data['SinLastContactMonth'] = np.sin(2 * np.pi * data['LastContactMonth'] / 12)
    data['CosLastContactMonth'] = np.cos(2 * np.pi * data['LastContactMonth'] / 12)
    return data

# LastContactDay

# CallStart # CallEnd # CallDuration
def featureCall(data):
    data['CallStart'] = pd.to_datetime(data['CallStart'], format='%H:%M:%S')
    data['CallEnd'] = pd.to_datetime(data['CallEnd'], format='%H:%M:%S')

    # Calculate call duration and create a new column 'CallDuration'
    data['CallDuration'] = data['CallEnd'] - data['CallStart']
    data['CallDuration'] = data['CallDuration'].dt.total_seconds()

    # after inspection, CallStart and CallEnd are not in the appropriate format to process
    data['CallStart'] = data['CallStart'].apply(lambda x: x.hour * 60 * 60 + x.minute * 60 + x.second)
    data['CallEnd'] = data['CallEnd'].apply(lambda x: x.hour * 60 * 60 + x.minute * 60 + x.second)

    # new feature on when the call happen
    data['CallCategory'] = data['CallStart'].apply(categorize_time).astype('category')
    return data

# NoOfContacts 

# DaysPassed 
def featureDaysPassed(data):
    data['DaysPassed_Simplify'] = data['DaysPassed'].apply(lambda x: 0 if x == -1 else 0)
    return data

# PrevAttempts
def featurePrevAttempts(data):
    data['PrevAttemp_Simplify'] = data['PrevAttempts'].apply(lambda x: 1 if x > 0 else 0)
    return data

# Outcome
def featureOutcome(data):
    # simplify outcome
    data['Outcome_Simplify'] = data['Outcome'].apply(lambda data: 1 if data == 'success' else 0)

    data['Outcome'] = data['Outcome'].astype('category')
    return data

# %%

def process_data(data):
    data = imputerJob(data)
    data = imputerCommunication(data)
    data = imputerEducation(data)
    data = imputerOutcome(data)

    data = featureJob(data)
    data = featureMarital(data)
    data = featureEducation(data)
    # data = featureBalance(data)
    data = featureCommunication(data)
    data = featureLastContactMonth(data)
    data = featureCall(data)
    # data = featureDaysPassed(data)
    # data = featurePrevAttempts(data)
    data = featureOutcome(data)
    return data

def save_process(data):
    data = process_data(data)
    data.to_csv(DATA_PROCESSED_PATH, 'train_processed.csv')
    return data






