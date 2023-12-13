# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:12:00 2023

@author: isaac
"""

# %%
import os
import pandas as pd
import process_pipeline
import encoder_pipeline
import feature_selection_pipeline

# %%
CURRENT_DIR = os.getcwd()
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)


DATA_RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_DIR, 'data', 'processed')
TRAIN_PATH = os.path.join(DATA_RAW_DIR, 'carInsurance_train.csv')
MODELS_PATH = os.path.join(PROJECT_DIR, 'models')
ENCODER_PATH = os.path.join(MODELS_PATH, 'encoders')

CATEG_PATH = os.path.join(PROJECT_DIR, 'references', 'categorical_columns.txt')
CONTI_PATH = os.path.join(PROJECT_DIR, 'references', 'continous_columns.txt')

PROJECT_NAME = 'pipeline_test'

# %% Helper Function
def get_content(txt_file):
    contents = []
    with open(txt_file) as file:
        for line in file:
            contents.append(line.strip())
            
    return contents

# %%
df = pd.read_csv(TRAIN_PATH)
# ensuring no missing data
df = process_pipeline.process_data(df)

# %%
# Encoding
le = encoder_pipeline.getEncoder(ENCODER_PATH, df, PROJECT_NAME)
# return numeric datatype DataFrame
df = le.label_encoder()

# Feature Selection
categ = get_content(CATEG_PATH)
conti = get_content(CONTI_PATH)


FS = feature_selection_pipeline.col_filter(df, 'CarInsurance', conti, categ)
categ_scores = FS.cat_filter()
conti_scores = FS.con_filter()

selected_labels = FS.get_columns(0.5)
















# %%
