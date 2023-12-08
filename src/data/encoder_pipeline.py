# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:08:04 2023

@author: isaac
"""

# %%
import os
import pickle
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# %%
CURRENT_DIR = os.getcwd()
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)
MODELS = os.path.join(PROJECT_DIR, 'models')
ENCODERS = os.path.join(MODELS, 'encoders')

# %%
class getEncoder:
    def __init__(self, path, df, project_name:str):
        self.path = path
        self.df = df
        self.project_name = project_name
        
    def label_encoder(self):
        # create file
        ENCODER_DIR = os.path.join(ENCODERS, self.project_name)
        if not os.path.exists(ENCODER_DIR):
            os.makedirs(ENCODER_DIR)
        
        # initialize label encoder
        le = LabelEncoder()
        cat_df = self.df.select_dtypes(include=[object])
        
        for col in cat_df:
            self.df[col] = le.fit_transform(self.df[col])
            
            encoderName = col.upper()
            PICKLE_SAVE_PATH = os.path.join(ENCODER_DIR, encoderName + "_ENCODER.pkl")
            with open(PICKLE_SAVE_PATH, 'wb') as file:
                pickle.dump(le, file)
        
        return self.df
    