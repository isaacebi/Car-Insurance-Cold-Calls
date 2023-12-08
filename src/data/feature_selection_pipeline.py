# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 08:58:32 2023

@author: isaac
"""

# %% Importing Libraries
import os
import numpy as np
import pandas as pd
import scipy.stats as ss

from sklearn.linear_model import LogisticRegression

# %% Pathing
CURRENT_DIR = os.getcwd()
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_DIR = os.path.dirname(SRC_DIR)

# %%
def cramers_corrected_stat(label, target):
    '''
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher, 
    Journal of the Korean Statistical Society 42 (2013): 323-328

    Parameters
    ----------
    confusion_matrix : DataFrame
        Pandas crosstab product between categorical.

    Returns
    -------
    int
        correlation.

    '''
    confusion_matrix = pd.crosstab(label, target).to_numpy()
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

# manual filter, if not needed can proceed using scikit-learn pipeline
class col_filter:
    def __init__(self, df, target, continous_list, categorical_list):
        self.df = df
        self.target = target
        self.conti_col = continous_list
        self.categ_col = categorical_list
    
    # create continous filter for input continous, output categorical
    def con_filter(self) -> dict:
        scores = {}
        
        for col in self.conti_col:
            if self.df[col].dtypes != '<M8[ns]':
                # using Logistic Regression to get scores
                lr = LogisticRegression()
                # fit model
                lr.fit(
                    np.expand_dims(self.df[col], axis=-1), 
                    self.df[self.target]
                    )
                # get accuracy score
                score = lr.score(
                    np.expand_dims(self.df[col], axis=-1), 
                    self.df[self.target]
                    )
                
                scores[col] = score
            
        return pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
        
    # create continous filter for input categorical, output categorical
    def cat_filter(self):
        scores = {}
        for col in self.categ_col:
            # using Crammer's V
            score = cramers_corrected_stat(self.df[col], self.df[self.target])
            scores[col] = score
        
        return pd.DataFrame.from_dict(scores, orient='index', columns=['Score'])
    
    # output selected features/labels
    def get_columns(self, threshold:float) -> list:
        labels_score = pd.concat([self.con_filter(), self.cat_filter()])
        col_selected = labels_score[labels_score['Score'] >= threshold].index.tolist()
        return col_selected



      
