# %% Importing libraries
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# %% Pathing
CURRENT_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW_PATH = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_DIR, 'data', 'processed')

TRAIN_PATH = os.path.join(DATA_RAW_PATH, 'carInsurance_train.csv')

# %%
from process_pipeline import process_data

# %%
def get_features_score():
    df = pd.read_csv(TRAIN_PATH)
    df = process_data(df)

    X = df.drop(columns=['CarInsurance'])
    y = df['CarInsurance']
    
    # some of dtypes in X is still in obj
    for col in X:
        # encoded object then scaling
        if X[col].dtype == 'category':
            X[col] = LabelEncoder().fit_transform(X[col])
            X[col] = StandardScaler().fit_transform(np.expand_dims(X[col], axis=-1))

        # scaling numeric
        else:
            X[col] = StandardScaler().fit_transform(np.expand_dims(X[col], axis=-1))

    # splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # define model
    model = XGBClassifier()
    # model = LogisticRegression()

    # fit the model
    model.fit(X_train, y_train)

    # using shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # visualize feature importance
    # ----- Reasoning ----- #
    # The reason between differences interpertation between model.feature_importance_ and shap values are mainly
    # due to the model.feature_importances are based on the train sets while shap are based on the X_test  
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.barh(X.columns.tolist(), model.feature_importances_)
    plt.show()

    shap.summary_plot(shap_values, X_test, plot_type='bar')
    shap.summary_plot(shap_values, X_test)

# %% Directly run function
get_features_score()

# %% This section need to be further explore in the future
# # define model
# model = XGBClassifier().fit(X, y)

# # explain the model's predictions using SHAP
# # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# explainer = shap.Explainer(model)
# shap_values = explainer(X)

# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
# shap.plots.waterfall(shap_values[2])

# %%
