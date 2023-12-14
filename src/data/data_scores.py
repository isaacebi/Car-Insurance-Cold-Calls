# %% Importing libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from xgboost import plot_importance

# %% Pathing
CURRENT_DIR = os.getcwd()
PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_RAW_PATH = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_PROCESSED_PATH = os.path.join(PROJECT_DIR, 'data', 'processed')

TRAIN_PATH = os.path.join(DATA_RAW_PATH, 'carInsurance_train.csv')

# %%
from process_pipeline import process_data

# %%
df = pd.read_csv(TRAIN_PATH)
df = process_data(df)

X = df.drop(columns=['CarInsurance'])
y = df['CarInsurance']

# some of dtypes in X is still in obj
for col in X:
    if X[col].dtype == 'O':
        X[col] = LabelEncoder().fit_transform(X[col])

# %%
# define model
model = XGBClassifier()

# fit the model
model.fit(X, y)

# get importance
importance = model.feature_importances_

# create pandas scores
score = pd.DataFrame(importance, index=X.columns.tolist(), columns=['Score'])

score.plot.bar(y='Score')
plt.show()

plot_importance(model)
plt.show()

# %%
