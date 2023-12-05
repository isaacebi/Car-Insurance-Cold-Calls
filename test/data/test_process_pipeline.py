# %% Libraries
import sys
import os
import unittest
import pandas as pd

# %% Pathing
# Get the current user's desktop directory
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
# Iterate through the desktop directory using os.walk
for root, dirs, files in os.walk(desktop_dir):
    if "Car-Insurance-Cold-Calls" in dirs:
        # Found the project directory
        project_dir = os.path.abspath(os.path.join(root, "Car-Insurance-Cold-Calls"))

data_raw_path = os.path.join(project_dir, 'data', 'raw')
data_processed_path = os.path.join(project_dir, 'data', 'processed')
train_path = os.path.join(data_raw_path, 'carInsurance_train.csv')

# adding src to the system path
sys.path.insert(0, project_dir)
print('---- SYSTEM PATH ----')
for i in range(len(sys.path)):
    print(f"{i}. {sys.path[i]}")

# %% Libraries import
from src.data import process_pipeline

# %%
class Test_process_pipeline(unittest.TestCase):
    def test_no_missing_data(self):
        df = pd.read_csv(train_path)
        df = process_pipeline.process_data(df)
        self.assertFalse(df.isnull().any().any())

# %%
if __name__ == '__main__':
    unittest.main()


# %%
