# %% Libraries
import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# %%
def copy_file(source_path=None, destination_path=None):
    # file path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    source_path = os.path.join(project_dir, 'credentials', 'kaggle.json')

    # windows path
    win_path = os.path.realpath(os.environ['HOMEPATH'])
    destination_path = os.path.join(win_path, '.kaggle')

    try:
        # Check if the source file exists
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"The file '{source_path}' does not exist.")

        if not os.path.exists(destination_path):
            # Copy the file to the destination path
            shutil.copy2(source_path, destination_path)
            print(f"File copied successfully from '{source_path}' to '{destination_path}'.")
        
    except Exception as e:
        print(f"Error: {e}")

def extract_all_zips(zip_path, extract_to):
    for root, dir, files in os.walk(zip_path):
        for file in files:
            if file.endswith('.zip'):
                zip_file_path = os.path.join(zip_path, file)

                with zipfile.ZipFile(zip_file_path, 'r') as inner_zip:
                    # Extract contents of inner zip file to the specified directory
                    inner_zip.extractall(extract_to)
                    print(f"Files from '{zip_file_path}' extracted successfully to '{extract_to}'.")

def delete_all_zip_files(directory_path):
    try:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.zip'):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"File '{file_path}' deleted successfully.")

        print(f"All zip files in '{directory_path}' deleted successfully.")
    
    except Exception as e:
        print(f"Error: {e}")


def load_datasets_name():
    """
    Load datasets name from a text file.

    """
    # file path
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    filePath = os.path.join(project_dir, 'references', 'dataset_name.txt')
    
    with open(filePath, 'r') as file:
        lines = file.readlines()
        datasets_name = lines[0].strip()

    return datasets_name

def download_kaggle_dataset_from_file(dataset_name):
    
    # Set Kaggle API credentials
    api = KaggleApi()
    api.authenticate()

    # Specify the destination path for downloading the dataset
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    raw_data_path = os.path.join(project_dir, 'data', 'raw')

    # Download the dataset
    api.dataset_download_files(dataset=dataset_name,
                              path=raw_data_path)
    
    extract_all_zips(raw_data_path, raw_data_path)
    delete_all_zip_files(raw_data_path)
    print(f"Dataset downloaded to: {raw_data_path}")


# %%

# create credential
copy_file()

# download, extract and delete ".zip" file
download_kaggle_dataset_from_file(load_datasets_name())