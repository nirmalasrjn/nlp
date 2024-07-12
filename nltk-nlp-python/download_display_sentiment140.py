import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_sentiment140_dataset():
    """
    Downloads the Sentiment140 dataset from Kaggle and extracts it to the current directory.

    The dataset is downloaded from the following link on Kaggle:
        https://www.kaggle.com/kazanova/sentiment140

    The dataset is extracted to the current directory with the name 'sentiment140'.
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Define the dataset and file paths
    dataset = 'kazanova/sentiment140'
    path = 'sentiment140'
    
    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Download the dataset
    api.dataset_download_files(dataset, path=path, unzip=True)
    
    print(f"Dataset downloaded and extracted to {path}")

def load_and_display_dataset():
    # Define the path to the dataset
    dataset_path = 'sentiment140/training.1600000.processed.noemoticon.csv'
    
    # Load the dataset into a Pandas DataFrame
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(dataset_path, encoding='latin-1', names=columns)
    
    # Display the first few rows of the DataFrame
    print(df.head())
    
    # Display DataFrame information
    print(df.info())
    
    # Display DataFrame shape
    print(df.shape)

if __name__ == "__main__":
    download_sentiment140_dataset()
    load_and_display_dataset()

