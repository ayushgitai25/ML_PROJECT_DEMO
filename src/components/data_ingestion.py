import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## To define classs variables we bascially use init, but using `dataclass` we will be directly
## be able to define my class variable
@dataclass  
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            ## Read data from somewhere, it can be APIs, MongoDB, etc.
            df = pd.read_csv('notebook\data\StudentsPerformance.csv') 
            logging.info('Read the dataset as Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            ## Convert read data into raw data path as CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            ## Save train and test into seperate files as CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path, ## paths used in next step which is `Data Transformation`
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

## Testing
if __name__ =="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()