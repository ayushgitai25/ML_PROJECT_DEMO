import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import os, sys

@dataclass ## to avoid using __init__ for class variables initialization
class DataTransformationConfig:
  preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTranformation:
  def __init__(self):
    self.data_tranformation_config = DataTransformationConfig()

  def get_data_transformer_object(self):
    '''This function is responsible for data transformation'''
    try:
      numerical_columns = ["writing score", "reading score"]
      categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
       'test preparation course']
      
      num_pipeline = Pipeline(
        steps = [
          ("impute", SimpleImputer(strategy= "median")),
          ("scaler", StandardScaler())
        ]
      )

      cat_pipeline = Pipeline(
        steps=[
          ("impute", SimpleImputer(strategy="most_frequent")),
          ("one hot encoder", OneHotEncoder()),
          ("scaler", StandardScaler(with_mean=False))
        ]
      )

      logging.info("Numerical columns standard scaling completed")
      logging.info("Categorical columns encoding completed")

      preprocessor = ColumnTransformer(
        [
          ("num_pipeline", num_pipeline, numerical_columns),
          ("cat_pipeline", cat_pipeline, categorical_columns)
        ]
      )

      return preprocessor
    
    except Exception as e:
      raise CustomException(e, sys)
    
  def initiate_data_transformation(self, train_path, test_path):
    try:
      train_df = pd.read_csv(train_path)
      test_df = pd.read_csv(test_path)

      logging.info("The train and test data completed")
      logging.info("Obtaining preprocessing object")

      preprocessing_obj = self.get_data_transformer_object()

      target_column_name = "math score"

      input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
      target_feature_train_df = train_df[target_column_name]

      input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
      target_feature_test_df = test_df[target_column_name]

      logging.info("Applying preprocessor objet on training dataframe and testing dataframe")

      input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

      ## Combine the training input features and the target column into one NumPy array.
      ## np.c_[] horizontally concatenates arrays.
      ## `train_arr` becomes a single array where the last column is the target (math score)
      ## use `np.array` as target_feature_train_df is a dataframe and not an array
      train_arr = np.c_[
        input_feature_train_arr, np.array(target_feature_train_df)
      ]
      test_arr = np.c_[
        input_feature_test_arr, np.array(target_feature_test_df)
      ]

      logging.info("Saved preprocessing object.")

      ## dump 'preprocessor.pkl' as well
      save_object(
        file_path= self.data_tranformation_config.preprocessor_obj_file_path,
        obj = preprocessing_obj
      )

      return (
        train_arr, test_arr,
        self.data_tranformation_config.preprocessor_obj_file_path, 
              ## `preprocessor_obj_file_path` is the path to 'preprocessor.pkl' so we have to return that as well
      )

    except Exception as e:
      raise CustomException(e, sys)

