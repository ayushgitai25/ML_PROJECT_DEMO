import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import evaluate_model
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
    try:
      logging.info("Splitting training and test input data")
      X_train, y_train, X_test, y_test = (
        train_array[:,:-1],
        train_array[:, -1],
        test_array[:,:-1],
        test_array[:, -1]
      )
      models = {
          "LinearRegression" : LinearRegression(),
          "GradientBoosting" : GradientBoostingRegressor(),
          "KNeighborsRegressor" : KNeighborsRegressor(),
          "DecisionTreeRegressor" : DecisionTreeRegressor(),
          "RandomForestRegressor" : RandomForestRegressor(),
          "XGBRegressor" : XGBRegressor(),
          "CatBoostRegressor" : CatBoostRegressor(verbose=0),
          "AdaBoostRegressor" : AdaBoostRegressor()
      }

      params = {
          "LinearRegression": {},   # no major hyperparams, kept empty intentionally

          "GradientBoosting": {
              "n_estimators": [100, 200, 300],
              "learning_rate": [0.01, 0.05, 0.1],
              "max_depth": [3, 4, 5, 6]
          },

          "KNeighborsRegressor": {
              "n_neighbors": [3, 5, 7, 9, 11],
              "weights": ["uniform", "distance"],
              "p": [1, 2]   # 1=manhattan, 2=euclidean
          },

          "DecisionTreeRegressor": {
              "criterion": ["squared_error", "friedman_mse", "absolute_error"],
              "max_depth": [None, 5, 10, 20, 30],
              "min_samples_split": [2, 5, 10]
          },

          "RandomForestRegressor": {
              "n_estimators": [100, 200, 300],
              "max_depth": [None, 10, 20, 30],
              "min_samples_split": [2, 5, 10]
          },

          "XGBRegressor": {
              "learning_rate": [0.01, 0.05, 0.1],
              "max_depth": [3, 5, 7],
              "n_estimators": [100, 200, 300],
              "subsample": [0.7, 0.8, 1.0]
          },

          "CatBoostRegressor": {
              "depth": [4, 6, 8, 10],
              "learning_rate": [0.01, 0.05, 0.1],
              "iterations": [200, 400, 600]
          },

          "AdaBoostRegressor": {
              "n_estimators": [50, 100, 200],
              "learning_rate": [0.01, 0.05, 0.1, 1.0]
          }
      }


      model_report:dict = evaluate_model(
        X_train=X_train, y_train=y_train, 
        X_test=X_test, y_test=y_test,
        models=models,
        params = params
      )

      best_model_score = max(sorted(model_report.values()))

      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]

      best_model = models[best_model_name]

      if best_model_score < 0.6:
        raise CustomException("No best model found")
      logging.info(f"Best model found : {best_model_name} with score : {best_model_score}")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj = best_model
      )

      predicted = best_model.predict(X_test)

      r2_square = r2_score(y_test, predicted)
      return r2_square

    except Exception as e:
      raise CustomException(e, sys)
        
