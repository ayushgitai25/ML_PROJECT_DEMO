import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.metrics import r2_score
import sys

from src.exception import CustomException

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      dill.dump(obj, file_obj)

  except Exception as e:
    raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for name, model in models.items():
            print(f"\n🔎 Tuning Model → {name}")

            param_grid = params.get(name, {})  # fetch hyperparams for model

            # If params exist → apply GridSearchCV, else use model directly
            if len(param_grid) > 0:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    refit=True,
                    verbose=1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            # Train test evaluation
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

            print(f"✔ {name} Score: {test_model_score:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
  try:
      with open(file_path, "rb") as file_obj:
         return dill.load(file_obj)
      
  except Exception as e:
    raise CustomException(e, sys)
