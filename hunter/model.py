import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from preprocessing import ExoHuntDataset

models = {
    "GradientBoosting": GradientBoostingClassifier,
    "RandomForest": RandomForestClassifier
}


class ExoHuntModel:
    def __init__(self, model_name="", model_params=None, random_state=None, model_path=""):
        self.model_name = model_name
        self.evaluation_result = None
        if model_path:
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
                print(f"[*] Model {model_name} was loaded from {model_path}!")

        else:
            self.model = models[model_name](random_state=random_state, **model_params)
            print(f"[*] Model {model_name} was initialized from scratch!")

    def train(self, train_x, train_y):
        print(f"[*] Training of {self.model_name} model.")
        self.model.fit(train_x, train_y)
        print(f"[*] Model fitted successfully.")
        return self.model

    def eval(self, eval_x, eval_y):
        pred_y = self.model.predict(eval_x)
        self.evaluation_result = classification_report(eval_y, pred_y)
        return self.evaluation_result

    def predict(self, X):
        # TODO model is not always trained on 24 components, but we viciously will assume so.
        if len(X) <= 24:
            X = np.vstack([X for _ in range(24)])  # appending same values for PCA transformation.
        X = ExoHuntDataset().pca.fit_transform(X)
        return int(np.median(self.model.predict(X)))

    def save(self, path, save_eval_only=False):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, f"{self.model_name}--{datetime.now()}.pkl")
        evaluation_report_path = os.path.join(path, f"{self.model_name}--classification-report--{datetime.now()}.txt")

        if save_eval_only:
            with open(evaluation_report_path, 'w') as file:
                file.writelines(self.evaluation_result)
        else:
            with open(file_path, 'wb') as file:
                pickle.dump(self.model, file)

            with open(evaluation_report_path, 'w') as file:
                file.writelines(self.evaluation_result)
        print(f"[*] Model saved at {file_path}")
