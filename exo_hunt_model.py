from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

models = {
    "GradientBoosting": GradientBoostingClassifier,
    "RandomForest":     RandomForestClassifier
}


class ExoHuntModel:
    def __init__(self, model_name="", random_state=None, model_path=""):
        if model_path:
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
                print(f"[*] Model {model_name} was loaded from {model_path}!")

        else:
            self.model = models[model_name](random_state=random_state)
            print(f"[*] Model {model_name} was initialized from scratch!")

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)
        return self.model

    def eval(self, eval_x, eval_y):
        # TODO: Add logging logic to the evaluation results
        pred_y = self.model.predict(eval_x)
        classification_report(eval_y, pred_y)
        plot_roc_curve(self.model, eval_x, eval_y)
        plot_precision_recall_curve(self.model, eval_x, eval_y)
        plot_roc_curve(self.model, eval_x, eval_y)
        return self.model

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
