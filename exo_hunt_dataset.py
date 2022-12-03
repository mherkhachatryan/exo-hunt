import pandas as pd
from sklearn.model_selection import train_test_split


class ExoHuntDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __load_data(self, mode=""):
        replace_categories = {2: 1, 1: 0}

        data = pd.read_csv(f'{self.dataset_path}/Exo{mode.capitalize()}.csv')
        data["LABEL"] = data["LABEL"].replace(replace_categories)

        X = data.drop("LABEL", axis=1)
        y = data.loc[:, "LABEL"]
        print("[*] Data Loaded successfully!")
        return X, y

    def load_train_data(self):
        return self.__load_data("train")

    def load_train_val_data(self, split_size, random_state):
        X, y = self.load_train_data()
        X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                              y,
                                                              test_size=split_size,
                                                              random_state=random_state)
        return X_train, X_valid, y_train, y_valid

    def load_test_data(self):
        return self.__load_data("test")
