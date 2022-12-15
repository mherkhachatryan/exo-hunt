import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


class ExoHuntDataset:
    def __init__(self, dataset_path=None, random_state=42):
        self.dataset_path = dataset_path
        self.over_sample = SMOTE(random_state=random_state)
        self.pca = PCA(n_components=24, random_state=random_state)

    def __load_data(self, mode="", **kwargs):
        replace_categories = {2: 1, 1: 0}

        data = pd.read_csv(f'{self.dataset_path}/Exo{mode.capitalize()}.csv', **kwargs)
        data["LABEL"] = data["LABEL"].replace(replace_categories)

        X = data.drop("LABEL", axis=1)
        y = data.loc[:, "LABEL"]

        X = self.pca.fit_transform(X)
        print("[*] Data Loaded successfully!")
        return X, y

    def load_train_data(self, **kwargs):
        X, y = self.__load_data("train", **kwargs)
        X, y = self.over_sample.fit_resample(X, y)
        return X, y

    def load_train_val_data(self, split_size, random_state, **kwargs):
        X, y = self.load_train_data(**kwargs)
        X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                              y,
                                                              test_size=split_size,
                                                              random_state=random_state)
        return X_train, X_valid, y_train, y_valid

    def load_test_data(self, **kwargs):
        return self.__load_data("test", **kwargs)

    def _load_mock_data(self, mode="train", nrows=10):
        if mode == "train":
            X, y = self.load_train_data(nrows=nrows)
        elif mode == "test":
            X, y = self.load_test_data(nrows=nrows)
        else:
            raise ValueError("Wrong mode")

        return X, y
