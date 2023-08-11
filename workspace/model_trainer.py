from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.model = None

    def train_model(self):
        X = self.data_loader.train_data.drop(["id", "Machine failure"], axis=1)
        y = self.data_loader.train_data["Machine failure"]

        self.model = RandomForestClassifier()
        self.model.fit(X, y)
