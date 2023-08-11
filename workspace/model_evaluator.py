from sklearn.metrics import f1_score


class ModelEvaluator:
    def __init__(self, data_loader, model_trainer):
        self.data_loader = data_loader
        self.model_trainer = model_trainer

    def evaluate_model(self):
        X_eval = self.data_loader.eval_data.drop(["id", "Machine failure"], axis=1)
        y_eval = self.data_loader.eval_data["Machine failure"]

        y_pred = self.model_trainer.model.predict(X_eval)
        f1 = f1_score(y_eval, y_pred)

        return f1
