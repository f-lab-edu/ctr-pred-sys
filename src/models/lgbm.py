import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from src.utils.load import load_yaml


class LightGBMModel:
    """
    A wrapper class of LightGBM model for Click-Through Rate (CTR) prediction.
    """

    def __init__(self, params=None):
        """
        Initialize the LightGBMModel.

        Parameters:
            params (dict, optional): Custom LightGBM parameters. If None, default parameters are loaded from a YAML file.
        """
        self.conf = load_yaml('lgbm')['model']
        self.params = {
            'objective': self.conf['objective'],
            'metric': self.conf['metric'],
            'boosting_type': self.conf['boosting_type'],
            'learning_rate': self.conf['learning_rate'],
            'num_leaves': self.conf['num_leaves'],
            'max_depth': self.conf['max_depth'],
            'min_data_in_leaf': self.conf['min_data_in_leaf'],
            'feature_fraction': self.conf['feature_fraction'],
            'bagging_fraction': self.conf['bagging_fraction'],
            'bagging_freq': self.conf['bagging_freq'],
            'verbose': self.conf['verbose'],
            'seed': self.conf['seed']
        }
        self.lgbm_model = None

    def fit(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_features
    ):
        """
        Train the LightGBM model on the provided training and testing datasets.

        Parameters:
            X_train (pd.DataFrame): The training feature set.
            X_test (pd.DataFrame): The testing feature set.
            y_train (pd.Series): The training target labels.
            y_test (pd.Series): The testing target labels.
            categorical_features (list): List of categorical feature names.
        """
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features, reference=train_data)

        self.lgbm_model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, test_data],
            num_boost_round=self.conf['num_boost_round'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.conf['early_stopping_rounds'])
            ]
        )

    def predict(self, X):
        """
        Predict target values for the given input features.

        Parameters:
            X (pd.DataFrame): The input feature set.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.lgbm_model:
            raise ValueError("Model has not been trained yet. Call `fit` first.")
        return self.lgbm_model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model using the ROC-AUC score.

        Parameters:
            X (pd.DataFrame): The input feature set.
            y (pd.Series): The true target labels.

        Returns:
            float: The ROC-AUC score of the model.
        """
        y_pred = self.predict(X)
        return roc_auc_score(y, y_pred)

    def save_model(self, path):
        """
        Save the trained LightGBM model to a file.

        Parameters:
            path (str): Path where the model should be saved.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if not self.model:
            raise ValueError("Model has not been trained yet. Call `fit` first.")
        self.lgbm_model.save_model(path)

    def load_model(self, path):
        """
        Load a LightGBM model from a file.

        Parameters:
            path (str): Path to the saved model file.
        """
        self.lgbm_model = lgb.Booster(model_file=path)
