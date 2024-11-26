import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from src.utils.load import load_yaml


class LightGBMModel:
    def __init__(self, params=None):
        self.conf = load_yaml()['model']['LGBM']
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
        self.model = None

    def fit(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_features
    ):

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, test_data],
            num_boost_round=self.conf['num_boost_round'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.conf['early_stopping_rounds'])
            ]
        )

    def predict(self, X):
        if not self.model:
            raise ValueError("Model has not been trained yet. Call `fit` first.")
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return roc_auc_score(y, y_pred)

    def save_model(self, path):
        if not self.model:
            raise ValueError("Model has not been trained yet. Call `fit` first.")
        self.model.save_model(path)

    def load_model(self, path):
        self.model = lgb.Booster(model_file=path)
