import bentoml
import torch

from __future__ import annotations
from bentoml.io import NumpyNdarray
from torch.utils.data import DataLoader

from src.utils.load import load_data, load_yaml
from src.preprocessing.preprocess import preprocess_data, split_data
from src.trainers.train import train_dl_model, train_lgbm_model
from src.trainers.eval import evaluate_dl_model, evaluate_lgbm_model
from src.datasets.criteo import CriteoDataset
from src.models.deepfm import DeepFM
from src.models.dcn_v2 import DCNv2
from src.models.lgbm import LightGBMModel

# Define a BentoML service with specified resource limits and traffic settings
@bentoml.service(
    resources={"cpu": "4", "nvidia.com/gpu": "1"},
    traffic={"timeout": 600},
)
class ModelService:
    """
    BentoML service for running machine learning models on the Criteo dataset.
    
    Supports three model types:
    - LightGBM (lgbm)
    - DeepFM (deepfm)
    - DCNv2 (dcn_v2)
    
    This service preprocesses the data, trains the specified model, and evaluates its performance.
    """

    def __init__(self) -> None:
        """
        Initialize the BentoML service.
        """
        self.model_type = "dcn_v2"
        self.conf = load_yaml(self.model_type)
        self.data_path = load_yaml("data")["data"]["dataset_path"]

    @bentoml.api(input=bentoml.io.Text(), output=bentoml.io.Text(), route="/run_model")
    def run_model(self, model_type: str) -> str:
        """
        Run the specified machine learning model.

        Args:
            model_type (str): The type of model to run. Options are 'lgbm', 'deepfm', or 'dcn_v2'.

        Returns:
            str: The evaluation results of the model.
        """
        print('[STEP 0] start')

        print('[STEP 1] load data')
        data = load_data(self.data_path)

        print('[STEP 2] preprocess data')
        X, y, num_cols, cat_cols = preprocess_data(data)

        if model_type == 'lgbm':
            """
            Run the LightGBM model.
            """
            print('[STEP 3] Running LightGBM model')
            X_train, X_test, y_train, y_test = split_data(X, y, num_cols, cat_cols, split_num_cat=False)
            model = LightGBMModel()
            model.fit(X_train, X_test, y_train, y_test, cat_cols)
            evaluation = evaluate_lgbm_model(model, X_test, y_test)
            return f"LightGBM evaluation: {evaluation}"

        elif model_type == 'deepfm':
            """
            Run the DeepFM model.
            """
            print('[STEP 3] Running DeepFM model')
            X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = split_data(X, y, num_cols, cat_cols, split_num_cat=True)

            num_numeric = len(num_cols)
            categorical_dims = [X[col].nunique() for col in cat_cols]
            embedding_dim = self.conf["model"]["embedding_dim"]
            deep_dims = self.conf["model"]["deep_dims"]
            dropout = self.conf["model"]["dropout"]
            batch_size = self.conf["training"]["batch_size"]
            epochs = self.conf["training"]["epochs"]
            lr = self.conf["training"]["learning_rate"]

            train_dataset = CriteoDataset(X_train_num, X_train_cat, y_train)
            test_dataset = CriteoDataset(X_test_num, X_test_cat, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DeepFM(num_numeric, categorical_dims, embedding_dim, deep_dims, dropout).to(device)
            train_dl_model(model, train_loader, test_loader, epochs, lr)

            evaluation = evaluate_dl_model(model, test_loader)
            return f"DeepFM evaluation: {evaluation}"

        elif model_type == 'dcn_v2':
            """
            Run the DCNv2 model.
            """
            print('[STEP 3] Running DCNv2 model')
            X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = split_data(X, y, num_cols, cat_cols, split_num_cat=True)

            num_numeric = len(num_cols)
            categorical_dims = [X[col].nunique() for col in cat_cols]
            embedding_dim = self.conf["model"]["embedding_dim"]
            deep_dims = self.conf["model"]["deep_dims"]
            cross_layers = self.conf["model"]["cross_layers"]
            dropout = self.conf["model"]["dropout"]
            batch_size = self.conf["training"]["batch_size"]
            epochs = self.conf["training"]["epochs"]
            lr = self.conf["training"]["learning_rate"]

            train_dataset = CriteoDataset(X_train_num, X_train_cat, y_train)
            test_dataset = CriteoDataset(X_test_num, X_test_cat, y_test)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = DCNv2(num_numeric, categorical_dims, embedding_dim, deep_dims, cross_layers, dropout).to(device)
            train_dl_model(model, train_loader, test_loader, epochs, lr)

            evaluation = evaluate_dl_model(model, test_loader)
            return f"DCNv2 evaluation: {evaluation}"

        else:
            return "Invalid model type. Please choose from ['lgbm', 'deepfm', 'dcn_v2']"