import torch
import numpy as np
import pandas as pd

from src.models.lgbm import LightGBMModel
from sklearn.metrics import log_loss, accuracy_score
from typing import Tuple, Union

def compute_log_loss(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute the Log Loss for the given true and predicted values.

    Parameters:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: Log Loss value.
    """
    return log_loss(y_true, y_pred)


def compute_accuracy(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute the accuracy for the given true and predicted values.

    Parameters:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: Accuracy score.
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_binary)


def evaluate_dl_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    """
    Evaluate a deep learning model on the given dataloader.

    Parameters:
        model (torch.nn.Module): The deep learning model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for evaluation data.

    Returns:
        tuple: Log Loss and Accuracy of the model on the evaluation data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for numeric, categorical, label in dataloader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            label = label.to(device)

            outputs = model(numeric, categorical)
            preds.extend(outputs.cpu().numpy())
            labels.extend(label.cpu().numpy())

    log_loss = compute_log_loss(np.array(labels), np.array(preds))
    accuracy = compute_accuracy(np.array(labels), np.array(preds))

    print(f"Log Loss: {log_loss:.4f}, Accuracy: {accuracy:.4f}")
    return log_loss, accuracy


def evaluate_lgbm_model(model: LightGBMModel, data: pd.DataFrame, labels: pd.Series) -> Tuple[float, float]:
    """
    Evaluate a LightGBM model on the given dataset.

    Parameters:
        model (lightgbm.Booster): The LightGBM model to evaluate.
        data (pd.DataFrame): Feature data for evaluation.
        labels (pd.Series): True binary labels.

    Returns:
        tuple: Log Loss and Accuracy of the model on the evaluation data.
    """
    preds = model.predict(data)

    log_loss = compute_log_loss(labels, preds)
    accuracy = compute_accuracy(labels, preds)

    print(f"Log Loss: {log_loss:.4f}, Accuracy: {accuracy:.4f}")
    return log_loss, accuracy
