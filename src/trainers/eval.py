import torch
import numpy as np

from sklearn.metrics import log_loss, accuracy_score


def compute_log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)


def compute_accuracy(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_binary)


def evaluate_dl_model(model, dataloader):
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


def evaluate_lgbm_model(model, data, labels):
    preds = model.predict(data)

    log_loss = compute_log_loss(labels, preds)
    accuracy = compute_accuracy(labels, preds)

    print(f"Log Loss: {log_loss:.4f}, Accuracy: {accuracy:.4f}")
    return log_loss, accuracy
