import torch
import torch.nn as nn
import torch.optim as optim


def train_dl_model(model, train_loader, test_loader, epochs):
    """
    Train a deep learning model.

    Parameters:
        model (torch.nn.Module): The deep learning model to train.
        train_loader (torch.utils.data.DataLoader): Dataloader for the training data.
        test_loader (torch.utils.data.DataLoader): Dataloader for the testing/validation data.
        epochs (int): The number of training epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for numeric, categorical, labels in train_loader:
            numeric, categorical, labels = (
                numeric.to(device),
                categorical.to(device),
                labels.to(device),
            )

            optimizer.zero_grad()
            outputs = model(numeric, categorical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for numeric, categorical, labels in test_loader:
                numeric, categorical, labels = (
                    numeric.to(device),
                    categorical.to(device),
                    labels.to(device),
                )
                outputs = model(numeric, categorical)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}"
        )


def train_lgbm_model(model, X_train, y_train, X_val, y_val):
    """
    Train a LightGBM model and evaluate it on the validation data.

    Parameters:
        model (lightgbm.Booster): The LightGBM model to train.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target labels.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.Series): Validation target labels.
    """
    model.fit(X_train, y_train, X_val, y_val)
    auc_score = model.evaluate(X_val, y_val)
    print(f"LightGBM Validation AUC: {auc_score:.4f}")
