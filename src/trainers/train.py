import torch
import torch.nn as nn
import torch.optim as optim


def train_dl_model(model, train_loader, test_loader, epochs):
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
    model.fit(X_train, y_train, X_val, y_val)
    auc_score = model.evaluate(X_val, y_val)
    print(f"LightGBM Validation AUC: {auc_score:.4f}")
