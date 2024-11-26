from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def preprocess_data(data):
    data.columns = (
        ["Label"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    )

    num_cols = [col for col in data.columns if col.startswith("I")]
    cat_cols = [col for col in data.columns if col.startswith("C")]

    data[num_cols] = data[num_cols].fillna(0)
    data[cat_cols] = data[cat_cols].fillna("missing")

    data[num_cols] = MinMaxScaler().fit_transform(data[num_cols])
    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    features = num_cols + cat_cols

    X = data[features]
    y = data["Label"]

    return X, y, num_cols, cat_cols


def split_data(X, y, num_cols, cat_cols, split_num_cat=True):
    if split_num_cat:
        X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = (
            train_test_split(
                X[num_cols], X[cat_cols], y, test_size=0.2, random_state=42
            )
        )
        return X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
