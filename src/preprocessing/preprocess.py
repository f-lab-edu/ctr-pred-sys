import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple, List, Union

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Preprocess the dataset by handling missing values, scaling numeric features,
    and encoding categorical features.

    Parameters:
        data (pd.DataFrame): The input raw dataset.

    Returns:
        tuple:
            - X (pd.DataFrame): The preprocessed features.
            - y (pd.Series): The target labels.
            - num_cols (list): List of numeric column names.
            - cat_cols (list): List of categorical column names.
    """
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


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    split_num_cat: bool = True,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
]:
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): The preprocessed feature set.
        y (pd.Series): The target labels.
        num_cols (list): List of numeric column names.
        cat_cols (list): List of categorical column names.
        split_num_cat (bool, optional): Whether to split numeric and categorical features separately.
                                         Default is True.

    Returns:
        tuple: Training and testing sets.
            If split_num_cat is True:
                - X_train_num (pd.DataFrame): Training numeric features.
                - X_test_num (pd.DataFrame): Testing numeric features.
                - X_train_cat (pd.DataFrame): Training categorical features.
                - X_test_cat (pd.DataFrame): Testing categorical features.
                - y_train (pd.Series): Training labels.
                - y_test (pd.Series): Testing labels.
            If split_num_cat is False:
                - X_train (pd.DataFrame): Training features.
                - X_test (pd.DataFrame): Testing features.
                - y_train (pd.Series): Training labels.
                - y_test (pd.Series): Testing labels.
    """
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
