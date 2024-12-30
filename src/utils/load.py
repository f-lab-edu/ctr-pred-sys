import pandas as pd
import yaml


def load_yaml(path):
    """
    Load a YAML file and return its contents.

    Parameters:
        path (str): The file path to the YAML file.

    Returns:
        dict: A dictionary containing the contents of the YAML file.
    """
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_data(path):
    """
    Load a csv file into a Pandas DataFrame.

    Parameters:
        path (str): The file path to the csv file.

    Returns:
        pd.DataFrame: A DataFrame containing the contents of the csv file.
    """
    data = pd.read_csv(path, sep="\t", header=None)
    return data
