import pandas as pd
import yaml
import os

def load_yaml(conf_name):
    """
    Load a YAML file and return its contents.

    Parameters:
        path (str): The file path to the YAML file.

    Returns:
        dict: A dictionary containing the contents of the YAML file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, f"configs/{conf_name}.yaml")

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
