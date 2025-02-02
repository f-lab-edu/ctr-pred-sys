import yaml
import os
import pandas as pd

from typing import Dict

def load_yaml(conf_name: str) -> Dict:
    """
    Load a YAML file and return its contents.

    Parameters:
        path (str): The file path to the YAML file.

    Returns:
        dict: A dictionary containing the contents of the YAML file.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(cur_dir, "../../"))

    path = os.path.join(root_dir, f"configs/{conf_name}.yaml")

    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data

def load_data(path: str) -> pd.DataFrame:
    """
    Load a csv file into a Pandas DataFrame.

    Parameters:
        path (str): The file path to the csv file.

    Returns:
        pd.DataFrame: A DataFrame containing the contents of the csv file.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(cur_dir, "../../"))
    path = os.path.join(root_dir, path)

    data = pd.read_csv(path, sep="\t", header=None)
    return data
