import pandas as pd
import yaml


def load_yaml():
    file_path = "/Users/euyeom/Desktop/Gigi/repo/ctr-pred-sys/configs/config.yaml"
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_data(path):
    data = pd.read_csv(path, sep="\t", header=None)
    return data
