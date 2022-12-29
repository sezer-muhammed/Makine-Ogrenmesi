import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

wine_dataset_path = r"winequality-red.csv"
wine_dataset = pd.read_csv(wine_dataset_path, sep=";")

print(wine_dataset.head())
print("asdsa")