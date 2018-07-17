import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math

df = pd.read_csv(
    filepath_or_buffer='../../Data/iris/iris.csv',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()

X = df.ix[:,0:4].values
y = df.ix[:,4].values

print(X)