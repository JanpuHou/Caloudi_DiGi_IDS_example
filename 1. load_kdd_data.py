import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import get_file

try:
    path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
except:
    print('Error downloading')
    raise
    
print(path) 

# This file is a CSV, just no CSV extension or headers
# Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
df = pd.read_csv(path, header=None)


print(df.head())



# Separate labels and network traffic profile data

#keep the last column as a series
last_column = df.iloc[: , -1]
print(last_column.head())
last_column.to_csv('kddcup_10_labels.csv',header=None,index=False)
last_column=pd.read_csv('kddcup_10_labels.csv')
print(last_column.head())

labels=last_column.to_numpy()
print(labels.shape)
print(labels[1:10])

#drop the last column of the dataframe
df = df.iloc[: , :-1]
print(df.head())


df.to_csv('kddcup_10_data.csv',header=None, index=False)
df1=pd.read_csv('kddcup_10_data.csv')
print(df1.head())

# Random sample some simulated network data for validation later on

df.sample(n=1000).to_csv('network1.csv',header=None,index=False)
df1=pd.read_csv('network1.csv')
print(df1.head())

df.sample(n=5000).to_csv('network2.csv',header=None,index=False)
df1=pd.read_csv('network2.csv')
print(df1.head())

df.sample(n=10000).to_csv('network3.csv',header=None,index=False)
df1=pd.read_csv('network3.csv')
print(df1.head())

df.sample(n=15000).to_csv('network4.csv',header=None,index=False)
df1=pd.read_csv('network4.csv')
print(df1.head())

df.sample(n=20000).to_csv('network5.csv',header=None,index=False)
df1=pd.read_csv('network5.csv')
print(df1.head())
