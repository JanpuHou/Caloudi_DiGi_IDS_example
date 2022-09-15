import pandas as pd
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import datetime as dt

try:
    path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
except:
    print('Error downloading')
    raise
    
print(path) 

# This file is a CSV, just no CSV extension or headers
# Download from: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
original_df = pd.read_csv(path, header=None)
df = original_df
df.to_csv('kddcup_10_data.csv',header=False, index=False)
print(df.shape)


df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

# Example of Attack Data Only (outcome = Neptune)
df = df[df['outcome'].str.contains('neptune.')]
print (df.shape)
print(df.iloc[: , -1].value_counts())

#drop the last column of the dataframe
df = df.iloc[: , :-1]
print(df.head())
print(df.shape)
df.to_csv('neptune_data.csv',header=False, index=False)
df=pd.read_csv(r'neptune_data.csv',header=None)
print(df.shape)
print(df.head())
df_new = df.iloc[:, 0:11]
print(df_new.head())

df = original_df


# Example of Normal Data Only (outcome = Normal)
df = df[df['outcome'].str.contains('normal.')]
print (df.shape)
print(df.iloc[: , -1].value_counts())

#drop the last column of the dataframe
df = df.iloc[: , :-1]
print(df.head())
print(df.shape)
df.to_csv('normal_data.csv',header=False, index=False)
df=pd.read_csv(r'normal_data.csv',header=None)
print(df.shape)
print(df.head())
df_new = df.iloc[:, 0:11]
print(df_new.head())