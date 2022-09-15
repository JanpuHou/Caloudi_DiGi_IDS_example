import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


df=pd.read_csv(r'kddcup_10_data.csv',header=None)


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
    'dst_host_srv_rerror_rate']


obj_df = df.select_dtypes(include=['object']).copy()
print(obj_df.head())
print(obj_df[obj_df.isnull().any(axis=1)])


def encode_data(df):
    oe_protocol = OneHotEncoder()
    oe_results = oe_protocol.fit_transform(df[["protocol_type"]])
    df1_P=pd.DataFrame(oe_results.toarray(), columns=oe_protocol.categories_)
#print(df1_P.head())
    df = df.join(df1_P)
#print(df.head())


    oe_service = OneHotEncoder()
    oe_results = oe_service.fit_transform(df[["service"]])
    df1_S=pd.DataFrame(oe_results.toarray(), columns=oe_service.categories_)
#print(df1_S.head())
    df = df.join(df1_S)
#print(df.head())


    oe_flag = OneHotEncoder()
    oe_results = oe_flag.fit_transform(df[["flag"]])
    df1_F=pd.DataFrame(oe_results.toarray(), columns=oe_flag.categories_)
#print(df1_F.head())
    df = df.join(df1_F)
#print(df.head())
    df=df.drop(['protocol_type','service','flag'], axis=1)
    return df


df=encode_data(df)
print(df.head())

# prepare data for training

x_columns = df.columns
x = df[x_columns].values

# prepare label for training

df=pd.read_csv('kddcup_10_labels.csv',header=None)
df.columns = ['outcome']
print(df.shape)

oe_outcome = OneHotEncoder()
oe_results = oe_outcome.fit_transform(df[["outcome"]])
df1_P=pd.DataFrame(oe_results.toarray(), columns=oe_outcome.categories_)
#print(df1_P.head())
df = df.join(df1_P)
print(df.head())

y=df.drop(['outcome'], axis=1)
print(df.head())


column_names = df.columns.values
print(column_names)




np.save('x.npy', x)    
x = np.load('x.npy')
np.save('y.npy', y)    
y = np.load('y.npy')
print(x.shape)
print(y.shape)


























