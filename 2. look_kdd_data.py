import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('network1.csv')
print(df.head())
print(df.info())

obj_df = df.select_dtypes(include=['object']).copy()
print(obj_df.head())
print(obj_df[obj_df.isnull().any(axis=1)])


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

# Look at the categorical data

output = df['protocol_type'].values
labels = set(output)
print('The different type of protocol are:', labels)
print('='*125)
print('The number of different protocol are:', len(labels))

output = df['service'].values
labels = set(output)
print('The different type of service are:', labels)
print('='*125)
print('The number of different service are:', len(labels))

output = df['flag'].values
labels = set(output)
print('The different type of flag are:', labels)
print('='*125)
print('The number of different flag are:', len(labels))

# Look at the numerical data

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(df['src_bytes'].head(500))
axs[0, 0].set_title('F5: scr_bytes')
axs[0, 1].plot(df['dst_bytes'].head(500))
axs[0, 1].set_title('F6: dst_bytes')
axs[1, 0].plot(df['dst_host_count'].head(500))
axs[1, 0].set_title('F31: dst_host_count')
axs[1, 1].plot(df['dst_host_same_src_port_rate'].head(500))
axs[1, 1].set_title('F36: dst_host_same_src_port_rat')

for ax in axs.flat:
    ax.set(xlabel='Time(Minutes)', ylabel='Enterprise Network Log')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
  
fig.suptitle('Data Samples from Network Traffic Profile: Numerical Features')
plt.show()
