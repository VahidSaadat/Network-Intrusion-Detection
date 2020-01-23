import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

headers = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
           "lnum_compromised", "lroot_shell", "lsu_attempted", "lnum_root", "lnum_file_creations",
           "lnum_shells", "lnum_access_files", "lnum_outbound_cmds", "is_host_login", "is_guest_login",
           "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

dataset_url = 'kddcup.data_10_percent_corrected.csv'

def lb_binarizer(col_name):
    lb = LabelBinarizer()
    lb_trans = lb.fit_transform(df[col_name])
    lb_frame = pd.DataFrame(lb_trans, columns=lb.classes_)
    #df.join(lb_frame)
    return lb_frame

# Min-Max normalization
def min_max_normalizer(feature_name):
    min_number = min(df[feature_name].values)
    max_number = max(df[feature_name].values)
    length = max_number - min_number
    for number in df[feature_name].unique():
        new_number = (number - min_number) / length
        df[feature_name].replace(number, new_number, inplace=True)

df = pd.read_csv(dataset_url, header=None, names=headers)

# label clean - normal = 0, others = 1
df.replace(to_replace =["normal."], value = 0, inplace=True)
for feature_type in df['label'].unique()[1:]:
    df.replace(to_replace =[feature_type], value = 1, inplace=True)

# merge types that have very lower number of records
for feature_type in df['service'].unique():
    if df['service'].value_counts()[feature_type] < 1000 :
        df.replace(to_replace =[feature_type], value = 'others_service', inplace=True) 
        
for feature_type in df['flag'].unique():
    if df['flag'].value_counts()[feature_type] < 1000 :
        df.replace(to_replace =[feature_type], value = 'others_flag', inplace=True)

low_count_feature_list = []        
for feature_type in df['src_bytes'].unique():
    if df['src_bytes'].value_counts()[feature_type] < 2500 :
        low_count_feature_list.append(feature_type)
    else:
        df['src_bytes'].replace(feature_type, str(feature_type) + '_src_bytes', inplace=True)      
df['src_bytes'].replace(low_count_feature_list, 'others_src_bytes', inplace=True)

low_count_feature_list = []        
for feature_type in df['dst_bytes'].unique():
    if df['dst_bytes'].value_counts()[feature_type] < 2500 :
        low_count_feature_list.append(feature_type)
    else:
        df['dst_bytes'].replace(feature_type, str(feature_type) + '_dst_bytes', inplace=True)      
df['dst_bytes'].replace(low_count_feature_list, 'others_dst_bytes', inplace=True)

for feature_type in df['duration'].unique()[1:]:
    df['duration'].replace(feature_type, 1, inplace=True)

# Useless features - very low number of records
df.drop(['is_host_login','is_guest_login', 'lnum_access_files',
         'lnum_outbound_cmds', 'lnum_shells', 'lnum_file_creations',
         'lnum_compromised', 'lroot_shell', 'lsu_attempted', 'lnum_root',
         'num_failed_logins', 'hot', 'land', 'wrong_fragment', 'urgent'], axis = 1, inplace = True)

need_normal_cols = ['count', 'srv_count', 'dst_host_srv_count', 'dst_host_count']
for col_name in need_normal_cols:
    min_max_normalizer(col_name)

obj_df = df.select_dtypes(include=['object']).copy()
for obj_header in obj_df.columns:
    lb_frame = lb_binarizer(obj_header)
    df = df.join(lb_frame)
    df.drop(obj_header, axis=1, inplace=True)

df.to_csv('kddcup_normal.csv')
