from typing import final
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import MAPPING

CSV_FILES: final = [
    'UNSW-NB15_1.csv',
    'UNSW-NB15_2.csv',
    'UNSW-NB15_3.csv',
    'UNSW-NB15_4.csv'
]

FEATURES: final = [
    'srcip',
    'sport',
    'dstip',
    'dsport',
    'proto',
    'state',
    'dur',
    'sbytes',
    'dbytes',
    'sttl',
    'dttl',
    'sloss',
    'dloss',
    'service',
    'sload',
    'dload',
    'spkts',
    'dpkts',
    'swin',
    'dwin',
    'stcpb',
    'dtcpb',
    'smeansz',
    'dmeansz',
    'trans_depth',
    'res_bdy_len',
    'sjit',
    'djit',
    'stime',
    'ltime',
    'sintpkt',
    'dintpkt',
    'tcprtt',
    'synack',
    'ackdat',
    'is_sm_ips_ports',
    'ct_state_ttl',
    'ct_flw_http_mthd',
    'is_ftp_login',
    'ct_ftp_cmd',
    'ct_srv_src',
    'ct_srv_dst',
    'ct_dst_ltm',
    'ct_src_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
    'attack_cat',
    'label'
]

FEATURES_TO_STANDARDIZE: final = [
    'dur',
    'sbytes',
    'dbytes',
    'sttl',
    'dttl',
    'sloss',
    'dloss',
    'sload',
    'dload',
    'spkts',
    'dpkts',
    'swin',
    'dwin',
    'stcpb',
    'dtcpb',
    'smeansz',
    'dmeansz',
    'trans_depth',
    'res_bdy_len',
    'sjit',
    'djit',
    'sintpkt',
    'dintpkt',
    'tcprtt',
    'synack',
    'ackdat',
    'is_sm_ips_ports',
    'ct_state_ttl',
    'ct_flw_http_mthd',
    'is_ftp_login',
    'ct_ftp_cmd',
    'ct_srv_src',
    'ct_srv_dst',
    'ct_dst_ltm',
    'ct_src_ltm',
    'ct_src_dport_ltm',
    'ct_dst_sport_ltm',
    'ct_dst_src_ltm',
]


def pre_processing(dataset_path: str):
    # Init global dataframe
    df = pd.DataFrame(columns=FEATURES)

    # Create global df from each local df
    for csv_file in CSV_FILES:
        df_local = pd.read_csv(os.path.join(dataset_path, csv_file), header=None, low_memory=False)
        df_local.columns = FEATURES
        df = pd.concat([df, df_local])

    # Remove switch information and time-related information
    for feature in ['srcip', 'sport', 'dstip', 'dsport', 'ltime', 'stime']:
        df = df.drop(feature, axis='columns')

    # Apply ohe on categorical features
    # Proto feature
    df['proto'] = df['proto'].apply(lambda x: x.lower())
    for value in ['tcp', 'udp']:
        df[f'proto_{value}'] = df.apply(lambda row: 1 if row['proto'] == value else 0, axis=1)
    df['proto_other'] = df.apply(
        lambda row: 1 if row['proto'] != 'tcp' and row['proto'] != 'udp' else 0, axis=1)
    # State feature
    df['state'] = df['state'].apply(lambda x: x.lower())
    for value in ['fin', 'con', 'int']:
        df[f'state_{value}'] = df.apply(lambda row: 1 if row['state'] == value else 0, axis=1)
    df['state_other'] = df.apply(
        lambda row: 1 if row['state'] != 'fin' and row['state'] != 'con' and row['state'] != 'int' else 0, axis=1)
    # Service feature
    df['service'] = df['service'].apply(lambda x: x.lower())
    for value in ['-', 'dns']:
        df[f'service_{value}'] = df.apply(lambda row: 1 if row['service'] == value else 0, axis=1)
    df['service_other'] = df.apply(lambda row: 1 if row['service'] != '-' and row['service'] != 'dns' else 0, axis=1)

    # Clean data
    df['attack_cat'] = df['attack_cat'].str.strip()
    df['attack_cat'] = df['attack_cat'].replace(np.nan, 'Normal')
    df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'], errors='coerce')
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace(np.nan, 0)
    df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].replace(np.nan, 0)
    df['is_ftp_login'] = df['is_ftp_login'].replace(np.nan, 0)
    df['attack_cat'] = df['attack_cat'].replace('Backdoor', 'Backdoors')

    # Mapping label
    df['label'] = df['attack_cat'].map(MAPPING)

    # Use Min Max Scaler
    scaler = MinMaxScaler()
    df[FEATURES_TO_STANDARDIZE] = scaler.fit_transform(df[FEATURES_TO_STANDARDIZE])

    # Create train, validation and test set
    train, val = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    train, test = train_test_split(train, test_size=0.25, shuffle=True, random_state=42)
    train.to_csv(os.path.join(dataset_path, 'UNSW-NB15-train.csv'), index=False)
    val.to_csv(os.path.join(dataset_path, 'UNSW-NB15-val.csv'), index=False)
    test.to_csv(os.path.join(dataset_path, 'UNSW-NB15-test.csv'), index=False)


if __name__ == '__main__':
    pre_processing(os.path.join('dataset', 'raw'))
