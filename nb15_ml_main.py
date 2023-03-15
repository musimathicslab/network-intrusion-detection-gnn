import pickle
from typing import final
import pandas as pd
import os

import xgboost as xgb

from nb15_pre_processing import MAPPING

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from utils import get_path_of_all
from utils.util import create_directory
from utils.util import plot_confusion_matrix
from utils.util import plot_recall
from utils.util import setup_logger

NAME_DIR: final = '02 - UNSW-NB15 XGB Classifier'


def test_xgb(dataset_path: str, binary: bool):
    # Get path of all file
    log_path, log_file_path, log_train_path, model_path, result_path, \
    confusion_matrix_path, detection_rate_path = get_path_of_all(NAME_DIR)
    for directory in [log_path, model_path, confusion_matrix_path, detection_rate_path]:
        create_directory(directory)

    # Create path of training and test dataset
    _file_name_training = f'UNSW-NB15-train{"-binary" if binary else ""}.csv'
    _file_path_training = os.path.join(dataset_path, _file_name_training)
    _file_name_val = f'UNSW-NB15-val{"-binary" if binary else ""}.csv'
    _file_path_val = os.path.join(dataset_path, _file_name_training)
    _file_name_testing = f'UNSW-NB15-test{"-binary" if binary else ""}.csv'
    _file_path_testing = os.path.join(dataset_path, _file_name_testing)

    # Init logger
    logger = setup_logger('logger', log_file_path)

    # Read al dataframe
    dfs = []
    dfs.append(pd.read_csv(_file_path_training, low_memory=False))
    dfs.append(pd.read_csv(_file_path_val, low_memory=False))
    dfs.append(pd.read_csv(_file_path_testing, low_memory=False))

    # Define model
    model = xgb.XGBClassifier()

    # Features to use
    extract_col = [
        'sttl', 'dload', 'dttl', 'sload', 'smeansz', 'sintpkt', 'dmeansz', 'dintpkt', 'tcprtt', 'ackdat',
        'synack', 'ct_state_ttl', 'ct_srv_src', 'ct_dst_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
        'proto_tcp', 'proto_udp', 'proto_other', 'state_fin', 'state_con', 'state_int', 'state_other',
        'service_-', 'service_dns', 'service_other'
    ]
    # Get x_train and y_train
    x_train = dfs[0][extract_col].values
    y_train = dfs[0]['label'].values

    # Load if model exist or train it
    model_name = f'model{"-binary" if binary else ""}'
    model_trained_path = os.path.join(model_path, f'{model_name}.h5')
    if os.path.exists(model_trained_path):
        model = pickle.load(open(model_trained_path, "rb"))
    else:
        # Train model
        model.fit(x_train, y_train)
        pickle.dump(model, open(model_trained_path, "wb"))

    # Get x_test and y_test
    x_test = dfs[2][extract_col].values
    y_test = dfs[2]['label'].values

    y_pred = model.predict(x_test)

    logger.info(f'Accuracy on test: {accuracy_score(y_test, y_pred)}')
    logger.info(f'Balanced accuracy on test: {balanced_accuracy_score(y_test, y_pred)}')
    logger.info(f'\n{classification_report(y_test, y_pred, digits=3)}')

    # Plot detection rate and confusion matrix if it is multi-classification task
    if not binary:
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, MAPPING.keys(), os.path.join(confusion_matrix_path, 'xgb-cm.png'))
        recall = recall_score(y_test, y_pred, average=None)
        plot_recall(MAPPING.keys(), recall, os.path.join(detection_rate_path, 'xgb-dr.png'))


if __name__ == "__main__":
    test_xgb(os.path.join(os.getcwd(), 'dataset', 'raw'), True)
