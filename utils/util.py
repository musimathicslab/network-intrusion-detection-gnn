import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np

import logging
mpl.use('TkAgg')


def log_train_test_status(list_attack_cat, y_train, y_test):
    unique, counts = np.unique(y_train, return_counts=True)
    logging.info('\ny_train status:')
    logging.info(dict(zip(list_attack_cat, counts)))

    unique, counts = np.unique(y_test, return_counts=True)
    logging.info('\ny_pred status:')
    logging.info(dict(zip(list_attack_cat, counts)))


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())

    return logger


def create_directory(p):
    from pathlib import Path
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, label, confusion_matrix_path):
    logging.disable(logging.CRITICAL)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index=[i for i in label], columns=[i for i in label])
    sns.heatmap(df_cm, annot=True, fmt='.2f')
    fig = plt.gcf()
    plt.subplots_adjust(bottom=0.3, left=0.2)
    fig.savefig(confusion_matrix_path, dpi=100)
    plt.close()


def plot_recall(labels, recall, recall_path):
    logging.disable(logging.CRITICAL)

    # create data
    x_pos = np.arange(0, len(labels) * 2, 2)
    _recall = np.round(recall * 100, 2)

    # create bars
    plt.bar(x_pos, _recall, width=0.5)

    # rotation of the bar names
    plt.xticks(x_pos, labels, rotation=70)
    # custom the subplot layout
    plt.subplots_adjust(bottom=0.3, top=0.8)
    # enable grid
    plt.grid(True)

    plt.title('Detection Rate')
    plt.ylabel('recall score')

    # print value on the top of bar
    x_locs, x_labs = plt.xticks()
    for i, v in enumerate(_recall):
        plt.text(x_locs[i] - 0.6, v + 5, str(v))

    # set limit on y label
    plt.ylim(0, max(_recall) + 15)

    # savefig
    fig = plt.gcf()
    fig.savefig(recall_path, dpi=100)
    plt.close()

    logging.disable(logging.NOTSET)
