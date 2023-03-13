import argparse
import os

SEPARATOR = '\n------------------------------------\n'
DETECTION_RATE = 'Detection Rate'
CONFUSION_MATRIX = 'Confusion Matrix'
BAR_STACKED = 'Bar Stacked'
LOGGER = 'logger.log'
TRAINING_LOGGER = 'training.log'
CWD = os.getcwd()

MAPPING = {
    "Normal": 0,
    "Worms": 1,
    "Backdoors": 2,
    "Shellcode": 3,
    "Analysis": 4,
    "Reconnaissance": 5,
    "DoS": 6,
    "Fuzzers": 7,
    "Exploits": 8,
    "Generic": 9
}


def get_path_of_all(name_dir, num_neigh=None, hid=0, n_convs=0, augmentation=False):
    log_path = os.path.join(CWD, 'log', name_dir)
    log_file_path = os.path.join(log_path, LOGGER if num_neigh is None else f'logger_{num_neigh}'
                                                                            f'hid_{hid}_n_convs_{n_convs}'
                                                                            f'{"_aug" if augmentation else ""}.log')
    log_train_path = os.path.join(log_path,
                                  TRAINING_LOGGER if num_neigh is None else f'training_{num_neigh}'
                                                                            f'hid_{hid}_n_convs_{n_convs}'
                                                                            f'{"_aug" if augmentation else ""}.log')
    model_path = os.path.join(CWD, 'model', name_dir)
    result_path = os.path.join(log_path, 'results.csv')
    image_path = os.path.join(CWD, 'image', name_dir)
    confusion_matrix_path = os.path.join(image_path, CONFUSION_MATRIX)
    detection_rate_path = os.path.join(image_path, DETECTION_RATE)

    return log_path, log_file_path, log_train_path, model_path, result_path, confusion_matrix_path, detection_rate_path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

