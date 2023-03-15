from typing import Union, List, Tuple
from collections import Counter

from tqdm import tqdm
import pandas as pd
import os

from torch_geometric.utils.convert import from_networkx
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from imblearn.over_sampling import SMOTE


class UNSWNB15NodeClassificationDataset(Dataset):
    def __init__(self, root, file_name, num_neighbors=2, binary: bool = False, augmentation: bool = False,
                 val: bool = False, test: bool = False, transform=None, pre_transform=None):

        self.file_name = file_name
        self.num_neighbors = num_neighbors
        self.binary = binary
        self.augmentation = augmentation
        self.df = None
        self.val = val
        self.test = test
        self.labels_encoder = []
        self.values_encoded = []

        super(UNSWNB15NodeClassificationDataset, self).__init__(
            root,
            transform,
            pre_transform
        )

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """ If these files are found in raw_dire, processing is skipped. """
        if self.val:
            file_path = f'nb15_val_{"binary_" if self.binary else ""}{self.num_neighbors}.pt'
        else:
            if self.test:
                file_path = f'nb15_test_{"binary_" if self.binary else ""}{self.num_neighbors}.pt'
            else:
                file_path = f'nb15_{"binary_" if self.binary else ""}{self.num_neighbors}' \
                            f'{"_aug" if self.augmentation else ""}.pt'
        return [file_path]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        """ If this file exist in raw_dir, the download is not triggered. """
        return self.file_name

    def download(self):
        pass

    def process(self):
        # Read csv file
        self.df = pd.read_csv(self.raw_paths[0])

        # Create global graph
        graph = nx.Graph()
        # For each entry of dataframe
        extract_col = [
            'sttl', 'dload', 'dttl', 'sload', 'smeansz', 'sintpkt', 'dmeansz', 'dintpkt', 'tcprtt', 'ackdat',
            'synack', 'ct_state_ttl', 'ct_srv_src', 'ct_dst_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
            'proto_tcp', 'proto_udp', 'proto_other', 'state_fin', 'state_con', 'state_int', 'state_other',
            'service_-', 'service_dns', 'service_other'
        ]
        iter_df = self.df[extract_col]
        y = self.df['label'].values

        # Apply smote if augmentation is set to True
        if self.augmentation and not self.test and not self.val and not self.binary:
            counter = Counter(y)
            n_normal = counter[0]
            n_attack = counter[9]
            sm = SMOTE(sampling_strategy={
                0: n_normal,
                1: n_attack,
                2: n_attack,
                3: n_attack,
                4: n_attack,
                5: n_attack,
                6: n_attack,
                7: n_attack,
                8: n_attack,
                9: n_attack,
            }, random_state=42)
            x_train, y_train = sm.fit_resample(iter_df.values, y)
            iter_df = pd.DataFrame(x_train, columns=iter_df.columns)
            y = y_train

        print(Counter(y))

        for index, flow_entry in tqdm(iter_df.iterrows(), total=iter_df.shape[0], desc=f'Creating nodes...'):
            # Create attr for each label
            node_attr = {}
            for label, value in flow_entry.items():
                node_attr[label] = value
            node_attr['y'] = y[index]
            graph.add_node(index, **node_attr)

        # Create edges
        if self.num_neighbors > 0:
            # Create edges
            features_to_link = ['proto', 'service', 'state']
            groups = self.df.groupby(features_to_link)
            for group in tqdm(groups, total=len(groups), desc=f'Creating edges for features: {features_to_link}'):
                idx_matches = group[1].index
                if len(idx_matches) < 1:
                    continue
                for idx in range(len(idx_matches)):
                    a = idx_matches[idx]
                    for i in range(self.num_neighbors):
                        if idx + 1 + i < len(idx_matches):
                            b = idx_matches[idx + 1 + i]
                            # If edge (a, b) not exist create
                            if not graph.has_edge(a, b):
                                graph.add_edge(a, b)

        # Create pytorch geometric data
        ptg = from_networkx(graph, group_node_attrs=extract_col)

        # Save data object
        if self.val:
            file_path = f'nb15_val_{"binary_" if self.binary else ""}{self.num_neighbors}.pt'
            torch.save(ptg, os.path.join(self.processed_dir, file_path))
        else:
            if self.test:
                file_path = f'nb15_test_{"binary_" if self.binary else ""}{self.num_neighbors}.pt'
                torch.save(ptg, os.path.join(self.processed_dir, file_path))
            else:
                file_path = f'nb15_{"binary_" if self.binary else ""}' \
                            f'{self.num_neighbors}{"_aug" if self.augmentation else ""}.pt'
                torch.save(ptg, os.path.join(self.processed_dir, file_path))

    def len(self) -> int:
        """ Return number of graph """
        return 1

    def get(self, idx: int) -> Data:
        """ Return the idx-th graph. """
        if self.val:
            file_path = f'nb15_val_{"binary_" if self.binary else ""}{self.num_neighbors}.pt'
            data = torch.load(os.path.join(self.processed_dir, file_path))
        else:
            if self.test:
                file_path = f'nb15_test_{"binary_" if self.binary else ""}{self.num_neighbors}.pt'
                data = torch.load(os.path.join(self.processed_dir, file_path))
            else:
                file_path = f'nb15_{"binary_" if self.binary else ""}' \
                            f'{self.num_neighbors}{"_aug" if self.augmentation else ""}.pt'
                data = torch.load(os.path.join(self.processed_dir, file_path))

        return data
