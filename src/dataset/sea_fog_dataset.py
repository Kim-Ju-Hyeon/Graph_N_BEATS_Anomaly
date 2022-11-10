import numpy as np
import random
import torch
from glob import glob

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils.scalers import Scaler
import os
import pandas as pd

from six.moves import urllib
import zipfile
import gdown
import pickle

from utils.dataset_utils import time_features_from_frequency_str


class Temporal_Graph_Signal(object):
    def __init__(self, scaler_type='std'):
        super(Temporal_Graph_Signal, self).__init__()
        self.num_workers = 4 * torch.cuda.device_count()
        self._set_dataset_parameters()
        self.scaler = Scaler(scaler_type)

    def preprocess_dataset(self):
        df = pd.read_csv(os.path.join(self.path, 'sea_fog_dataset.csv'), index_col=0)
        col_names = df.columns
        index_val = df.index

        scaled_df = self.scaler.scale(df.to_numpy().T)
        self.dataframe = pd.DataFrame(scaled_df.T, columns=list(col_names), index=index_val)

        if not os.path.isfile(os.path.join(self.path, f'scaler.pickle')):
            pickle.dump(self.scaler, open(os.path.join(self.path, f'scaler.pickle'), 'wb'))

    def _get_timestamp(self, y_df):
        dataframe = pd.DataFrame()
        time_cls = time_features_from_frequency_str(self.freq)
        for cls_ in time_cls:
            cls_name = cls_.__class__.__name__
            dataframe[cls_name] = cls_(y_df['date'].dt)
        time_stamp = dataframe.to_numpy().T

        return time_stamp

    def _set_dataset_parameters(self):
        self.nodes_num = 55
        self.path = './data/SeaFog/graph_signal'
        self.node_features = 1
        self.freq = '10min'
        self.url = None

    def _generate_dataset(self, indices, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        features, target, anomaly = [], [], []

        for i, j in indices:
            features.append(self.dataframe.iloc[i: i + num_timesteps_in].T)
            target.append(self.dataframe.iloc[i + num_timesteps_in: j].T)

        features = torch.FloatTensor(np.array(features))
        targets = torch.FloatTensor(np.array(target))

        _data = []
        for batch in range(len(indices)):
            _data.append(Data(x=features[batch], y=targets[batch], time_stamp=None))

        return _data

    def get_dataset(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, batch_size: int = 32,
                    return_loader=True):
        if not os.path.isfile(os.path.join(self.path, f'indices.pickle')):
            self.dataframe.index = pd.to_datetime(self.dataframe.index)
            self.indices = [
                (i, i + (num_timesteps_in + num_timesteps_out))
                for i in range(self.dataframe.shape[0] - (num_timesteps_in + num_timesteps_out))
                if (self.dataframe.index[i + (num_timesteps_in + num_timesteps_out)] - self.dataframe.index[
                    i]).seconds / 600 == num_timesteps_in + num_timesteps_out
            ]

            random.shuffle(self.indices)
            pickle.dump(self.indices, open(os.path.join(self.path, f'indices.pickle'), 'wb'))

        else:
            self.indices = pickle.load(open(os.path.join(self.path, f'indices.pickle'), 'rb'))

        total_length_dataset = len(self.indices)
        train_idx = int(total_length_dataset * 0.7)
        valid_idx = int(total_length_dataset * 0.2)
        train_indices = self.indices[:train_idx]
        validation_indices = self.indices[train_idx:train_idx+valid_idx]
        test_indices = self.indices[train_idx+valid_idx:]

        train_dataset = self._generate_dataset(train_indices, num_timesteps_in, num_timesteps_out)
        valid_dataset = self._generate_dataset(validation_indices, num_timesteps_in, num_timesteps_out)
        test_dataset = self._generate_dataset(test_indices, num_timesteps_in, num_timesteps_out)

        if return_loader:
            train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                               num_workers=self.num_workers, pin_memory=True)
            valid = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                               num_workers=self.num_workers, pin_memory=True)
            test = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=self.num_workers, pin_memory=True)

            return train, valid, test

        else:
            return train_dataset, valid_dataset, test_dataset

    def get_scaler(self):
        return self.scaler
