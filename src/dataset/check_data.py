from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

# 단순히 자료 보고싶을 때
path = '/home/parksh/share/project/fog-dnn/data/new_data/last_2h/'
file = joblib.load(path + 'SF_0002_last_2h.pkl') 

# 관측자료만 불러오기
data = file['x'].copy()
# 현재 시점(lag00)만 보기
data_lag00 = data.iloc[:, :15]


class DNNDataset(Dataset):
    def __init__(
        self, X: np.ndarray, y: pd.Series, transform: Optional[Callable] = None
    ):
        self.X = X.astype(np.float32)
        self.y = y.values.astype(np.int64)
        self.len = self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.len


def prepare_data(self):
        # data = joblib.load(self.hparams.dataset_path, mmap_mode='r')
        data = joblib.load(
            self.hparams.dataset_path
        )
        X = data["x"]
        scaler = StandardScaler()
        X.loc[:, :] = scaler.fit_transform(X)

        y = data["y"]
        label_name = f'y_{self.hparams.pred_hour}'
        X = X.join(y.filter(like="time"))  # time related feature
        y = y[label_name]  # y_1 | y_3 | y_6
        nan_mask = np.isnan(X).any(axis=1) | y.isna().values

        # dropna
        X = X[~nan_mask].values
        y = y[~nan_mask]
        self.input_dim = X.shape[-1]


        # train / test split
        test_mask = y.index > "2020-07-01"
        test_split = np.flatnonzero(test_mask)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        groups = y.index.year * 366 + y.index.dayofyear
        groups = groups[~test_mask]

        # test set을 제외한 Train(train+val) 세트를 년도로 구분하면 연도에 따라 해무발생 빈도가 많이 다르니 validation set이 train set을 대표하지 못함
        # 그러므로 최소 unit을 하루 단위로 하여 train/valid로 shuffle split함. group shuffle하면 하루가 train/valid로 나누어지지 않음
        train_split, val_split = list(
            gss.split(y[~test_mask], y[~test_mask], groups=groups)
        )[0]
        splits = [train_split, val_split, test_split]

        self.train_ds = DNNDataset(X[splits[0]], y.iloc[splits[0]])
        self.val_ds = DNNDataset(X[splits[1]], y.iloc[splits[1]])
        self.test_ds = DNNDataset(X[splits[2]], y.iloc[splits[2]])
        pos_label_weight = 1 / y.mean()

        # calc weight
        self.weight = (
            np.where(y == 1, pos_label_weight, 1) / 2
        )  # pos_label_weight을 class 비율의 역수로할 때 2로 나누어주면 sample_weight 평균이 1이 되어 나눔
        self.weight = self.weight.astype(np.float32)[splits[0]]
        self.test_index = y.index[splits[-1]]

        self.num_batch = (
            splits[0].size + splits[1].size
        ) // self.hparams.batch_size + 1  # 무시


def train_dataloader(self):
    return DataLoader(
        self.train_ds,
        # sampler=WeightedRandomSampler(self.weight, self.hparams.batch_size, replacement=True),
        sampler=WeightedRandomSampler(
            self.weight, len(self.weight), replacement=True
        ),  # sample weight에 따라 해무를 데이터 비율보다 더 많이 샘플링함.
        batch_size=self.hparams.batch_size,
        drop_last=True,
        # num_workers=1
    )


# val dataloader
def val_dataloader(self):
    return DataLoader(
        self.val_ds,
        batch_size=self.hparams.batch_size,
        shuffle=False,
        # num_workers=1
    )


# test dataloader
def test_dataloader(self):
    return DataLoader(
        self.test_ds,
        batch_size=self.hparams.batch_size,
        shuffle=False,
        # num_workers=1
    )