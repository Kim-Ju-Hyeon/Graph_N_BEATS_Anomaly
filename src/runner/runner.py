import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn

from collections import defaultdict
import torch.optim as optim

from models.IC_PN_BEATS import IC_PN_BEATS
from models.N_model import N_model
from utils.train_helper import model_snapshot, load_model
from utils.logger import get_logger
from torch.autograd import Variable

from dataset.sea_fog_dataset import Temporal_Graph_Signal

from utils.score import get_score
import yaml
from utils.train_helper import edict2dict


class Runner(object):
    def __init__(self, config):
        self.get_dataset(config)
        self.config = config
        self.exp_dir = config.exp_dir
        self.model_save = config.model_save
        self.logger = get_logger(logger_name=str(config.seed))
        self.seed = config.seed
        self.use_gpu = config.use_gpu
        self.device = config.device

        self.best_model_dir = os.path.join(self.model_save, 'best.pth')
        self.ck_dir = os.path.join(self.model_save, 'training.ck')

        self.train_conf = config.train
        self.dataset_conf = config.dataset
        self.nodes_num = config.dataset.nodes_num
        self.target_col_num = [10, 21, 32, 43, 54]
        self.combine_loss = config.train.combine_loss

        if self.train_conf.loss_type == 'classification' or not self.combine_loss:
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=self.normedWeight.to(device=self.device))

        if (self.train_conf.loss_type == 'regression_all') or (self.train_conf.loss_type == 'regression_vis') or self.combine_loss:
            if self.train_conf.loss_function == 'MAE':
                self.regression_loss = nn.L1Loss()
            elif self.train_conf.loss_function == 'MSE':
                self.regression_loss = nn.MSELoss()
            else:
                raise ValueError('Non-supported Loss Function')

        if self.config.model_name == 'IC_PN_BEATS':
            self.model = IC_PN_BEATS(self.config)
        elif self.config.model_name == 'N_model':
            self.model = N_model(self.config)
        else:
            raise ValueError("Non-supported Model")

        if self.use_gpu and (self.device != 'cpu'):
            self.model = self.model.to(device=self.device)

    def get_dataset(self, config):
        loader = Temporal_Graph_Signal(config.dataset.scaler_type)

        config.dataset.nodes_num = loader.nodes_num
        config.dataset.node_features = loader.node_features
        config.dataset.root = loader.path
        config.dataset.freq = loader.freq
        save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
        yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

        loader.preprocess_dataset()
        _train_dataset, self.valid_dataset, self.test_dataset = loader.get_dataset(
            num_timesteps_in=config.forecasting_module.backcast_length,
            num_timesteps_out=config.forecasting_module.forecast_length,
            batch_size=config.train.batch_size)

        self.train_dataset = _train_dataset[0]
        self.normedWeight = _train_dataset[1]

        self.scaler = loader.get_scaler()

    def train(self):
        # create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        results = defaultdict(list)
        best_val_loss = np.inf

        if self.config.train_resume:
            checkpoint = load_model(self.ck_dir)
            self.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_val_loss = checkpoint['best_valid_loss']
            self.train_conf.epoch -= checkpoint['epoch']

        # ========================= Training Loop ============================= #
        for epoch in range(self.train_conf.epoch):
            # ====================== training ============================= #
            self.model.train()

            train_loss = []

            for i, data_batch in enumerate(tqdm(self.train_dataset)):

                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                backcast, model_output, _ = self.model(data_batch.x, interpretability=False)
                forecast = model_output[0].view(self.train_conf.batch_size, self.dataset_conf.nodes_num, -1)
                groud_truth = data_batch.y.view(self.train_conf.batch_size, self.dataset_conf.nodes_num, -1)

                if self.train_conf.loss_type == 'classification' or self.combine_loss:
                    anomaly = model_output[1]
                    anomaly_target = data_batch.anomaly.view(self.train_conf.batch_size, 5)
                    classi_loss = self.classification_loss(anomaly, anomaly_target)

                if self.train_conf.loss_type == 'regression_vis':
                    forecast = forecast[:, self.target_col_num, :]
                    groud_truth = groud_truth[:, self.target_col_num, :]

                regress_loss = self.regression_loss(forecast, groud_truth)

                if self.combine_loss:
                    loss = 0.5 * classi_loss + 0.5 * regress_loss
                else:
                    if self.train_conf.loss_type == 'classification':
                        loss = classi_loss
                    else:
                        loss = regress_loss

                # backward pass (accumulates gradients).
                loss.backward()

                # performs a single update step.
                optimizer.step()
                optimizer.zero_grad()

                train_loss += [float(loss.data.cpu().numpy())]

                # display loss
                if (i + 1) % 500 == 0:
                    self.logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, i + 1,
                                                                         float(loss.data.cpu().numpy())))

            train_loss = np.stack(train_loss).mean()
            results['train_loss'] += [train_loss]

            # ===================== validation ============================ #
            self.model.eval()

            val_loss = []
            for data_batch in tqdm(self.valid_dataset):
                if self.use_gpu and (self.device != 'cpu'):
                    data_batch = data_batch.to(device=self.device)

                with torch.no_grad():
                    _, model_output, _ = self.model(data_batch.x, interpretability=False)

                forecast = model_output[0].view(self.train_conf.batch_size, self.dataset_conf.nodes_num, -1)
                groud_truth = data_batch.y.view(self.train_conf.batch_size, self.dataset_conf.nodes_num, -1)

                if self.train_conf.loss_type == 'classification' or self.combine_loss:
                    anomaly = model_output[1]
                    anomaly_target = data_batch.anomaly.view(self.train_conf.batch_size, 5)
                    classi_loss = self.classification_loss(anomaly, anomaly_target)

                if self.train_conf.loss_type == 'regression_vis':
                    forecast = forecast[:, self.target_col_num, :]
                    groud_truth = groud_truth[:, self.target_col_num, :]

                regress_loss = self.regression_loss(forecast, groud_truth)

                if self.combine_loss:
                    loss = 0.5 * classi_loss + 0.5 * regress_loss
                else:
                    if self.train_conf.loss_type == 'classification':
                        loss = classi_loss
                    else:
                        loss = regress_loss

                val_loss += [float(loss.data.cpu().numpy())]

            val_loss = np.stack(val_loss).mean()
            results['val_loss'] += [val_loss]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_dir)

            self.logger.info("Epoch {} Avg. Validation Loss = {:.6}".format(epoch + 1, val_loss, 0))
            self.logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            model_snapshot(epoch=epoch, model=self.model, optimizer=optimizer, scheduler=None,
                           best_valid_loss=best_val_loss, exp_dir=self.ck_dir)

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'training_result.pickle'), 'wb'))

    def test(self):
        # self.config.train.batch_size = 1

        if self.config.model_name == 'IC_PN_BEATS':
            self.best_model = IC_PN_BEATS(self.config)
        elif self.config.model_name == 'N_model':
            self.best_model = N_model(self.config)
        else:
            raise ValueError("Non-supported Model")

        best_snapshot = load_model(self.best_model_dir)
        self.best_model.load_state_dict(best_snapshot)

        if self.use_gpu and (self.device != 'cpu'):
            self.best_model = self.best_model.to(device=self.device)

        # ===================== validation ============================ #
        self.best_model.eval()

        test_loss = []
        results = defaultdict()
        target = []
        forecast_list = []
        backcast_list = []

        if self.config.model_name == 'IC_PN_BEATS':
            per_trend_backcast = []
            per_trend_forecast = []
            per_seasonality_backcast = []
            per_seasonality_forecast = []
            singual_backcast = []
            singual_forecast = []
        else:
            stack_backcast = []
            stack_forecast = []

        attention_matrix = []
        anomaly_list = []

        for data_batch in tqdm(self.test_dataset):
            if self.use_gpu and (self.device != 'cpu'):
                data_batch = data_batch.to(device=self.device)

            with torch.no_grad():
                _backcast_output, model_output, outputs = self.best_model(data_batch.x, interpretability=True)

            forecast = model_output[0].view(self.train_conf.batch_size, self.dataset_conf.nodes_num, -1)
            groud_truth = data_batch.y.view(self.train_conf.batch_size, self.dataset_conf.nodes_num, -1)

            forecast = forecast[:, self.target_col_num, :]
            groud_truth = groud_truth[:, self.target_col_num, :]

            loss = self.regression_loss(forecast, groud_truth)

            test_loss += [float(loss.data.cpu().detach().numpy())]

            forecast_list += [forecast.cpu().detach().numpy()]
            backcast_list += [_backcast_output.cpu().detach().numpy()]
            if self.train_conf.loss_type == 'classification' or self.combine_loss:
                anomaly_list += [model_output[1].cpu().detach().numpy()]

            if self.config.model_name == 'IC_PN_BEATS':
                per_trend_backcast += [outputs['per_trend_backcast']]
                per_trend_forecast += [outputs['per_trend_forecast']]
                per_seasonality_backcast += [outputs['per_seasonality_backcast']]
                per_seasonality_forecast += [outputs['per_seasonality_forecast']]
                singual_backcast += [outputs['singual_backcast']]
                singual_forecast += [outputs['singual_forecast']]

            else:
                stack_backcast += [outputs['per_stack_backcast']]
                stack_forecast += [outputs['per_stack_forecast']]

            attention_matrix += [outputs['attention_matrix']]

        if self.config.model_name == 'IC_PN_BEATS':
            results['per_trend_backcast'] = np.stack(per_trend_backcast, axis=0)
            results['per_trend_forecast'] = np.stack(per_trend_forecast, axis=0)
            results['per_seasonality_backcast'] = np.stack(per_seasonality_backcast, axis=0)
            results['per_seasonality_forecast'] = np.stack(per_seasonality_forecast, axis=0)
            results['singual_backcast'] = np.stack(singual_backcast, axis=0)
            results['singual_forecast'] = np.stack(singual_forecast, axis=0)

        else:
            results['stack_backcast'] = np.stack(stack_backcast, axis=0)
            results['stack_forecast'] = np.stack(stack_forecast, axis=0)

        results['test_loss'] = np.stack(test_loss).mean()
        results['forecast'] = np.stack(forecast_list, axis=0)
        results['backcast'] = np.stack(backcast_list, axis=0)
        results['attention_matrix'] = np.stack(attention_matrix, axis=0)

        if self.train_conf.loss_type == 'classification' or self.combine_loss:
            results['anomaly_pred'] = np.stack(anomaly_list, axis=0)

        self.logger.info(f"Avg. Test Loss = {results['test_loss']}")

        pickle.dump(results, open(os.path.join(self.config.exp_sub_dir, 'test_result.pickle'), 'wb'))
