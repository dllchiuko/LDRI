import pandas as pd
import numpy as np
from time import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from .metric import TopK
from .analysis import Summary
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import warnings
import os

warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


class TrainAndTest(object):
    def __init__(self, model, model_name, backbone, device, strategy,
                 optimizer, criterion, seed, cold,
                 train_loader, test_loader, train_df, test_df,
                 confounder_info, params_dict,
                 dataset, num_samples, epochs,
                 is_train=True, is_valid=True, load_epoch=32,
                 pred_mode='do', alpha=0.5, beta=0.4, gamma=0.5):

        self.model = model
        self.model_name = model_name
        self.backbone = backbone
        self.device = device
        self.model.to(self.device)

        self.optimizer = optimizer
        self.criterion = criterion

        self.epochs = epochs
        self.dataset = dataset
        self.num_samples = num_samples

        self.is_train = is_train
        self.is_valid = is_valid
        self.load_epoch = load_epoch

        self.save_path = os.path.join(parent_dir, 'logs')
        self.log_file = open(os.path.join(self.save_path,
                                          f'{self.model_name}_{self.backbone}_{self.dataset}.txt'), 'a')
        self.save_path_pth = os.path.join(self.save_path, 'pth_files')
        if strategy == 'smooth':
            self.save_name_pth = os.path.join(self.save_path_pth,
                                              f'{self.model_name}_{self.backbone}_{self.dataset}_{self.num_samples}_{seed}_smooth')
            if seed == 2024:
                self.save_name_pth = os.path.join(self.save_path_pth,
                                                  f'{self.model_name}_{self.backbone}_{self.dataset}_{self.num_samples}_smooth')
        elif strategy == 'raw':
            self.save_name_pth = os.path.join(self.save_path_pth,
                                              f'{self.model_name}_{self.backbone}_{self.dataset}_{self.num_samples}_{seed}')
            if seed == 2024:
                self.save_name_pth = os.path.join(self.save_path_pth,
                                                  f'{self.model_name}_{self.backbone}_{self.dataset}_{self.num_samples}')
        print(self.save_name_pth)

        print('\n\n',
              '\nTime:', str(datetime.now()),
              '\nSeed:', seed,
              '\nTrain or Test:', 'Train' if self.is_train else 'Test',
              '\nNum of samples:', self.num_samples if self.num_samples is not None else 'full',
              '\nNum of epochs:', self.epochs if self.is_train else None,
              '\nPrediction mode:', 'backbone only, perceptron only, do-calculus, w/o do'
              '\nUpdating strategy:', strategy,
              '\nAlpha:', alpha,
              '\nBeta(OURS):', beta,
              '\nGamma(TaFR)', gamma,
              '\nCold start:', cold,
              # '\nPrediction mode:', 'do-calculus' if pred_mode == 'do' else 'None',
              # '\nParameters:', params_dict,
              '\n',
              file=self.log_file
              )

        if is_train > 0:
            self.train(self.model, alpha, beta, gamma, train_loader, test_loader, train_df, test_df, confounder_info,
                       pred_mode)
        else:
            if self.model_name == 'OURS':
                self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='ym', beta=beta)
                self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='yt', beta=beta)
                self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='do', beta=beta)
                self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='w/o_ba', beta=beta)

            elif self.model_name == 'TaFR':
                self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='!', gamma=gamma)
                # self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='~', gamma=gamma)

    def train(self, model, alpha, beta, gamma, train_loader, test_loader, train_df, test_df, confounder_info,
              pred_mode):
        global train_loss, bce_loss_ym, bce_loss_yt, valid_loss
        optimizer = self.optimizer

        # model.load_state_dict(torch.load(self.save_name_pth + '_epoch64.pth'))
        model.train()
        print(f'Start training the {self.model_name} with {self.backbone} on {self.dataset} dataset...')
        # print(f'Start training the {model.__class__.__name__} with {backbone} on {self.dataset} dataset...')
        start_time = time()
        min_loss = 2.0

        for epoch in tqdm(range(self.epochs)):
            for idx, (x, diff, y) in enumerate(train_loader):
                if self.model_name == 'OURS':
                    ym, yt, yt_mat = model.forward(x.to(self.device), diff.to(self.device))
                    bce_loss_ym = nn.BCELoss()(ym, y.to(self.device))
                    bce_loss_yt = nn.BCELoss()(yt, y.to(self.device))
                    train_loss = alpha * bce_loss_ym + (1 - alpha) * bce_loss_yt
                elif self.model_name == 'TaFR':
                    ym = model.forward(x.to(self.device), diff.to(self.device))
                    bce_loss_ym = nn.BCELoss()(ym, y.to(self.device))
                    bce_loss_yt = 0.
                    train_loss = bce_loss_ym

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            if (epoch + 1) % 2 == 0 or (epoch + 1) == self.epochs:
                if self.model_name == 'OURS':
                    valid_loss = self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='ym',
                                                     beta=beta)
                    valid_loss = self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='yt',
                                                     beta=beta)
                    valid_loss = self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='do',
                                                     beta=beta)
                    valid_loss = self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='w/o_ba',
                                                     beta=beta)
                elif self.model_name == 'TaFR':
                    valid_loss = self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='!',
                                                     beta=beta)
                    valid_loss = self.valid_and_test(model, test_loader, test_df, confounder_info, pred_mode='~',
                                                     beta=beta)

                print(
                    'epoch {:04d}/{:04d} | ym loss {:.4f} | yt loss {:.4f} | valid loss {:.4f} | time {:.4f} \n'
                    .format(epoch + 1, self.epochs, bce_loss_ym, bce_loss_yt, valid_loss, time() - start_time),
                    file=self.log_file
                )
                print(
                    'epoch {:04d}/{:04d} | ym loss {:.4f} | yt loss {:.4f} | valid loss {:.4f} | time {:.4f} \n'
                    .format(epoch + 1, self.epochs, bce_loss_ym, bce_loss_yt, valid_loss, time() - start_time),
                )

            if (epoch + 1) % 2 == 0 or (epoch + 1) == self.epochs:
                torch.save(model.state_dict(), f'{self.save_name_pth}_epoch{epoch + 1}.pth')

        print('Total time cost for train and validation: {:.4f}'.format(time() - start_time), file=self.log_file)

    def valid_and_test(self, model, loader, df, confounder_info, pred_mode='do', beta=0.5, gamma=0.5, entity_pop=None,
                       tag_pop=None):
        global total_loss
        if not self.is_train:
            loaded_file = self.save_name_pth + f'_epoch{self.load_epoch}.pth'
            model.load_state_dict(
                torch.load(loaded_file, map_location=self.device)
            )
            print('Loaded pth file:', str(loaded_file), file=self.log_file)

        model.eval()

        eval_pred = []
        eval_true = []
        # time_mat = []
        with (torch.no_grad()):
            if self.model_name == 'OURS':
                for idx, (x, diff, y) in enumerate(loader):
                    ym, yt, yt_mat = model.forward(x.to(self.device), diff.to(self.device))
                    y_pred = ym
                    if pred_mode == 'ym':
                        y_pred = ym
                    elif pred_mode == 'yt':
                        y_pred = yt
                    elif pred_mode == 'do':
                        # beta = 0.3
                        # out_new = yt_mat.to(self.device) + ym * yt_mat.to(self.device)
                        out_new = beta * ym + (1 - beta) * nn.Sigmoid()(yt_mat).to(self.device)
                        out = torch.matmul(out_new, confounder_info.float().T.to(self.device)).to(
                            self.device)  # (batch_size,)
                        out = out.unsqueeze(-1)

                        # out_new = yt_mat + yt_mat * torch.exp(ym)  # this one!!!
                        # out = torch.matmul(out_new, confounder_info.float().T.to(self.device)).to(
                        #     self.device)  # (batch_size,)
                        # out = nn.Sigmoid()(out).unsqueeze(-1)  # (batch_size, 1)
                        y_pred = out

                    elif pred_mode == 'w/o_ba':
                        out = beta * ym + (1 - beta) * yt
                        y_pred = out

                    y_pred = y_pred.detach().cpu()
                    loss = nn.BCELoss()(y_pred, y)
                    eval_pred.append(y_pred)
                    eval_true.append(y)
                    # time_mat.append(yt_mat.detach().cpu().view(-1, yt_mat.shape[1]))

            elif self.model_name == 'TaFR':
                for idx, (x, diff, surv, y) in enumerate(loader):
                    ym = model.forward(x.to(self.device), diff.to(self.device))
                    if pred_mode == '!':  # 数据集不全
                        y_pred = ym
                    elif pred_mode == '~':
                        yt = surv[torch.arange(diff.shape[0]), diff].float().view(-1, 1)
                        y_pred = (gamma * ym.detach().cpu() + (1 - gamma) * yt.detach().cpu()).view(-1, 1)

                    y_pred = y_pred.detach().cpu()
                    loss = nn.BCELoss()(y_pred, y)
                    eval_pred.append(y_pred)
                    eval_true.append(y)

            eval_pred = torch.cat(eval_pred).squeeze()
            # time_mat = nn.Sigmoid()(torch.cat(time_mat, dim=0))

            if (self.is_train > 0 and self.is_valid > 0) or self.is_train == 0:
                df['pred'] = eval_pred.tolist()
                dataset_file = os.path.join(parent_dir, 'dataset')
                # df.to_csv(os.path.join(parent_dir, 'dataset') + f'/{self.dataset}_{self.backbone}_epoch{self.load_epoch}.csv')
                # print(len(df))
                # # df = df.fillna(0.)
                # print(os.path.join(dataset_file,
                #                    f'/{self.dataset}_{self.backbone}_epoch{self.load_epoch}.csv'))
                # print('Successfully saved csv.')

                pred = df['pred'].to_numpy()
                pred = (pred >= 0.5).astype(int)
                true = df['test_label'].to_numpy()
                auc = round(roc_auc_score(true, pred), 4)
                f1 = round(f1_score(true, pred), 4)
                print('AUC for all:', auc)
                print('F1-Score for all:', f1)
                print('AUC for all:', auc, file=self.log_file)
                print('F1-Score for all:', f1, file=self.log_file)

                if self.dataset == 'kuairand_pure':
                    TopK(1, df, self.log_file).evaluate()
                    TopK(3, df, self.log_file).evaluate()
                    TopK(5, df, self.log_file).evaluate()
                    TopK(10, df, self.log_file).evaluate()
                    TopK(20, df, self.log_file).evaluate()

                elif self.dataset == 'kuairand_1k':
                    # TopK(100, df, self.log_file).evaluate()
                    TopK(100, df, self.log_file).evaluate()
                    TopK(300, df, self.log_file).evaluate()
                    TopK(500, df, self.log_file).evaluate()
                    TopK(800, df, self.log_file).evaluate()
                    TopK(1000, df, self.log_file).evaluate()

            if self.is_train == 0:
                summary = Summary(df, self.log_file, self.dataset)
                # summary.tag_curv()
                # summary.get_auc_diff()
                # summary.get_play_rate()
                summary.get_metric_diff('MAP')
                summary.get_metric_diff('RECALL')
                summary.get_metric_diff('NDCG')
                summary.get_metric_diff('HR')

            else:
                summary = Summary(df, self.log_file, self.dataset)
                # summary.tag_curv()
                # summary.get_auc_diff()

        return loss
