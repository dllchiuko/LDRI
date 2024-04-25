# -*- coding: utf-8 -*-
from datetime import date

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .loader import *
import sys

sys.path.append('..')


class Generator(Loader):
    def __init__(self, dataset, model_name, tafr_acc, use_dense=True, cold_start=0):
        super(Generator, self).__init__(dataset)

        self.model_name = model_name
        self.use_dense = use_dense
        self.cold_start = cold_start
        self.tafr_acc = tafr_acc
        print('Dataset:', self.dataset)
        print('Model Name:', self.model_name)
        print('Cold_start:', self.cold_start)

        # selected feats of user and item
        self.id_feature_user = ['user_id']
        self.cate_feature_user = ['user_active_degree', 'is_live_streamer', 'is_video_author',
                                  'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range']
        self.sparse_feature_user = self.id_feature_user + self.cate_feature_user
        self.dense_feature_user = []

        self.id_feature_item = ['music_id', 'video_id']
        self.cate_feature_item = ['tag_pop', 'video_type', 'upload_type', 'music_type']
        self.sparse_feature_item = self.id_feature_item + self.cate_feature_item
        self.dense_feature_item = ['watch_ratio', 'play_time_ms', 'comment_stay_time', 'profile_stay_time'] \
            if self.use_dense else []
        self.dense_field_dim_item = [len(self.dense_feature_item)] if self.use_dense else []

        self.id_feature = self.id_feature_user + self.id_feature_item
        self.cate_feature = self.cate_feature_user + self.cate_feature_item
        self.sparse_feature = self.sparse_feature_user + self.sparse_feature_item
        self.dense_feature = self.dense_feature_user + self.dense_feature_item
        self.dense_field_dim = [len(self.dense_feature)] if self.use_dense else []

        self.feature = self.sparse_feature + self.dense_feature
        self.feature_map = dict({'id_feature': self.id_feature,
                                 'cate_feature': self.cate_feature,
                                 'dense_feature': self.dense_feature})
        self.n_day = 30
        # self.n_day = 60 if self.dataset == 'kuairand_1k' else 30

    def wrapper(self, batch_size=512, num_samples=None):
        # load preprocessed dataframe
        global train_df, test_df, train_loader, test_loader
        # log_df = pd.read_csv(self.save_path + self.dataset + '.csv', low_memory=False)
        log_df = pd.read_csv(self.save_path + self.dataset + '.csv')
        log_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        log_df = log_df.fillna(value=0.0)

        # imputer = SimpleImputer(missing_values=np.inf, strategy='most_frequent')
        # log_df[self.dense_feature] = pd.DataFrame(imputer.fit_transform(log_df[self.dense_feature].astype(float)),
        #                                           columns=self.dense_feature)
        scaler = StandardScaler()
        log_df[self.dense_feature] = pd.DataFrame(scaler.fit_transform(log_df[self.dense_feature].astype(float)),
                                                  columns=self.dense_feature)
        log_df = log_df.fillna(value=0.0)
        log_df['date_diff'] = log_df['date_diff'].apply(lambda x: min(x, self.n_day - 1))

        # print(log_df.isna().any())
        # tag_info_dict = TimelinessEmbedding(self.n_day).get_topic_time_embed_mat(log_df)
        confounder_info = []
        for day in range(self.n_day):
            log_df1 = log_df[log_df.date_diff == day]
            confounder_info.append(len(log_df1) / len(log_df))
        confounder_info = torch.from_numpy(np.array(confounder_info))

        # set test label
        train_label = 'test_label'
        test_label = 'test_label'

        # get statistics
        self.get_statistics(log_df)

        # baseline setting
        if self.model_name == 'OURS':
            log_df = log_df
            # surv_df = pd.read_csv(
            #     self.save_path + 'TaFR_predict_' + self.dataset + '_acc' + str(self.tafr_acc) + '.csv')
            # day_range = [str(i) for i in range(self.n_day)]
            # day = list(set(day_range) - set(surv_df.columns))
            # for d in day:
            #     surv_df[d] = str(1.0)
            # surv_df = surv_df[['photo_id'] + day_range]
            # log_df = pd.merge(log_df, surv_df, left_on='video_id', right_on='photo_id', how='left').dropna(how='any')
            # log_df.drop(columns=['photo_id'], inplace=True)
        elif self.model_name == 'TaFR':
            print('read file:', self.save_path + 'TaFR_predict_' + self.dataset + '_acc' + str(self.tafr_acc) + '.csv')
            surv_df = pd.read_csv(self.save_path + 'TaFR_predict_' + self.dataset + '_acc' + str(self.tafr_acc) + '.csv')
            day_range = [str(i) for i in range(self.n_day)]
            day = list(set(day_range) - set(surv_df.columns))
            for d in day:
                surv_df[d] = str(1.0)
            surv_df = surv_df[['photo_id'] + day_range]
            log_df = pd.merge(log_df, surv_df, left_on='video_id', right_on='photo_id', how='left').dropna(how='any')
            log_df.drop(columns=['photo_id'], inplace=True)
            print('died length:', len(list(log_df['video_id'].unique())))

        # split data by interaction date
        log_df['date'] = pd.to_datetime(log_df['date'], format='%Y-%m-%d').dt.date
        if self.dataset == 'kuairand_1k':
            train_df = log_df[(log_df['date'] >= date(2022, 4, 8)) & (log_df['date'] <= date(2022, 4, 28))]
            valid_df = log_df[(log_df['date'] >= date(2022, 4, 29)) & (log_df['date'] <= date(2022, 5, 1))]
            test_df = log_df[(log_df['date'] >= date(2022, 5, 2)) & (log_df['date'] <= date(2022, 5, 8))]
            print('Length of logs:', len(log_df))
            if self.cold_start:
                # test_df = test_df[test_df.date_diff <= 2]
                # cold_video = test_df.groupby(by=['video_id'])['date_diff'].max() <= 2
                # test_df = test_df[test_df.video_id.isin(cold_video[cold_video].index)]

                cold_video = test_df.groupby(by=['video_id'])['date_diff'].agg(['min', 'max'])
                filtered_video = cold_video[(cold_video['max'] <= 2) & (cold_video['min'] == 0)]

                print(filtered_video)

                # 最终的筛选结果
                test_df = test_df[test_df['video_id'].isin(filtered_video.index)]

                print('Length of cold_start logs:', len(test_df))
                print('Number of cold_start videos:', len(test_df['video_id'].unique()))

        elif self.dataset == 'kuairand_pure':
            log_df = log_df.sample(frac=1)
            train_df = log_df[:int(len(log_df) * 0.6)]
            valid_df = log_df[int(len(log_df) * 0.6):int(len(log_df) * 0.7)]
            test_df = log_df[int(len(log_df) * 0.7):]

        if num_samples is not None:
            train_df = train_df.iloc[:int(num_samples * 0.6), :]
            test_df = test_df.iloc[:int(num_samples * 0.4), :]
            shuffle = True
        else:
            shuffle = False

        train_x = np.concatenate([np.array(train_df[self.id_feature]),
                                  np.array(train_df[self.cate_feature]),
                                  np.array(train_df[self.dense_feature])], axis=1)  # [train_size, 5+5+6+2=18]
        test_x = np.concatenate([np.array(test_df[self.id_feature]),
                                 np.array(test_df[self.cate_feature]),
                                 np.array(test_df[self.dense_feature])], axis=1)

        if self.model_name == 'OURS':
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_x),
                              torch.from_numpy(np.array(train_df['date_diff'])),
                              torch.from_numpy(np.array(train_df[train_label])).float().unsqueeze(1)),
                shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_x),
                              torch.from_numpy(np.array(test_df['date_diff'])),
                              torch.from_numpy(np.array(test_df[test_label])).float().unsqueeze(1)),
                shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
            return train_loader, test_loader, train_df, test_df, confounder_info

        elif self.model_name == 'TaFR':
            day_range = [str(i) for i in range(self.n_day)]
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_x),
                              torch.from_numpy(np.array(train_df['date_diff'])),
                              torch.from_numpy(np.array(train_df[train_label])).float().unsqueeze(1)),
                shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_x),
                              torch.from_numpy(np.array(test_df['date_diff'])),
                              torch.from_numpy(np.array(test_df[day_range]).astype(float)),
                              torch.from_numpy(np.array(test_df[test_label])).float().unsqueeze(1)),
                shuffle=shuffle, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)
            return train_loader, test_loader, train_df, test_df, confounder_info

    def get_statistics(self, data_df):
        id_feature_size = [int(max(data_df[i]) + 1) for i in self.id_feature]
        cate_feature_size = [int(max(data_df[i]) + 1) for i in self.cate_feature]

        feature_size = id_feature_size + cate_feature_size + self.dense_field_dim
        feature_size_map = dict({'id_feature': id_feature_size,
                                 'cate_feature': cate_feature_size,
                                 'dense_feature': self.dense_field_dim})
        # setattr(Generator, 'feature_size', feature_size)
        setattr(Generator, 'feature_size_map', feature_size_map)

        feature_size_map_item = dict(
            {'id_feature_item': [int(max(data_df[i]) + 1) for i in self.id_feature_item],
             'cate_feature_item': [int(max(data_df[i]) + 1) for i in self.cate_feature_item],
             'dense_feature_item': self.dense_field_dim_item
             })
        setattr(Generator, 'feature_size_map_item', feature_size_map_item)

    def pad(self, df, length_list):
        global cate_tensor_list
        video_type_list = df[['video_type']].value.tolist()
        video_type = nn.utils.rnn.pad_sequence([torch.Tensor(x) for x in video_type_list]).t()

        upload_type_list = df[['upload_type']].value.tolist()
        upload_type = nn.utils.rnn.pad_sequence([torch.Tensor(x) for x in upload_type_list]).t()

        music_type_list = df[['music_type']].value.tolist()
        music_type = nn.utils.rnn.pad_sequence([torch.Tensor(x) for x in music_type_list]).t()

        tag_list = df[['tag']].value.tolist()
        tag_list = list(map(lambda x: str(x[0]).strip().split(','), tag_list))
        tag_list = list(map(lambda x: int(x) + 1, list_) for list_ in tag_list)
        tag_tensor_list = [torch.Tensor(list(x)) for x in tag_list]
        tag = nn.utils.rnn.pad_sequence(tag_tensor_list).t()

        cate_tensor_list = [video_type, upload_type, music_type, tag]

        cate_tensor_list = [cate_tensor_list[i][:, :length_list[i]] for i in range(len(length_list))]
        cate_feature_sizes = [int(cate_tensor_list[i].max().item())
                              for i in range(len(cate_tensor_list))]
        return np.concatenate(cate_tensor_list, axis=1), \
            cate_feature_sizes

# Generator('kuairand_pure', 'TaFR').wrapper()
