# -*- coding:utf-8 -*-
# @Time: 2020/3/18 21:39
# @Author: jockwang, jockmail@126.com
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset
import logging

class TopK():
    def __init__(self, train_loader, test_loader, n_item, batch_size, graph, device):
        self.user_list, self.train_record, self.test_record, self.item_set, self.k_list = \
            self.topk_settings(train_loader, test_loader)
        self.item_enc = OneHotEncoder().fit([[i] for i in range(n_item)])
        self.batch_size = batch_size
        self.device = device
        self.graph = graph

    def get_user_record(self, data_loader, is_train):
        user_history_dict = dict()
        for users, _, labels, items in data_loader:
            interactions = np.array([users.tolist(), items.tolist(), labels.tolist()])
            interactions = interactions.transpose()
            for interaction in interactions:
                user = interaction[0]
                item = interaction[1]
                label = interaction[2]
                if is_train or label[0] == 1:
                    if user not in user_history_dict:
                        user_history_dict[user] = set()
                    user_history_dict[user].add(item)
        return user_history_dict

    def topk_settings(self, train_loader, test_loader):
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]

        train_record = self.get_user_record(train_loader, True)
        test_record = self.get_user_record(test_loader, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = []
        for i in test_record.items():
            item_set.extend(i[1])
        return user_list, train_record, test_record, set(item_set), k_list

    def eval(self, model):
        precision_list = {k: [] for k in self.k_list}
        recall_list = {k: [] for k in self.k_list}
        hr_list = {k: [] for k in self.k_list}

        for user in self.user_list:
            test_item_list = list(self.item_set-self.train_record[user])
            item_scroe_map = dict()
            dataloader = torch.utils.data.DataLoader(dataset=miniDataset(user, test_item_list, self.item_enc),
                                                     batch_size=self.batch_size, shuffle=False)
            for _, [u, i, item] in enumerate(dataloader):
                u, i = u.to(self.device), i.to(self.device)
                outs = model(u, i, self.graph).tolist()
                for ie, score in zip(item.tolist(), outs):
                    item_scroe_map[ie] = score

            sorted_items = sorted(item_scroe_map.items(), key = lambda x: x[1], reverse=True)
            items_sorted = [s[0] for s in sorted_items]

            for k in self.k_list:
                hit_num = len(set(items_sorted[:k]) & self.test_record[user])
                precision_list[k].append(hit_num / k)
                recall_list[k].append(hit_num / len(self.test_record[user]))
                hr_list[k].append(hit_num)
        precision = [np.mean(precision_list[k]) for k in self.k_list]
        recall = [np.mean(recall_list[k]) for k in self.k_list]
        hr = [np.mean(hr_list[k]) for k in self.k_list]

        for i in range(len(self.k_list)):
            k = self.k_list[i]
            logging.info('Pre@%d:%.6f, Recall@%d:%.6f, HR@%d:%.6f'%(k, precision[i], k, recall[i], k, hr[i]))

class miniDataset(Dataset):
    def __init__(self, user, test_item_list, item_enc):
        super(miniDataset, self).__init__()

        self.item_enc = item_enc
        self.user = user
        self.test_item_list = test_item_list

    def __getitem__(self, index):
        item = self.item_enc.transform([[self.test_item_list[index]]]).toarray()
        return torch.tensor(self.user, dtype=torch.long), torch.tensor(item, dtype=torch.float), torch.tensor(
            self.test_item_list[index], dtype=torch.long)

    def __len__(self):
        return len(self.test_item_list)




