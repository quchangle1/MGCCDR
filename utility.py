import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class StreamerTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_streamers, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_streamers = num_streamers
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_streamer = self.u_b_pairs[index]
        all_streamers = [pos_streamer]

        while True:
            i = np.random.randint(self.num_streamers)
            if self.u_b_graph[user_b, i] == 0 and not i in all_streamers:                                                          
                all_streamers.append(i)                                                                                                   
                if len(all_streamers) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_streamers)


    def __len__(self):
        return len(self.u_b_pairs)


class StreamerTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_streamers):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_streamers = num_streamers

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.streamers = torch.arange(num_streamers, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_streamers, self.num_videos = self.get_data_size()

        b_i_pairs, b_i_graph = self.get_sv()
        u_i_pairs, u_i_graph = self.get_uv()

        u_b_pairs_train, u_b_graph_train = self.get_us("train")
        u_b_pairs_val, u_b_graph_val = self.get_us("tune")
        u_b_pairs_test, u_b_graph_test = self.get_us("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.streamer_train_data = StreamerTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_streamers, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.streamer_val_data = StreamerTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_streamers)
        self.streamer_test_data = StreamerTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_streamers)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.streamer_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10, drop_last=True)
        self.val_loader = DataLoader(self.streamer_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(self.streamer_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)


    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_sv(self):
        with open(os.path.join(self.path, self.name, 'streamer_video.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_streamers, self.num_videos)).tocsr()

        print_statistics(b_i_graph, 'S-V statistics')

        return b_i_pairs, b_i_graph


    def get_uv(self):
        with open(os.path.join(self.path, self.name, 'user_video.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_videos)).tocsr()

        print_statistics(u_i_graph, 'U-V statistics')

        return u_i_pairs, u_i_graph


    def get_us(self, task):
        with open(os.path.join(self.path, self.name, 'user_streamer_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_streamers)).tocsr()

        print_statistics(u_b_graph, "U-S statistics in %s" %(task))

        return u_b_pairs, u_b_graph
