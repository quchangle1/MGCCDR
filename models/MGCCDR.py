import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 


def cal_bpr_loss(pred):
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) 
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph

class MGCCDR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_streamers = conf["num_streamers"]
        self.num_videos = conf["num_videos"]
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]


        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.us_graph, self.uv_graph, self.sv_graph = raw_graph

        self.US_propagation_graph = self.get_propagation_graph(self.us_graph)

        self.UV_propagation_graph = self.get_propagation_graph(self.uv_graph)
        self.UV_aggregation_graph = self.get_aggregation_graph(self.uv_graph)

        self.SV_propagation_graph = self.get_propagation_graph(self.sv_graph)
        self.SV_aggregation_graph = self.get_aggregation_graph(self.sv_graph)

        self.init_noise_eps()


    def init_md_dropouts(self):
        self.US_dropout = nn.Dropout(self.conf["US_ratio"], True)
        self.UV_dropout = nn.Dropout(self.conf["UV_ratio"], True)
        self.SV_dropout = nn.Dropout(self.conf["SV_ratio"], True)
        self.mess_dropout_dict = {
            "US": self.US_dropout,
            "UV": self.UV_dropout,
            "SV": self.SV_dropout
        }


    def init_noise_eps(self):
        self.US_eps = self.conf["US_ratio"]
        self.UV_eps = self.conf["UV_ratio"]
        self.SV_eps = self.conf["SV_ratio"]
        self.eps_dict = {
            "US": self.US_eps,
            "UV": self.UV_eps,
            "SV": self.SV_eps
        }


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.streamers_feature = nn.Parameter(torch.FloatTensor(self.num_streamers, self.embedding_size))
        nn.init.xavier_normal_(self.streamers_feature)
        self.videos_feature = nn.Parameter(torch.FloatTensor(self.num_videos, self.embedding_size))
        nn.init.xavier_normal_(self.videos_feature)


    def init_fusion_weights(self):

        self.lin = nn.Linear(self.embedding_size*2, 1)
        nn.init.xavier_uniform_(self.lin.weight)


    def get_propagation_graph(self, bipartite_graph):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        return to_tensor(laplace_transform(propagation_graph)).to(device)


    def get_aggregation_graph(self, bipartite_graph):
        device = self.device

        streamer_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/streamer_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)


    def propagate(self, graph, A_feature, B_feature, graph_type, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if not test:
                random_noise = torch.rand_like(features).to(self.device)
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) 
        all_features = torch.mean(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        if not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature

    def mga(self, users_feature, streamers_feature):
        emb_dict = {behavior: torch.cat([user_feat, streamer_feat], dim=0) for behavior, user_feat, streamer_feat in zip(['US', 'UV', 'SV'], users_feature, streamers_feature)}

        key = emb_dict['US']  
        key = key.unsqueeze(1).repeat(1, len(users_feature), 1)  
    
        query_emb = torch.stack([emb_dict[behavior] for behavior in ['US', 'UV', 'SV']], dim=1)  

        concat_emb = torch.cat([key, query_emb], dim=2)  

        attention = self.lin(concat_emb).softmax(dim=1) 

        updated_emb = (attention * query_emb).sum(dim=1) 
        all_features = updated_emb
        user_feature, streamer_feature = torch.split(all_features, (users_feature[0].shape[0], streamers_feature[0].shape[0]), 0)
        return user_feature, streamer_feature


    def get_multi_modal_representations(self, test=False):
        US_users_feature, US_streamers_feature = self.propagate(self.US_propagation_graph, self.users_feature, self.streamers_feature, "US", test)

        UV_users_feature, UV_videos_feature = self.propagate(self.UV_propagation_graph, self.users_feature, self.videos_feature, "UV", test)
        UV_streamers_feature = self.aggregate(self.SV_aggregation_graph, UV_videos_feature, "SV", test)

        SV_streamers_feature, SV_videos_feature = self.propagate(self.SV_propagation_graph, self.streamers_feature, self.videos_feature, "SV", test)
        SV_users_feature = self.aggregate(self.UV_aggregation_graph, SV_videos_feature, "UV", test)

        users_feature = [US_users_feature, UV_users_feature, SV_users_feature]
        streamers_feature = [US_streamers_feature, UV_streamers_feature, SV_streamers_feature]

        users_rep, streamers_rep = self.mga(users_feature, streamers_feature)

        return users_rep, streamers_rep


    def cal_c_loss(self, pos, aug):
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) 
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) 

        pos_score = torch.exp(pos_score / self.c_temp) 
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1)

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, streamers_feature):
        pred = torch.sum(users_feature * streamers_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        b_view_cl = self.cal_c_loss(streamers_feature, streamers_feature)

        c_losses = [u_view_cl, b_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch):

        users, streamers = batch
        users_rep, streamers_rep = self.get_multi_modal_representations()

        users_embedding = users_rep[users].expand(-1, streamers.shape[1], -1)
        streamers_embedding = streamers_rep[streamers]

        bpr_loss, c_loss = self.cal_loss(users_embedding, streamers_embedding)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, streamers_feature = propagate_result
        scores = torch.mm(users_feature[users], streamers_feature.t())
        return scores
