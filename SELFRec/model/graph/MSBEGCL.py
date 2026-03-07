import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
import random
import numpy as np
import scipy.sparse as sp
from util.msbe_helper import load_msbe_neighbors

class MSBEGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MSBEGCL, self).__init__(conf, training_set, test_set)
        args = self.config['MSBEGCL']
        
        # --- 超参数配置 ---
        self.cl_rate = float(args.get('lambda', 0.15))       # SimGCL (Uniformity)
        self.glscl_rate = float(args.get('gamma', 0.1))      # GLSCL (Local Similarity)
        self.lgcl_rate = float(args.get('lgcl_lambda', 0.2)) # LightGCL (Global Denoising)
        self.eps = float(args.get('eps', 0.1))
        self.n_layers = int(args.get('n_layer', 2))
        self.tau = float(args.get('tau', 0.2))
        self.svd_q = int(args.get('svd_q', 5))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. LightGCL 核心：计算低秩 SVD (全局建模) ---
        adj = self.data.interaction_mat
        print(f"--- Computing SVD for LightGCL (q={self.svd_q}) ---")
        u, s, v = self._compute_svd(adj, q=self.svd_q)
        svd_dict = {
            'u': u.to(self.device), 
            's': s.to(self.device), 
            'v': v.to(self.device)
        }

        # 初始化 Encoder
        self.model = MSBEGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, svd_dict, self.device)
        
        # --- 2. GLSCL 核心：加载同质相似邻居 (Biclique 挖掘) ---
        self.user_msb, self.item_msb = load_msbe_neighbors(
            args['biclique.file'], 
            self.data.user, 
            self.data.item, 
            self.data.interaction_mat, 
            float(args.get('sim_threshold', 0.2))
        )

    def _compute_svd(self, adj, q=5):
        coo = adj.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data).float()
        t = torch.sparse.FloatTensor(indices, values, coo.shape).to(self.device)
        # 使用低秩 SVD 提取全局结构
        u, s, v = torch.svd_lowrank(t, q=q)
        return u, s, v

    def info_nce(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.exp(torch.sum(view1 * view2, dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(view1, view2.t()) / self.tau).sum(dim=1)
        return -torch.log(pos_score / (ttl_score + 1e-8) + 1e-8).mean()

    # --- GLSCL 思想：局部同质对比损失 ---
    def local_sim_loss(self, nodes, neighbor_dict, embeddings):
        """
        GLSCL: 将具有相同局部结构（Biclique）的同质节点拉近。
        """
        if len(nodes) == 0: return 0.0
        node_list = nodes.cpu().tolist()
        pos_neighbors = []
        valid_idx = []
        for i, node in enumerate(node_list):
            if node in neighbor_dict and neighbor_dict[node]:
                # 随机选一个挖掘出的相似邻居作为正样本
                pos_neighbors.append(random.choice(neighbor_dict[node]))
                valid_idx.append(i)
        if not valid_idx: return 0.0
        
        anchor_embs = F.normalize(embeddings[nodes[valid_idx]], dim=1)
        neighbor_embs = F.normalize(embeddings[torch.tensor(pos_neighbors).to(self.device)], dim=1)
        
        pos_score = torch.exp(torch.sum(anchor_embs * neighbor_embs, dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(anchor_embs, anchor_embs.T) / self.tau).sum(dim=1)
        return -torch.log(pos_score / (ttl_score + 1e-8) + 1e-8).mean()

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                u_unique = torch.unique(torch.tensor(user_idx).to(self.device))
                i_unique = torch.unique(torch.tensor(pos_idx).to(self.device))
                
                # 1. Forward + 获取 SVD 增强视图
                rec_u, rec_i, g_u_l, g_i_l, z_u_l, z_i_l = model(return_svd=True)
                
                # 2. 主损失：BPR Loss
                l_main = bpr_loss(rec_u[user_idx], rec_i[pos_idx], rec_i[neg_idx])
                
                # 3. SimGCL Loss：均匀性对比
                u_v1, i_v1 = model(perturbed=True)
                u_v2, i_v2 = model(perturbed=True)
                l_unif = self.cl_rate * (self.info_nce(u_v1[u_unique], u_v2[u_unique]) + 
                                       self.info_nce(i_v1[i_unique], i_v2[i_unique]))

                # 4. LightGCL Loss：全局降噪对比 (对齐 GNN 和 SVD 视图)
                l_lgcl = 0
                for l in range(self.n_layers):
                    l_lgcl += self.info_nce(z_u_l[l][u_unique], g_u_l[l][u_unique])
                    l_lgcl += self.info_nce(z_i_l[l][i_unique], g_i_l[l][i_unique])
                l_lgcl *= self.lgcl_rate

                # 5. GLSCL Loss：局部结构相似性对比
                l_local = self.glscl_rate * (self.local_sim_loss(u_unique, self.user_msb, rec_u) + 
                                           self.local_sim_loss(i_unique, self.item_msb, rec_i))

                total_loss = l_main + l_unif + l_lgcl + l_local + l2_reg_loss(self.reg, rec_u[user_idx], rec_i[pos_idx], rec_i[neg_idx])
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)

    def predict(self, u):
        u = self.data.get_user_id(u)
        return torch.matmul(self.user_emb[u], self.item_emb.T).cpu().numpy()

class MSBEGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, svd_dict, device):
        super().__init__()
        self.data, self.eps, self.n_layers, self.device = data, eps, n_layers, device
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(device)
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(nn.init.xavier_uniform_(torch.empty(data.user_num, emb_size))),
            'item_emb': nn.Parameter(nn.init.xavier_uniform_(torch.empty(data.item_num, emb_size))),
        })
        # LightGCL SVD 预计算组件
        self.u_mul_s = svd_dict['u'] @ torch.diag(svd_dict['s'])
        self.v_mul_s = svd_dict['v'] @ torch.diag(svd_dict['s'])
        self.ut, self.vt = svd_dict['u'].T, svd_dict['v'].T

    def forward(self, perturbed=False, return_svd=False):
        ego = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embs = [ego]
        g_u_l, g_i_l, z_u_l, z_i_l = [], [], [], []

        for k in range(self.n_layers):
            if return_svd:
                # 提取 LightGCL 奇异值过滤视图
                eu, ei = torch.split(ego, [self.data.user_num, self.data.item_num])
                g_u_l.append(self.u_mul_s @ (self.vt @ ei))
                g_i_l.append(self.v_mul_s @ (self.ut @ eu))
            
            ego = torch.sparse.mm(self.sparse_norm_adj, ego)
            
            if return_svd:
                # 提取原始 GNN 视图用于对比
                zu, zi = torch.split(ego, [self.data.user_num, self.data.item_num])
                z_u_l.append(zu); z_i_l.append(zi)

            if perturbed:
                # SimGCL 随机噪声扰动
                noise = torch.rand_like(ego).to(self.device)
                ego += torch.sign(ego) * F.normalize(noise, dim=-1) * self.eps
            all_embs.append(ego)

        avg_emb = torch.mean(torch.stack(all_embs, dim=1), dim=1)
        res_u, res_i = torch.split(avg_emb, [self.data.user_num, self.data.item_num])
        if return_svd: return res_u, res_i, g_u_l, g_i_l, z_u_l, z_i_l
        return res_u, res_i