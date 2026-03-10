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
import os

class MSBEGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MSBEGCL, self).__init__(conf, training_set, test_set)
        
        # [Device Safe]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- MSBEGCL IS USING DEVICE: {self.device} ---") 
        
        if not self.config.contain('MSBEGCL'):
            print("WARNING: MSBEGCL config section not found! Using hardcoded defaults.")
            args = {'n_layer': 2, 'lambda': 0.2, 'lgcl_lambda': 0.2, 'gamma': 0.1, 'eps': 0.1, 'tau': 0.2, 'biclique.file': ''}
        else:
            args = self.config['MSBEGCL']
        
        #参数获取
        self.cl_rate = float(args.get('lambda', 0.2))      # SimGCL
        self.lgcl_rate = float(args.get('lgcl_lambda', 0.2)) # LightGCL (SVD)
        self.glscl_rate = float(args.get('gamma', 0.1))    # GLSCL (Local Similarity)
        self.eps = float(args.get('eps', 0.1))
        self.n_layers = int(args.get('n_layer', 2))
        self.tau = float(args.get('tau', 0.2))
        
        # 1. LightGCL 核心：预计算 SVD
        print("Pre-computing SVD for LightGCL...")
        adj = self.data.interaction_mat
        # Convert to float for SVD
        adj_float = adj.astype(float)
        u, s, v = torch.svd_lowrank(self._to_torch_sparse(adj_float), q=int(args.get('svd_q', 5)))
        svd_dict = {'u': u, 's': s, 'v': v}
        print("SVD computation complete.")

        self.model = MSBEGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, svd_dict, self.device)
        
        # 2. GLSCL 核心：加载同质相似邻居
        self.biclique_file = args.get('biclique.file', '')
        self.user_sim_neighbors, self.item_sim_neighbors = self._load_neighbors(self.biclique_file)

    def _to_torch_sparse(self, mat):
        coo = mat.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(indices, values, coo.shape).to(self.device)

    def _load_neighbors(self, file_path):
        user_dict, item_dict = {}, {}
        if not os.path.exists(file_path):
            print(f"!!! CRITICAL WARNING: {file_path} NOT FOUND. GLSCL WILL NOT WORK !!!")
            return user_dict, item_dict
        
        print(f"Loading bicliques from {file_path}")
        count = 0
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 2: continue
                # Parse users and items
                users = parts[0].strip().split()
                items = parts[1].strip().split()
                
                try:
                    user_ids = [self.data.get_user_id(u) for u in users if u in self.data.user]
                    item_ids = [self.data.get_item_id(i) for i in items if i in self.data.item]
                    
                    # GLSCL logic: Users in same biclique are similar to each other
                    # Items in same biclique are similar to each other
                    if len(user_ids) > 1:
                        for u in user_ids:
                            if u not in user_dict: user_dict[u] = []
                            # Add other users in biclique
                            user_dict[u].extend([x for x in user_ids if x != u])
                            
                    if len(item_ids) > 1:
                        for i in item_ids:
                            if i not in item_dict: item_dict[i] = []
                            item_dict[i].extend([x for x in item_ids if x != i])
                    
                    count += 1
                except Exception as e:
                    continue

        print(f"--- Loaded {count} bicliques. Users with neighbors: {len(user_dict)}, Items with neighbors: {len(item_dict)} ---")
        return user_dict, item_dict

    def info_nce(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.exp(torch.sum(view1 * view2, dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(view1, view2.t()) / self.tau).sum(dim=1)
        return -torch.log(pos_score / (ttl_score + 1e-8) + 1e-8).mean()

    def glscl_loss(self, nodes, neighbor_dict, embs):
        """ GLSCL 核心思想：对比同质节点（User-User, Item-Item）的局部相似性 """
        valid_nodes = []
        pos_neighbors = []
        
        nodes_list = nodes.cpu().tolist()
        for node in nodes_list:
            if node in neighbor_dict and neighbor_dict[node]:
                valid_nodes.append(node)
                pos_neighbors.append(random.choice(neighbor_dict[node]))
        
        if not valid_nodes: return torch.tensor(0.0).to(self.device)
        
        anchor = F.normalize(embs[torch.tensor(valid_nodes).to(self.device)], dim=1)
        positive = F.normalize(embs[torch.tensor(pos_neighbors).to(self.device)], dim=1)
        
        pos_score = torch.exp(torch.sum(anchor * positive, dim=1) / self.tau)
        
        # Denominator: Contrast with other anchors in the batch? 
        # Or contrast with other nodes?
        # Standard InfoNCE usually contrasts with batch negatives.
        # Here we can contrast anchor with other anchors.
        ttl_score = torch.exp(torch.matmul(anchor, anchor.T) / self.tau).sum(dim=1)
        return -torch.log(pos_score / (ttl_score + 1e-8) + 1e-8).mean()

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        
        for epoch in range(self.maxEpoch):
            epoch_loss = 0
            epoch_rec = 0
            epoch_sim = 0
            epoch_light = 0
            epoch_glscl = 0
            batch_count = 0
            
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                u_idx, p_idx, n_idx = batch
                u_unique = torch.unique(torch.tensor(u_idx).to(self.device))
                i_unique = torch.unique(torch.tensor(p_idx).to(self.device))
                
                # Forward
                rec_u, rec_i, g_u_l, g_i_l, z_u_l, z_i_l = model(return_svd=True)
                
                # 1. BPR Loss
                l_main = bpr_loss(rec_u[u_idx], rec_i[p_idx], rec_i[n_idx])
                
                # 2. SimGCL (Uniformity)
                u_v1, i_v1 = model(perturbed=True)
                u_v2, i_v2 = model(perturbed=True)
                l_sim = self.cl_rate * (self.info_nce(u_v1[u_unique], u_v2[u_unique]) + 
                                       self.info_nce(i_v1[i_unique], i_v2[i_unique]))

                # 3. LightGCL (Global SVD Alignment)
                l_light = torch.tensor(0.0).to(self.device)
                for l in range(self.n_layers):
                    l_light += self.info_nce(z_u_l[l][u_unique], g_u_l[l][u_unique])
                    l_light += self.info_nce(z_i_l[l][i_unique], g_i_l[l][i_unique])
                l_light *= self.lgcl_rate

                # 4. GLSCL (Local Homogeneous Similarity)
                l_glscl = torch.tensor(0.0).to(self.device)
                l_glscl = self.glscl_rate * (self.glscl_loss(u_unique, self.user_sim_neighbors, rec_u) + 
                                           self.glscl_loss(i_unique, self.item_sim_neighbors, rec_i))

                total_loss = l_main + l_sim + l_light + l_glscl + l2_reg_loss(self.reg, rec_u[u_idx], rec_i[p_idx], rec_i[n_idx])
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_rec += l_main.item()
                epoch_sim += l_sim.item()
                epoch_light += l_light.item()
                epoch_glscl += l_glscl.item()
                batch_count += 1
            
            print(f'Epoch {epoch} Loss={epoch_loss/batch_count:.4f} Rec={epoch_rec/batch_count:.4f} Sim={epoch_sim/batch_count:.4f} Light={epoch_light/batch_count:.4f} GLSCL={epoch_glscl/batch_count:.4f}')

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model(perturbed=False)
            
    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class MSBEGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, svd_dict, device):
        super().__init__()
        self.data, self.eps, self.n_layers, self.device = data, eps, n_layers, device
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(device)
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(nn.init.xavier_uniform_(torch.empty(data.user_num, emb_size))),
            'item_emb': nn.Parameter(nn.init.xavier_uniform_(torch.empty(data.item_num, emb_size))),
        })
        # LightGCL: SVD components
        # svd_dict['u']: (N, q), svd_dict['s']: (q,)
        self.u_mul_s = svd_dict['u'] @ torch.diag(svd_dict['s'])
        self.v_mul_s = svd_dict['v'] @ torch.diag(svd_dict['s'])
        self.ut, self.vt = svd_dict['u'].t(), svd_dict['v'].t()

    def forward(self, perturbed=False, return_svd=False):
        ego = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embs = [ego]
        g_u_l, g_i_l, z_u_l, z_i_l = [], [], [], []

        for k in range(self.n_layers):
            if return_svd:
                eu, ei = torch.split(ego, [self.data.user_num, self.data.item_num])
                g_u_l.append(self.u_mul_s @ (self.vt @ ei))
                g_i_l.append(self.v_mul_s @ (self.ut @ eu))
            
            ego = torch.sparse.mm(self.sparse_norm_adj, ego)
            
            if return_svd:
                zu, zi = torch.split(ego, [self.data.user_num, self.data.item_num])
                z_u_l.append(zu); z_i_l.append(zi)

            if perturbed:
                random_noise = torch.rand_like(ego).to(self.device)
                ego += torch.sign(ego) * F.normalize(random_noise, dim=-1) * self.eps
            all_embs.append(ego)

        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.mean(all_embs, dim=1)
        res_u, res_i = torch.split(all_embs, [self.data.user_num, self.data.item_num])
        
        if return_svd: return res_u, res_i, g_u_l, g_i_l, z_u_l, z_i_l
        return res_u, res_i
