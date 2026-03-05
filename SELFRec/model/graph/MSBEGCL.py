import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import random
from util.msbe_helper import load_msbe_neighbors

class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d,d)))

    def forward(self,x):
        return x @ self.W

class MSBEGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MSBEGCL, self).__init__(conf, training_set, test_set)
        args = self.config['MSBEGCL']
        self.cl_rate = float(args['lambda'])  # Uniformity Weight (SimGCL)
        self.msb_rate = float(args['gamma'])   # Similarity Weight (GLSCL/MSBE)
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        self.temp = float(args.get('tau', 0.2))
        
        # New LightGCL params
        self.svd_q = int(args.get('svd_q', 5))
        self.lightgcl_rate = float(args.get('lightgcl_lambda', 0.1))

        # Compute SVD for LightGCL
        svd_dict = None
        if self.lightgcl_rate > 0:
            adj = self.data.interaction_mat
            # Convert to torch sparse coo
            coo = adj.tocoo()
            indices = torch.LongTensor([coo.row, coo.col])
            values = torch.from_numpy(coo.data).float()
            adj_tensor = torch.sparse.FloatTensor(indices, values, coo.shape).cuda()
            adj_tensor = adj_tensor.coalesce()
            
            # SVD
            u, s, v = torch.svd_lowrank(adj_tensor, q=self.svd_q)
            
            svd_dict = {
                'u_mul_s': u @ torch.diag(s),
                'v_mul_s': v @ torch.diag(s),
                'ut': u.T,
                'vt': v.T
            }

        # Encoder (SimGCL Style)
        self.model = MSBEGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, svd_dict)
        
        # Load MSB Neighbors
        self.biclique_file = args['biclique.file']
        self.sim_threshold = float(args.get('sim_threshold', 0.1))
        
        self.user_msb_neighbors, self.item_msb_neighbors = load_msbe_neighbors(
            self.biclique_file,
            self.data.user,
            self.data.item,
            self.data.interaction_mat,
            self.sim_threshold
        )

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                
                # 1. Main Task: BPR (Clean Embeddings)
                # Modified to get intermediate views for LightGCL
                if self.lightgcl_rate > 0:
                    rec_user_emb, rec_item_emb, g_u_list, g_i_list, z_u_list, z_i_list = model(perturbed=False, return_intermediate=True)
                else:
                    rec_user_emb, rec_item_emb = model(perturbed=False)

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                
                l_main = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                # 2. L_uniform: SimGCL Noise Contrastive Loss
                # Generate two independent perturbed views
                user_v1, item_v1 = model(perturbed=True)
                user_v2, item_v2 = model(perturbed=True)
                
                u_idx_unique = torch.unique(torch.tensor(user_idx).cuda())
                i_idx_unique = torch.unique(torch.tensor(pos_idx).cuda())
                
                l_uniform = InfoNCE(user_v1[u_idx_unique], user_v2[u_idx_unique], self.temp) + \
                            InfoNCE(item_v1[i_idx_unique], item_v2[i_idx_unique], self.temp)

                # 3. L_sim: Local Similarity Contrastive Loss
                # Compare Clean Embedding of Node vs Clean Embedding of Neighbor
                sim_user_idx = self.get_msb_samples(user_idx, self.user_msb_neighbors)
                sim_item_idx = self.get_msb_samples(pos_idx, self.item_msb_neighbors)
                
                l_sim = InfoNCE(rec_user_emb[user_idx], rec_user_emb[sim_user_idx], self.temp) + \
                        InfoNCE(rec_item_emb[pos_idx], rec_item_emb[sim_item_idx], self.temp)

                # 4. L_lightgcl: Global Contrastive Loss (SVD vs GNN)
                l_lightgcl = torch.tensor(0.0).cuda()
                if self.lightgcl_rate > 0 and model.Ws is not None:
                    batch_u = torch.tensor(user_idx).cuda()
                    batch_i = torch.unique(torch.tensor(pos_idx + neg_idx).cuda())

                    for l in range(self.n_layers):
                         # Users
                         gnn_u = F.normalize(z_u_list[l][batch_u], p=2, dim=1)
                         hyper_u = F.normalize(g_u_list[l][batch_u], p=2, dim=1)
                         hyper_u = model.Ws[l](hyper_u)
                         
                         pos_score = torch.exp((gnn_u * hyper_u).sum(1) / self.temp)
                         neg_score = torch.exp(gnn_u @ hyper_u.T / self.temp).sum(1)
                         l_lightgcl += -torch.log(pos_score / (neg_score + 1e-8) + 1e-8).mean()
                         
                         # Items
                         gnn_i = F.normalize(z_i_list[l][batch_i], p=2, dim=1)
                         hyper_i = F.normalize(g_i_list[l][batch_i], p=2, dim=1)
                         hyper_i = model.Ws[l](hyper_i)
                         
                         pos_score = torch.exp((gnn_i * hyper_i).sum(1) / self.temp)
                         neg_score = torch.exp(gnn_i @ hyper_i.T / self.temp).sum(1)
                         l_lightgcl += -torch.log(pos_score / (neg_score + 1e-8) + 1e-8).mean()

                # 5. Total Loss
                batch_loss = l_main + \
                             self.cl_rate * l_uniform + \
                             self.msb_rate * l_sim + \
                             self.lightgcl_rate * l_lightgcl + \
                             l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                if n % 100 == 0 and n > 0:
                     print('training:', epoch + 1, 'batch', n, 'rec_loss:', l_main.item(), 'cl_uniform:', l_uniform.item(), 'cl_sim:', l_sim.item(), 'cl_light:', l_lightgcl.item())
            
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model(perturbed=False)
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def get_msb_samples(self, indices, neighbor_dict):
        res = []
        for i in indices:
            if i in neighbor_dict and len(neighbor_dict[i]) > 0:
                res.append(random.choice(neighbor_dict[i]))
            else:
                res.append(i) # Fallback to Self
        return torch.tensor(res).cuda()

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model(perturbed=False)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class MSBEGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, svd_dict=None):
        super(MSBEGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).cuda()
        self.embedding_dict = self._init_model()

        # LightGCL components
        if svd_dict is not None:
            self.u_mul_s = svd_dict['u_mul_s']
            self.v_mul_s = svd_dict['v_mul_s']
            self.ut = svd_dict['ut']
            self.vt = svd_dict['vt']
            self.act = nn.LeakyReLU(0.5)
            self.Ws = nn.ModuleList([W_contrastive(emb_size) for i in range(n_layers)])
        else:
            self.Ws = None

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False, return_intermediate=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        
        g_u_list = []
        g_i_list = []
        z_u_list = []
        z_i_list = []
        
        for k in range(self.n_layers):
            if return_intermediate and self.Ws is not None:
                # SVD Propagation (LightGCL)
                eu, ei = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num])
                
                vt_ei = self.vt @ ei
                gu = self.act(self.u_mul_s @ vt_ei)
                
                ut_eu = self.ut @ eu
                gi = self.act(self.v_mul_s @ ut_eu)
                
                g_u_list.append(gu)
                g_i_list.append(gi)

            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            
            if return_intermediate and self.Ws is not None:
                zu, zi = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num])
                z_u_list.append(self.act(zu))
                z_i_list.append(self.act(zi))
                
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        
        if return_intermediate:
            return user_all_embeddings, item_all_embeddings, g_u_list, g_i_list, z_u_list, z_i_list
            
        return user_all_embeddings, item_all_embeddings
