import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import random
from util.msbe_helper import load_msbe_neighbors

class MSBEGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MSBEGCL, self).__init__(conf, training_set, test_set)
        args = self.config['MSBEGCL']
        self.cl_rate = float(args['lambda'])  # Uniformity Weight (SimGCL)
        self.msb_rate = float(args['gamma'])   # Similarity Weight (GLSCL/MSBE)
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        self.temp = float(args.get('tau', 0.2))
        
        # Encoder (SimGCL Style)
        self.model = MSBEGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        
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

                # 4. Total Loss
                batch_loss = l_main + \
                             self.cl_rate * l_uniform + \
                             self.msb_rate * l_sim + \
                             l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                if n % 100 == 0 and n > 0:
                     print('training:', epoch + 1, 'batch', n, 'rec_loss:', l_main.item(), 'cl_uniform:', l_uniform.item(), 'cl_sim:', l_sim.item())
            
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
    def __init__(self, data, emb_size, eps, n_layers):
        super(MSBEGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).cuda()
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
