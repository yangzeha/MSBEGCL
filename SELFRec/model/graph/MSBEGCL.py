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
        self.msb_rate = float(args['gamma'])   # Structure Weight (MSBE)
        self.fallback_weight = float(args.get('fallback_weight', 0.0)) # Disable Fallback by default (0.0)
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        self.temp = float(args.get('tau', 0.2)) # Temperature
        
        # Encoder (SimGCL Style)
        self.model = MSBEGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        
        # Load MSBE Neighbors
        self.biclique_file = args['biclique.file']
        self.sim_threshold = float(args.get('sim_threshold', 0.1))
        
        # user_msb_neighbors: {uid: [nid1, nid2, ...]}
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
                
                # 1. Main Task: BPR (No Noise)
                rec_user_emb, rec_item_emb = model(perturbed=False)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                l_main = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                # 2. L_uniform: SimGCL Noise Contrastive Loss
                user_v1, item_v1 = model(perturbed=True)
                user_v2, item_v2 = model(perturbed=True)
                
                u_idx_unique = torch.unique(torch.tensor(user_idx).cuda())
                i_idx_unique = torch.unique(torch.tensor(pos_idx).cuda())
                
                l_uniform = InfoNCE(user_v1[u_idx_unique], user_v2[u_idx_unique], self.temp) + \
                            InfoNCE(item_v1[i_idx_unique], item_v2[i_idx_unique], self.temp)

                # 3. L_sim: Weighted Structure-Confidence-Aware Loss
                # We simply supply the perturbed views (view2) as the fallback target
                # This makes Scenario B = InfoNCE(Clean, Perturbed) -> Reinforce consistency locally
                l_sim_u = self.cal_msbe_loss(u_idx_unique, self.user_msb_neighbors, rec_user_emb, user_v2)
                l_sim_i = self.cal_msbe_loss(i_idx_unique, self.item_msb_neighbors, rec_item_emb, item_v2)
                l_sim = l_sim_u + l_sim_i

                # 4. Total Loss Combination
                batch_loss = l_main + \
                             self.cl_rate * l_uniform + \
                             self.msb_rate * l_sim + \
                             l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                if n % 100 == 0 and n > 0:
                    print(f'training: {epoch + 1} batch {n} rec_loss: {l_main.item():.4f} cl_uniform: {l_uniform.item():.4f} cl_sim: {l_sim.item():.4f}')
            
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model(perturbed=False)
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_msbe_loss(self, independent_indices, neighbor_dict, all_embeddings, perturbed_embeddings=None):
        loss = 0.0
        nodes_cpu = independent_indices.cpu().tolist()
        
        anchors_A = []
        targets_A = []
        
        # Scenario B: if fallback is enabled, we use Perturbed View as target
        # If fallback_weight is 0, we skip B computation
        anchors_B = []
        targets_B = []
        
        for idx in nodes_cpu:
            if idx in neighbor_dict and len(neighbor_dict[idx]) > 0:
                # Scenario A: Mean Pooling of Biclique Neighbors
                neighbors = neighbor_dict[idx]
                
                # Sample 5 neighbors to reduce noise/compute
                if len(neighbors) > 5:
                    sampled = random.sample(neighbors, 5)
                else:
                    sampled = neighbors
                
                anchor_emb = all_embeddings[idx]
                # CRITICAL FIX: Mean pooling + Normalization
                mean_emb = torch.mean(all_embeddings[sampled], dim=0)
                target_emb = F.normalize(mean_emb, dim=0) 
                
                anchors_A.append(anchor_emb)
                targets_A.append(target_emb)
                
            elif self.fallback_weight > 0 and perturbed_embeddings is not None:
                # Scenario B: Fallback to SimGCL logic (Clean vs Perturbed Self)
                anchor_emb = all_embeddings[idx]
                target_emb = perturbed_embeddings[idx]
                
                anchors_B.append(anchor_emb)
                targets_B.append(target_emb)
        
        # Compute Loss A (High Weight: 1.0)
        if len(anchors_A) > 0:
            a_emb = torch.stack(anchors_A)
            t_emb = torch.stack(targets_A)
            loss += InfoNCE(a_emb, t_emb, self.temp)
            
        # Compute Loss B (Low Weight)
        if len(anchors_B) > 0:
            a_emb = torch.stack(anchors_B)
            t_emb = torch.stack(targets_B)
            # This effectively adds more weight to standard CL for these nodes
            loss += self.fallback_weight * InfoNCE(a_emb, t_emb, self.temp)
            
        return loss

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
