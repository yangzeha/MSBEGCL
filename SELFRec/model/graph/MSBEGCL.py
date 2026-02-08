import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import random
import numpy as np
from util.msbe_helper import load_msbe_neighbors

class MSBEGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MSBEGCL, self).__init__(conf, training_set, test_set)
        args = self.config['MSBEGCL']
        self.cl_rate = float(args['lambda'])  # SimGCL Uniformity Weight
        self.msb_rate = float(args['gamma'])   # Similarity Weight (GLSCL/MSBE)
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        self.tau = float(args.get('tau', 0.2))
        
        # New: Fallback Weight & Threshold
        self.fallback_weight = float(args.get('fallback_weight', 0.2))
        self.sim_threshold = float(args.get('sim_threshold', 0.1))
        
        # Encoder
        self.model = MSBEGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)
        
        # Load MSB Neighbors
        self.biclique_file = args['biclique.file']
        self.user_msb, self.item_msb = load_msbe_neighbors(
            self.biclique_file,
            self.data.user,
            self.data.item,
            self.data.interaction_mat,
            self.sim_threshold
        )

    def cal_msbe_loss(self, nodes, neighbors_dict, embeddings, is_user=True):
        """
        Idea 2: Mean of all members + Weighted Fallback
        """
        # 1. Identify which nodes have bicliques
        has_msbe = []
        no_msbe = []
        nodes_np = nodes.cpu().numpy()
        
        for idx, node in enumerate(nodes_np):
            if node in neighbors_dict and len(neighbors_dict[node]) > 0:
                has_msbe.append(idx)
            else:
                no_msbe.append(idx)
        
        loss = 0.0
        # --- Case A: Node has Biclique (Weight = 1.0) ---
        if len(has_msbe) > 0:
            target_indices = torch.tensor(has_msbe).cuda()
            target_nodes = nodes[target_indices]
            pos_embs = []
            
            for node in target_nodes.cpu().numpy():
                # Mean of all members
                m_list = neighbors_dict[node]
                group_emb = torch.mean(embeddings[m_list], dim=0) 
                pos_embs.append(group_emb)
            
            pos_embs = torch.stack(pos_embs)
            loss += InfoNCE(embeddings[target_nodes], pos_embs, 0.2)

        # --- Case B: No Biclique, Fallback to GLSCL (Weight = fallback_weight) ---
        if len(no_msbe) > 0:
            target_indices = torch.tensor(no_msbe).cuda()
            target_nodes = nodes[target_indices]
            
            # Get Top-1 Neighbors (GLSCL logic)
            top1_embs = self.get_top1_neighbors(target_nodes, embeddings, is_user)
            loss += self.fallback_weight * InfoNCE(embeddings[target_nodes], top1_embs, 0.2)
            
        return loss

    def get_top1_neighbors(self, nodes, all_embeddings, is_user):
        # Simplified: Random fallback as per instructions.
        # TODO: Replace with pre-calculated top-1 index lookup.
        rand_indices = torch.randint(0, len(all_embeddings), (len(nodes),)).cuda()
        return all_embeddings[rand_indices]

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                
                # 1. SimGCL Global View (Perturbed)
                u_view1, i_view1 = model(perturbed=True)
                u_view2, i_view2 = model(perturbed=True)
                
                # 2. Basic Rec View (Clean)
                rec_user_emb, rec_item_emb = model()
                # Use only necessary embeddings
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx] 
                neg_item_emb = rec_item_emb[neg_idx]
                
                # --- Loss Calculation ---
                # (1) Recommendation BPR Loss
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                
                # (2) SimGCL Global Contrastive Loss (Uniformity)
                u_idx_unique = torch.unique(torch.tensor(user_idx).cuda())
                i_idx_unique = torch.unique(torch.tensor(pos_idx).cuda())
                
                cl_loss = self.cl_rate * (InfoNCE(u_view1[u_idx_unique], u_view2[u_idx_unique], self.tau) + 
                                         InfoNCE(i_view1[i_idx_unique], i_view2[i_idx_unique], self.tau))
                
                # (3) MSBE Structural Contrastive Loss (Idea 2)
                msbe_loss = self.msb_rate * (
                    self.cal_msbe_loss(u_idx_unique, self.user_msb, rec_user_emb, True) +
                    self.cal_msbe_loss(i_idx_unique, self.item_msb, rec_item_emb, False)
                )
                
                total_loss = batch_loss + cl_loss + msbe_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if n % 100 == 0 and n > 0:
                     print('training:', epoch + 1, 'batch', n, 'rec_loss:', batch_loss.item(), 'cl_uniform:', cl_loss.item(), 'msbe_struct:', msbe_loss.item())
            
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model(perturbed=False)
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

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
