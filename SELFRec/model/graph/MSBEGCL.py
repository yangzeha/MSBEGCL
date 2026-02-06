import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.msbe_helper import load_msbe_adj

class MSBEGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MSBEGCL, self).__init__(conf, training_set, test_set)
        args = self.config['MSBEGCL']
        self.cl_rate = float(args['lambda'])
        self.msbe_rate = float(args['gamma'])
        self.eps = float(args['eps'])
        self.n_layers = int(args['n_layer'])
        self.biclique_file = args['biclique.file']
        
        self.msbe_adj = load_msbe_adj(
            self.biclique_file, 
            self.data.user, 
            self.data.item, 
            self.data.user_num, 
            self.data.item_num
        )
        
        self.model = MSBEGCL_Encoder(
            self.data, 
            self.emb_size, 
            self.eps, 
            self.n_layers, 
            norm_adj=self.data.norm_adj
        )
        self.model.set_msbe_adj(self.msbe_adj)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                
                rec_user_emb, rec_item_emb = model(view='global')
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                
                cl_loss_noise = self.cl_rate * self.cal_cl_loss_noise([user_idx, pos_idx])
                cl_loss_msbe = self.msbe_rate * self.cal_cl_loss_msbe([user_idx, pos_idx])
                
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss_noise + cl_loss_msbe
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_noise:', cl_loss_noise.item(), 'cl_msbe:', cl_loss_msbe.item())
            
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model(view='global')
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss_noise(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(view='global', perturbed=True)
        user_view_2, item_view_2 = self.model(view='global', perturbed=True)
        return InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2) + InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

    def cal_cl_loss_msbe(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_global, item_global = self.model(view='global', perturbed=False)
        user_local, item_local = self.model(view='local', perturbed=False)
        # Using 0.2 as temperature, same as SimGCL
        return InfoNCE(user_global[u_idx], user_local[u_idx], 0.2) + InfoNCE(item_global[i_idx], item_local[i_idx], 0.2)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward(view='global')

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class MSBEGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, norm_adj):
        super(MSBEGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_msbe_adj = None

    def set_msbe_adj(self, msbe_adj):
        self.sparse_msbe_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(msbe_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, view='global', perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        
        if view == 'global':
            adj = self.sparse_norm_adj
        elif view == 'local':
            if self.sparse_msbe_adj is None:
                # If local structure is missing, fallback to global to avoid crash
                adj = self.sparse_norm_adj
            else:
                adj = self.sparse_msbe_adj
        else:
            adj = self.sparse_norm_adj

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
