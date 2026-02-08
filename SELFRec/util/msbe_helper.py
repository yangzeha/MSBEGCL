
import os
import numpy as np
import scipy.sparse as sp

def load_msbe_adj(file_path, user_map, item_map, user_num, item_num):
    """
    Load bicliques from file and construct a sparse adjacency matrix.
    File format expected:
    user_id1 user_id2 ... | item_id1 item_id2 ...
    or
    u1,u2...:i1,i2...
    """
    if not os.path.exists(file_path):
        print(f"Warning: Biclique file not found at {file_path}. Structural CL will be disabled.")
        # Return empty matrix to allow running without the file
        return sp.csr_matrix((user_num + item_num, user_num + item_num))

    rows = []
    cols = []
    
    print(f"Loading bicliques from {file_path}...")
    
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if '|' in line:
                parts = line.split('|')
            elif ':' in line:
                parts = line.split(':')
            else:
                continue
                
            if len(parts) != 2:
                continue
                
            u_part = parts[0].strip().split()
            i_part = parts[1].strip().split()
            
            u_ids = [user_map.get(u) for u in u_part if u in user_map]
            i_ids = [item_map.get(i) for i in i_part if i in item_map]
            
            if not u_ids or not i_ids:
                continue
            
            for u in u_ids:
                for i in i_ids:
                    rows.append(u)
                    cols.append(i + user_num)
                    rows.append(i + user_num)
                    cols.append(u)
            
            count += 1
            
    print(f"Loaded {count} bicliques.")
    
    data = np.ones(len(rows))
    adj = sp.csr_matrix((data, (rows, cols)), shape=(user_num + item_num, user_num + item_num))
    
    # Normalize
    adj.data = np.ones_like(adj.data)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    return norm_adj

    def load_msbe_neighbors(file_path, user_map, item_map, interaction_mat=None, sim_threshold=0.1):
    """
    Load Global Top-1 neighbor based on Interaction Similarity (Cosine).
    If max similarity < sim_threshold, returned list is empty (triggering SimGCL fallback).
    
    Args:
        interaction_mat: User-Item sparse matrix.
        sim_threshold: Minimum cosine similarity.
    
    Returns:
        user_neighbors: dict {userid: [best_sim_uid]}
        item_neighbors: dict {itemid: [best_sim_iid]}
    """
    
    print(f"Calculating Global Top-1 Nearest Neighbor (Threshold={sim_threshold})...")
    
    def get_knn(mat, transpose=False):
        # mat is sparse CSR
        target_mat = mat.T if transpose else mat
        target_mat = target_mat.tocsr()
        
        # 1. Normalize rows (L2) -> Dot product becomes Cosine Similarity
        # Compute row norms
        row_sums = np.array(target_mat.power(2).sum(axis=1)).flatten()
        row_norms = np.sqrt(row_sums)
        row_norms[row_norms == 0] = 1.0 # Avoid div by zero
        
        diag_norm = sp.diags(1.0 / row_norms)
        norm_mat = diag_norm.dot(target_mat)
        
        # 2. Compute Similarity: S = R * R^T (Sparse dot product)
        # Note: This can be memory intensive for very large datasets (N^2). 
        # For Yelp2018 (30k users), 30k*30k float32 ~ 3.6GB dense, but sparse likely fits.
        print(f"  Computing full similarity matrix ({'Items' if transpose else 'Users'})...")
        sim_mat = norm_mat.dot(norm_mat.T)
        
        # 3. Extract Top-1 for each row
        neighbors = {}
        sim_mat = sim_mat.tocsr()
        num_nodes = sim_mat.shape[0]
        
        for i in range(num_nodes):
            row = sim_mat[i]
            if row.nnz == 0: continue
            
            # Get indices and data
            indices = row.indices
            data = row.data
            
            # Filter self-loop
            mask = indices != i
            indices = indices[mask]
            data = data[mask]
            
            if len(data) == 0: continue
            
            # Filter by threshold
            # User requirement: If similarity < threshold, filtering happens here -> empty list -> SimGCL
            mask_thresh = data >= sim_threshold
            indices = indices[mask_thresh]
            data = data[mask_thresh]
            
            if len(data) == 0: continue
                
            # Find Top-1 (Global Max)
            max_idx_local = np.argmax(data)
            best_neighbor = indices[max_idx_local]
            
            neighbors[i] = [best_neighbor]
            
            if i % 10000 == 0 and i > 0:
                print(f"    Processed {i}/{num_nodes} nodes...")
                
        return neighbors

    user_neighbors = get_knn(interaction_mat, transpose=False)
    item_neighbors = get_knn(interaction_mat, transpose=True)
    
    print(f"Global Top-1 Search complete. Found valid neighbors for {len(user_neighbors)} users and {len(item_neighbors)} items.")
    return user_neighbors, item_neighbors

