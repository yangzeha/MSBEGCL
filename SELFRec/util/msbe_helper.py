
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

def load_msbe_neighbors(file_path, user_map, item_map, interaction_mat=None):
    """
    Load neighbor dictionaries from bicliques.
    Sorts neighbors by interaction similarity (Cosine Similarity).
    Returns:
        user_neighbors: dict {userid: [most_similar_uid, ...]}
        item_neighbors: dict {itemid: [most_similar_iid, ...]}
    """
    if not os.path.exists(file_path):
        return {}, {}
    
    # 1. Collect biclique members (assuming disjoint or overlapping is fine, we just collect candidates)
    # User said: "User will only appear in one maximal similar biclique" (Strict Partition? Or disjoint)
    # We will assume they are grouped.
    
    user_candidates = {} # u -> set of neighbors
    item_candidates = {} # i -> set of neighbors
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if '|' in line: parts = line.split('|')
            elif ':' in line: parts = line.split(':')
            else: continue
            if len(parts) != 2: continue
            
            u_part = parts[0].strip().split()
            i_part = parts[1].strip().split()
            
            u_ids = [user_map.get(u) for u in u_part if u in user_map]
            i_ids = [item_map.get(i) for i in i_part if i in item_map]
            
            # Map every node to all other nodes in the same clique
            if len(u_ids) > 1:
                for u in u_ids:
                    if u not in user_candidates: user_candidates[u] = set()
                    for v in u_ids:
                        if u != v:
                            user_candidates[u].add(v)
                            
            if len(i_ids) > 1:
                for i in i_ids:
                    if i not in item_candidates: item_candidates[i] = set()
                    for j in i_ids:
                        if i != j:
                            item_candidates[i].add(j)

    # 2. Sort neighbors by Cosine Similarity on Interaction Matrix
    print("Calculating similarities for neighbor ranking...")
    
    # Helper for fast cosine similarity between one row and many rows
    def sort_by_similarity(candidates_dict, mat, transpose=False):
        final_dict = {}
        # Pre-compute norms if possible, or computing on the fly
        # mat is sparse csr
        # For item similarity, we need columns. If transpose=False (user), mat is U x I.
        # If transpose=True (item), we want Cosine(Col_i, Col_j).
        # Better to transpose matrix once for items.
        
        target_mat = mat.T if transpose else mat
        # Ensure CSR for fast slicing
        target_mat = target_mat.tocsr()
        
        # Norms
        # row norms: sqrt(sum(x^2))
        inv_norms = np.array(np.sqrt(target_mat.multiply(target_mat).sum(axis=1))).flatten()
        with np.errstate(divide='ignore'):
            inv_norms = 1.0 / inv_norms
        inv_norms[np.isinf(inv_norms)] = 0.0
        
        # Process each node
        processed = 0
        total = len(candidates_dict)
        
        for node_id, neighbors_set in candidates_dict.items():
            if not neighbors_set:
                final_dict[node_id] = []
                continue
                
            neighbors = list(neighbors_set)
            
            # Vectorized Sim Calculation: 
            # query_vec * neighbors_matrix.T
            
            # 1. Get query vector
            q_vec = target_mat[node_id]
            q_norm = inv_norms[node_id]
            
            if q_vec.nnz == 0:
                final_dict[node_id] = neighbors # random order
                continue
            
            # 2. Get neighbors submatrix
            # Using fancy indexing might be slow if list is long, but here it's small (clique size)
            nb_indices = neighbors
            nb_mat = target_mat[nb_indices]
            
            if nb_mat.nnz == 0:
                final_dict[node_id] = neighbors
                continue
                
            # 3. Dot product
            # (1 x F) * (K x F)^T -> (1 x K)
            dot_prods = q_vec.dot(nb_mat.T).toarray().flatten()
            
            # 4. Normalize
            nb_norms = inv_norms[nb_indices]
            sims = dot_prods * q_norm * nb_norms
            
            # 5. Sort
            # Zip and sort descending
            sorted_pairs = sorted(zip(neighbors, sims), key=lambda x: x[1], reverse=True)
            final_dict[node_id] = [p[0] for p in sorted_pairs]
            
            processed += 1
            if processed % 5000 == 0:
                print(f"Processed {processed}/{total} nodes...")
                
        return final_dict

    user_neighbors = sort_by_similarity(user_candidates, interaction_mat, transpose=False)
    item_neighbors = sort_by_similarity(item_candidates, interaction_mat, transpose=True)
    
    return user_neighbors, item_neighbors

