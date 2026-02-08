
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
    Load neighbors from Maximal Similar Bicliques (MSBE).
    Constraint: Candidates must be from the same Biclique.
    Selection: Top-1 neighbor with highest Cosine Similarity (Interaction-based).
    Fallback: If max similarity < threshold, no neighbor is returned (model handles fallback).
    
    Args:
        file_path: Path to bicliques.txt
        interaction_mat: User-Item sparse matrix (for similarity calculation)
        sim_threshold: Filtering threshold
    """
    if not os.path.exists(file_path):
        return {}, {}
    
    print(f"Loading MSBE neighbors from {file_path} (Threshold={sim_threshold})...")
    
    # 1. Parse Bicliques into Clusters
    # User said: "User will only appear in one maximal similar biclique"
    user_clusters = [] # List of lists of user IDs
    item_clusters = [] # List of lists of item IDs
    
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
            
            if len(u_ids) > 1: user_clusters.append(u_ids)
            if len(i_ids) > 1: item_clusters.append(i_ids)

    # 2. Process Clusters: Find best neighbor within cluster
    print(f"Processing {len(user_clusters)} user clusters and {len(item_clusters)} item clusters...")
    
    def process_clusters(clusters, mat, transpose=False):
        final_neighbors = {}
        target_mat = mat.T if transpose else mat
        target_mat = target_mat.tocsr()
        
        # Pre-compute row norms
        row_sums = np.array(target_mat.power(2).sum(axis=1)).flatten()
        row_norms = np.sqrt(row_sums)
        row_norms[row_norms == 0] = 1.0
        
        # Diagonal norm matrix
        diag_norm = sp.diags(1.0 / row_norms)
        norm_mat = diag_norm.dot(target_mat)
        
        processed_count = 0
        
        for cluster in clusters:
            if len(cluster) < 2: continue
            
            # Extract sub-matrix for this cluster
            # (K x F)
            cluster_mat = norm_mat[cluster] 
            
            # Compute similarity block: K x K
            # dense is fine for small clusters (e.g. < 1000)
            sim_block = cluster_mat.dot(cluster_mat.T).toarray()
            
            # For each node in cluster, find max in same row (excluding self)
            for idx, node_id in enumerate(cluster):
                # Row in sim_block
                row_sims = sim_block[idx]
                
                # Mask self
                row_sims[idx] = -1.0
                
                # Find max
                best_local_idx = np.argmax(row_sims)
                best_sim = row_sims[best_local_idx]
                
                if best_sim >= sim_threshold:
                    best_neighbor_id = cluster[best_local_idx]
                    final_neighbors[node_id] = [best_neighbor_id]
            
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count} clusters...")
                
        return final_neighbors

    user_neighbors = process_clusters(user_clusters, interaction_mat, transpose=False)
    item_neighbors = process_clusters(item_clusters, interaction_mat, transpose=True)
    
    print(f"MSBE Neighbor Search complete. Found neighbors for {len(user_neighbors)} users and {len(item_neighbors)} items.")
    return user_neighbors, item_neighbors

