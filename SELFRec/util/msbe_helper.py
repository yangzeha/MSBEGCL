
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

def load_msbe_neighbors(file_path, user_map, item_map):
    """
    Load neighbor dictionaries from bicliques.
    Returns:
        user_neighbors: dict {userid: [similar_userid_1, ...]}
        item_neighbors: dict {itemid: [similar_itemid_1, ...]}
    """
    if not os.path.exists(file_path):
        return {}, {}
    
    user_neighbors = {}
    item_neighbors = {}
    
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
            
            # Build User-User connections (Clique expansion)
            # Count frequency to determine similarity strength
            # Frequency = How many bicliques do they share?
            pass # We first need to collect raw edges then process frequency

    # Re-scan to build weighted graph
    user_pair_counts = {} # (u, v) -> count
    item_pair_counts = {} # (i, j) -> count
    
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
            
            if len(u_ids) > 1:
                u_ids.sort()
                for idx1 in range(len(u_ids)):
                    for idx2 in range(idx1 + 1, len(u_ids)):
                        pair = (u_ids[idx1], u_ids[idx2])
                        user_pair_counts[pair] = user_pair_counts.get(pair, 0) + 1
                        
            if len(i_ids) > 1:
                i_ids.sort()
                for idx1 in range(len(i_ids)):
                    for idx2 in range(idx1 + 1, len(i_ids)):
                        pair = (i_ids[idx1], i_ids[idx2])
                        item_pair_counts[pair] = item_pair_counts.get(pair, 0) + 1

    # Convert counts to sorted adjacency lists
    user_neighbors = {}
    for (u, v), count in user_pair_counts.items():
        if u not in user_neighbors: user_neighbors[u] = []
        if v not in user_neighbors: user_neighbors[v] = []
        user_neighbors[u].append((v, count))
        user_neighbors[v].append((u, count))
        
    item_neighbors = {}
    for (i, j), count in item_pair_counts.items():
        if i not in item_neighbors: item_neighbors[i] = []
        if j not in item_neighbors: item_neighbors[j] = []
        item_neighbors[i].append((j, count))
        item_neighbors[j].append((i, count))

    # Sort by count (descending) and stripe counts
    for u in user_neighbors:
        user_neighbors[u].sort(key=lambda x: x[1], reverse=True)
        user_neighbors[u] = [x[0] for x in user_neighbors[u]]
        
    for i in item_neighbors:
        item_neighbors[i].sort(key=lambda x: x[1], reverse=True)
        item_neighbors[i] = [x[0] for x in item_neighbors[i]]
    
    return user_neighbors, item_neighbors

