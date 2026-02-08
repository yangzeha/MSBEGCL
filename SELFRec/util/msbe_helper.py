import os
import numpy as np
import scipy.sparse as sp

def load_msbe_adj(file_path, user_map, item_map, user_num, item_num):
    """
    Deprecated / Dummy function to satisfy interface.
    No Biclique Structure aggregation (GLSCL style).
    """
    return sp.csr_matrix((user_num + item_num, user_num + item_num))

def load_msbe_neighbors(file_path, user_map, item_map, interaction_mat=None, sim_threshold=0.1):
    """
    Compute Global Top-1 Similarity Neighbors.
    Ignores strict biclique structure as requested.
    Uses Sparse Matrix Multiplication batch-wise for global similarity search.
    
    Returns:
        user_neighbors: {user_id: [neighbor_id]}
        item_neighbors: {item_id: [neighbor_id]}
    """
    print(f'Calculating Global Top-1 Neighbors (Threshold={sim_threshold})...')
    
    if interaction_mat is None:
        print('Error: interaction_mat required for global similarity.')
        return {}, {}
        
    def get_global_top1(mat, threshold):
        # Normalize rows
        mat = mat.tocsr()
        row_sums = np.array(mat.power(2).sum(axis=1)).flatten()
        row_norms = np.sqrt(row_sums)
        row_norms[row_norms == 0] = 1.0
        diag_norm = sp.diags(1.0 / row_norms)
        norm_mat = diag_norm.dot(mat)
        
        num_nodes = norm_mat.shape[0]
        neighbors = {}
        
        # Batch processing with Sparse Matrix Multiplication
        # Tune batch_size based on memory. 2000 is usually safe.
        batch_size = 2000 
        
        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)
            
            # Batch: (B, F)
            batch_mat = norm_mat[start_idx:end_idx]
            
            # Sim: (B, N)
            sim_batch = batch_mat.dot(norm_mat.T)
            
            # Convert to dense to find max
            sim_dense = sim_batch.toarray()
            
            # Mask self
            for i in range(len(sim_dense)):
                global_id = start_idx + i
                sim_dense[i, global_id] = -1.0
            
            # Find max
            max_indices = np.argmax(sim_dense, axis=1)
            max_values = np.max(sim_dense, axis=1)
            
            for i in range(len(sim_dense)):
                if max_values[i] > threshold:
                    global_u = start_idx + i
                    best_n = max_indices[i]
                    neighbors[global_u] = [int(best_n)]
                    
            if (start_idx // batch_size) % 5 == 0:
                print(f'Processed {end_idx}/{num_nodes} nodes...')
                
        return neighbors

    # User Neighbors
    user_neighbors = get_global_top1(interaction_mat, sim_threshold)
    print(f'Found {len(user_neighbors)} users with similar neighbors > {sim_threshold}')
    
    # Item Neighbors
    item_neighbors = get_global_top1(interaction_mat.T, sim_threshold)
    print(f'Found {len(item_neighbors)} items with similar neighbors > {sim_threshold}')
    
    return user_neighbors, item_neighbors
