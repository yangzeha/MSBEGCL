import os

notebook_content = r'''
<VSCode.Cell id="cell_1" language="markdown">
# Run MSBEGCL on Kaggle

This notebook sets up the environment, compiles the necessary C++ mining tools, prepares the data, runs the mining algorithm to generate bicliques, and finally trains the MSBEGCL recommender system.
</VSCode.Cell>
<VSCode.Cell id="cell_2" language="python">
import os, sys, subprocess, time, shutil

# --- Configuration ---
repo_url = 'https://github.com/yangzeha/MSBEGCL.git'
repo_dir = 'MSBEGCL'
model_name = 'MSBEGCL'
dataset_name = 'yelp2018'

# 1. Clean and Clone Repository
# Force delete existing folder to avoid stale/broken state from previous failed runs
if os.path.exists(repo_dir):
    print(f"Removing existing '{repo_dir}' to ensure a fresh clone...")
    try:
        shutil.rmtree(repo_dir)
        print("Removal successful.")
    except Exception as e:
        print(f"Error removing directory: {e}")
        # Try shell command if python fails (sometimes permission issues on windows/linux vary)
        subprocess.run(['rm', '-rf', repo_dir])

print(f'Cloning {repo_dir} from {repo_url} (branch: master)...')
try:
    subprocess.run(['git', 'clone', '-b', 'master', repo_url], check=True)
    print("Clone successful.")
except subprocess.CalledProcessError as e:
    print(f"Git clone failed: {e}")
    sys.exit(1)

# 2. Setup Directories
if os.path.basename(os.getcwd()) != repo_dir:
    os.chdir(repo_dir)
print(f'Initial working directory: {os.getcwd()}')

# [Robustness Fix]: Auto-detect nested structure (e.g. MSBEGCL/MSBEGCL/...)
# Detailed log to debug Kaggle environment
roots = os.listdir('.')
print(f'Files in root: {roots}')

target_structure_found = False
possible_subdirs = ['.', 'MSBEGCL', 'msbegcl', repo_dir]

for d in possible_subdirs:
    if d == '.':
        path_to_check = '.'
    else:
        path_to_check = d
        if not os.path.exists(d) or not os.path.isdir(d):
            continue
            
    # Check if this dir contains the project keys
    contents = os.listdir(path_to_check)
    if 'SELFRec' in contents and 'Similar-Biclique-Idx' in contents:
        print(f"Project root found in: '{path_to_check}'")
        if d != '.':
            os.chdir(d)
        target_structure_found = True
        break

if not target_structure_found:
    print("WARNING: Standard project structure (SELFRec + Similar-Biclique-Idx) NOT FOUND in root or expected subdirs.")
    print("Searching recursively for SELFRec...")
    found = False
    for root, dirs, files in os.walk('.'):
        if 'SELFRec' in dirs:
            print(f"Found SELFRec in {root}")
            os.chdir(root)
            found = True
            break
    if not found:
        print("CRITICAL ERROR: Could not locate SELFRec directory anywhere.")

print(f'Final working directory: {os.getcwd()}')

# Ensure we are in the root of MSBEGCL which should contain SELFRec and Similar-Biclique-Idx
selfrec_path = 'SELFRec'
msbe_path = 'Similar-Biclique-Idx'

# Debug: Verify Content
print('\n--- Directory Structure Check ---')
print(f"Current contents: {os.listdir('.')}")

if os.path.exists(msbe_path):
    # ... check datasets ...
    datasets_path = os.path.join(msbe_path, 'datasets')
    if os.path.exists(datasets_path):
        print(f"Contents of {datasets_path}: {os.listdir(datasets_path)}")
    else:
        print(f"Error: {datasets_path} does not exist.")
else:
    print(f"Error: {msbe_path} does not exist.")


# 3. Install Dependencies
print('\n--- Installing Python Dependencies ---')
subprocess.run([sys.executable, '-m', 'pip', 'install', 'PyYAML==6.0.2', 'scipy==1.14.1', '-q'], check=True)
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'faiss-cpu', '-q'], check=True)
except:
    print("faiss-cpu install failed, continuing...")

# 4. Compile C++ Tools (MSBE Mining)
print('\n--- Compiling C++ Mining Tools ---')

# [Dependency]: Install sparsehash locally
sparsez_dir = 'sparsehash'
if not os.path.exists(sparsez_dir):
    print("Cloning sparsehash...")
    subprocess.run(['git', 'clone', 'https://github.com/sparsehash/sparsehash.git'], check=True)
    
    # sparsehash needs configuration to generate specific headers
    print("Configuring sparsehash...")
    cwd_backup = os.getcwd()
    os.chdir(sparsez_dir)
    try:
        # ./configure && make (to generate sparseconfig.h source files)
        # Check permissions for configure script
        subprocess.run(['chmod', '+x', 'configure']) 
        subprocess.run(['./configure'], check=True)
        # make is usually needed to build the intermediate files
        subprocess.run(['make'], check=True)
    except Exception as e:
        print(f"Warning: sparsehash configure/make failed: {e}. Trying to proceed with raw source...")
    finally:
        os.chdir(cwd_backup)

# Compile msbe
msbe_src = os.path.join(msbe_path, 'main.cpp')
msbe_exe = './msbe'
if not os.path.exists(msbe_src):
    print(f"CRITICAL ERROR: Source file {msbe_src} not found!")
else:
    # Added -I sparsehash/src to include path
    subprocess.run(['g++', '-O3', msbe_src, '-o', msbe_exe, '-I', msbe_path, '-I', 'sparsehash/src', '-D_PrintResults_', '-D_CheckResults_'], check=True)
    subprocess.run(['chmod', '+x', msbe_exe])
    print('msbe compiled.')

# 5. Data Preprocessing (Text -> Binary for Mining)
print(f'\n--- Preprocessing {dataset_name} for Mining ---')
train_file = os.path.join(selfrec_path, 'dataset', dataset_name, 'train.txt')
mining_graph_txt = 'graph.txt'

if not os.path.exists(train_file):
    print(f"CRITICAL ERROR: Data file {train_file} not found!")
else:
    # Read Train Data & Map IDs
    users = set()
    items = set()
    edges = []
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, i = parts[0], parts[1]
                users.add(u)
                items.add(i)
                edges.append((u, i))

    sorted_users = sorted(list(users))
    sorted_items = sorted(list(items))
    u_map = {u: idx for idx, u in enumerate(sorted_users)}
    i_map = {i: idx for idx, i in enumerate(sorted_items)}

    n1 = len(users)
    n2 = len(items)
    m_val = len(edges) * 2 
   
    with open(mining_graph_txt, 'w') as f:
        f.write(f"{n1} {n2} {len(edges)}\n")
        for u, i in edges:
            # Write UserID and ItemID+n1
            f.write(f"{u_map[u]} {i_map[i] + n1}\n")

    print(f'Generated {mining_graph_txt} with {n1} users, {n2} items, {len(edges)} edges.')

    # Python-based Binary Generation (Imitating ai_project logic)
    import struct
    
    # 1. Build Adjacency List (Undirected/Bipartite)
    # n1 users [0, n1-1], n2 items [n1, n1+n2-1]
    total_nodes = n1 + n2
    adj = [[] for _ in range(total_nodes)]
    edge_count = 0
    
    for u, i in edges:
        uid = u_map[u]
        iid = i_map[i] + n1
        
        # Add undirected edge
        adj[uid].append(iid)
        adj[iid].append(uid)
        edge_count += 2
        
    # Sort adjacency lists (MSBE requirement)
    for k in range(total_nodes):
        adj[k].sort()
        
    # 2. Write _b_degree.bin
    degree_file = 'graph_b_degree.bin'
    with open(degree_file, 'wb') as f:
        f.write(struct.pack('I', 4)) # sizeof(ui)
        f.write(struct.pack('I', n1))
        f.write(struct.pack('I', n2))
        f.write(struct.pack('I', edge_count))
        
        degrees = [len(adj[k]) for k in range(total_nodes)]
        f.write(struct.pack(f'{total_nodes}I', *degrees))
        
    # 3. Write _b_adj.bin
    adj_file = 'graph_b_adj.bin'
    with open(adj_file, 'wb') as f:
        flat_adj = []
        for k in range(total_nodes):
            flat_adj.extend(adj[k])
        f.write(struct.pack(f'{edge_count}I', *flat_adj))
        
    print(f"Generated binary graph files: {degree_file}, {adj_file}")
    
    # Create dummy text file to satisfy MSBE input check
    with open(mining_graph_txt, 'w') as f:
        f.write("dummy")

# 6. Run Mining
print('\n--- Mining Bicliques ---')
# Parameters from yaml or paper defaults
sim_threshold = 0.1  # epsilon (Lowered to 0.1 to capture more bicliques)
size_threshold = 2   # tau

if os.path.exists(msbe_exe) and os.path.exists(mining_graph_txt):
    # A. Build Index
    print('Building Index...')
    # ./msbe graph.txt 1 1 0.3 GRL3
    subprocess.run([msbe_exe, mining_graph_txt, '1', '1', '0.3', 'GRL3'], check=True)

    # B. Enumerate
    print('Enumerating...')
    raw_bicliques_file = 'bicliques_raw.txt'
    with open(raw_bicliques_file, 'w') as outfile:
        # ./msbe graph.txt 0 1 0.3 GRL3 1 GRL3 0 0 heu 4 epsilon tau 2
        subprocess.run([
            msbe_exe, mining_graph_txt, 
            '0', '1', '0.3', 'GRL3', 
            '1', 'GRL3', 
            '0', '0', 'heu', 
            '4', str(sim_threshold), str(size_threshold), '2'
        ], stdout=outfile, check=True)
    
    # [Debug]: Check the output file content
    if os.path.exists(raw_bicliques_file):
        size = os.path.getsize(raw_bicliques_file)
        print(f"Mining output file size: {size} bytes")
        # Print content regardless of size for debugging this time (limit to 2000 chars)
        print("--- Content of bicliques_raw.txt (Debug snippet) ---")
        try:
             with open(raw_bicliques_file, 'r') as f:
                content = f.read()
                print(content[:2000])
                if len(content) > 2000: print("... (truncated)")
        except Exception as e:
            print(f"Error reading debug file: {e}")
        print("------------------------------------------")
else:
    print("Skipping mining due to compliation or data failure.")

# 7. Process Bicliques -> Model Format
print('\n--- Formatting Bicliques for Model ---')
final_biclique_path = os.path.join(selfrec_path, 'dataset', dataset_name, 'bicliques.txt')
count = 0

if os.path.exists('bicliques_raw.txt'):
    with open('bicliques_raw.txt', 'r') as fr, open(final_biclique_path, 'w') as fw:
        current_users = []
        current_items = []
        
        for line in fr:
            line = line.strip()
            if not line:
                continue
            
            # Format: u1 u2 ... | i1 i2 ...
            if '|' in line:
                try:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        u_part = parts[0].strip()
                        i_part = parts[1].strip()
                        current_users = [int(x) for x in u_part.split() if x.isdigit()]
                        current_items = [int(x) for x in i_part.split() if x.isdigit()]
                    else:
                        continue
                except ValueError:
                    continue

                if len(current_users) > 0 and len(current_items) > 0:
                    real_users = []
                    real_items = []
                    
                    for uid in current_users:
                        if uid < n1:
                             real_users.append(sorted_users[uid])
                    
                    for iid in current_items:
                        # Raw item ID is encoded as iid (>= n1 usually in the combined graph)
                        # So items valid are [n1, n1+n2-1]
                        if iid >= n1:
                            raw_iid = iid - n1
                            if raw_iid < n2:
                                real_items.append(sorted_items[raw_iid])
                    
                    if len(real_users) > 0 and len(real_items) > 0:
                        fw.write(f"{' '.join(real_users)} | {' '.join(real_items)}\n")
                        count += 1
                
                # Reset
                current_users = []
                current_items = []
                
    print(f"Processed {count} bicliques into {final_biclique_path}")
    
    # [Robustness] Verify output content
    if count == 0:
         print("CRITICAL ERROR: No bicliques were extracted (count == 0).")
         print("Check sim_threshold (epsilon) or data preprocessing.")
         print("Terminating execution as requested.")
         sys.exit(1)
else:
    print("Warning: bicliques_raw.txt not found.")

# 8. Update Configuration
conf_path = os.path.join(selfrec_path, 'conf', 'MSBEGCL.yaml')

# Verify conf file exists before reading
if not os.path.exists(conf_path):
    print(f"Error: Config file {conf_path} not found. CWD: {os.getcwd()}")
else:
    with open(conf_path, 'r') as f:
        conf_content = f.read()

    # Update path to be local to SELFRec execution
    new_path = f'./dataset/{dataset_name}/bicliques.txt'
    import re
    
    conf_content = re.sub(r'biclique\.file:.*', f'biclique.file: {new_path}', conf_content)
    # Increase epochs to ensure convergence
    conf_content = re.sub(r'max\.epoch:.*', 'max.epoch: 60', conf_content)

    with open(conf_path, 'w') as f:
        f.write(conf_content)
    print("Updated MSBEGCL.yaml with correct biclique path and max.epoch=60.")

# 9. Run MSBEGCL
print('\n--- Starting Traning ---')

# Must be inside SELFRec to run main.py usually, BUT if we move there, relative paths change
# The script above assumes we are in ROOT (containing SELFRec folders)
# main.py is in SELFRec/main.py.

# Check where main.py is
main_py_path = os.path.join(selfrec_path, 'main.py')
if not os.path.exists(main_py_path):
    print(f"CRITICAL: {main_py_path} not found.")

# We switch to SELFRec directory to run the model, as it likely depends on relative imports
os.chdir(selfrec_path)
print(f"Changed directory to {os.getcwd()} for training.")

process = subprocess.Popen(
    [sys.executable, '-u', 'main.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, 
    text=True,
    bufsize=1
)

try:
    process.stdin.write(f'{model_name}\n')
    process.stdin.flush()
    process.stdin.close()
except Exception as e:
    print(f"Error writing to stdin: {e}")

# Streaming Output
while True:
    line = process.stdout.readline()
    if not line and process.poll() is not None:
        break
    if line:
        print(line.strip())

if process.poll() != 0:
    print("Training failed.")
else:
    print("Training finished successfully.")
</VSCode.Cell>
'''

with open(r'c:\Users\LENOVO\Desktop\论文代码\MSBEGCL(我的论文代码)\SELFRec\run_SimGCL_kaggle.ipynb', 'w', encoding='utf-8') as f:
    f.write(notebook_content)
