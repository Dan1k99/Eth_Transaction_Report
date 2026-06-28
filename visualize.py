import os
import sys
import json
import random
import pickle
import sys
import webbrowser
import numpy as np
import networkx as nx
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Add T-EDGE directory to sys.path so we can import tGraph
sys.path.append(os.path.join(os.path.dirname(__file__), 'T-EDGE'))
from tGraph import tGraph

# Global variable to store the currently loaded graph object
CURRENT_TGRAPH = None
CURRENT_DATASET = None

def get_available_datasets():
    """Scan T-EDGE directory for df_train_0.5 pickle files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    t_edge_dir = os.path.join(script_dir, "T-EDGE")
    if not os.path.exists(t_edge_dir):
        return []
    
    datasets = []
    for file in os.listdir(t_edge_dir):
        if file.endswith("_df_train_0.5.pickle"):
            dataset_name = file.replace("_df_train_0.5.pickle", "")
            datasets.append(dataset_name)
    return sorted(datasets)

def load_dataset(dataset_name, exclude_failed=False):
    """Load the specified dataset using tGraph."""
    global CURRENT_TGRAPH, CURRENT_DATASET
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "T-EDGE", f"{dataset_name}_df_train_0.5.pickle")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset pickle file not found at {file_path}")
        
    print(f"Loading dataset: {dataset_name} (exclude_failed={exclude_failed})")
    CURRENT_TGRAPH = tGraph(file_path, "pkl_f", exclude_failed=exclude_failed)
    CURRENT_DATASET = dataset_name
    
    # Calculate stats
    num_nodes = CURRENT_TGRAPH.G.number_of_nodes()
    num_edges = CURRENT_TGRAPH.G.number_of_edges()
    
    contract_nodes = sum(1 for n, attr in CURRENT_TGRAPH.G.nodes(data=True) if attr.get('is_contract', 0) == 1)
    eoa_nodes = num_nodes - contract_nodes
    
    return {
        "dataset": dataset_name,
        "nodes": num_nodes,
        "edges": num_edges,
        "contract_nodes": contract_nodes,
        "eoa_nodes": eoa_nodes,
        "min_time": CURRENT_TGRAPH.min_time,
        "max_time": CURRENT_TGRAPH.max_time
    }

def linear_rank_mapping(original_array, order='ascending'):
    x = np.array(original_array)
    if len(x) == 0:
        return np.array([])
    if order == 'ascending':
        return (np.argsort(x) + 1).tolist()
    elif order == 'descending':  
        return (np.argsort(-x) + 1).tolist()

def normalized_probs(unnormalized_probs):
    if len(unnormalized_probs) > 0:
        norm_const = sum(unnormalized_probs)
        if norm_const == 0: 
            norm_const = 1.0
        return [u_prob / norm_const for u_prob in unnormalized_probs]
    return []

def combine_probs(p1, p2, alpha, p3=None, gamma=None):
    probs1 = normalized_probs(p1)
    probs2 = normalized_probs(p2)
    
    if p3 is not None and gamma is not None:
        probs3 = normalized_probs(p3)
        beta = 1.0 - alpha - gamma
        # Avoid zero/negative power issues
        comb = np.multiply(
            np.multiply(np.power(probs1, alpha), np.power(probs2, beta)),
            np.power(probs3, gamma)
        )
    else:
        comb = np.multiply(np.power(probs1, alpha), np.power(probs2, 1.0 - alpha))
    return comb.tolist()

def weight_choice(probs):
    """Pick an index based on the probability distribution."""
    if len(probs) == 0:
        return -1
    s = sum(probs)
    if s == 0:
        return random.randint(0, len(probs) - 1)
    
    # Scale to 1.0
    normalized = [p / s for p in probs]
    r = random.random()
    accum = 0.0
    for idx, p in enumerate(normalized):
        accum += p
        if r <= accum:
            return idx
    return len(probs) - 1

class InteractiveWalker:
    """A visual-friendly version of temporal random walks that logs the trace of probabilities."""
    def __init__(self, G, min_time, max_time, params):
        self.G = G
        self.min_time = min_time
        self.max_time = max_time
        
        self.time_biased_type = params.get('time_biased_type', 'time_close_exp')
        self.first_biased_type = params.get('first_biased_type', 'time_uniform')
        self.amount_biased = params.get('amount_biased', 'amount_exp')
        self.gas_biased = params.get('gas_biased', None) # 'friction_inverse' or None
        self.alpha = float(params.get('alpha', 0.5))
        self.gamma = float(params.get('gamma', 0.0))
        self.state_penalty = float(params.get('state_penalty', 1.0))
        
        self.typology_weights = params.get('typology_weights', {
            'eoa_to_eoa': 1.0,
            'eoa_to_contract': 1.0,
            'contract_to_eoa': 1.0,
            'contract_to_contract': 1.0
        })

    def get_first_step(self, cur):
        G = self.G
        tmp_node = []
        tmp_time = []
        tmp_key = []
        unnormalized_probs_t = []
        
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) == 0:
            return None, None, None, []
            
        for nbr in cur_nbrs:
            nbr_keys = list(G.get_edge_data(cur, nbr))
            for k in nbr_keys:
                t = k
                if self.first_biased_type == "time_uniform":
                    unnormalized_probs_t.append(1.0)
                elif self.first_biased_type in ["time_freq", "time_close_linear"]:
                    unnormalized_probs_t.append(float(self.max_time - t + 1))
                elif self.first_biased_type == "time_far":
                    unnormalized_probs_t.append(float(t - self.min_time + 1)) 
                elif self.first_biased_type == "time_far_linear":
                    unnormalized_probs_t.append(float(t))
                else:
                    unnormalized_probs_t.append(1.0)
                
                tmp_node.append(nbr)
                tmp_time.append(t)
                tmp_key.append(k)
                
        if self.first_biased_type == "time_close_linear":
            unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t, order='descending')
        elif self.first_biased_type == "time_far_linear":
            unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t)
            
        if len(unnormalized_probs_t) == 0:
            return None, None, None, []
            
        # Apply scaling
        scaled_probs = []
        candidates_info = []
        for i in range(len(unnormalized_probs_t)):
            prob = float(unnormalized_probs_t[i])
            nbr = tmp_node[i]
            k = tmp_key[i]
            edge_data = G[cur][nbr][k]
            
            # State Penalty
            is_failed = edge_data.get('is_failed', 0)
            penalty = self.state_penalty if is_failed == 1 else 1.0
            
            # Node Typology
            from_is_contract = G.nodes[cur].get('is_contract', 0)
            to_is_contract = G.nodes[nbr].get('is_contract', 0)
            
            if from_is_contract == 0 and to_is_contract == 0:
                w_typo = self.typology_weights.get('eoa_to_eoa', 1.0)
                typo_type = 'eoa_to_eoa'
            elif from_is_contract == 0 and to_is_contract == 1:
                w_typo = self.typology_weights.get('eoa_to_contract', 1.0)
                typo_type = 'eoa_to_contract'
            elif from_is_contract == 1 and to_is_contract == 0:
                w_typo = self.typology_weights.get('contract_to_eoa', 1.0)
                typo_type = 'contract_to_eoa'
            else:
                w_typo = self.typology_weights.get('contract_to_contract', 1.0)
                typo_type = 'contract_to_contract'
                
            final_prob = prob * penalty * w_typo
            scaled_probs.append(final_prob)
            
            candidates_info.append({
                "neighbor": nbr,
                "timestamp": k,
                "weight": edge_data.get('weight', 0.0),
                "is_failed": is_failed,
                "friction": edge_data.get('friction', 1.0),
                "raw_time_prob": prob,
                "raw_amount_prob": 1.0,
                "raw_friction_prob": 1.0,
                "penalty": penalty,
                "typo_weight": w_typo,
                "typo_type": typo_type,
                "final_prob": final_prob
            })
            
        # Normalize final probabilities for trace display
        tot = sum(scaled_probs)
        if tot == 0: 
            tot = 1.0
        for info in candidates_info:
            info["normalized_prob"] = info["final_prob"] / tot
            
        selected = weight_choice(scaled_probs)
        if selected == -1:
            return None, None, None, []
            
        # Mark chosen one
        for idx, info in enumerate(candidates_info):
            info["selected"] = (idx == selected)
            
        return tmp_node[selected], tmp_time[selected], tmp_key[selected], candidates_info

    def get_next_step(self, cur, prevtime):
        G = self.G
        tmp_node = []
        tmp_time = []
        tmp_key = []
        unnormalized_probs_t = []
        unnormalized_probs_a = []
        unnormalized_probs_g = []
        
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) == 0:
            return None, None, None, []
            
        for nbr in cur_nbrs:
            nbr_keys = list(G.get_edge_data(cur, nbr))
            for k in nbr_keys:
                t = k
                a = G[cur][nbr][k].get('weight', 0.0)
                
                # Biased random walk constraint: t >= prevtime
                if t >= prevtime:
                    unnormalized_probs_a.append(float(a))
                    if self.gas_biased == "friction_inverse":
                        unnormalized_probs_g.append(float(G[cur][nbr][k].get('friction', 1.0)))
                    else:
                        unnormalized_probs_g.append(1.0)
                        
                    if self.time_biased_type == "time_uniform":
                        unnormalized_probs_t.append(1.0)
                    elif self.time_biased_type == "time_close_raw":
                        unnormalized_probs_t.append(float(self.max_time - t + 1))
                    elif self.time_biased_type == "time_close_exp":
                        unnormalized_probs_t.append(float(t - prevtime))
                    else:
                        unnormalized_probs_t.append(float(t - prevtime + 1))
                        
                    tmp_node.append(nbr)
                    tmp_time.append(t)
                    tmp_key.append(k)
                    
        if len(unnormalized_probs_t) == 0:
            return None, None, None, []
            
        # Process component probabilities
        # 1. Time Mapping
        if self.time_biased_type == "time_close_linear":
            unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t, order='descending')
        elif self.time_biased_type == "time_far_linear":
            unnormalized_probs_t = linear_rank_mapping(unnormalized_probs_t)
        elif self.time_biased_type == "time_close_exp":
            # Softmax
            exp_t = np.exp(unnormalized_probs_t - np.max(unnormalized_probs_t))
            unnormalized_probs_t = (exp_t / exp_t.sum()).tolist()
            
        # 2. Amount Mapping
        if self.amount_biased == "amount_linear":
            unnormalized_probs_a = linear_rank_mapping(unnormalized_probs_a)
        elif self.amount_biased == "amount_exp":
            # Softmax
            exp_a = np.exp(unnormalized_probs_a - np.max(unnormalized_probs_a))
            unnormalized_probs_a = (exp_a / exp_a.sum()).tolist()
            
        # 3. Gas/Friction Mapping
        if self.gas_biased == "friction_inverse" and len(unnormalized_probs_g) > 0:
            max_f = max(unnormalized_probs_g)
            if max_f == 0: 
                max_f = 1.0
            # Inverse: smaller friction has higher probability
            unnormalized_probs_g = [max(1e-6, 1.0 - (f / max_f)) for f in unnormalized_probs_g]
            
        # Combine probabilities
        if self.gas_biased == "friction_inverse":
            unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha, unnormalized_probs_g, self.gamma)
        elif self.amount_biased != "amount_uniform":
            unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha)
        else:
            unnormalized_probs = normalized_probs(unnormalized_probs_t)
            
        # Apply State-Penalty and Node Typology
        scaled_probs = []
        candidates_info = []
        for i in range(len(unnormalized_probs)):
            prob = float(unnormalized_probs[i])
            nbr = tmp_node[i]
            k = tmp_key[i]
            edge_data = G[cur][nbr][k]
            
            # State Penalty
            is_failed = edge_data.get('is_failed', 0)
            penalty = self.state_penalty if is_failed == 1 else 1.0
            
            # Typology Weight
            from_is_contract = G.nodes[cur].get('is_contract', 0)
            to_is_contract = G.nodes[nbr].get('is_contract', 0)
            
            if from_is_contract == 0 and to_is_contract == 0:
                w_typo = self.typology_weights.get('eoa_to_eoa', 1.0)
                typo_type = 'eoa_to_eoa'
            elif from_is_contract == 0 and to_is_contract == 1:
                w_typo = self.typology_weights.get('eoa_to_contract', 1.0)
                typo_type = 'eoa_to_contract'
            elif from_is_contract == 1 and to_is_contract == 0:
                w_typo = self.typology_weights.get('contract_to_eoa', 1.0)
                typo_type = 'contract_to_eoa'
            else:
                w_typo = self.typology_weights.get('contract_to_contract', 1.0)
                typo_type = 'contract_to_contract'
                
            final_prob = prob * penalty * w_typo
            scaled_probs.append(final_prob)
            
            candidates_info.append({
                "neighbor": nbr,
                "timestamp": k,
                "weight": edge_data.get('weight', 0.0),
                "is_failed": is_failed,
                "friction": edge_data.get('friction', 1.0),
                "raw_time_prob": float(unnormalized_probs_t[i]),
                "raw_amount_prob": float(unnormalized_probs_a[i]) if len(unnormalized_probs_a) > i else 1.0,
                "raw_friction_prob": float(unnormalized_probs_g[i]) if len(unnormalized_probs_g) > i else 1.0,
                "penalty": penalty,
                "typo_weight": w_typo,
                "typo_type": typo_type,
                "final_prob": final_prob
            })
            
        tot = sum(scaled_probs)
        if tot == 0: 
            tot = 1.0
        for info in candidates_info:
            info["normalized_prob"] = info["final_prob"] / tot
            
        selected = weight_choice(scaled_probs)
        if selected == -1:
            return None, None, None, []
            
        for idx, info in enumerate(candidates_info):
            info["selected"] = (idx == selected)
            
        return tmp_node[selected], tmp_time[selected], tmp_key[selected], candidates_info

    def walk(self, walk_length, start_node):
        if start_node not in self.G:
            return []
            
        walk_steps = []
        cur = start_node
        
        # Add initial node
        is_contract = self.G.nodes[cur].get('is_contract', 0)
        walk_steps.append({
            "step": 0,
            "node": cur,
            "is_contract": is_contract,
            "timestamp": None,
            "weight": None,
            "friction": None,
            "is_failed": None,
            "candidates": []
        })
        
        # Step 1
        nxt, nxt_t, nxt_k, candidates = self.get_first_step(cur)
        if nxt is not None:
            edge_data = self.G[cur][nxt][nxt_k]
            walk_steps.append({
                "step": 1,
                "node": nxt,
                "is_contract": self.G.nodes[nxt].get('is_contract', 0),
                "timestamp": nxt_t,
                "weight": edge_data.get('weight', 0.0),
                "friction": edge_data.get('friction', 1.0),
                "is_failed": edge_data.get('is_failed', 0),
                "candidates": candidates
            })
            cur = nxt
            prevtime = nxt_t
        else:
            return walk_steps
            
        # Subsequent steps
        step = 2
        while len(walk_steps) < walk_length:
            nxt, nxt_t, nxt_k, candidates = self.get_next_step(cur, prevtime)
            if nxt is not None:
                edge_data = self.G[cur][nxt][nxt_k]
                walk_steps.append({
                    "step": step,
                    "node": nxt,
                    "is_contract": self.G.nodes[nxt].get('is_contract', 0),
                    "timestamp": nxt_t,
                    "weight": edge_data.get('weight', 0.0),
                    "friction": edge_data.get('friction', 1.0),
                    "is_failed": edge_data.get('is_failed', 0),
                    "candidates": candidates
                })
                cur = nxt
                prevtime = nxt_t
                step += 1
            else:
                break
                
        return walk_steps

def get_k_order_subgraph(G, start_nodes, k_in=1, k_out=2, max_nodes=150):
    """
    Subsample the graph dynamically around center nodes so it renders beautifully.
    Extends k-order BFS search, capped at a maximum node count.
    """
    sampled_nodes = set(start_nodes)
    
    # Forward search
    if k_out > 0:
        queue = [(n, 0) for n in start_nodes if n in G]
        visited = set(start_nodes)
        while queue and len(sampled_nodes) < max_nodes:
            curr, depth = queue.pop(0)
            if depth < k_out:
                for neighbor in G.neighbors(curr):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        sampled_nodes.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        if len(sampled_nodes) >= max_nodes:
                            break
                            
    # Backward search
    if k_in > 0:
        G_rev = G.reverse(copy=False)
        queue = [(n, 0) for n in start_nodes if n in G_rev]
        visited = set(start_nodes)
        while queue and len(sampled_nodes) < max_nodes:
            curr, depth = queue.pop(0)
            if depth < k_in:
                for neighbor in G_rev.neighbors(curr):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        sampled_nodes.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        if len(sampled_nodes) >= max_nodes:
                            break
                            
    # Make sure we induce the subgraph correctly
    subG = G.subgraph(sampled_nodes).copy()
    return subG

class APIHandler(BaseHTTPRequestHandler):
    
    def end_headers(self):
        # Allow CORS for easy development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
        
    def serve_static(self, file_path, content_type):
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return
            
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.end_headers()
        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())
            
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Serve UI Assets
        if path == "/" or path == "/index.html":
            self.serve_static(os.path.join(script_dir, "visualize.html"), "text/html")
            return
            
        # 2. API: Get Available Datasets
        elif path == "/api/datasets":
            datasets = get_available_datasets()
            self.send_json({"datasets": datasets})
            return
            
        # 3. API: Load Dataset
        elif path == "/api/load_graph":
            dataset_name = query.get('dataset', [None])[0]
            exclude_failed = query.get('exclude_failed', ['false'])[0].lower() == 'true'
            
            if not dataset_name:
                self.send_json({"error": "No dataset specified"}, 400)
                return
                
            try:
                stats = load_dataset(dataset_name, exclude_failed)
                self.send_json({"success": True, "stats": stats})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
            return
            
        # 4. API: Get Subgraph
        elif path == "/api/subgraph":
            if CURRENT_TGRAPH is None:
                self.send_json({"error": "No dataset loaded. Call /api/load_graph first."}, 400)
                return
                
            # If no center node is specified, pick top nodes by degree
            center_node = query.get('center_node', [None])[0]
            max_nodes = int(query.get('max_nodes', [100])[0])
            
            G = CURRENT_TGRAPH.G
            
            if center_node and center_node in G:
                subG = get_k_order_subgraph(G, [center_node], k_in=1, k_out=2, max_nodes=max_nodes)
            else:
                # Top nodes by total degree
                degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
                top_nodes = [node for node, deg in degrees[:15]]
                subG = get_k_order_subgraph(G, top_nodes, k_in=1, k_out=1, max_nodes=max_nodes)
                
            # Prepare elements for vis.js
            nodes_out = []
            for n, attr in subG.nodes(data=True):
                deg = G.degree(n)
                nodes_out.append({
                    "id": n,
                    "label": f"Node {n}",
                    "is_contract": attr.get('is_contract', 0),
                    "degree": deg
                })
                
            edges_out = []
            for u, v, k, attr in subG.edges(keys=True, data=True):
                edges_out.append({
                    "from": u,
                    "to": v,
                    "timestamp": k,
                    "weight": float(attr.get('weight', 1.0)),
                    "gas": float(attr.get('gas', 1.0)),
                    "friction": float(attr.get('friction', 1.0)),
                    "gas_limit": float(attr.get('gas_limit', 21000.0)),
                    "is_failed": int(attr.get('is_failed', 0))
                })
                
            # Also fetch standard degrees for user context
            self.send_json({
                "nodes": nodes_out,
                "edges": edges_out,
                "center_node": center_node
            })
            return
            
        # 5. API: Search Node
        elif path == "/api/search_node":
            if CURRENT_TGRAPH is None:
                self.send_json({"error": "No dataset loaded"}, 400)
                return
                
            node_id = query.get('query', [''])[0].strip()
            G = CURRENT_TGRAPH.G
            
            if node_id in G:
                attr = G.nodes[node_id]
                self.send_json({
                    "found": True,
                    "node": {
                        "id": node_id,
                        "is_contract": attr.get('is_contract', 0),
                        "degree": G.degree(node_id),
                        "in_degree": G.in_degree(node_id) if hasattr(G, 'in_degree') else 0,
                        "out_degree": G.out_degree(node_id) if hasattr(G, 'out_degree') else 0,
                    }
                })
            else:
                # Find top matches (string prefix)
                matches = []
                for n, attr in G.nodes(data=True):
                    if n.startswith(node_id):
                        matches.append({
                            "id": n,
                            "is_contract": attr.get('is_contract', 0),
                            "degree": G.degree(n)
                        })
                        if len(matches) >= 10:
                            break
                self.send_json({"found": False, "suggestions": matches})
            return

        # 6. API: Simulate Walks
        elif path == "/api/simulate_walk":
            if CURRENT_TGRAPH is None:
                self.send_json({"error": "No dataset loaded"}, 400)
                return
                
            start_node = query.get('start_node', [None])[0]
            if not start_node or start_node not in CURRENT_TGRAPH.G:
                # Pick a high degree EOA/Contract node as default
                degrees = sorted(CURRENT_TGRAPH.G.degree(), key=lambda x: x[1], reverse=True)
                start_node = degrees[0][0]
                
            walk_length = int(query.get('walk_length', [15])[0])
            
            # Walk Strategy weights
            params = {
                'time_biased_type': query.get('time_biased_type', ['time_close_exp'])[0],
                'first_biased_type': query.get('first_biased_type', ['time_uniform'])[0],
                'amount_biased': query.get('amount_biased', ['amount_exp'])[0],
                'gas_biased': query.get('gas_biased', [None])[0], # 'friction_inverse' or None
                'alpha': float(query.get('alpha', [0.5])[0]),
                'gamma': float(query.get('gamma', [0.0])[0]),
                'state_penalty': float(query.get('state_penalty', [1.0])[0]),
                'typology_weights': {
                    'eoa_to_eoa': float(query.get('eoa_to_eoa', [1.0])[0]),
                    'eoa_to_contract': float(query.get('eoa_to_contract', [1.0])[0]),
                    'contract_to_eoa': float(query.get('contract_to_eoa', [1.0])[0]),
                    'contract_to_contract': float(query.get('contract_to_contract', [1.0])[0])
                }
            }
            
            if params['gas_biased'] == 'none' or params['gas_biased'] == '':
                params['gas_biased'] = None
                
            walker = InteractiveWalker(CURRENT_TGRAPH.G, CURRENT_TGRAPH.min_time, CURRENT_TGRAPH.max_time, params)
            
            try:
                walk_steps = walker.walk(walk_length, start_node)
                self.send_json({
                    "success": True,
                    "start_node": start_node,
                    "walk_steps": walk_steps
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json({"error": str(e)}, 500)
            return
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"API Endpoint not found")

def run(port=8050):
    server_address = ('', port)
    httpd = HTTPServer(server_address, APIHandler)
    print(f"\n=======================================================")
    print(f"  T-EDGE TEMPORAL GRAPH VISUALIZATION SERVER ACTIVE")
    print(f"  Url: http://localhost:{port}")
    print(f"=======================================================\n")
    
    # Try to open browser
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
        
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visualization server...")
        httpd.server_close()

if __name__ == "__main__":
    # Load first available dataset on boot
    avail = get_available_datasets()
    if avail:
        print(f"Preloading default dataset: {avail[0]}")
        try:
            load_dataset(avail[0])
        except Exception as e:
            print(f"Could not preload {avail[0]}: {e}")
            
    run(8050)
