import os
import sys
import random
import numpy as np
import networkx as nx

# Add T-EDGE to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'T-EDGE'))
from tGraph import tGraph

def get_k_order_subgraph(G, start_nodes, k_in=1, k_out=1, max_nodes=60):
    """Sample a compact subgraph so the visual plot is highly readable and clean."""
    sampled_nodes = set(start_nodes)
    
    # Forward BFS
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
                    
    # Backward BFS
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
                    
    return G.subgraph(sampled_nodes).copy()

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
        comb = np.multiply(
            np.multiply(np.power(probs1, alpha), np.power(probs2, beta)),
            np.power(probs3, gamma)
        )
    else:
        comb = np.multiply(np.power(probs1, alpha), np.power(probs2, 1.0 - alpha))
    return comb.tolist()

def weight_choice(probs):
    if len(probs) == 0:
        return -1
    s = sum(probs)
    if s == 0:
        return random.randint(0, len(probs) - 1)
    normalized = [p / s for p in probs]
    r = random.random()
    accum = 0.0
    for idx, p in enumerate(normalized):
        accum += p
        if r <= accum:
            return idx
    return len(probs) - 1

class StaticWalker:
    """Replicates the random walk logic for static image logging."""
    def __init__(self, G, min_time, max_time, gas_biased=None, alpha=0.5, gamma=0.0, state_penalty=1.0):
        self.G = G
        self.min_time = min_time
        self.max_time = max_time
        self.gas_biased = gas_biased
        self.alpha = alpha
        self.gamma = gamma
        self.state_penalty = state_penalty

    def walk(self, walk_length, start_node):
        if start_node not in self.G:
            return []
            
        walk = [start_node]
        walk_times = []
        cur = start_node
        
        # Step 1
        nxt, nxt_t = self.get_first_step(cur)
        if nxt is not None:
            walk.append(nxt)
            walk_times.append(nxt_t)
            cur = nxt
            prevtime = nxt_t
        else:
            return walk
            
        # Steps 2+
        while len(walk) < walk_length:
            nxt, nxt_t = self.get_next_step(cur, prevtime)
            if nxt is not None:
                walk.append(nxt)
                walk_times.append(nxt_t)
                cur = nxt
                prevtime = nxt_t
            else:
                break
        return walk

    def get_first_step(self, cur):
        G = self.G
        tmp_node = []
        tmp_time = []
        tmp_key = []
        unnormalized_probs_t = []
        
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) == 0:
            return None, None
            
        for nbr in cur_nbrs:
            nbr_keys = list(G.get_edge_data(cur, nbr))
            for k in nbr_keys:
                t = k
                unnormalized_probs_t.append(1.0)
                tmp_node.append(nbr)
                tmp_time.append(t)
                tmp_key.append(k)
                
        scaled_probs = []
        for i in range(len(unnormalized_probs_t)):
            prob = float(unnormalized_probs_t[i])
            nbr = tmp_node[i]
            k = tmp_key[i]
            edge_data = G[cur][nbr][k]
            is_failed = edge_data.get('is_failed', 0)
            penalty = self.state_penalty if is_failed == 1 else 1.0
            scaled_probs.append(prob * penalty)
            
        selected = weight_choice(scaled_probs)
        if selected == -1:
            return None, None
        return tmp_node[selected], tmp_time[selected]

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
            return None, None
            
        for nbr in cur_nbrs:
            nbr_keys = list(G.get_edge_data(cur, nbr))
            for k in nbr_keys:
                t = k
                a = G[cur][nbr][k].get('weight', 0.0)
                
                if t >= prevtime:
                    unnormalized_probs_a.append(float(a))
                    if self.gas_biased == "friction_inverse":
                        unnormalized_probs_g.append(float(G[cur][nbr][k].get('friction', 1.0)))
                    else:
                        unnormalized_probs_g.append(1.0)
                    unnormalized_probs_t.append(float(t - prevtime))
                    
                    tmp_node.append(nbr)
                    tmp_time.append(t)
                    tmp_key.append(k)
                    
        if len(unnormalized_probs_t) == 0:
            return None, None
            
        # Process component probabilities (Softmax equivalent)
        exp_t = np.exp(unnormalized_probs_t - np.max(unnormalized_probs_t))
        unnormalized_probs_t = (exp_t / exp_t.sum()).tolist()
        
        exp_a = np.exp(unnormalized_probs_a - np.max(unnormalized_probs_a))
        unnormalized_probs_a = (exp_a / exp_a.sum()).tolist()
        
        if self.gas_biased == "friction_inverse" and len(unnormalized_probs_g) > 0:
            max_f = max(unnormalized_probs_g)
            if max_f == 0: 
                max_f = 1.0
            unnormalized_probs_g = [max(1e-6, 1.0 - (f / max_f)) for f in unnormalized_probs_g]
            
        # Combine
        if self.gas_biased == "friction_inverse":
            unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha, unnormalized_probs_g, self.gamma)
        else:
            unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha)
            
        # Apply State-Penalty
        scaled_probs = []
        for i in range(len(unnormalized_probs)):
            prob = float(unnormalized_probs[i])
            nbr = tmp_node[i]
            k = tmp_key[i]
            edge_data = G[cur][nbr][k]
            
            is_failed = edge_data.get('is_failed', 0)
            penalty = self.state_penalty if is_failed == 1 else 1.0
            scaled_probs.append(prob * penalty)
            
        selected = weight_choice(scaled_probs)
        if selected == -1:
            return None, None
        return tmp_node[selected], tmp_time[selected]

def generate_static_plot(tG, start_node, output_filename, walk_type="baseline"):
    import matplotlib.pyplot as plt
    
    G = tG.G
    
    # 1. Simulate Walk
    if walk_type == "baseline":
        walker = StaticWalker(G, tG.min_time, tG.max_time, gas_biased=None, alpha=0.5, state_penalty=1.0)
        title_str = "T-EDGE Temporal Walk: Baseline Strategy"
    elif walk_type == "folding":
        walker = StaticWalker(G, tG.min_time, tG.max_time, gas_biased=None, alpha=0.5, state_penalty=1.0)
        title_str = "T-EDGE Temporal Walk: Bipartite Folding (Router Bypass) Strategy"
    else:
        # State Penalty (0.2) + Friction Inverse (gamma=0.2)
        walker = StaticWalker(G, tG.min_time, tG.max_time, gas_biased="friction_inverse", alpha=0.4, gamma=0.2, state_penalty=0.2)
        title_str = "T-EDGE Temporal Walk: State-Penalty & Friction-Gas Strategy"
        
    print(f"Simulating {walk_type} temporal random walk...")
    walk_path = walker.walk(walk_length=15, start_node=start_node)
    print(f"Generated Walk Path: {' -> '.join(walk_path)}")
    
    # 2. Extract Subgraph around the walk nodes so the drawing is beautiful and clear
    subG = get_k_order_subgraph(G, walk_path, k_in=1, k_out=1, max_nodes=50)
    
    # 3. Setup coordinates using force-directed layout
    plt.figure(figsize=(12, 10), facecolor='#0b0d19')
    ax = plt.gca()
    ax.set_facecolor('#0b0d19')
    
    # We use spring_layout for attractive layouting
    pos = nx.spring_layout(subG, k=0.35, seed=42)
    
    # 4. Classify nodes in subgraph
    eoa_nodes = []
    contract_nodes = []
    node_colors = []
    
    for node in subG.nodes():
        is_contract = subG.nodes[node].get('is_contract', 0) == 1
        if is_contract:
            contract_nodes.append(node)
        else:
            eoa_nodes.append(node)
            
    # Draw Background EOA Nodes (indigo blue)
    nx.draw_networkx_nodes(
        subG, pos, 
        nodelist=[n for n in eoa_nodes if n not in walk_path],
        node_color='#3b82f6', 
        node_shape='o',
        node_size=200, 
        alpha=0.6,
        edgecolors='#60a5fa',
        linewidths=1.0
    )
    
    # Draw Background Contract Nodes (magenta pink)
    nx.draw_networkx_nodes(
        subG, pos, 
        nodelist=[n for n in contract_nodes if n not in walk_path],
        node_color='#ec4899', 
        node_shape='h',
        node_size=250, 
        alpha=0.7,
        edgecolors='#f472b6',
        linewidths=1.5
    )
    
    # Highlight Nodes in Walk Path (gold/yellow)
    nx.draw_networkx_nodes(
        subG, pos, 
        nodelist=walk_path,
        node_color='#fbbf24', 
        node_shape='o',
        node_size=350, 
        edgecolors='#ffffff',
        linewidths=2.0
    )
    
    # 5. Draw normal background edges
    normal_edges = []
    failed_edges = []
    folded_edges = []
    
    for u, v, k, attr in subG.edges(keys=True, data=True):
        # Don't draw walk edges as background edges
        is_walk_edge = False
        for i in range(len(walk_path)-1):
            if (u == walk_path[i] and v == walk_path[i+1]) or (v == walk_path[i] and u == walk_path[i+1]):
                is_walk_edge = True
                break
        
        if not is_walk_edge:
            if attr.get('router') is not None:
                folded_edges.append((u, v))
            elif attr.get('is_failed', 0) == 1:
                failed_edges.append((u, v))
            else:
                normal_edges.append((u, v))
                
    # Draw successful normal edges
    nx.draw_networkx_edges(
        subG, pos,
        edgelist=normal_edges,
        edge_color='#4b5563',
        width=1.0,
        alpha=0.4,
        arrows=True,
        arrowsize=10
    )
    
    # Draw failed normal edges (dashed red)
    nx.draw_networkx_edges(
        subG, pos,
        edgelist=failed_edges,
        edge_color='#ef4444',
        width=1.0,
        style='dashed',
        alpha=0.4,
        arrows=True,
        arrowsize=10
    )
    
    # Draw bypassed folded edges (cyan)
    if folded_edges:
        nx.draw_networkx_edges(
            subG, pos,
            edgelist=folded_edges,
            edge_color='#06b6d4',
            width=1.5,
            alpha=0.6,
            arrows=True,
            arrowsize=10
        )
    
    # 6. Draw Walk Path Edges (Gold/Yellow thick lines)
    walk_edges = [(walk_path[i], walk_path[i+1]) for i in range(len(walk_path)-1)]
    nx.draw_networkx_edges(
        subG, pos,
        edgelist=walk_edges,
        edge_color='#f59e0b',
        width=3.5,
        alpha=0.9,
        arrows=True,
        arrowsize=16
    )
    
    # 7. Add Labels (Node IDs)
    labels = {node: str(node) for node in subG.nodes()}
    nx.draw_networkx_labels(
        subG, pos,
        labels=labels,
        font_color='#ffffff',
        font_size=8,
        font_family='sans-serif',
        font_weight='bold'
    )
    
    # Add step order overlays on walk nodes
    for step_idx, node in enumerate(walk_path):
        x, y = pos[node]
        plt.text(
            x, y + 0.08, 
            s=f"#{step_idx}", 
            color='#fbbf24', 
            fontsize=10, 
            fontweight='bold',
            horizontalalignment='center',
            bbox=dict(facecolor='#0b0d19', edgecolor='#fbbf24', boxstyle='round,pad=0.2', alpha=0.9)
        )
        
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', label='EOA Node', markerfacecolor='#3b82f6', markeredgecolor='#60a5fa', markersize=10),
        Line2D([0], [0], marker='h', color='none', label='Contract Node', markerfacecolor='#ec4899', markeredgecolor='#f472b6', markersize=12),
        Line2D([0], [0], marker='o', color='none', label='Walk Traversed Node', markerfacecolor='#fbbf24', markeredgecolor='#ffffff', markersize=12),
        Line2D([0], [0], color='#4b5563', lw=1, label='Successful Tx'),
        Line2D([0], [0], color='#ef4444', lw=1, ls='--', label='Failed Tx'),
    ]
    if walk_type == "folding":
        legend_elements.append(
            Line2D([0], [0], color='#06b6d4', lw=1.5, label='Bypassed Contract Hop')
        )
    legend_elements.append(
        Line2D([0], [0], color='#f59e0b', lw=3, label='Walk Path Highlight')
    )
    plt.legend(handles=legend_elements, loc='upper left', facecolor='#161a34', edgecolor='#2e3456', labelcolor='#ffffff', fontsize=10)
    
    plt.title(title_str, color='#ffffff', fontsize=14, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(output_filename, facecolor='#0b0d19', dpi=150)
    plt.close()
    print(f"Success! Saved static visualization plot to {output_filename}")

if __name__ == "__main__":
    import os
    dataset = "Eth_1M_1_99M"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "T-EDGE", f"{dataset}_df_train_0.5.pickle")
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        sys.exit(1)
        
    print(f"Loading graph dataset (baseline mode): {dataset}...")
    tG_baseline = tGraph(file_path, "pkl_f", exclude_failed=False, graph_mode="baseline")
    
    print(f"Loading graph dataset (router_bypass mode): {dataset}...")
    tG_bypass = tGraph(file_path, "pkl_f", exclude_failed=False, graph_mode="router_bypass", delta_t=120)
    
    # Pick a start node with relatively high degree from baseline
    degrees = sorted(tG_baseline.G.degree(), key=lambda x: x[1], reverse=True)
    start_node = degrees[0][0]
    print(f"Selected starting node for walks: {start_node} (degree: {tG_baseline.G.degree(start_node)})")
    
    # Generate Baseline walk plot
    output_1 = os.path.join(script_dir, "baseline_walk.png")
    generate_static_plot(tG_baseline, start_node, output_1, walk_type="baseline")
    
    # Generate Friction-Gas walk plot
    output_2 = os.path.join(script_dir, "friction_walk.png")
    generate_static_plot(tG_baseline, start_node, output_2, walk_type="friction")
    
    # Generate Bipartite Folding walk plot
    output_3 = os.path.join(script_dir, "bypassed_walk.png")
    # Ensure start node exists in the folded graph, or fallback to its highest degree node
    start_node_bypass = start_node if start_node in tG_bypass.G else sorted(tG_bypass.G.degree(), key=lambda x: x[1], reverse=True)[0][0]
    generate_static_plot(tG_bypass, start_node_bypass, output_3, walk_type="folding")
    
    print("\nVisualizations are completed successfully!")
    print(f"1. Baseline walk image: {output_1}")
    print(f"2. State-Penalty + Friction Gas walk image: {output_2}")
    print(f"3. Bipartite Folding (Router Bypass) walk image: {output_3}")
