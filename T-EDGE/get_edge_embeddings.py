# -*- coding: utf-8 -*-
import numpy as np
from utils import sigmoid, softmax,  tanh

def get_edge_embeddings(edge_list, emb_matrix, edge_score_mode, use_gas_features=False, G=None):
    
    embs = []
    
    node_avg_gas_price = {}
    global_avg_gas_price = 1e-5
    
    node_avg_gas_limit = {}
    global_avg_gas_limit = 21000.0
    
    node_error_ratio = {}
    global_avg_error_ratio = 0.0
    
    if use_gas_features and G is not None:
        node_gas_prices = {}
        node_gas_limits = {}
        node_failed_flags = {}
        
        for u, v, k, d in G.edges(keys=True, data=True):
            gas_price = float(d.get('gas', 0.0))
            gas_limit = float(d.get('gas_limit', 21000.0))
            is_failed = float(d.get('is_failed', 0.0))
            
            for node in (u, v):
                if node not in node_gas_prices:
                    node_gas_prices[node] = []
                    node_gas_limits[node] = []
                    node_failed_flags[node] = []
                node_gas_prices[node].append(gas_price)
                node_gas_limits[node].append(gas_limit)
                node_failed_flags[node].append(is_failed)
                
        all_gas_price_avgs = []
        for node, prices in node_gas_prices.items():
            avg_p = np.mean(prices)
            node_avg_gas_price[node] = avg_p
            all_gas_price_avgs.append(avg_p)
        if all_gas_price_avgs:
            global_avg_gas_price = np.mean(all_gas_price_avgs)
            
        all_gas_limit_avgs = []
        for node, limits in node_gas_limits.items():
            avg_l = np.mean(limits)
            node_avg_gas_limit[node] = avg_l
            all_gas_limit_avgs.append(avg_l)
        if all_gas_limit_avgs:
            global_avg_gas_limit = np.mean(all_gas_limit_avgs)
            
        all_error_ratios = []
        for node, flags in node_failed_flags.items():
            avg_err = np.mean(flags)
            node_error_ratio[node] = avg_err
            all_error_ratios.append(avg_err)
        if all_error_ratios:
            global_avg_error_ratio = np.mean(all_error_ratios)
            
    for edge in edge_list:
        node1 = str(edge[0])
        node2 = str(edge[1])
        
        emb1 = np.array(emb_matrix[node1]) if node1 in emb_matrix else np.zeros(128)
        emb2 = np.array(emb_matrix[node2]) if node2 in emb_matrix else np.zeros(128)
        
        if edge_score_mode == "multiply":        
            baseline_features = np.multiply(emb1, emb2)
        elif edge_score_mode == "subtract":   
            baseline_features = np.subtract(emb1, emb2)
        elif edge_score_mode == "subtract_sigmoid":   
            baseline_features = sigmoid(np.subtract(emb1, emb2))
        elif edge_score_mode == "subtract_tanh":   
            baseline_features = tanh(np.subtract(emb1, emb2))
        elif edge_score_mode == "append":   
            baseline_features = np.concatenate([emb1, emb2])
        else:
            print("ERROR!!! No suitable edge_score_mode")
            baseline_features = np.concatenate([emb1, emb2])
            
        if use_gas_features and G is not None:
            # Look up node-level historical averages
            u_price = node_avg_gas_price.get(node1, global_avg_gas_price)
            v_price = node_avg_gas_price.get(node2, global_avg_gas_price)
            
            u_limit = node_avg_gas_limit.get(node1, global_avg_gas_limit)
            v_limit = node_avg_gas_limit.get(node2, global_avg_gas_limit)
            
            u_err = node_error_ratio.get(node1, global_avg_error_ratio)
            v_err = node_error_ratio.get(node2, global_avg_error_ratio)
            
            # Ensure safety
            u_price_safe = np.nan_to_num(u_price, nan=0.0, posinf=0.0, neginf=0.0)
            v_price_safe = np.nan_to_num(v_price, nan=0.0, posinf=0.0, neginf=0.0)
            
            u_limit_safe = np.nan_to_num(u_limit, nan=0.0, posinf=0.0, neginf=0.0)
            v_limit_safe = np.nan_to_num(v_limit, nan=0.0, posinf=0.0, neginf=0.0)
            
            u_err_safe = np.nan_to_num(u_err, nan=0.0, posinf=0.0, neginf=0.0)
            v_err_safe = np.nan_to_num(v_err, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Concatenate baseline features + 6 gas price, gas limit, and error ratio features
            final_feature_vector = np.append(baseline_features, [
                u_price_safe, v_price_safe, 
                u_limit_safe, v_limit_safe, 
                u_err_safe, v_err_safe
            ])
        else:
            final_feature_vector = baseline_features
            
        embs.append(final_feature_vector)
        
    return np.array(embs)