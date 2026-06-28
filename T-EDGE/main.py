import os
from tGraphNE import tGraphNE
from tGraph import tGraph
from link_prediction import LP

def log_result(dataset, walks, method, roc, ap):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "ab_test_results.csv")
    with open(csv_path, "a") as f:
        f.write(f"{dataset},{walks},{method},{roc:.6f},{ap:.6f}\n")

def main():
    datasets = ["Eth_0_999k", "Eth_1M_1_99M"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filetype = "pkl_f"
    
    # Parameter setting
    dimensions = 128
    window_size = 4
    workers = 8
    num_walks = 10
    walk_length = 10
    
    # Generate Temporal Walks (WS1 + WS5): time_close_exp and amount_exp with alpha=0.5
    first_biased_type = "time_uniform"
    time_biased_type = "time_close_exp"
    amount_biased = "amount_exp"
    alpha = 0.5
    edge_score_mode = "append"
    model_type = "XGBoost"
    gamma = 0.2
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"Executing A/B Testing Pipeline for Dataset: {dataset}")
        print(f"{'#'*60}")
        
        file_ = os.path.join(script_dir, dataset + "_df_train_0.5.pickle")
        if not os.path.exists(file_):
            print(f"Dataset {file_} not found. Please run prepare_dataset.py first.")
            continue
            
        out_dir = os.path.join(script_dir, "..", "data", dataset)
        os.makedirs(out_dir, exist_ok=True)
        
        # RUN 1: Baseline (Control)
        print("\n========== RUN 1: Baseline (Control) ==========")
        tG_baseline = tGraph(file_, filetype, exclude_failed=False, graph_mode="baseline")
        print(tG_baseline)
        output_1 = os.path.join(out_dir, f"vec_baseline_{dataset}.txt")
        tGraphNE(tG_baseline, time_biased_type, first_biased_type, amount_biased, alpha=0.5, dimensions=dimensions, num_walks=num_walks, walk_length=walk_length, output=output_1)
        roc_1, ap_1 = LP(dataset, output_1, edge_score_mode, tG=tG_baseline, model_type=model_type, use_gas_features=False)
        log_result(dataset, num_walks, "Baseline", roc_1, ap_1)
        
        # RUN 2: Contract Failure-Rate Bias
        print("\n========== RUN 2: Contract Failure-Rate Bias ==========")
        tG_fail_rate = tGraph(file_, filetype, exclude_failed=False, graph_mode="baseline")
        print(tG_fail_rate)
        output_2 = os.path.join(out_dir, f"vec_fail_rate_{dataset}.txt")
        tGraphNE(tG_fail_rate, time_biased_type, first_biased_type, amount_biased, alpha=0.4, dimensions=dimensions, num_walks=num_walks, walk_length=walk_length, output=output_2, gas_biased="fail_rate", gamma=0.2)
        roc_2, ap_2 = LP(dataset, output_2, edge_score_mode, tG=tG_fail_rate, model_type=model_type, use_gas_features=False)
        log_result(dataset, num_walks, "Contract Failure-Rate", roc_2, ap_2)
        
        # RUN 3: Bipartite Folding (Router Bypass)
        print("\n========== RUN 3: Bipartite Folding (Router Bypass) ==========")
        tG_bypass = tGraph(file_, filetype, exclude_failed=False, graph_mode="router_bypass", delta_t=120)
        print(tG_bypass)
        output_3 = os.path.join(out_dir, f"vec_router_bypass_{dataset}.txt")
        tGraphNE(tG_bypass, time_biased_type, first_biased_type, amount_biased, alpha=0.5, dimensions=dimensions, num_walks=num_walks, walk_length=walk_length, output=output_3)
        roc_3, ap_3 = LP(dataset, output_3, edge_score_mode, tG=tG_bypass, model_type=model_type, use_gas_features=False)
        log_result(dataset, num_walks, "Bipartite Folding", roc_3, ap_3)
        
        # RUN 4: Gas-Friction
        print("\n========== RUN 4: Gas-Friction ==========")
        tG_friction = tGraph(file_, filetype, exclude_failed=False, graph_mode="baseline")
        print(tG_friction)
        output_4 = os.path.join(out_dir, f"vec_friction_{dataset}.txt")
        tGraphNE(tG_friction, time_biased_type, first_biased_type, amount_biased, alpha=0.4, dimensions=dimensions, num_walks=num_walks, walk_length=walk_length, output=output_4, gas_biased="friction_inverse", gamma=0.2)
        roc_4, ap_4 = LP(dataset, output_4, edge_score_mode, tG=tG_friction, model_type=model_type, use_gas_features=False)
        log_result(dataset, num_walks, "Gas-Friction", roc_4, ap_4)
        
        # RUN 5: Above-Median Gas Premium
        print("\n========== RUN 5: Above-Median Gas Premium ==========")
        tG_rel_premium = tGraph(file_, filetype, exclude_failed=False, graph_mode="baseline")
        print(tG_rel_premium)
        output_5 = os.path.join(out_dir, f"vec_rel_premium_{dataset}.txt")
        tGraphNE(tG_rel_premium, time_biased_type, first_biased_type, amount_biased, alpha=0.4, dimensions=dimensions, num_walks=num_walks, walk_length=walk_length, output=output_5, gas_biased="relative_premium", gamma=0.2)
        roc_5, ap_5 = LP(dataset, output_5, edge_score_mode, tG=tG_rel_premium, model_type=model_type, use_gas_features=False)
        log_result(dataset, num_walks, "Relative Gas Premium", roc_5, ap_5)

        # Comparative Report
        print(f"\n{'='*120}")
        print(f"FINAL COMPARATIVE REPORT: {dataset}")
        print(f"{'='*120}")
        print(f"Metric\t\tBaseline\tFailure-Rate\tBypass\t\tFriction\tRelGasPrem")
        print(f"ROC-AUC\t\t{roc_1:.6f}\t\t{roc_2:.6f}\t\t{roc_3:.6f}\t\t{roc_4:.6f}\t\t{roc_5:.6f}")
        print(f"Avg Precision\t{ap_1:.6f}\t\t{ap_2:.6f}\t\t{ap_3:.6f}\t\t{ap_4:.6f}\t\t{ap_5:.6f}")
        print(f"{'='*120}\n")

if __name__ == "__main__":
    main()
