
import pickle as pickle
import os
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score, classification_report

from load_embeddings import load_embeddings
from get_edge_embeddings import get_edge_embeddings

def LP(dataset, emb_file, edge_score_mode, tG=None, model_type="SVM", use_gas_features=False):
    print("dataset", dataset)
    print("emb_file", emb_file)
    print("edge_score_mode", edge_score_mode)
    print("model_type", model_type)
    print("use_gas_features", use_gas_features)

    emb_matrix = load_embeddings(emb_file)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pklfile_train_test_split = os.path.join(script_dir, dataset + "_train_test_split_0.5.pickle")
    if os.path.exists(pklfile_train_test_split):    
        with open( pklfile_train_test_split,"rb") as f:     
            train_test_split = pickle.load(f)
            train_edges_pos = train_test_split['train_edges_pos']
            train_edges_false = train_test_split['train_edges_false']
            test_edges_pos = train_test_split['test_edges_pos']
            test_edges_false = train_test_split['test_edges_false']            
        print("train_test_split{} is loaded as pkl file")   
        print("len(train_edges_pos)", len(train_edges_pos) )
        print("len(train_edges_false)", len(train_edges_false) )
        print("len(test_edges_pos)", len(test_edges_pos) )
        print("len(test_edges_false)", len(test_edges_false) )

    else:           
        print("ERROR")     
        print(pklfile_train_test_split, "is not exists") 
        print("Excuting function LP_tGraph first")  
        
    print("get_edge_embeddings for train data...")    
    pos_train_edge_embs = get_edge_embeddings( train_edges_pos, emb_matrix, edge_score_mode, use_gas_features=use_gas_features, G=tG.G if tG else None )   
    neg_train_edge_embs = get_edge_embeddings( train_edges_false, emb_matrix, edge_score_mode, use_gas_features=use_gas_features, G=tG.G if tG else None )   
    train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

    # Create train-set edge labels: 1 = real edge, 0 = false edge
    train_edge_labels = np.concatenate([np.ones(len(pos_train_edge_embs), int), np.zeros(len(neg_train_edge_embs), int)])
    #print("train_edge_labels", train_edge_labels)
    print("edge_classifier training...")    
    
    if model_type == "XGBoost":
        edge_classifier = XGBClassifier(eval_metric='logloss')
    else:
        edge_classifier = SVC(kernel='linear',C=0.4)
    
    print(f"X_train shape: {np.shape(train_edge_embs)}")
    print(f"Contains NaN: {np.isnan(train_edge_embs).any()}")
    print(f"Contains Inf: {np.isinf(train_edge_embs).any()}")
    if np.isnan(train_edge_embs).any() or np.isinf(train_edge_embs).any():
        raise Exception("Validation Failed: X_train contains NaN or Inf!")

    print(f"ACTIVE CLASSIFIER TYPE: {type(edge_classifier)}")
    edge_classifier.fit(train_edge_embs, train_edge_labels)
    print("edge_classifier finish training!")
    
    if model_type == "XGBoost":
        importances = edge_classifier.feature_importances_
        if use_gas_features:
            gas_price_weight = sum(importances[-6:-4])
            gas_limit_weight = sum(importances[-4:-2])
            error_ratio_weight = sum(importances[-2:])
            embedding_weight = sum(importances[:-6])
            print("\n" + "="*50)
            print("XGBOOST FEATURE IMPORTANCE SCORES:")
            print("="*50)
            print(f"  - Graph Embeddings (latent representations) : {embedding_weight:.4f}")
            print(f"  - Gas Price (transaction fee rate)          : {gas_price_weight:.4f}")
            print(f"  - Gas Limit (computational budget limit)    : {gas_limit_weight:.4f}")
            print(f"  - Error Ratio (historical failed ratio)     : {error_ratio_weight:.4f}")
            print("="*50 + "\n")
        else:
            embedding_weight = sum(importances)
            print("\n" + "="*50)
            print("XGBOOST FEATURE IMPORTANCE SCORES:")
            print("="*50)
            print(f"  - Graph Embeddings (latent representations) : {embedding_weight:.4f}")
            print("="*50 + "\n")
    
    # Test-set edge embeddings, labels
    print("get_edge_embeddings for test data...")  
    pos_test_edge_embs = get_edge_embeddings(test_edges_pos, emb_matrix, edge_score_mode, use_gas_features=use_gas_features, G=tG.G if tG else None )   
    neg_test_edge_embs = get_edge_embeddings(test_edges_false, emb_matrix, edge_score_mode, use_gas_features=use_gas_features, G=tG.G if tG else None )   
    test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

    # Create val-set edge labels: 1 = real edge, 0 = false edge
    test_edge_labels = np.concatenate([np.ones(len(pos_test_edge_embs), int), np.zeros(len(neg_test_edge_embs), int)])
    #print("test_edge_labels", test_edge_labels)
    # Predicted edge scores: probability of being of class "1" (real edge)
    print("predicting...")
    #test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]
    test_preds = edge_classifier.predict(test_edge_embs)
    #print("test_preds", test_preds)

    test_roc = roc_auc_score(test_edge_labels, test_preds)
    print("test_roc", test_roc)
    # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
    test_ap = average_precision_score(test_edge_labels, test_preds)
    print("test_ap", test_ap) 
    print(classification_report(test_edge_labels, test_preds))

    print("Finishing prediction!")

    return test_roc, test_ap



