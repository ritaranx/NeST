import faiss 
import numpy as np 
import os 

def inference_knn(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo,k,  gamma = 0.1, beta=0.1, prev_val = None):
    train_pred = np.array(train_pred)
    unlabeled_pred = np.array(unlabeled_pred)
    d = train_feat.shape[-1]
    index = faiss.IndexFlatL2(d)
    index.add(train_feat)
    D, I = index.search(unlabeled_feat, k)
    unlabeled_pred =  np.expand_dims(unlabeled_pred, axis = 1)
    # [#unlabel, 1]
    # train_pred[I] ---> [#unlabel, k]
    # print(unlabeled_pred.shape)
    score = np.log((1e-10 + train_pred[I])/ (1e-10 + unlabeled_pred)) * train_pred[I]
    # print(score.shape)
    mean_kl = np.mean(np.sum(score, axis = -1), axis = -1)

    # mean_mse =  np.mean((train_pred[I] - unlabeled_pred)**2, axis = -1)
    # train pred (n_samples, n_class)
    # train pred[I] (n_samples, n_neighbor, n_class)
    var_mse =  np.var(train_pred[I], axis = -1)

    if prev_val is not None:
        current_val = prev_val * gamma + (1- gamma) * (mean_kl + var_mse * beta)
    else:
        current_val = mean_kl + var_mse * beta
    idx = np.argsort(current_val)

    return idx

def inference_conf(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, gamma = 0.1, prev_val = None):
    train_pred = np.array(train_pred)
    unlabeled_pred = np.array(unlabeled_pred)
    current_val = -np.max(unlabeled_pred, axis = -1)
    if prev_val is not None:
        current_val = prev_val * gamma + (1- gamma) * (current_val)
    else:
        current_val = current_val
    idx = np.argsort(current_val)

    return idx

def inference_uncertainty(unlabeled_label, unlabeled_pseudo, mutual_info, gamma = 0.1, prev_val = None):
    if prev_val is not None:
        current_val = prev_val * gamma + (1- gamma) * (mutual_info)
    else:
        current_val = mutual_info
    idx = np.argsort(current_val)
   
    return idx

def save_data(train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo, dataset = 'agnews',  n_labels = 10, n_iter = 0):
    if n_iter == 0:
        path = f"{dataset}/{n_labels}"
        
    else:
        path = f"{dataset}/{n_labels}_{n_iter}"
    os.makedirs(path, exist_ok = True)
    
    with open(f"{path}/train_pred.npy", 'wb') as f:
        np.save(f, train_pred)
    
    with open(f"{path}/train_feat.npy", 'wb') as f:
        np.save(f, train_feat)
    
    with open(f"{path}/train_label.npy", 'wb') as f:
        np.save(f, train_label)

    with open(f"{path}/unlabeled_pred.npy", 'wb') as f:
        np.save(f, unlabeled_pred)

    with open(f"{path}/unlabeled_feat.npy", 'wb') as f:
        np.save(f, unlabeled_feat)
    
    with open(f"{path}/unlabeled_label.npy", 'wb') as f:
        np.save(f, unlabeled_label)
    
    with open(f"{path}/unlabeled_pseudo.npy", 'wb') as f:
        np.save(f, unlabeled_pseudo)




def load_pred_data(dataset = 'agnews', n_labels = 10, n_iter = 0):
    # os.makedirs(f"{dataset}/{n_labels}", exist_ok = True)
    # with open(f"{dataset}/{n_labels}/train_pred.npy", 'rb') as f:
    if n_iter == 0:
        path = f"{dataset}/{n_labels}"
    else:
        path = f"{dataset}/{n_labels}_{n_iter}"
    train_pred = np.load(f"{path}/train_pred.npy")

    train_feat = np.load(f"{path}/train_feat.npy")

    train_label = np.load(f"{path}/train_label.npy")

    unlabeled_pred = np.load(f"{path}/unlabeled_pred.npy")

    unlabeled_feat = np.load(f"{path}/unlabeled_feat.npy")

    unlabeled_label = np.load(f"{path}/unlabeled_label.npy")
    
    unlabeled_pseudo = np.load(f"{path}/unlabeled_pseudo.npy")

    return train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_pseudo
