import numpy as np
import torch

import matplotlib.pyplot as plt

def plot_pr_curves(PR_list, save_name=''):
    ## Plot results, and log
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.grid(alpha=0.5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    for PR in PR_list:
        ax.plot(PR['recall'], PR['precision'], label=PR['label'])
    #ax.plot(PR_LaBNE['recall'], PR_LaBNE['precision'], label='LaBNE')
    #ax.plot(R_SAGE, P_SAGE, label='GraphSAGE')
    #ax.plot(R_SAGE_all, P_SAGE_all, label='GraphSAGE+Attr')
    ax.legend(loc='upper right')

    #ax.set_title(f'PS network: N={N}, avg_k={avg_k}, gamma={gamma}, Temp={Temp}, seed={seed}')

    fig.savefig('./figs/PR/'+save_name+'.pdf', bbox_inches='tight')
    plt.close()

def precision_recall(test_data, val_data, predictions):
    """Generates a Precision-Recall curve, given the predictions 
    and the test data. tp (true positive) cases are added until 
    recall is 1 (tp == test_pos_edge_cases).

    Args:
        test_data (torch.data): _description_
        predictions (list): a list of 3-tuples, each containing i,j node indices and the predicted score (sorted).

    Returns:
        tuple: a tuple given by (R, P, predictions) where R is the recall, P is the precision and predictions is the passed list of predictions.
    """

    # Concatenate test and validation edges
    total_index = np.concatenate((test_data.pos_edge_label_index.T.detach().numpy(), val_data.pos_edge_label_index.T.detach().numpy()), axis=0)

    R = [] #recall
    P = [] #precision
    tp = 0 #true positive cases (i.e. correctly predicted positive cases)
    fp = 0 #false positive cases (i.e. incorrectly predicted positive cases)
    for i, p in enumerate(predictions):
        # this checks if the prediction is a true positive by looking for it
        # in the test_data positive edge cases
        if np.any(np.all(np.array([p[0], p[1]]) == total_index, axis=1)):
            tp += 1
        else:
            fp += 1
        P.append(tp / (tp + fp))
        R.append(tp / total_index.shape[0])
        # if all positive test cases have been accounted for (i.e. R = 1), break
        if tp == total_index.shape[0]:
            break
    return R, P, predictions

def precision_recall_trained_model(model, train_data, val_data, test_data, norm='prob_adj'):
    """Generates a Precision-Recall curve, given the trained model, train data and test data.
    For now, this assumes the model generates a per-node feature vector (``z``), and uses that
    to generate a probability matrix (``prob_adj``) which indicates the probability of an edge 
    being present. 

    Args:
        model (torch.nn.Model): A GNN model, already trained.
        train_data (torch.Tensor): Train data
        test_data (torch.Tensor): Associated test data.

    Returns:
        tuple: a tuple given by (R, P, predictions) where R is the recall, P is the precision 
               and predictions is the list of predictions.
    """
    z = model.forward(train_data.x, train_data.pos_edge_label_index).detach().numpy() #forward
    # generate a probability matrix
    prob_adj = z @ z.T
    # prob_list will be built by scanning the entries of the probability matrix and sorting
    prob_list = []
    for i in range(prob_adj.shape[0]):
        for j in range(i+1, prob_adj.shape[0]):
            # don't include train data edges in the list
            if not np.any(np.all(np.array([i, j]) == train_data.pos_edge_label_index.T.detach().numpy(), axis=1)):
                if norm == 'prob_adj':
                    prob_list.append([i,j, float(prob_adj[i,j])])
                elif norm == 'dist':
                    prob_list.append([i,j, float(np.linalg.norm(z[i] - z[j]))])
    # sort the list by probability
    prob_list = sorted(prob_list, key=lambda x: x[2], reverse=True)
    return precision_recall(test_data, val_data, prob_list)