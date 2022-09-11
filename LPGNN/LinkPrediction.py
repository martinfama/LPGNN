from cgi import test
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
import torch_geometric as pyg

from .distances import *
from .graph_metrics import *


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

def sort_distance_file(filename, sort_dir='asc'):
    """ Sort a distance file by distance, and save it to a new file. Assumes
        the file is in the format of a list of 3-tuples, each containing i,j node 
        indices and the corresponding distance.

    Args:
        filename (_type_): The file to sort. The output file will be saved to the same directory, with the name {filename}_sorted.
        sort_dir (str, optional): whether to sort in ascending or descending order. Defaults to 'asc'.
    """
    if sort_dir == 'asc': sort_dir = ''
    else: sort_dir = 'r'
    os.system(f"LC_ALL=C sort -t',' -gs{sort_dir}k4 {filename} -o {filename}_sorted")

def precision_recall_score_file(data:pyg.data.Data, position_name:str, filename:str, chunk_size=10000, step_size=1, skip_file_prep=False, dist='poincare'):
    """ Generates a Precision-Recall curve, given a file of sorted scores.

    Args:
        data (pyg.data.Data): The graph to analyze. Is assumed to contain train and test masks.
        filename (str): The file to read predictions from. The file should be in the format of a list of 3-tuples, each containing i,j node indices and the corresponding score.

    Returns:
        tuple: a tuple given by (R, P, predictions) where R is the recall, P is the precision
               and predictions is the list of predictions.
    """

    total_test_edges = data.test_pos_edge_label_index.shape[1]
    
    N = data.num_nodes
    if not skip_file_prep:
        print('Generating test mask...')
        test_mask = th.zeros( N*(N-1)//2 )
        def d2_index_to_d1(i, j, N):
            """ Convert a 2D index to a 1D index, considering the upper triangular matrix (main diagonal not included). """
            # convert a 2d index to a 1d index, considering the upper triangular matrix (main diagonal not included)
            return i*N - th.div(i*(i+1), 2, rounding_mode='floor') + j - i - 1
            #return i*N + j - (i+1)*(i+2)/2
        # Set the test mask, which is passed to `hyperbolic_distance_list_to_file` to save if edge exists alongside the distance
        test_mask_idx = d2_index_to_d1(data.test_pos_edge_label_index[0], data.test_pos_edge_label_index[1], N).type(th.long)
        test_mask[test_mask_idx] = 1
        print("Total test mask: ", test_mask.sum(), " . Total test edges: ", total_test_edges)
        # save the predictions to a file, and sort it
        print('Generating hyperbolic distances and saving to file ', filename)
        hyperbolic_distance_list_to_file(getattr(data, position_name), chunk_size=chunk_size, filename=filename, extra_info_tensor=test_mask, dist=dist)
        print('Sorting file...')
        sort_distance_file(filename, sort_dir='asc')

    tp = 0
    fp = 0
    R = 0
    P = 0
    R_list = []
    P_list = []

    print('Reading file and generating precision-recall curve...')
    # read the file in chunks
    index = 0
    for chunk in pd.read_csv(filename+'_sorted', header=None, chunksize=step_size, iterator=True):
        # get the 3rd column, which is indicates whether the edge exists in the test set
        chunk = chunk.to_numpy()[:,2].astype(np.int)
        tp_ = chunk.sum() # number of true positives in this chunk
        tp += tp_ # add to the total number of true positives
        fp += chunk.shape[0] - tp_ # add to the total number of false positives
        P = tp / (tp + fp) 
        R = tp / total_test_edges
        print(f"\rindex: {index}, R: {R:.3f}, P: {P:.3f}", end='')
        R_list.append(R)
        P_list.append(P)
        # if all positive test cases have been accounted for (i.e. R = 1), break
        if tp == total_test_edges: return R_list, P_list
        index += chunk.shape[0]
    return R_list, P_list