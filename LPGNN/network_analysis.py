import igraph
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.optimize
import collections
from . import popularity_similarity as pop_sim

import copy

import os

import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.neighbors
from sklearn.metrics import PrecisionRecallDisplay

from .graph_metrics import *

from . import debug_print

#power law function. used to determine a scale free network's scaling exponent, gamma.
def power_law(x, A, gamma):
    return A*x**(-gamma)

def infer_gamma(graph=None):
    graph_bins_transposed = np.transpose([x for x in list(graph.degree_distribution().bins()) if x[2] > 1])
    popt, pcov = scipy.optimize.curve_fit(f=power_law, xdata=graph_bins_transposed[0], ydata=graph_bins_transposed[2], p0=[1, 2])
    return popt[1] #return gamma

#plot the distribution of node degrees in the graph
#x-axis: node degree
#y-axis: # of occurrences of that degree
#! Logarithmic bins....
def graph_degree_plot(graph=None, logScale=False, fit_power_law=False, A=None, gamma=None):
    graph_bins_transposed = np.transpose(list(graph.degree_distribution().bins()))
    #graph_bins_transposed = np.transpose([x for x in list(graph.degree_distribution().bins())[1:] if x[2] > 1])
    fig, ax = plt.subplots(figsize=(8,5))

    if logScale:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$p(k)$')

    ax.scatter(graph_bins_transposed[0], graph_bins_transposed[2]/sum(graph_bins_transposed[2]), s=6)

    if fit_power_law:
        popt, pcov = scipy.optimize.curve_fit(f=power_law, xdata=graph_bins_transposed[0], ydata=graph_bins_transposed[2], p0=[1, 2])
        x = np.linspace(graph_bins_transposed[0][0], graph_bins_transposed[0][-1], 100)
        ax.plot(x, power_law(x, popt[0], popt[1]), label=f"A*x^(-gamma)\nA={popt[0]:.2f}\ngamma={popt[1]:.2f}", color='orange', linewidth=1)
        ax.legend(loc='upper right')
    elif A != None and gamma != None:
        x = np.linspace(graph_bins_transposed[0][0], graph_bins_transposed[0][-1], 100)
        ax.plot(x, power_law(x, A, gamma), label=f"A*x^(-gamma)\nA={A:.2f}\ngamma={gamma:.2f}", color='orange', linewidth=1)
    
    return fig

#plot node degrees
#x-axis: node label "i", which is essentially a measure of age in the graph (1 = oldest node)
#y-axis: node i's degree
def graph_node_degrees_plot(graph):
    fig, ax = plt.subplots()

    ax.grid(alpha=0.5)
    ax.set_xlabel('Node label [i]')
    ax.set_ylabel('Node degree')
    ax.scatter(range(1,  graph.vcount()+1), graph.degree(), s=4)
    
    return fig, ax

#get the Pearson correlations between the original PS generated graph node distances
#and the inferred node distances after LaBNE
#def LaBNE_distance_Pearson(PSgraph=None, LaBNE_graph=None):

def inferred_angles(PS_graph=None, LaBNE_graph=None, colors=False, c='black'):
    if PS_graph.vcount() != LaBNE_graph.vcount():
        raise ValueError('PS Graph and LaBNE Graph have different vertex counts.')
    
    LaBNE_theta = LaBNE_graph.vs['theta']

    pearson_correlation = np.corrcoef(PS_graph.vs['theta'], LaBNE_theta)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$\theta_{\text{real}}$ [rad]', fontsize=15)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=14)
    ax.set_ylabel(r'$\theta_{\text{inferido}}$ [rad]', fontsize=15)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=14)
    #ax.scatter(PS_graph.vs['theta'], LaBNE_theta, s=6, cmap='rainbow', c=PS_graph.vs['theta'])
    if colors == True:
        ax.scatter(PS_graph.vs['theta'], LaBNE_theta, s=4, c=PS_graph.vs['theta'])
    else:
        ax.scatter(PS_graph.vs['theta'], LaBNE_theta, s=4, c=c)
    #ax.scatter(PS_graph.vs['theta'], LaBNE_theta, s=6)

    return fig, ax, pearson_correlation

def precision_recall_snapshots(snapshot_t0 = igraph.Graph(), snapshot_t1 = igraph.Graph(), metric=None, step=1, plot=False):
    
    """
    if snapshot_t0.vcount() != snapshot_t1.vcount():
        raise ValueError('Snapshot t0 and snapshot t1 have different vertex counts.')
    if snapshot_t0.is_connected() == False:
        raise ValueError('Snapshot t0 is not connected.')
    if snapshot_t1.is_connected() == False:
        raise ValueError('Snapshot t1 is not connected.')
    """
    N = snapshot_t0.vcount()
    
    # create copy of the first snapshot, to be used as training graph
    train_graph = copy.deepcopy(snapshot_t0)

    metric_function = None
    sort_dir = 'descending'
    if metric == 'CN':
        metric_function = common_neighbors
    elif metric == 'DS':
        metric_function = dice_similarity
    elif metric == 'AA':
        metric_function = adamic_adar_index
    elif metric == 'PA':
        metric_function = preferential_attachment
    elif metric == 'JC':
        metric_function = jaccard_coefficient
    elif metric == 'LaBNE':
        metric_function = LaBNE_metric
        sort_dir = 'ascending'
        train_graph = pop_sim.generateLaBNE(graph=train_graph, eigenvector_k=3)
    else:
        raise ValueError('Metric not recognized. Valid metrics are: CN, DS, AA, PA, JC, LaBNE')

    #array to save link scores, along with the corresponding edge and true/false label (which corresponds to whether
    # the edge is in the original graph or not)
    link_scores = []
    for i in range(N):
        for j in range(i+1, N):
            if not train_graph.are_connected(i, j):
                #save a three element tuple, which contains the tuple (i, j), the boolean label (0 or 1) and the link score
                #the boolean label is 0 if the edge is in the original graph, and 1 if it is not
                #the link score is the link score between the two nodes, _given_ by the train graph which has the test edges removed
                link_scores.append([(i, j), snapshot_t1.are_connected(i, j), metric_function(train_graph, i, j)])

    link_scores = np.array(link_scores)
    #sort by link scores
    if sort_dir == 'descending':
        link_scores = link_scores[link_scores[:,2].argsort()[::-1]]
    elif sort_dir == 'ascending':
        link_scores = link_scores[link_scores[:,2].argsort()]
        

    #generate the precision-recall curve by thresholding the link scores
    #and then calculating the precision and recall at each threshold, using
    #the true positive and false positive counts. we recall that precision
    #is the true positive rate, and recall is the relevance.
    tp = 0
    fp = 0
    fn = sum(np.transpose(link_scores)[1])
    tn = len(link_scores) - fn
    precision = []
    recall = []
    matrix = []
    non_train_edgecount = snapshot_t1.ecount()-train_graph.ecount()
    for i in range(0, len(link_scores)):
        if link_scores[i,1]:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        if i % step == 0:
            precision.append(tp/(tp+fp))
            recall.append(tp/non_train_edgecount)
            matrix.append(np.array([[tn, fp],[fn, tp]]))
        if tp >= non_train_edgecount:
            break

    # the confusion matrix:
    #   0,0 is TN
    #   1,0 is FP
    #   0,1 is FN
    #   1,1 is TP
    # it is given as snapshots for each precision-recall point
    matrix = np.transpose(matrix)

    # create dictionary of precision, recall, and link scores. we return this dictionary
    return_dict = {'precision':precision, 'recall':recall, 'C_matrix':matrix, 'link_scores':link_scores}

    # add a precision-recall curve to the return dictionary. returned as a matplotlib figure and axis
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.grid(alpha=0.5)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        #ax.scatter(recall, precision, c=np.log(np.arange(len(precision))), s=8, label=metric)
        ax.scatter(recall, precision, s=8, label=metric)
        return_dict['plot'] = {'fig':fig, 'ax':ax}

    return return_dict

# def precision_recall_snapshot_files(snapshot_t0 = ng.Network, snapshot_t1 = ng.Network, header='', metric=None, debug_print_flag=False):

#     debug_print.debug_print_flag = debug_print_flag

#     N = snapshot_t0.graph.vcount()
#     E_t0 = snapshot_t0.graph.get_edgelist()
#     E_t1 = snapshot_t1.graph.get_edgelist()
    
#     train_graph = snapshot_t0.graph
#     metric_function = None
#     sort_dir = 'descending'
#     if metric == 'CN':
#         metric_function = common_neighbors
#     elif metric == 'DS':
#         metric_function = dice_similarity
#     elif metric == 'AA':
#         metric_function = adamic_adar_index
#     elif metric == 'PA':
#         metric_function = preferential_attachment
#     elif metric == 'JC':
#         metric_function = jaccard_coefficient
#     elif metric == 'LaBNE' or metric == 'LaBNE_M':
#         metric_function = LaBNE_metric
#         train_graph = ng.generateLaBNE(network=snapshot_t0)
#         sort_dir = 'ascending'
#     else:
#         raise ValueError('Metric not recognized. Valid metrics are: CN, DS, AA, PA, JC, LaBNE')

#     debug_print.DebugPrint(f'Using metric: {metric}\n')
#     #debug_print.DebugPrint(f'    Train graph is connected before pruning: {train_graph.is_connected()} | ')
#     #get the largest cluster of the train graph, discarding any other clusters or loose nodes
#     #train_graph = train_graph.clusters().giant()
#     #debug_print.DebugPrint(f'After pruning: {train_graph.is_connected()}\n')
#     #debug_print.DebugPrint(f'    Original graph edgecount: {edgecount}, Train set edgecount: {len(E_train)}, Test set edgecount: {len(E_test)}\n')
#     #array to save link scores, along with the corresponding edge and true/false label (which corresponds to whether
#     # the edge is in the original graph or not)
#     link_scores = []
#     debug_print.DebugPrint('    Calculating link scores... ')
#     with open(f'./obj/{header}_PR_{metric}.csv', 'wb') as file:
#         for i in range(0, N):
#             for j in range(i+1, N):
#                 if not snapshot_t0.graph.are_connected(i, j):
#                     #save a three element tuple, which contains the tuple (i, j), the boolean label (0 or 1) and the link score
#                     #the boolean label is 0 if the edge is in the original graph, and 1 if it is not
#                     #the link score is the link score between the two nodes, _given_ by the train graph which has the test edges removed
#                     #link_scores.append([(i, j), PGP_t1_Network.graph.are_connected(i, j), metric_function(PGP_t0_Network.graph, i, j)])
#                     #file.write(f'{i:05},{j:05} {1*snapshot_t1.graph.are_connected(i, j)} {metric_function(train_graph, i, j)}\n'.encode())
#                     file.write(f'{i:05},{j:05} {1*snapshot_t1.graph.are_connected(i, j)} {metric_function(train_graph, i, j)}\n'.encode())
#     file.close()
    
#     debug_print.DebugPrint('Done. Sorting file... ')
#     #os.system(f"LC_ALL=C sort -t' ' -gk1 ./obj/{header}_PR_{metric}.csv > ./obj/{header}_PR_{metric}_sorted.csv")
#     if sort_dir == 'descending':
#         os.system(f"LC_ALL=C sort -t' ' -grk3 -s ./obj/{header}_PR_{metric}.csv > ./obj/{header}_PR_{metric}_sorted_final.csv")
#         #os.system(f"LC_ALL=C sort -t' ' -grk3 ./obj/{header}_PR_{metric}.csv > ./obj/{header}_PR_{metric}_sorted_.csv")
#     elif sort_dir == 'ascending':
#         os.system(f"LC_ALL=C sort -t' ' -gk3 -s ./obj/{header}_PR_{metric}.csv > ./obj/{header}_PR_{metric}_sorted_final.csv")
#         #os.system(f"LC_ALL=C sort -t' ' -gk3 ./obj/{header}_PR_{metric}.csv > ./obj/{header}_PR_{metric}_sorted_.csv")
#     debug_print.DebugPrint('Done.\n')
    
#     """
#     PR_LP_sorted = open(f'./obj/{header}_PR_{metric}_sorted.csv')
#     tp = 0
#     fp = 0
#     fn = snapshot_t1.graph.ecount() - snapshot_t0.graph.ecount()
#     tn = N*(N-1)/2 - snapshot_t1.graph.ecount()
#     precision = []
#     recall = []
#     C_matrix = []
#     non_train_edges = fn
#     for i in range(0, int(N*(N-1)/2-snapshot_t0.graph.ecount())):
#         line = PR_LP_sorted.readline()
#         line = line.split(' ')
#         line = [line[0] == '1']
    
#         if line[0]:
#             tp += 1
#             fn -= 1
#         else:
#             fp += 1
#             tn -= 1
#         if i % step == 0:
#             precision.append(tp/(tp+fp))
#             recall.append(tp/non_train_edges)
#             C_matrix.append([[tn, fp], [fn, tp]])

#     return precision, recall, C_matrix
#     """
#     return

def precision_recall_train_set(graph=igraph.Graph(), metric=None, test_size=0.33, plot=False, random_state=47, step=1, debug_print_flag=False):
    
    debug_print.debug_print_flag = debug_print_flag

    N = graph.vcount()
    E = graph.get_edgelist()
    #get graph edge count
    edgecount = len(E)

    train_graph = copy.deepcopy(graph)
    # split the list of edges into training and testing sets
    E_test = []
    E_train = None
    #E_train, E_test = sklearn.model_selection.train_test_split(E, test_size=test_size, random_state=random_state)
    removed_edges = 0
    while removed_edges < len(E)*test_size:

        while 1:
            _ = copy.deepcopy(train_graph)
            train_edgelist = train_graph.get_edgelist()
            removed_edge = train_edgelist[np.random.choice(range(len(train_edgelist)),1)[0]]

            _.delete_edges([(removed_edge[0], removed_edge[1])])
            if _.is_connected():
                break

        train_graph = _
        E_test.append(removed_edge)
        removed_edges += 1
        # create copy of graph, but removing the test set of edges
        #train_graph.delete_edges(E_test)
        #train_graph = copy.deepcopy(graph)
        #train_graph = train_graph.clusters().giant()
    E_train = [edge for edge in E if edge not in E_test]
        
    metric_function = None
    sort_dir = 'descending'
    if metric == 'CN':
        metric_function = common_neighbors
    elif metric == 'DS':
        metric_function = dice_similarity
    elif metric == 'AA':
        metric_function = adamic_adar_index
    elif metric == 'PA':
        metric_function = preferential_attachment
    elif metric == 'JC':
        metric_function = jaccard_coefficient
    elif metric == 'LaBNE':
        metric_function = LaBNE_metric
        sort_dir = 'ascending'
        train_graph = ng.generateLaBNE(graph=train_graph)
    else:
        raise ValueError('Metric not recognized. Valid metrics are: CN, DS, AA, PA, JC, LaBNE')

    debug_print.DebugPrint(f'Using metric: {metric}\n')
    debug_print.DebugPrint(f'    Original graph edgecount: {edgecount}, Train set edgecount: {len(E_train)}, Test set edgecount: {len(E_test)}\n')
    #array to save link scores, along with the corresponding edge and true/false label (which corresponds to whether
    # the edge is in the original graph or not)
    link_scores = []
    debug_print.DebugPrint('    Calculating link scores... ')
    for i in range(train_graph.vcount()):
        for j in range(i+1, train_graph.vcount()):
            if not train_graph.are_connected(i, j):
                #save a three element tuple, which contains the tuple (i, j), the boolean label (0 or 1) and the link score
                #the boolean label is 0 if the edge is in the original graph, and 1 if it is not
                #the link score is the link score between the two nodes, _given_ by the train graph which has the test edges removed
                link_scores.append([(i, j), graph.are_connected(i, j), metric_function(train_graph, i, j)])
    debug_print.DebugPrint('Done. ')

    link_scores = np.array(link_scores)
    #sort by link scores
    if sort_dir == 'descending':
        link_scores = link_scores[link_scores[:,2].argsort(kind='stable')[::-1]]
    elif sort_dir == 'ascending':
        link_scores = link_scores[link_scores[:,2].argsort(kind='stable')]

    #generate the precision-recall curve by thresholding the link scores
    #and then calculating the precision and recall at each threshold, using
    #the true positive and false positive counts. we recall that precision
    #is the true positive rate, and recall is the relevance.
    
    """
    p = 0
    fp = 0
    fn = snapshot_t1.graph.ecount() - snapshot_t0.graph.ecount()
    tn = N*(N-1)/2 - snapshot_t1.graph.ecount()
    precision = []
    recall = []
    C_matrix = []
    non_train_edges = fn
    for i in range(0, int(N*(N-1)/2-snapshot_t0.graph.ecount())):
        line = PR_LP_sorted.readline()
        line = line.split(' ')
        line = [line[0] == '1']
    
        if line[0]:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        if i % step == 0:
            precision.append(tp/(tp+fp))
            recall.append(tp/non_train_edges)
            C_matrix.append([[tn, fp], [fn, tp]])

    return precision, recall, C_matrix
    """


    tp = 0
    fp = 0
    fn = graph.ecount() - train_graph.ecount()
    tn = N*(N-1)/2 - graph.ecount()
    precision = []
    recall = []
    C_matrix = []
    debug_print.DebugPrint('Calculating precision-recall... ')
    non_train_edgecount = fn
    for i in range(0, len(link_scores)):
        if link_scores[i,1]:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        if i % step == 0:
            precision.append(tp/(tp+fp))
            recall.append(tp/non_train_edgecount)
            C_matrix.append(np.array([[tn, fp],[fn, tp]]))
    debug_print.DebugPrint('Done. ')

    # the confusion matrix:
    #   0,0 is TN
    #   1,0 is FP
    #   0,1 is FN
    #   1,1 is TP
    # it is given as snapshots for each precision-recall point
    C_matrix = np.transpose(C_matrix)

    # create dictionary of precision, recall, and link scores. we return this dictionary
    return_dict = {'precision':precision, 'recall':recall, 'C_matrix':C_matrix, 'link_scores':link_scores}

    # add a precision-recall curve to the return dictionary. returned as a matplotlib figure and axis
    if plot:
        debug_print.DebugPrint('Plotting PR curve... ')
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.grid(alpha=0.5)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        #ax.scatter(recall, precision, c=np.log(np.arange(len(precision))), s=8, label=metric)
        ax.scatter(recall, precision, s=8, label=metric)
        debug_print.DebugPrint('Done. ')
        return_dict['plot'] = {'fig':fig, 'ax':ax}

    debug_print.DebugPrint('\n')
    debug_print.debug_print_flag = False
    return return_dict

"""
def linear_SVC_precision_recall(graph=igraph.Graph(), random_state=42):

    E = []
    y = []
    A = graph.get_adjacency()
    for i in range(graph.vcount()):
        for j in range(i, graph.vcount()):
            E.append((i,j))
            y.append(1*graph.are_connected(i, j))
    
    E_train, E_test, y_train, y_test = sklearn.model_selection.train_test_split(E, y, test_size=0.33, random_state=random_state)

    classifier = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.svm.LinearSVC(random_state=random_state))
    classifier.fit(E_train, y_train)

    display = PrecisionRecallDisplay.from_estimator(classifier, E_test, y_test, name='LinearSVC')
    display.ax_.legend(loc='upper right')
    display.ax_.grid(alpha=0.5)
    #_ = display._ax.set_title('Precision-Recall')

    return display, E_train, E_test, y_train, y_test
"""