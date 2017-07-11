'''
Modified PC algorithm code (original code: https://github.com/keiichishima/pcalg) 
by incorporating KRESIT as the conditional independence test.

Original code: A graph generator based on the PC algorithm [Kalisch2007].
[Kalisch2007] Markus Kalisch and Peter Bhlmann. Estimating
high-dimensional directed acyclic graphs with the pc-algorithm. In The
Journal of Machine Learning Research, Vol. 8, pp. 613-636, 2007.
License: BSD

Example run in terminal:
1) KRESIT:
$ python PCalg_twostep_flags.py 400 --dimZ 4 --data_filename "Synthetic_DAGexample" 
--kernelX_use_median --kernelY_use_median 
--results_filename "test_run_KRESIT" --figure_filename "test_graph_KRESIT" 

(It takes the first 400 samples and the first 4 dimensions from the Synthetic_DAGexample.csv
and run KRESIT with Gaussian kernel median Heuristic on the variables X and Y. The kernel on Z
is set by default to be Gaussian median Heuristic. The regularisation parameters is 
set by default to use grid search. The resulting CPDAG is saved "test_graph_KRESIT.pdf".)

2) RESIT:
$ python PCalg_twostep_flags.py 400 --dimZ 4 --data_filename "Synthetic_DAGexample"
--kernelX --kernelY 
--kernelRxz --kernelRyz
--kernelRxz_use_median --kernelRyz_use_median 
--RESIT_type
--result_filename "test_run_RESIT" --figure_filename "test_graph_RESIT"

(It takes the first 400 samples and the first 4 dimensions from the Synthetic_DAGexample.csv
and run RESIT. The kernels on X and Y are set to be linear. The kernels on the residuals Rxz and 
Ryz are Gaussian with median Heuristic bandwidth.The regularisation parameters is 
set by default to use grid search. The resulting CPDAG is saved "test_graph_RESIT.pdf".)

'''
from __future__ import print_function

# Remote_running 
#import matplotlib
#matplotlib.use('Agg')

import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)


from itertools import combinations, permutations
import logging

import networkx as nx
import cPickle as pickle
from pickle import load, dump

from kerpy.GaussianKernel import GaussianKernel
from kerpy.LinearKernel import LinearKernel
from TwoStepCondTestObject import TwoStepCondTestObject
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
import numpy as np
from numpy import arange

_logger = logging.getLogger(__name__)






def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.
    Args:
        node_ids: a list of node ids
    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
        pass
    return g




def estimate_skeleton( data_matrix, alpha, **kwargs):
    # originally first argument is indep_test_func
    # now this version uses HSIC Spectral Test for independence 
    # and KRESIT for conditional independence.
    """Estimate a skeleton graph from the statistis information.
    Args:
        indep_test_func: the function name for a conditional
            independency test.
        data_matrix: data (as a numpy array).
        alpha: the significance level.
        kwargs:
            'max_reach': maximum value of l (see the code).  The
                value depends on the underlying distribution.
            'method': if 'stable' given, use stable-PC algorithm
                (see [Colombo2014]).
            other parameters may be passed depending on the
                indep_test_func()s.
    Returns:
        g: a skeleton graph (as a networkx.Graph).
        sep_set: a separation set (as an 2D-array of set()).
    [Colombo2014] Diego Colombo and Marloes H Maathuis. Order-independent
    constraint-based causal structure learning. In The Journal of Machine
    Learning Research, Vol. 15, pp. 3741-3782, 2014.
    """
    
    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"
    
    node_ids = range(data_matrix.shape[1])
    g = _create_complete_graph(node_ids)
    
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    
    
    X_idx_list_init = []
    Y_idx_list_init = []
    Z_idx_list_init = []
    pval_list_init = []
    
    l = 0
    completed_xy_idx_init = 0
    completed_z_idx_init = 0 
    remove_edges_current = []
    
    
    results_filename = kwargs['results_filename']
    myfolder = "pcalg_results/"
    save_filename = myfolder + results_filename + ".bin"
    #print("save_filename:", save_filename)
    #sys.exit(1)
    if not os.path.exists(myfolder):
        os.mkdir(myfolder)
    elif os.path.exists(save_filename):
        load_f = open(save_filename,"r")
        [X_idx_list_init, Y_idx_list_init, Z_idx_list_init, pval_list_init, \
         l, completed_xy_idx_init, completed_z_idx_init,\
         remove_edges_current, g] = load(load_f)
        load_f.close()
        print("Found exitising results")
    
    X_idx_list = X_idx_list_init
    Y_idx_list = Y_idx_list_init
    Z_idx_list = Z_idx_list_init
    pval_list = pval_list_init
    completed_xy_idx = completed_xy_idx_init
    completed_z_idx = completed_z_idx_init
    
    
    
    while True:
        cont = False
        remove_edges = remove_edges_current
        perm_iteration_list = list(permutations(node_ids,2))
        length_iteration_list = len(perm_iteration_list)
        
        for ij in arange(completed_xy_idx, length_iteration_list):
            (i,j) = perm_iteration_list[ij]
            adj_i = g.neighbors(i)
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
                pass
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
                if len(adj_i) < l:
                    continue
                
                cc = list(combinations(adj_i, l))
                length_cc = len(cc)
                
                
                for kk in arange(completed_z_idx, length_cc):
                    k = cc[kk]
                    _logger.debug('indep prob of %s and %s with subset %s'
                                  % (i, j, str(k)))
                    if l == 0: # independence testing 
                        print("independence testing", (i,j))
                        data_x = data_matrix[:,[i]]
                        data_y = data_matrix[:,[j]]
                        
                        num_samples = np.shape(data_matrix)[0]
                        kernelX_hsic = GaussianKernel(1.)
                        kernelY_hsic = GaussianKernel(1.)
                        kernelX_use_median_hsic = True 
                        kernelY_use_median_hsic = True
                        
                        myspectraltestobj = HSICSpectralTestObject(num_samples, None, kernelX_hsic, kernelY_hsic, 
                            kernelX_use_median = kernelX_use_median_hsic,
                            kernelY_use_median = kernelY_use_median_hsic, 
                            num_nullsims=1000, unbiased=False)
                        
                        p_val, _ = myspectraltestobj.compute_pvalue_with_time_tracking(data_x,data_y)
                        
                        
                        X_idx_list.append((i))
                        Y_idx_list.append((j))
                        Z_idx_list.append((0))
                        pval_list.append((p_val))
                        
                        
                    else: # conditional independence testing
                        print("conditional independence testing",(i,j,k))
                        data_x = data_matrix[:,[i]]
                        data_y = data_matrix[:,[j]]
                        data_z = data_matrix[:,k]
                        
                        num_samples = np.shape(data_matrix)[0]
                        #kernelX = GaussianKernel(1.)
                        #kernelY = GaussianKernel(1.)
                        #kernelX_use_median = True
                        #kernelY_use_median = True
                        #kernelX = LinearKernel()
                        #kernelY = LinearKernel()
                        kernelX = kwargs['kernelX']
                        kernelY = kwargs['kernelY']
                        kernelZ = GaussianKernel(1.)
                        kernelX_use_median = kwargs['kernelX_use_median']
                        kernelY_use_median = kwargs['kernelY_use_median']
                        kernelRxz = kwargs['kernelRxz']
                        kernelRyz = kwargs['kernelRyz']
                        kernelRxz_use_median = kwargs['kernelRxz_use_median']
                        kernelRyz_use_median = kwargs['kernelRyz_use_median']
                        RESIT_type = kwargs['RESIT_type']
                        optimise_lambda_only = kwargs['optimise_lambda_only']
                        grid_search = kwargs['grid_search']
                        GD_optimise = kwargs['GD_optimise']
                        
                        
                        
                        num_lambdaval = 30
                        lambda_val = 10**np.linspace(-6,1, num=num_lambdaval)
                        z_bandwidth = None
                        #num_bandwidth = 20
                        #z_bandwidth = 10**np.linspace(-5,1,num = num_bandwidth)
                        
                        mytestobj = TwoStepCondTestObject(num_samples, None, 
                                                 kernelX, kernelY, kernelZ, 
                                                 kernelX_use_median=kernelX_use_median,
                                                 kernelY_use_median=kernelY_use_median, 
                                                 kernelZ_use_median=True, 
                                                 kernelRxz = kernelRxz, kernelRyz = kernelRyz,
                                                 kernelRxz_use_median = kernelRxz_use_median, 
                                                 kernelRyz_use_median = kernelRyz_use_median,
                                                 RESIT_type = RESIT_type,
                                                 num_shuffles=800,
                                                 lambda_val=lambda_val,lambda_X = None, lambda_Y = None,
                                                 optimise_lambda_only = optimise_lambda_only, 
                                                 sigmasq_vals = z_bandwidth ,sigmasq_xz = 1., sigmasq_yz = 1.,
                                                 K_folds=5, grid_search = grid_search,
                                                 GD_optimise=GD_optimise, learning_rate=0.1,max_iter=300,
                                                 initial_lambda_x=0.5,initial_lambda_y=0.5, initial_sigmasq = 1)
                        
                        
                        
                        p_val, _ = mytestobj.compute_pvalue(data_x, data_y, data_z)
                        
                        X_idx_list.append((i))
                        Y_idx_list.append((j))
                        Z_idx_list.append(k)
                        pval_list.append((p_val))
                        
                        
                    
                    
                    completed_z_idx = kk + 1
                    
                    save_f = open(save_filename,"w")
                    dump([X_idx_list, Y_idx_list, Z_idx_list, pval_list, l, completed_xy_idx, completed_z_idx,\
                          remove_edges, g], save_f)
                    save_f.close()
                    
                    
                    _logger.debug('p_val is %s' % str(p_val))
                    if p_val > alpha:
                        if g.has_edge(i, j):
                            _logger.debug('p: remove edge (%s, %s)' % (i, j))
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                            pass
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                    pass
                completed_z_idx = 0
                completed_xy_idx = ij + 1
                cont = True
                pass
            pass
        
        
        
        l += 1
        completed_xy_idx = 0
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break
        
        
        
        save_f = open(save_filename,"w")
        dump([X_idx_list, Y_idx_list, Z_idx_list, pval_list, l, completed_xy_idx, completed_z_idx,\
                          remove_edges, g], save_f)
        save_f.close()
        
        pass
    
    return (g, sep_set)




def estimate_cpdag(skel_graph, sep_set):
    """Estimate a CPDAG from the skeleton graph and separation sets
    returned by the estimate_skeleton() function.
    Args:
        skel_graph: A skeleton graph (an undirected networkx.Graph).
        sep_set: An 2D-array of separation set.
            The contents look like something like below.
                sep_set[i][j] = set([k, l, m])
    Returns:
        An estimated DAG.
    """
    dag = skel_graph.to_directed()
    node_ids = skel_graph.nodes()
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        if j in adj_i:
            continue
        adj_j = set(dag.successors(j))
        if i in adj_j:
            continue
        common_k = adj_i & adj_j
        for k in common_k:
            if k not in sep_set[i][j]:
                if dag.has_edge(k, i):
                    _logger.debug('S: remove edge (%s, %s)' % (k, i))
                    dag.remove_edge(k, i)
                    pass
                if dag.has_edge(k, j):
                    _logger.debug('S: remove edge (%s, %s)' % (k, j))
                    dag.remove_edge(k, j)
                    pass
                pass
            pass
        pass
    
    def _has_both_edges(dag, i, j):
        return dag.has_edge(i, j) and dag.has_edge(j, i)
    
    def _has_any_edge(dag, i, j):
        return dag.has_edge(i, j) or dag.has_edge(j, i)
    
    def _has_one_edge(dag, i, j):
        return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
                (not dag.has_edge(i, j)) and dag.has_edge(j, i))
    
    def _has_no_edge(dag, i, j):
        return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))
    
    # For all the combination of nodes i and j, apply the following
    # rules.
    for (i, j) in combinations(node_ids, 2):
        # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
        # such that k and j are nonadjacent.
        #
        # Check if i-j.
        if _has_both_edges(dag, i, j):
            # Look all the predecessors of i.
            for k in dag.predecessors(i):
                # Skip if there is an arrow i->k.
                if dag.has_edge(i, k):
                    continue
                # Skip if k and j are adjacent.
                if _has_any_edge(dag, k, j):
                    continue
                # Make i-j into i->j
                _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                dag.remove_edge(j, i)
                break
            pass
        
        # Rule 2: Orient i-j into i->j whenever there is a chain
        # i->k->j.
        #
        # Check if i-j.
        if _has_both_edges(dag, i, j):
            # Find nodes k where k is i->k.
            succs_i = set()
            for k in dag.successors(i):
                if not dag.has_edge(k, i):
                    succs_i.add(k)
                    pass
                pass
            # Find nodes j where j is k->j.
            preds_j = set()
            for k in dag.predecessors(j):
                if not dag.has_edge(j, k):
                    preds_j.add(k)
                    pass
                pass
            # Check if there is any node k where i->k->j.
            if len(succs_i & preds_j) > 0:
                # Make i-j into i->j
                _logger.debug('R2: remove edge (%s, %s)' % (j, i))
                dag.remove_edge(j, i)
                break
            pass
        
        # Rule 3: Orient i-j into i->j whenever there are two chains
        # i-k->j and i-l->j such that k and l are nonadjacent.
        #
        # Check if i-j.
        if _has_both_edges(dag, i, j):
            # Find nodes k where i-k.
            adj_i = set()
            for k in dag.successors(i):
                if dag.has_edge(k, i):
                    adj_i.add(k)
                    pass
                pass
            # For all the pairs of nodes in adj_i,
            for (k, l) in combinations(adj_i, 2):
                # Skip if k and l are adjacent.
                if _has_any_edge(dag, k, l):
                    continue
                # Skip if not k->j.
                if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                    continue
                # Skip if not l->j.
                if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                    continue
                # Make i-j into i->j.
                _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                dag.remove_edge(j, i)
                break
            pass
        
        # Rule 4: Orient i-j into i->j whenever there are two chains
        # i-k->l and k->l->j such that k and j are nonadjacent.
        #
        # However, this rule is not necessary when the PC-algorithm
        # is used to estimate a DAG.
        pass
    
    return dag




def run():#if __name__ == '__main__':
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt
    from SimDataGen import SimDataGen
    
    from tools.ProcessingObject import ProcessingObject
    args = ProcessingObject.parse_arguments()
    num_samples = args.num_samples
    kernelX = args.kernelX #Default: GaussianKernel(1.)
    kernelY = args.kernelY #Default: GaussianKernel(1.)
    kernelX_use_median = args.kernelX_use_median #Default: False
    kernelY_use_median = args.kernelY_use_median #Default: False
    kernelRxz = args.kernelRxz #Default: LinearKernel 
    kernelRyz = args.kernelRyz #Default: LinearKernel
    kernelRxz_use_median = args.kernelRxz_use_median #Default: False 
    kernelRyz_use_median = args.kernelRyz_use_median #Default: False
    RESIT_type = args.RESIT_type #Default: False
    optimise_lambda_only = args.optimise_lambda_only #Default: True
    grid_search = args.grid_search #Default: True
    GD_optimise = args.GD_optimise #Default: False 
    results_filename = args.results_filename
    figure_filename = args.figure_filename
    data_filename = args.data_filename
    num_var = args.dimZ
    
    
    
    datafile = data_filename + ".csv"
    data = np.loadtxt(datafile,delimiter = ',')
    dm = data[range(num_samples),:][:, range(num_var)]
    
    
    (g, sep_set) = estimate_skeleton(data_matrix=dm, alpha=0.05,
                                     kernelX = kernelX, kernelY = kernelY,
                                     kernelX_use_median = kernelX_use_median,
                                     kernelY_use_median = kernelY_use_median,
                                     kernelRxz = kernelRxz, kernelRyz = kernelRyz,
                                     kernelRxz_use_median = kernelRxz_use_median,
                                     kernelRyz_use_median = kernelRyz_use_median,
                                     RESIT_type = RESIT_type,
                                     results_filename = results_filename,
                                     optimise_lambda_only = optimise_lambda_only, 
                                     grid_search = grid_search, 
                                     GD_optimise = GD_optimise)
    
    
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    
    
    if num_var == 7:
        labels={}
        labels[0]=r'$X$'
        labels[1]=r'$Y$'
        labels[2]=r'$Z$'
        labels[3]=r'$A$'
        labels[4]=r'$B$'
        labels[5]=r'$Cx$'
        labels[6]=r'$Cy$'
    elif num_var == 6:
        labels={}
        labels[0] = r'$X$'
        labels[1] = r'$Y$'
        labels[2] = r'$Z$'
        labels[3] = r'$A$'
        labels[4] = r'$Cx$'
        labels[5] = r'$Cy$'
    elif num_var == 5:
        labels={}
        labels[0] = r'$X$'
        labels[1] = r'$Y$'
        labels[2] = r'$Z$'
        labels[3] = r'$A$'
        labels[4] = r'$B$'
    elif num_var == 4:
        labels={}
        labels[0] = r'$X$'
        labels[1] = r'$Y$'
        labels[2] = r'$Z$'
        labels[3] = r'$A$'
    else:
        raise NotImplementedError
    
    nx.draw_networkx(g, pos=nx.spring_layout(g), labels=labels, with_labels=True)
    figure_name = figure_filename + ".pdf"
    plt.savefig(figure_name)
    #plt.show()


if __name__ == '__main__':
    run()