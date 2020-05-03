from __future__ import absolute_import
from __future__ import print_function


import sys
import numpy as np
import numpy.linalg as LA
import random
import itertools
import multiprocessing as mp
from .tree import Tree
#from ..tools.ete_diff_lib._lapjv import lapjv 
from lap import lapjv # install lapjv from external
import logging
from tqdm import tqdm
log = logging.Logger("main")



DESC = ""

def EUCL_DIST(a,b):  
    return 1 - (float(len(a[1] & b[1])) / max(len(a[1]), len(b[1])))

def EUCL_DIST_B(a,b): 

    dist_a = sum([descendant.dist for descendant in a[0].iter_leaves() if descendant.name in(a[1] - b[1])])
    dist_b = sum([descendant.dist for descendant in b[0].iter_leaves() if descendant.name in(b[1] - a[1])])
    
    return 1 - (float(len(a[1] & b[1])) / max(len(a[1]), len(b[1]))) + abs(dist_a - dist_b)

# checks distances of non shared leaves, so as to compare trees which generated from same resource
def EUCL_DIST_B_ALL(*args): 
    
    a = args[0]
    b = args[1]
    #attr1 = args[2]
    #attr2 = args[3]

    dist_a = sum([descendant.dist for descendant in a[0].iter_leaves()])
    dist_b = sum([descendant.dist for descendant in b[0].iter_leaves()])
    
    return 1 - (float(len(a[1] & b[1])) / max(len(a[1]), len(b[1]))) + abs(dist_a - dist_b)

def RF_DIST(*args):
    if len(a[1] & b[1]) < 2:
        return 1.0
    (a, b) = (b, a) if len(b[1]) > len(a[1]) else (a,b)
    rf, rfmax, names, side1, side2, d1, d2 = a[0].robinson_foulds(b[0])
    return (rf/rfmax if rfmax else 0.0)



def get_distances1(t1,t2): #better names
    def _get_leaves_paths(t):
        leaves = t.get_leaves()
        leave_branches = set()

        for n in leaves:
            if n.is_root():
                continue
            movingnode = n
            length = 0
            while not movingnode.is_root():
                length += movingnode.dist
                movingnode = movingnode.up
            leave_branches.add((n.name,length))

        return leave_branches

    def _get_distances(leaf_distances1,leaf_distances2):

        unique_leaves1 = leaf_distances1 - leaf_distances2
        unique_leaves2 = leaf_distances2 - leaf_distances1
        
        return abs(sum([leaf[1] for leaf in unique_leaves1]) - sum([leaf[1] for leaf in unique_leaves2]))

    return _get_distances(_get_leaves_paths(t1),_get_leaves_paths(t2))    


def get_distances2(t1,t2): #better names
    def cophenetic_compared_matrix(t_source,t_compare):

        leaves = t_source.get_leaves()
        paths = {x.name: set() for x in leaves}

        # get the paths going up the tree
        # we get all the nodes up to the last one and store them in a set

        for n in leaves:
            if n.is_root():
                continue
            movingnode = n
            while not movingnode.is_root():
                paths[n.name].add(movingnode)
                movingnode = movingnode.up

        # We set the paths for leaves not in the source tree as empty to indicate they are non-existent

        for i in (set(x.name for x in t_compare.get_leaves()) - set(x.name for x in t_source.get_leaves())):
            paths[i] = set()

        # now we want to get all pairs of nodes using itertools combinations. We need AB AC etc but don't need BA CA

        leaf_distances = {x: {} for x in paths.keys()}

        for (leaf1, leaf2) in itertools.combinations(paths.keys(), 2):
            # figure out the unique nodes in the path
            if len(paths[leaf1]) > 0 and len(paths[leaf2]) > 0:
                uniquenodes = paths[leaf1] ^ paths[leaf2]
                distance = sum(x.dist for x in uniquenodes)
            else:
                distance = 0
            leaf_distances[leaf1][leaf2] = leaf_distances[leaf2][leaf1] = distance

        allleaves = sorted(leaf_distances.keys()) # the leaves in order that we will return

        output = [] # the two dimensional array that we will return

        for i, n in enumerate(allleaves):
            output.append([])
            for m in allleaves:
                if m == n:
                    output[i].append(0) # distance to ourself = 0
                else:
                    output[i].append(leaf_distances[n][m])
        return np.asarray(output)

    ccm1 = cophenetic_compared_matrix(t1,t2)
    ccm2 = cophenetic_compared_matrix(t2,t1)
    
    return LA.norm(ccm1-ccm2)


def sepstring(items, sep=", "):
    return sep.join(sorted(map(str, items)))



### Treediff ###

def treediff(t1, t2, attr1, attr2, dist_fn=EUCL_DIST, reduce_matrix=False,branchdist=None, jobs=1):
    log = logging.getLogger()
    log.info("Computing distance matrix...")

    t1_cached_content = t1.get_cached_content(store_attr=attr1)
    t2_cached_content = t2.get_cached_content(store_attr=attr2)
    
    #parts1 = [(k, v) for k, v in t1_cached_content.items() if k.children]
    #parts2 = [(k, v) for k, v in t2_cached_content.items() if k.children]

    parts1 = [(k, v) for k, v in t1_cached_content.items()]
    parts2 = [(k, v) for k, v in t2_cached_content.items()]

    parts1 = sorted(parts1, key = lambda x : len(x[1]))
    parts2 = sorted(parts2, key = lambda x : len(x[1]))

    pool = mp.Pool(jobs)
    matrix = [[pool.apply_async(dist_fn,args=((n1,x),(n2,y))) for n2,y in parts2] for n1,x in parts1] 
    pool.close()
    
    # progress bar
    for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = matrix[i][j].get()

    # with tqdm(total=len(matrix[0])*len(matrix)) as pbar:
    #     for i in range(len(matrix)):
    #         for j in range(len(matrix[0])):
    #             matrix[i][j] = matrix[i][j].get()
    #             pbar.update(1)
    
    # Reduce matrix to avoid useless comparisons
    if reduce_matrix:
        log.info( "Reducing distance matrix...")
        cols_to_include = set(range(len(matrix[0])))
        rows_to_include = []
        for i, row in enumerate(matrix):
            try:
                cols_to_include.remove(row.index(0.0))
            except ValueError:
                rows_to_include.append(i)
            except KeyError:
                pass
        
        cols_to_include = sorted(cols_to_include)

        parts1 = [parts1[row] for row in rows_to_include]
        parts2 = [parts2[col] for col in cols_to_include]
        
        new_matrix = []
        for row in rows_to_include:
            new_matrix.append([matrix[row][col] for col in cols_to_include])
 
        if len(new_matrix) < 1:
            return new_matrix
        
        log.info("Distance matrix reduced from %dx%d to %dx%d" %\
                (len(matrix), len(matrix[0]), len(new_matrix), len(new_matrix[0])))
            
        matrix = new_matrix

    log.info("Comparing trees...")

    matrix = np.asarray(matrix, dtype=np.float32)

    #print(lapjv(matrix,extend_cost=True))
    _, cols, _ = lapjv(matrix,extend_cost=True) #not extend a non-square matrix, return opt, x_c, y_c

    difftable = []
    b_dist = -1 #should be others
    for r in range(len(matrix)):
        c = cols[r]
        if matrix[r][c] != 0:
            if branchdist:
                b_dist = branchdist(parts1[r][0], parts2[c][0])
            else:
                pass
            dist, side1, side2, diff, n1, n2 = (matrix[r][c], 
                                                parts1[r][1], parts2[c][1],
                                                parts1[r][1].symmetric_difference(parts2[c][1]),
                                                parts1[r][0], parts2[c][0])
            
            n1 = Tree(n1.write(features=[attr1]))  #in order to gain the names of internal nodes
            n2 = Tree(n2.write(features=[attr2])) 
            difftable.append([dist, b_dist, side1, side2, diff, n1, n2])
    return difftable