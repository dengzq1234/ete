# #START_LICENSE###########################################################
#
#
# This file is part of the Environment for Tree Exploration program
# (ETE).  http://etetoolkit.org
#
# ETE is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ETE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ETE.  If not, see <http://www.gnu.org/licenses/>
#
#
#                     ABOUT THE ETE PACKAGE
#                     =====================
#
# ETE is distributed under the GPL copyleft license (2008-2015).
#
# If you make use of ETE in published work, please cite:
#
# Jaime Huerta-Cepas, Joaquin Dopazo and Toni Gabaldon.
# ETE: a python Environment for Tree Exploration. Jaime BMC
# Bioinformatics 2010,:24doi:10.1186/1471-2105-11-24
#
# Note that extra references to the specific methods implemented in
# the toolkit may be available in the documentation.
#
# More info at http://etetoolkit.org. Contact: huerta@embl.de
#
#
# #END_LICENSE#############################################################
from __future__ import absolute_import
from __future__ import print_function


import sys
import numpy as np
import numpy.linalg as LA
import random
import itertools
import multiprocessing as mp
from ..coretype.tree import Tree
from ..utils import print_table, color
from .ete_diff_lib._lapjv import lapjv
import textwrap
import argparse
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

def RF_DIST(a, b):
    if len(a[1] & b[1]) < 2:
        return 1.0
    (a, b) = (b, a) if len(b[1]) > len(a[1]) else (a,b)
    rf, rfmax, names, side1, side2, d1, d2 = a[0].robinson_foulds(b[0])
    return (rf/rfmax if rfmax else 0.0)



def get_distances1(t1,t2):
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


def get_distances2(t1,t2):
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

#     parts1 = [(k, v) for k, v in t1_cached_content.items() if k.children]
#     parts2 = [(k, v) for k, v in t2_cached_content.items() if k.children]

    parts1 = [(k, v) for k, v in t1_cached_content.items()]
    parts2 = [(k, v) for k, v in t2_cached_content.items()]

    parts1 = sorted(parts1, key = lambda x : len(x[1]))
    parts2 = sorted(parts2, key = lambda x : len(x[1]))

    pool = mp.Pool(jobs)
    matrix = [[pool.apply_async(dist_fn,args=((n1,x),(n2,y))) for n2,y in parts2] for n1,x in parts1]
    pool.close()
    
    with tqdm(total=len(matrix[0])*len(matrix)) as pbar:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = matrix[i][j].get()
                pbar.update(1)
    
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

    cols , _ = lapjv(matrix,extend_cost=True)

    difftable = []
    b_dist = -1
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
            difftable.append([dist, b_dist, side1, side2, diff, n1, n2])

    return difftable



### REPORTS ###

def show_difftable_summary(difftable, rf=-1, rf_max=-1, branchdist=False):
    total_dist = 0
    total_bdist = 0
    for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
        total_dist += dist
        total_bdist += b_dist
        
    if branchdist:
        log.info("\n"+"\t".join(["Dist", "Branch Dist", "Mismatches", "RF", "maxRF"]))
        print("%0.6f\t%0.6f\t%10d\t%d\t%d" %(total_dist,total_bdist, len(difftable), rf, rf_max))
    else:
        log.info("\n"+"\t".join(["Dist", "Mismatches", "RF", "maxRF"]))
        print("%0.6f\t%10d\t%d\t%d" %(total_dist, len(difftable), rf, rf_max))

def show_difftable(difftable, branchdist=False):
    showtable = []
    
    if branchdist:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            showtable.append([dist, b_dist, len(side1), len(side2), len(diff), sepstring(diff)])
        print_table(showtable, header=["Dist", "Branch Dist", "Size1", "Size2", "ndiffs", "Diff"],
                    max_col_width=80, wrap_style="wrap", row_line=True)
    else:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            showtable.append([dist, len(side1), len(side2), len(diff), sepstring(diff)])
        print_table(showtable, header=["Dist", "Size1", "Size2", "ndiffs", "Diff"],
                    max_col_width=80, wrap_style="wrap", row_line=True)

def show_difftable_tab(difftable, branchdist=False):
    showtable = []
    
    if branchdist:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            showtable.append([dist, b_dist, len(side1), len(side2), len(diff),
                              sepstring(side1, "|"), sepstring(side2, "|"),
                              sepstring(diff, "|")])
        print('#' + '\t'.join(["Dist", "Branch Dist", "Size1", "Size2", "ndiffs", "Diff", "refTree", "targetTree"]))
    else:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            showtable.append([dist, len(side1), len(side2), len(diff),
                              sepstring(side1, "|"), sepstring(side2, "|"),
                              sepstring(diff, "|")])
        print('#' + '\t'.join(["Dist", "Size1", "Size2", "ndiffs", "Diff", "refTree", "targetTree"]))
        
    print('\n'.join(['\t'.join(map(str, items)) for items in showtable]))
    
def show_difftable_topo(difftable, attr1, attr2, usecolor=False, branchdist=False):
    if not difftable:
        return
    showtable = []
    maxcolwidth = 80
    total_dist = 0
    for dist, b_dist, side1, side2, diff, n1, n2 in sorted(difftable, reverse=True):
        total_dist += dist
        n1 = Tree(n1.write(features=[attr1]))
        n2 = Tree(n2.write(features=[attr2]))
        n1.ladderize()
        n2.ladderize()
        for leaf in n1.iter_leaves():
            leaf.name = getattr(leaf, attr1)
            if leaf.name in diff:
                leaf.name += " ***"
                if usecolor:
                    leaf.name = color(leaf.name, "red")
        for leaf in n2.iter_leaves():
            leaf.name = getattr(leaf, attr2)
            if leaf.name in diff:
                leaf.name += " ***"
                if usecolor:
                    leaf.name = color(leaf.name, "red")

        topo1 = n1.get_ascii(show_internal=False, compact=False)
        topo2 = n2.get_ascii(show_internal=False, compact=False)

        # This truncates too large topology strings pretending to be
        # scrolled to the right margin
        topo1_lines = topo1.split("\n")
        topowidth1 = max([len(l) for l in topo1_lines])
        if topowidth1 > maxcolwidth:
            start = topowidth1 - maxcolwidth
            topo1 = '\n'.join([line[start+1:] for line in topo1_lines])
        
        topo2_lines = topo2.split("\n")
        topowidth2 = max([len(l) for l in topo2_lines])
        if topowidth2 > maxcolwidth:
            start = topowidth2 - maxcolwidth
            topo2 = '\n'.join([line[start+1:] for line in topo2_lines])
        
        if branchdist:
            showtable.append([dist, b_dist, "%d/%d (%d)" %(len(side1), len(side2),len(diff)), topo1, topo2])
        else:
            showtable.append([dist, "%d/%d (%d)" %(len(side1), len(side2),len(diff)), topo1, topo2])
    
    if branchdist:
        print_table(showtable, header=["Dist", "Branch Dist", "#Diffs", "Tree1", "Tree2"],
                    max_col_width=maxcolwidth, wrap_style="wrap", row_line=True)
    else:
        print_table(showtable, header=["Dist", "#Diffs", "Tree1", "Tree2"],
                    max_col_width=maxcolwidth, wrap_style="wrap", row_line=True)    
        
    log.info("Total euclidean distance:\t%0.4f\tMismatching nodes:\t%d" %(total_dist, len(difftable)))
       
def populate_args(diff_args_p):

    diff_args = diff_args_p.add_argument_group("DIFF GENERAL OPTIONS")
        
    diff_args.add_argument("--ref_attr", dest="ref_attr",
                        default = "name", 
                        help=("Defines the attribute in REFERENCE tree that will be used"
                              " to perform the comparison"))
    
    diff_args.add_argument("--target_attr", dest="target_attr",
                        default = "name",
                        help=("Defines the attribute in TARGET tree that will be used"
                              " to perform the comparison"))
    
    diff_args.add_argument("--fullsearch", dest="fullsearch",
                        action="store_true",
                        help=("Enable this option if duplicated attributes (i.e. name)"
                              "exist in reference or target trees."))
    
    diff_args.add_argument("--quiet", dest="quiet",
                        action="store_true",
                        help="Do not show process information")
    
    diff_args.add_argument("--report", dest="report",
                        choices=["topology", "diffs", "diffs_tab", "summary","table"],
                        default = "topology",
                        help="Different format for the comparison results")

    diff_args.add_argument("--ncbi", dest="ncbi",
                        action="store_true",
                        help="If enabled, it will use the ETE ncbi_taxonomy module to for ncbi taxid translation")

    diff_args.add_argument("--color", dest="color",
                        action="store_true",
                        help="If enabled, it will use colors in some of the report")
    
    diff_args.add_argument("--distance", dest="distance",
                           type=str, choices= ['e', 'rf', 'eb'], default='e',
                           help=('Distance measure: e = Euclidean distance, rf = Robinson-Foulds symetric distance'
                                 ' eb = Euclidean distance + branch length difference between disjoint leaves'))
    
    diff_args.add_argument("--branch-distance", dest="branchdist",
                           choices=[None,"get_distances1","get_distances2"],
                           default=None,
                           help=("Extend report with branch distance after node comparison."
                                " Select between None, get_distances1 and get_distances2."
                                " None by default"))
    
    diff_args.add_argument("-C", "--cpu", dest="maxjobs", type=int,
                            default=1, help="Maximum number of CPU/jobs"
                            " available in the execution host. If higher"
                            " than 1, tasks with multi-threading"
                            " capabilities will enabled. Note that this"
                            " number will work as a hard limit for all applications,"
                            " regardless of their specific configuration.")
    
def run(args):
    
    if (not args.ref_trees or not args.src_trees):
        logging.warning("Target tree (-s) or reference tree (-r) weren't introduced. You can find the arguments in the help command (-h)")
               
    else:
           
        for rtree in args.ref_trees:
        
            t1 = Tree(rtree,format=args.ref_newick_format)
            
            for ttree in args.src_trees:

                t2 = Tree(ttree,format=args.src_newick_format)         

                if args.quiet:
                    logging.basicConfig(format='%(message)s', level=logging.WARNING)
                else:
                    logging.basicConfig(format='%(message)s', level=logging.INFO)
                log = logging

                # Set maximun number of jobs
                if args.maxjobs == -1:
                    maxjobs = mp.cpu_count()
                else:
                    maxjobs = args.maxjobs          

                if args.ncbi:
                    from common import ncbi
                    ncbi.connect_database()

                rattr, tattr = args.ref_attr, args.target_attr

                if args.ncbi:

                    taxids = set([getattr(leaf, rattr) for leaf in t1.iter_leaves()])
                    taxids.update([getattr(leaf, tattr) for leaf in t2.iter_leaves()])
                    taxid2name = ncbi.get_taxid_translator(taxids)
                    for leaf in  t1.get_leaves()+t2.get_leaves():
                        try:
                            leaf.name=taxid2name.get(int(leaf.name), leaf.name)
                        except ValueError:
                            pass       

                if args.distance == 'e':
                    dist_fn = EUCL_DIST
                elif args.distance == 'rf':
                    dist_fn = RF_DIST
                elif args.distance == 'eb':
                    dist_fn = EUCL_DIST_B
                    
                if args.branchdist == "get_distances1":
                    branchdist = get_distances1
                elif args.branchdist == "get_distances2":
                    branchdist = get_distances2
                else:
                    branchdist = None

                difftable = treediff(t1, t2, rattr, tattr, dist_fn, args.fullsearch, branchdist=branchdist,jobs=maxjobs)

                if len(difftable) != 0:
                    if args.report == "topology":
                        show_difftable_topo(difftable, rattr, tattr, usecolor=args.color,branchdist=branchdist)
                    elif args.report == "diffs":
                        show_difftable(difftable, branchdist=branchdist)
                    elif args.report == "diffs_tab":
                        show_difftable_tab(difftable, branchdist=branchdist)
                    elif args.report == 'table':
                        rf, rf_max, _, _, _, _, _ = t1.robinson_foulds(t2, attr_t1=rattr, attr_t2=tattr)
                        show_difftable_summary(difftable, rf, rf_max, branchdist=branchdist)
                else:
                    log.info("Difference between (Reference) %s and (Target) %s returned no results" % (rtree, ttree))


