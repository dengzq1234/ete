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
import textwrap
import argparse
import logging
from tqdm import tqdm
log = logging.Logger("main")

from ..coretype.tree_diff import treediff, EUCL_DIST, EUCL_DIST_B, RF_DIST, get_distances1, get_distances2

DESC = ""

def sepstring(items, sep=", "):
    return sep.join(sorted(map(str, items)))


### REPORTS ###

def show_difftable_summary(difftable, rf=-1, rf_max=-1, branchdist=False):
    showtable = []

    total_dist = 0
    total_bdist = 0
    avg_dist = 0
    
    for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
        total_dist += dist
        total_bdist += b_dist
    
    if len(difftable) > 0:
        avg_dist = total_dist/len(difftable)

    if branchdist:
        log.info("\n"+"\t".join(["Average Dist", "Total Dist", "Branch Dist", "Mismatches", "RF", "maxRF"]))
        #print("\n"+"\t".join(["Dist", "Branch Dist", "Mismatches", "RF", "maxRF"]))
        #print("%0.6f\t%0.6f\t%10d\t%d\t%d" %(total_dist,total_bdist, len(difftable), rf, rf_max))
        showtable.append([avg_dist, total_dist, total_bdist, len(difftable), rf, rf_max])

    else:
        log.info("\n"+"\t".join(["Average Dist", "Total Dist", "Branch Dist", "Mismatches", "RF", "maxRF"]))
        #print("\n"+"\t".join(["Dist", "Mismatches", "RF", "maxRF"]))
        #print("%0.6f\t%10d\t%d\t%d" %(total_dist, len(difftable), rf, rf_max))
        showtable.append([avg_dist, total_dist, len(difftable), rf, rf_max])

    return showtable

def show_difftable_topo(difftable, attr1, attr2, usecolor=False, branchdist=False):
    if not difftable:
        return
    showtable = []
    maxcolwidth = 80
    total_dist = 0
    for dist, b_dist, side1, side2, diff, n1, n2 in sorted(difftable, reverse=True):
        total_dist += dist
        #n1 = Tree(n1.write(features=[attr1]))
        #n2 = Tree(n2.write(features=[attr2]))
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

        topo1 = n1.get_ascii(show_internal=True, compact=False)
        topo2 = n2.get_ascii(show_internal=True, compact=False)

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
    
    # if branchdist:
    #     print_table(showtable, header=["Dist", "Branch Dist", "#Diffs", "Tree1", "Tree2"],
    #                 max_col_width=maxcolwidth, wrap_style="wrap", row_line=True) 
    # else:
    #     print_table(showtable, header=["Dist", "#Diffs", "Tree1", "Tree2"],
    #                 max_col_width=maxcolwidth, wrap_style="wrap", row_line=True)    
        
    log.info("Total euclidean distance:\t%0.4f\tMismatching nodes:\t%d" %(total_dist, len(difftable)))

    return showtable #ziqi

def show_difftable(difftable, branchdist=False):
    showtable = []
    if branchdist:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            if n1.name == '':
                n1.name = 'Null'
            if n2.name == '':
                n2.name = 'Null'
            # header: ["Dist", "Branch Dist", "Internal1", "Internal2", "Size1", "Size2", "ndiffs", "refTree", "targetTree", "Diff"]    
            showtable.append([dist, b_dist, n1.name, n2.name, len(side1), len(side2), len(diff), sepstring(diff)])
        # print_table(showtable, header=["Dist", "Branch Dist", "Size1", "Size2", "ndiffs", "Diff"],
        #             max_col_width=80, wrap_style="wrap", row_line=True)
        
    else:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            if n1.name == '':
                n1.name = 'Null'
            if n2.name == '':
                n2.name = 'Null'

            showtable.append([dist, n1.name, n2.name, len(side1), len(side2), len(diff), sepstring(diff)])
        # print_table(showtable, header=["Dist", "Size1", "Size2", "ndiffs", "Diff"],
        #             max_col_width=80, wrap_style="wrap", row_line=True)
    return showtable

def show_difftable_tab(difftable, branchdist=False):
    showtable = []
    
    if branchdist:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            if n1.name == '':
                n1.name = 'Null'
            if n2.name == '':
                n2.name = 'Null'

            # header: ["Dist", "Branch Dist", "Internal1", "Internal2", "Size1", "Size2", "ndiffs", "refTree", "targetTree", "Diff"]
            showtable.append([dist, b_dist, n1.name, n2.name, len(side1), len(side2), len(diff),
                            sepstring(side1, "|"), sepstring(side2, "|"),
                            sepstring(diff, "|")])
        
        #print('#' + '\t'.join(["Dist", "Branch Dist", "Size1", "Size2", "ndiffs", "refTree", "targetTree", "Diff"]))
    else:
        for dist, b_dist, side1, side2, diff, n1, n2 in difftable:
            if n1.name == '':
                n1.name = 'Null'
            if n2.name == '':
                n2.name = 'Null'
                
            # header: ["Dist", "Internal1", "Internal2", "Size1", "Size2", "ndiffs", "refTree", "targetTree", "Diff"]
            showtable.append([dist, n1.name, n2.name, len(side1), len(side2), len(diff),
                            sepstring(side1, "|"), sepstring(side2, "|"),
                            sepstring(diff, "|")])

        
    #print('\n'.join(['\t'.join(map(str, items)) for items in showtable]))
    return showtable

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
                        choices=["topology", "diffs", "diffs_tab","table"], #remove summary because it's part of the table
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
    
    diff_args.add_argument("--unrooted", dest="unrooted",
                              action = "store_true",
                              help="""compare trees as unrooted""")

    diff_args.add_argument("-C", "--cpu", dest="maxjobs", type=int,
                            default=1, help="Maximum number of CPU/jobs"
                            " available in the execution host. If higher"
                            " than 1, tasks with multi-threading"
                            " capabilities will enabled. Note that this"
                            " number will work as a hard limit for all applications,"
                            " regardless of their specific configuration.")
    
def run(args):
    
    if (not args.ref_trees or not args.src_trees):
        logging.warning("Target tree (-t) or reference tree (-r) weren't introduced. You can find the arguments in the help command (-h)") 
               
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
                        display_result = show_difftable_topo(difftable, rattr, tattr, usecolor=args.color,branchdist=branchdist)
                        if branchdist:
                            print_table(display_result, header=["Dist", "Branch Dist", "#Diffs", "Tree1", "Tree2"],
                                        max_col_width=80, wrap_style="wrap", row_line=True) 
                        else:
                            print_table(display_result, header=["Dist", "#Diffs", "Tree1", "Tree2"],
                                        max_col_width=80, wrap_style="wrap", row_line=True) 

                    elif args.report == "diffs":
                        display_result = show_difftable(difftable, branchdist=branchdist)
                        if branchdist:
                            print_table(display_result, header=["Dist", "Branch Dist", "Internal1", "Internal2", "Size1", "Size2", "ndiffs", "Diff"], 
                            max_col_width=80, wrap_style="wrap", row_line=True)
                        else:
                            print_table(display_result, header=["Dist", "Internal1", "Internal2", "Size1", "Size2", "ndiffs", "Diff"],
                            max_col_width=80, wrap_style="wrap", row_line=True)

                    elif args.report == "diffs_tab":
                        display_result = show_difftable_tab(difftable, branchdist=branchdist)
                        if branchdist:
                            print('#' + '\t'.join(["Dist", "Branch Dist", "Internal1", "Internal2", "Size1", "Size2", "ndiffs", "refTree", "targetTree", "Diff"]))
                        else:
                            print('#' + '\t'.join(["Dist", "Internal1", "Internal2", "Size1", "Size2", "ndiffs",  "refTree", "targetTree", "Diff" ])) 
                        print('\n'.join(['\t'.join(map(str, items)) for items in display_result]))

                    elif args.report == 'table':
                        unrooted_trees = args.unrooted
                        rf, rf_max, _, _, _, _, _ = t1.robinson_foulds(t2, attr_t1=rattr, attr_t2=tattr, unrooted_trees=unrooted_trees)
                        display_results = show_difftable_summary(difftable, rf, rf_max, branchdist=branchdist)
                        for display_result in display_results:
                            if branchdist:
                                print("\n"+"\t".join(["Average Dist", "Total Dist", "Branch Dist", "Mismatches", "RF", "maxRF"]))
                                print("%0.6f\t%0.6f\t%0.6f\t%10d\t%d\t%d" %(display_result[0],display_result[1], display_result[2], display_result[3], display_result[4], display_result[5]))

                            else:
                                print("\n"+"\t".join(["Average Dist", "Total Dist", "Mismatches", "RF", "maxRF"]))
                                print("%0.6f\t%0.6f\t%10d\t%d\t%d" %(display_result[0],display_result[1], display_result[2], display_result[3], display_result[4]))


                else:
                    log.info("Difference between (Reference) %s and (Target) %s returned no results" % (rtree, ttree))


