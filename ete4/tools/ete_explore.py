from .common import src_tree_iterator

from ete4 import PhyloTree
from ete4.smartview import TreeStyle
from ete4.smartview.gui.server import run_smartview

DESC = "Launches an instance of the ETE smartview tree explorer server."

def populate_args(explore_args_p):
    explore_args_p.add_argument(
        "--face", action="append",
        help=("adds a face to the selected nodes; example: --face "
              "'value:@dist, pos:b-top, color:red, size:10, if:@dist>0.9'"))

def parse_metadata(metadata):
    metatable = []
    tsv_file = open(metadata)
    read_tsv = csv.DictReader(tsv_file, delimiter="\t")

    for row in read_tsv:
        metatable.append(row)
    tsv_file.close()

    return metatable, read_tsv.fieldnames

def parse_fasta(fastafile):
    fasta_dict = {}
    with open(fastafile,'r') as f:
        head = ''
        seq = ''
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    fasta_dict[head] = seq
                    seq = ''
                    head = line[1:]
                else:
                    head = line[1:]
            else:
                seq += line
    fasta_dict[head] = seq

    return fasta_dict

def add_annotations(t, metadata):
    # add props to leaf
    annotations, columns = parse_metadata(metadata)

    #['#query', 'seed_ortholog', 'evalue', 'score', 'eggNOG_OGs', 'max_annot_lvl', 'COG_category', 'Description', \
    # 'Preferred_name', 'GOs', 'EC', 'KEGG_ko', 'KEGG_Pathway', 'KEGG_Module', 'KEGG_Reaction', 'KEGG_rclass', \
    # 'BRITE', 'KEGG_TC', 'CAZy', 'BiGG_Reaction', 'PFAMs']
    for annotation in annotations:
        gene_name = next(iter(annotation.items()))[1] #gene name must be on first column
        try:
            target_node = t.search_nodes(name=gene_name)[0]
            for _ in range(1, len(columns)):
                if columns[_] == 'seed_ortholog': # only for emapper annotations
                    taxid, gene = annotation[columns[_]].split('.', 1)
                    target_node.add_prop('taxid', taxid)
                    target_node.add_prop('gene', gene)
                target_node.add_prop(columns[_], annotation[columns[_]])
        except:
            pass


def run(args):

    # VISUALIZATION
    # Basic tree style
    ts = TreeStyle()
    ts.show_leaf_name = True
    
    try:
        tfile = next(src_tree_iterator(args))
    except StopIteration:
        run_smartview()
    else:
        t = PhyloTree(open(tfile), parser=args.src_newick_format)
        t.explore(name=tfile)
        try:
            input('Running ete explorer. Press enter to finish the session.\n')
        except KeyboardInterrupt:
            pass  # it's okay if the user exits with Ctrl+C too
