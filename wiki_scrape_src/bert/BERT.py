'''

module for BERT relation extractor model

relations_dict (in relations.pkl)
{'Component-Whole(e2,e1)\n': 0, 'Other\n': 1, 'Instrument-Agency(e2,e1)\n': 2,
'Member-Collection(e1,e2)\n': 3, 'Cause-Effect(e2,e1)\n':
4, 'Entity-Destination(e1,e2)\n': 5, 'Content-Container(e1,e2)\n':
6, 'Message-Topic(e1,e2)\n': 7, 'Product-Producer(e2,e1)\n':
8, 'Member-Collection(e2,e1)\n': 9, 'Entity-Origin(e1,e2)\n':
10, 'Cause-Effect(e1,e2)\n': 11, 'Component-Whole(e1,e2)\n':
12, 'Message-Topic(e2,e1)\n': 13, 'Product-Producer(e1,e2)\n':
14, 'Entity-Origin(e2,e1)\n': 15, 'Content-Container(e2,e1)\n':
16, 'Instrument-Agency(e1,e2)\n': 17, 'Entity-Destination(e2,e1)\n': 18}

making decisions on BERT relations, principal: E1, component under
consideration: E2:
    probably not used outright: (1) (other); (2,17) (instrument-agency);
    (4,11) (cause-effect); (7,13) (message-topic)
    0: E2 is a component of E1: recurse
    3: E1 is a member of collection E2: pass
    5: E1 is for destination E2: pass (recall also the ~confusion re 'brick'
        and "bricks go in a kiln"--lots of 5's)
    6: E1 goes in container E2: pass
    8: E1 produces E2: pass (eg with 'foundry' and "foundries make metal"
        --though imperfect)
    9: E2 is a member of collection E1: recurse
    10: E1 comes from origin E2: recurse (recall example re 'brick' and
        "bricks come from clay")
    12: E1 is a component of E2: pass
    14: E2 produces E1: recurse
    15: E2 comes from origin E1: pass
    16: E2 goes in container E1: recurse
    18: E2 is for destination E1: recurse
    SUMMARY:
    () recurse on: {0,9,10,14,16,18}
    () pass on: {1,2,3,4,5,6,7,8,11,12,13,15,17}

'''

import sys
import pickle
import math
import logging


logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
frmtr = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
hdlr.setFormatter(frmtr)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)
logger.propagate = False


# TODO: assign pathname with BERT-Relation-Extraction folder to BERT_repo_path
# eg: BERT_repo_path = '/mycomputer/home/github_clones/BERT-Relation-Extraction
BERT_repo_path = ''


sys.path.append(BERT_repo_path)
from src.tasks.infer import infer_from_trained


# kludge for passing arguments to infer_from_trained function
class args:
    pass


class AskBert:

    def __init__(self):
        # fetch relations lookup dictionary
        with open(BERT_repo_path+'/data/'
                  'relations.pkl','rb') as fp:
            rel_map = pickle.load(fp)   # object of Relations_Mapper class
        self.relations_dict = rel_map.idx2rel
        args.model_no = 0
        args.model_size = 'bert-base-uncased'
        args.num_classes = 19
        self.inferer = infer_from_trained(args, detect_entities=False)

    def get_relation(self,labeled_sentence,text_form = False):
        # expects a properly labeled sentence containing two phrases,
        # one delimited by [E1]...[/E1] and the other by [E2]...[/E2] tags
        # if text_form is True, returns textual form of the relation
        # (eg 'Entity-Destination(e1,e2)\n'), otherwise integer corresponding
        # to entry in relations_dict dictionary
        x = self.inferer.infer_sentence(labeled_sentence, detect_entities=False)
        if text_form:
            return self.relations_dict[x]
        else:
            return x

    def consensus_relation_new(self,rel_lst):
        # given one or more relation classes in list rel_lst (in integer
        # form; so eg 0 = 'Component-Whole(e2,e1)\n', etc, from relations_dict),
        # try to form a consensus (w/ perhaps degree of confidence indicator)
        # --most relevant relations are of type
        #       container-contained:
        #       Component-Whole (0,e2/e1; 12,e1/e2)
        #       Member-Collection (3,e1/e2; 9,e2/e1)
        #       Content-Container (6,e1/e2; 16,e2/e1)
        #       Entity-Destination (5,e1/e2; 18,e2/e1)
        #       used by:
        #       Instrument-Agency (2,e2/e1; 17,e1/e2)
        #       factory-product:
        #       Entity-Origin (10,e1/e2; 15,e2/e1)
        #       Product-Producer (8,e2/e1; 14,e1/e2)
        if not rel_lst: # list should not be empty
            logger.warning("empty list sent to consensus_relation")
        '''
        # without instrument-agency:
        ambiguous_labels = {1,2,4,7,11,13,17}   # categories that will not be
        # used for determining consensus; includes 1 (Other), obviously,
        # along with cause/effect, message/topic, and possibly
        # instrument/agency
        recurse_labels = {0,9,10,14,16,18} # categories to recurse on
        pass_labels = {3,5,6,8,12,15} # categories to pass (don't recurse) on
        '''
        # with instrument-agency
        ambiguous_labels = {1, 4, 7, 11, 13,
                            }  # categories that will not be
        # used for determining consensus; includes 1 (Other), obviously,
        # along with cause/effect, message/topic, and possibly
        # instrument/agency
        recurse_labels = {0, 2, 9, 10, 14, 16, 18}  # categories to recurse on
        pass_labels = {3, 5, 6, 8, 12,
                       15, 17}  # categories to pass (don't recurse) on
        usable_labels = [ix for ix in rel_lst if ix not in ambiguous_labels]
        if not usable_labels:
            return 'pass'
        unique_labels = {}
        for label in usable_labels:
            if label not in unique_labels:
                unique_labels[label] = 1
            else:
                unique_labels[label] += 1
        # check for simple majority
        ul_flat = [(key,value) for key,value in unique_labels.items()]
        mx_fq = max([x[1] for x in ul_flat])
        max_entries = [x for x in ul_flat if x[1]==mx_fq]    # all
        # flattened dictionary tuples that have occurrence frequencies in
        # rel_lst equal to the max frequency
        if len(max_entries)==1: # has a simple majority entry
            maj_cat = max_entries[0][0] # integer label for majority category
            if maj_cat in recurse_labels:
                return 'recurse'
            else:
                return 'pass'
        else:   # there are ties
            maj_cats = [x[0] for x in max_entries]
            num_rec = len([x for x in maj_cats if x in recurse_labels])
            prp_rec = num_rec / len(maj_cats)
            if prp_rec>0.5: # require more than half of tied categories to be
                # in the recurse group
                return 'recurse'
            else:
                return 'pass'

    def get_rel_dct(self):
        # returns relations dictionary, {'Component-Whole(e2,e1)\n': 0,
        #  'Other\n': 1, ...}
        return self.relations_dict
