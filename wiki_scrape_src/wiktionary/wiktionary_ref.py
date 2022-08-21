'''
a fairly robust singular / plural complementation module; this uses wiktionary

member function check_ps_comp accepts a phrase and returns a tuple
corresponding to result of plural / singular complementarity check

note, this does not deal with initialisms--ie it precludes from checking
wiktionary definitions of type "Initialism of ..."

note, wiktionary however is not perfect; there are some ~creative plurals
listed (eg "house"->"hice" (to the logic of mouse->mice))
'''

import json
import logging
import importlib.resources


logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
frmtr = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
hdlr.setFormatter(frmtr)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def check_ps_comp(in_phrase):
    # in_phrase is the phrase we're interested in finding the plural / singular
    # complement of (usually it will/should be a single word, usually
    # lowercased--case is determined prior to calling this)
    # returns a three element ~tuple:
    # flag--
    #   NNF--in wiktionary but no noun form;
    #   FNF--found in wiktionary, and has noun form
    # plurals_list--list of plural forms of word
    # singulars_list--list of singulars forms of word
    '''
    possibilities:
        --in_phrase is not in wiktionary--this will basically cause immediate
        return; if the word is not found in wiktionary, it's probably
        ~unusual, and/or not worth pursuing for ps complement
        --in dictionary, with noun form, but no complement found (both
        lists empty)--this suggests (a) ps complement exists, but wiktionary
        doesn't know about it (~unlikely); (b) the ps complement exists but
        has unique meaning (eg brickwork / brickworks); (c) there is no ps
        complement (eg certain uncountable nouns, like 'news')
        --in dictionary, with noun form, both lists non-empty--words where
        this can occur?
        --in dictionary, with noun form, only one list non-empty--~ideal
        case; eg in_phrase = octopus, giving ~longish list in plurals_list
    '''
    if in_phrase not in ps_dict: # ps_dict from outer scope
        return ('NIW',[],[])
    else:
        return ps_dict[in_phrase]


# load singular / plural complementation dictionary, parsed from a dump ca
# early 2022 of Wiktionary; note, this file / dictionary can be ~sizeable,
# ca 40MB
raw_txt = importlib.resources.read_text("wiki_scrape_src.data",
                                    "ps_wiktionary.json")   # intra-package
                                                            # fetch
ps_dict = json.loads(raw_txt)
