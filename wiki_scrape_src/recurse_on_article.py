'''
module for main function (and sub-functions) to handle processing an article;
this will produce:
    a list of valid wikipedia URL suffixes for recursion (ie URL text
    after /wiki/)

    nb: there was some debate re case sensitivity and phrase searches in
word_annot_list; for now, (a) make multi-word phrase searches case-insensitive;
(b) leave case sensitivity for 1-word phrases intact

    a note on URL suffixes, for eg URL objects in anchor tag lists:
    () in general intra-wiki URL's will be stored with only the trailing
        suffix--only retaining what's past the '/wiki/'
    () in general, URL suffixes will be stored in readable text (no %26, etc.)
        the only place there will be a URL-raw format is ".can_url" attribute
        in anchor tag objects (?)
    () the parsed phrase associated with any URL in an anchor tag object
        (ie in individual url_obj objects) will be of readable text (no %'s),
        and only retain words past the last #-sign (and ~likely remove
        disambiguation qualifiers--not 'Mortar (cement)' but 'Mortar')
'''

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
#import wikipediaapi
import logging
import urllib
import copy
import csv
import importlib.resources
import wiki_scrape_src.wiktionary.wiktionary_ref as wiktionary_ref
import wiki_scrape_src.article_create as article_create   # for WikiArticle
# class
import wiki_scrape_src.bert.BERT as BERT   # for relation extraction model

logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
frmtr = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
hdlr.setFormatter(frmtr)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)    # set to DEBUG for most / all info
logger.propagate = False

wordnet_lemmatizer = WordNetLemmatizer()
#wiki_API = wikipediaapi.Wikipedia('en')
ask_bert = BERT.AskBert()

stop_words = set(stopwords.words('english'))

FREQ_DICT_MAX = 50000   # how many top entries to include from main corpus
# frequency dictionary
LEM_FREQ_CUTOFF = 2 # for word lift threshold; require more than this number
# of lemma instances for a word to be counted in word lift calculations
CORPUS_FREQ_CUTOFF = 0.000005  # eg word 35000 in the corpus has freq ca 1e-6
ACR_ANC_CUTOFF = 10 # frequency cutoff for acronyms and anchors
SNG_WRD_LIFT_CUTOFF = 250   # lift / TF/IDF score for selecting 1-grams
#word_annot_list = []  # main ~workhorse word list, containing tokenized
# elements from the text






class WalWord:

    # class for individual word-tokens

    def __init__(self, wrd, pos='', NE=False, lem='', snt_idx=None):
        self.wrd = wrd
        self.lem = lem  # lemmatized form of word; nb: this ~should be
        # lower-cased (not checked for, more of an instantiation requirement)
        self.anch = []
        self.acro = []
        self.pos = pos  # part of speech
        self.NE = NE  # named entity boolean
        self.sent_num = snt_idx    # sentence # word is in, if any

    def get_wrd(self):
        return self.wrd

    def get_lem(self):
        return self.lem

    def __str__(self):
        if len(self.anch) > 0:
            anch = '<a>'
        else:
            anch = ''
        if len(self.acro) > 0:
            acro = '<n>'
        else:
            acro = ''
        an_str = anch + ' ' + acro
        if an_str:
            return "%s: %s" % (self.wrd, self.pos) + " / " + an_str
        else:
            return "%s: %s" % (self.wrd, self.pos)

    def __repr__(self):
        return str(self)


class WikiUrlElement:
    # class for individual intra-wiki url objects; can handle both full
    # url's, and shortened '/wiki/...' url's
    def __init__(self,in_url):
        in_url = re.sub(r'.*/wiki/', '', in_url) # in suffix form,
        # eg Integrated_circuit or Snoopy#Red_Baron (subsection)
        self.url = urllib.parse.unquote(in_url)  # resolves URL encoding, %'s
        self.url_to_phrase(in_url)    # phrase parsed from url
        # suffix; note, this may include hyphens
        #self.phrase--set by url_to_phrase

    def get_url(self):
        return self.url

    def url_to_phrase(self,in_url):
        #    converts in_url to a phrase (eg Integrated_circuit to integrated
        # circuit); note, it doesn't make much sense to retain subheading
        # pound (#) signs, so only retain last of phrase beyond last # sign
        #     note this also removes disambiguation parentheticals for the
        #  .phrase element (eg 'Mortar_(masonry)' is saved as 'mortar')
        #     a ~problem is the ~URL-friendly wiki hyperlink shorthand uses
        # "%.." syntax for ~unusual symbols (like dashes or apostrophes)
        # URL encoding replaces unsafe ASCII characters with a "%"
        # followed by two hexadecimal digits.
        url_txt = in_url.replace('_', ' ')  # switch _ for space
        url_txt = url_txt.split('#')[-1]
        url_txt = re.sub(r'\(.*\)$','',url_txt) # cut out deambiguifiers
        tmp_spl = url_txt.split()   # split on spaces; always a list
        # the following attempts to figure out capitalization from
        # the parsed wiki URL phrase
        if len(tmp_spl) > 1:
            if re.match(r'[A-Z]',  # 2nd word starts w/ UC
                      tmp_spl[1][0]):
                self.phrase = url_txt   # preserve case
            else:
                self.phrase = tmp_spl[0].lower()+' '+' '.join(tmp_spl[1:])
        else:
            self.phrase = tmp_spl[0].lower()
        # NB: this will sometimes get capitalization wrong--the
        # problem is wiki intra-links start first word with capital
        # letter, always, so it's hard to tell whether it's truly
        # capitalized or not

    def get_phrase(self):
        # returns ~cleaned phrase, derived from url suffix
        return self.phrase


class PhraseElement:

    def __init__(self,phrase):
        self.phrase = phrase    # a string
        self.ph_check = 'NC' # flag for whether the phrase was searched for
        # in the text
        self.ps_check = 'NC' # flag for reporting whether plural/singular
        # complement has been checked for this phrase; 'NC' =
        # not checked; 'CK' = checked
        self.ps_comp = [] # list of ps complements for the phrase; if
        # .ps_check = 'CK' and this list is empty, this means there are no ps
        # complements (eg "news"; otoh octopus might have several)
        self.freq = 0   # frequency count of this phrase in the text; w/
        # phrases_checked keeping track of an individual phrase's frequency
        # in the text, the data for this value comes from looking up .phrase
        # in phrases_checked

    def get_phrase(self):
        return self.phrase

    def set_ph_checked(self):
        # to indicate this phrase was searched for in the text
        self.ps_check = 'CK'

    def get_ps_status(self):
        return self.ps_check

    def get_ps_comp(self):
        # returns list of ps complements
        return self.ps_comp

    def set_ps_comp(self,in_psc):
        # expects a single phrase or a list of phrases (strings)
        # this will treat receiving None to mean no ps complement exists or
        # is intended (for .phrase)
        if in_psc:
            if type(in_psc) is list:
                self.ps_comp = in_psc
            else:
                self.ps_comp = [in_psc]
        self.ps_check = 'CK'

    def set_freq(self,in_num):
        self.freq = in_num

    def get_freq(self):
        return self.freq


class AnchorElement:

    # class for html anchor elements
    # .can_url--canonical url in raw form--can include #'s, hyphens,
    #     and URL-text % symbols
    # url_list--list of wiki_url_obj objects
    # .acro--any acronyms associated with this entity
    # .freq--frequency the phrases of url_list appear in the text
    # .NEN--whether detected a named entity or not
    # .status--'normal' for typical anchor tags encountered in the text;
    # 'principal' for the article itself--ie the url and phrases associated
    # with the wikipedia article under examination
    # .wal_ix--for saving a set of tuples re finding phrases for this
    # anchor element in a given article's text (may be subsumed by individual
    # phrase elements keeping track of this though)
    #
    # note, the canonical URL is as-is (mixed case), while url_obj in
    # url_list have their phrases (but not URL's) all lower-cased
    # note, the canonical URL will be stored as a post-/wiki/ suffix, in the
    # .can_url attribute; the article suffix url will be stored as a url
    # element in the url_list

    def __init__(self, strs, links=None, can_url=None, status='normal'):
        # strs should be a list of phrases (strings)
        # can URL (if supplied) should be the as-is wiki URL suffix string
        self.phrases = []   # list of phrase element objects
        for phrase in strs:
            self.phrases.append(PhraseElement(phrase))
            # phrases assoc w/ anchor tag;
            # strs should be a compatible iterable
        if links:
            self.url_list = links  # list of wikipedia URL objects; *includes*
        # canonical url too (if not 'None')
        if can_url:
            self.set_canonical(can_url) # this converts can_url into just
            # suffix, post /wiki/ form
        else:
            self.can_url = None
        self.freq = 0
        self.NEN = False
        self.status = status
        self.acro = []  # for cross-referencing matches to acro list entries;
        # holds index values of elements in acro list
        self.ps_comp = 'NC' # for possibility of just having a general ps
        # complement 'global anchor object' flag for the whole anchor tag (vs
        # per phrase)
        # ps_comp is 'NC' if complementation not checked; 'OK' if complements
        # are the same entity; 'DF' if complements refer to different entities

    def inspect_anchor(self,phrases_checked):
        print("anchor phrases:")
        for phrase_el in self.phrases:
            frq = len(phrases_checked[phrase_el.get_phrase()])
            print("%s: ps--%s: freq--%s" % (phrase_el.get_phrase(),
                  phrase_el.get_ps_comp(),frq))

    def set_canonical(self,in_url):
        # sets canonical URL for this anchor tag
        self.can_url = re.sub(r'.*/wiki/', '', in_url) # in suffix form,
        # eg Integrated_circuit or Snoopy#Red_Baron (subsection); can also
        # include % symbols (not converted to printable text)

    def get_canonical(self):
        # returns printable URL suffix from canonical URL string (recall
        # canonical URL is stored ~raw)
        if self.can_url:
            return urllib.parse.unquote(self.can_url)
        else:
            return None

    def get_phrases(self):
        # fetch list of phrase elements
        return self.phrases

    def get_all_phrases(self):
        # for a common interface w/ acronym elements; for anchor tags,
        # this method is same as get_phrases
        return self.get_phrases()

    def get_can_print(self):
        # to fetch a printable form of raw can_url
        return urllib.parse.unquote(self.can_url)

    def add_url(self,in_url):
        # places another url in the anchor tag object; auto-check for redundancy
        in_url = re.sub(r'.*/wiki/', '', in_url)
        wiki_suffix = urllib.parse.unquote(in_url)  # resolves URL encoding, %'s
        if wiki_suffix not in [x.url for x in self.url_list]:
            self.url_list.append(WikiUrlElement(wiki_suffix))

    def add_acro(self,in_acro):
        # add another acronym list index to acro list for this anchor tag
        if in_acro not in self.acro:
            self.acro.append(in_acro)

    def get_acro(self):
        # return list of indices in acronym list that are linked to this
        # anchor element
        return self.acro

    def add_phrase_element(self,in_phr_el):
        # add another phrase element to the .phrases list
        if in_phr_el.phrase not in [x.phrase for x in self.phrases]:
            self.phrases.append(in_phr_el)
            return True
        else:
            return False    # flags if phrase is already in the phrase
            # elements list; allows error checking

    def set_phrase_checked(self,in_phr):
        # sets checked flag of phrase element referenced by in_phr
        ix = self.fetch_phrase_ix(in_phr)
        if ix=='NA':
            return False    # phrase not found
        else:
            self.phrases[ix].set_ph_checked()
            return True

    def fetch_phrase_ix(self,in_phr):
        # internal function, get index of phrase in phrase list
        for ii,ph_el in enumerate(self.phrases):
            if ph_el.phrase == in_phr:
                break
        else:
            return 'NA'    # flags if in_phr is not in the phrases element list
        return ii

    def get_urls(self):
        # return the whole list of url objects
        return self.url_list

    def get_url(self,ix=0):
        # returns url of url_list[ix]; returns a string, not a url object
        try:
            return self.url_list[ix].url
        except IndexError:
            return None

    def set_nen(self,NEN):
        self.NEN = NEN

    def get_nen(self):
        return self.NEN

    def get_rep_phrase(self):
        # gets a representative phrase (requires phrase elements to have
        # assigned frequencies to really work), phrase most frequently
        # encountered in text
        pe_sort = sorted(self.get_phrases(),key=lambda x: x.get_freq(),
                         reverse=True)
        return pe_sort[0].get_phrase()

    def update_freq(self):
        # totals frequency of all phrase elements in the text
        tot_frq = 0
        for ph_el in self.phrases:
            tot_frq += ph_el.get_freq()
        self.freq = tot_frq
        return tot_frq

    def get_freq(self):
        # nb: .freq is not updated automatically
        return self.freq

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def __str__(self):
        pass
        #return str(self.phrases) + ", " + self.link + ", " + str(self.freq)

    def __repr__(self):
        pass
        #return str(self)


class AcronymElement:

    # class for suspected acronyms (eg CPU)

    def __init__(self, acro, st='', status='normal'):
        if acro[-1]=='s':    # acro for acronym proper; should be in singular
            # form prior, but can check here too (CPUs->CPU)
            acro = acro[:-1]
        phr_elm_sing = PhraseElement(acro)
        phr_elm_sing.set_ps_comp(acro+'s')
        phr_elm_plur = PhraseElement(acro+'s')
        phr_elm_plur.set_ps_comp(acro)
        self.acro = [phr_elm_sing,phr_elm_plur] # acronym literals associated
        # with the acronym
        self.phrases = [PhraseElement(st)]   # list of phrase elements
        # associated with acronym; st is string associated with acronym
        self.NEN = False
        self.status = status
        # .status--'ordinary' for typical anchor tags encountered in the text;
        # 'principal' for the article itself--ie the url and phrases associated
        # with the wikipedia article under examination
        self.freq = 0
        self.anch = []  # for cross-referencing matches to anchor list entries

    def get_phrases(self):
        # fetch associated phrases
        return self.phrases

    def get_all_phrases(self):
        # this returns all phrase elements associated with the acronym,
        # including acronym literals (for ~duck typing w/ anchor elements)
        return self.get_phrases() + self.get_acro()

    def set_phrase_checked(self,in_phr):
        # sets checked flag of phrase element referenced by in_phr
        ix = self.fetch_phrase_ix(in_phr)
        if ix=='NA':
            return False    # phrase not found
        else:
            self.phrases[ix].set_ph_checked()
            return True

    def add_anch(self,in_anch):
        # add another anchor list index to anch list for this acro tag
        if in_anch not in self.anch:
            self.anch.append(in_anch)

    def add_phrase_element(self,in_phr_el):
        # add another phrase element to the .phrases list
        if in_phr_el.phrase not in [x.phrase for x in self.phrases]:
            self.phrases.append(in_phr_el)
            return True
        else:
            return False    # flags if phrase is already in the phrase
            # elements list; allows error checking

    def fetch_phrase_ix(self,in_phr,acro=False):
        # internal function, get index of phrase in phrase
        if acro:
            phr_lst = self.acro
        else:
            phr_lst = self.phrases
        for ii,ph_el in enumerate(phr_lst):
            if ph_el.phrase == in_phr:
                break
        else:
            return 'NA'    # flags if in_phr is not in the phrases element list
        return ii

    def get_acro(self):
        return self.acro    # returns list of phrase objects containing acronym
        # literals (a plural / singular pair)

    def get_acro_sing(self):
        # returns singular form of acronym for this acro element
        tmp_phr = self.acro[0].get_phrase()
        if tmp_phr[-1]=='s':
            tmp_phr = tmp_phr[:-1]
        return tmp_phr

    def inspect_acro(self,phrases_checked):
        print("acronyms and phrases:")
        for phrase_el in self.acro + self.phrases:
            frq = len(phrases_checked[phrase_el.get_phrase()])
            print("%s: ps--%s: freq--%s" % (phrase_el.get_phrase(),
                  phrase_el.get_ps_comp(),frq))

    def get_rep_phrase(self):
        # gets a representative phrase (requires phrase elements to have
        # assigned frequencies to really work), phrase most frequently
        # encountered in text
        pe_sort = sorted(self.get_phrases()+self.get_acro(),key=lambda x:
        x.get_freq(),reverse=True)
        return pe_sort[0].get_phrase()

    def set_nen(self,NEN):
        self.NEN = NEN

    def get_nen(self):
        return self.NEN

    def get_anch(self):
        return self.anch

    def update_freq(self):
        # totals frequency of all phrase and acro elements in the text
        tot_frq = 0
        for ph_el in self.phrases+self.acro:
            tot_frq += ph_el.get_freq()
        self.freq = tot_frq
        return tot_frq

    def get_freq(self):
        # nb: .freq is not updated automatically
        return self.freq

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    #def __str__(self):
    #    pass

    #def __repr__(self):
    #    pass


class TextArticleProcess:
    '''
    this is a consolidating class for the principal article processing
    methods of eg recurse_on_article_old_10, and previous versions
    primary attributes:
    word_annot_list--a list of word element objects, each containing a token
    from post-tokenized article text
    phrases_checked--a dictionary for keeping track of all phrases /
    n-grams checked in the text; <phrase>:<index set> w/ phrase a string and
    <index set> a set of tuples holding information on all indices in
    word_annot_list where <phrase> has been found
        tuples in <index set> are of form (start_index,len(phrase),sentence#,
    NE_bool)
        len(<index set>) gives the frequency (count) of <phrase> being found in
    the text
    sentence_list--a list of all sentences parsed via sentence tokenizer from
    the text
    acro_dict--a dictionary of acronyms, <acronym literal>:<acronym phrase>
    fw_indices--quick lookup of where in w.a.l. a given sentence begins; list
    index ii corresponds to ii'th sentence in sentence_list
    anch_dict--directly from article, <URL suffix>:<anchor tag phrase>
    acro_list--list of acronym elements
    anch_list--list of anchor elements
    acro_list_principal, anch_list_principal--for holding principal elements
    word_freq_dict--after anchor and acro phrases, NENs, etc. have been
    removed, this dict holds frequency counts of ~relevant remaining words
    word_lift_dict--a TF/IDF-type "lift" score for remaining words (1-grams)
    in word_freq_dict
    article_word_count--est count of words in article being processed
    '''

    def __init__(self,article):
        self.acro_dict = {}  # dictionary with key=acronym (eg CPU) and value=(
        # <acro phrase (de-hyphenated?)>,<plural boolean>) (w/ boolean for
        # initialization--eg CPUs)
        self.anch_dict = article.base_anch_dict # b.a.d. is very basic,
        # <wiki html suffix (no /wiki/)>:<anchor tag text>
        self.sentence_list = []
        self.fw_indices = [0]  # list of indices in word_annot_list of 1st
        # words of sentences
        self.word_annot_list = []   # primary list for tracking instances of
        # word tokens, their part of speech, relations to anchor or acronym
        # elements, in the text
        self.phrases_checked = {}  # principal container for tracking what
        # phrases have already been searched-for in the text, and associated
        # tuples of where to find each occurrence of a given phrase
        self.wrd_lst_gen(article.wiki_text, article.extra_text)
        self.anchor_list_principal = [] # for principal anchor elements
        self.acro_list_principal = []   # for principal acronym elements
        self.anchor_list = []   # list of anchor elements
        self.acro_list = []  # list of acro elements
        self.word_freq_dict = {}
        self.word_lift_dict = {}
        self.article_word_count = 0

    def wrd_lst_gen(self, w_text, e_text):
        # parses text and populates word_annot_list, a list of word element
        # objects
        # --sentence_list is for holding sentences tokenized from the clean
        # wikipedia text portion in WikiArticle object
        # --fw_indices are word_annot_list indexes of 1st words of sentences
        len_lst_sen = 0  # for fw_indices
        self.sentence_list = sent_tokenize(w_text)  # ~sentence-by-sentence
        # iterable for clean wiki text (this is just a list of sentence strings)
        for ii, sent in enumerate(self.sentence_list):  # sentence-by-sentence
            # words_list will be a list of words and punctuation, in order
            # the tokens appear in the text; eg "I like pie, cheese, and fruit."
            # becomes ['I','like','pie',',','cheese',',',...]; note, it's smart
            # enough to parse 'didn't' as 'did','n\'t' but not smart enough to
            # convert the conjunction (ie no(?) 'n\'t'->'not'); it also retains
            # hyphenated words whole
            words_list = nltk.word_tokenize(sent)
            self.sent_phrs_proc(words_list, sentence_index=ii)
            tmp_dct = self.acro_search(words_list)
            if tmp_dct:
                self.acro_dict.update(tmp_dct)  # will overwrite overlaps w/
                # tmp_dict
                # (shouldn't be any); ie acro_dict is updated every
                # sentence-pass through the main loop
            if ii > 0:
                self.fw_indices.append(len_lst_sen + self.fw_indices[ii - 1])
            len_lst_sen = len(words_list)

        for phrs in e_text.splitlines():  # phrase by phrase in extra text
            words_list = nltk.word_tokenize(phrs)
            self.sent_phrs_proc(words_list, sentence_index=None)
            tmp_dct = self.acro_search(words_list)
            if tmp_dct:
                self.acro_dict.update(tmp_dct)

    def sent_phrs_proc(self, words_list, sentence_index=None):
        # tokenizes and processes a complete sentence, or a phrase
        # tag parts of speech (POS); will be an iterable of tuples, (word,POS)
        # --principal attribute affected/updated is word_annot_list
        tagged = nltk.pos_tag(words_list)
        # chunk for named entity recognition
        # binary=True just tags everything it thinks as a NE with 'NE'; if
        # not setting binary=True, will try to resolve specifics--eg
        # George Washington gets tagged with 'PERSON', etc.
        parse_tree = nltk.ne_chunk(tagged, binary=True)  # uses case sensitive

        # here is a ~good example of a ~successful POS / ne_chunk tagging; the
        # input sentence is "Most paper clips are variations of the Gem type":
        # Tree('S', [('Most', 'JJS'), ('paper', 'NN'), ('clips', 'NNS'),
        # ('are', 'VBP'), ('variations', 'NNS'), ('of', 'IN'), ('the', 'DT'),
        # Tree('NE', [('Gem', 'NNP')]), ('type', 'NN')...
        # note how it successfully tags "Gem" as a NE; also note the tree
        # structure retains word order (but just groups terms by ~sublisting
        # them)

        # named entity parsing
        # flatten NE tree/subtree list, including NE tags; so a list
        # element will be eg ['variations','NNS','nNE'] or
        # ['Gem','NNP','NE'] etc.
        # NB: this assumes ~flat layout, w/ only binary NE tagging--no noun
        # phrase or prepositional phrase subtrees, etc. (as in stackoverflow
        # 31689621)
        for comp in parse_tree:
            if type(comp) != nltk.tree.Tree:  # not a subtree
                # WalWord constructor args: wrd,pos='',NE=False,lem=''
                self.make_lem_wp(comp, ne_bool=False,
                            sentence_index=sentence_index)  # creates new
                # WalWord object and appends it to word_annot_list
            elif comp.label() == 'NE':  # subtree detected; check label
                for sub in comp:
                    self.make_lem_wp(sub, ne_bool=True,
                                sentence_index=sentence_index)  # creates new
                    # WalWord obj and appends to word_annot_list
            else:
                print('Problem parsing nltk trees...')
                raise AttributeError

    def make_lem_wp(self, in_list, ne_bool=False, sentence_index=None):
        # expects in_list of form [<word>,<POS>]; boolean is for NE or not
        # handles creation of a new word_list_obj object, and appends into
        # word_annot_list
        # returns no value; acts on attribute word_annot_list
        tmp_pos = pos_convert(in_list[1])
        if tmp_pos is not None:
            lem_str = wordnet_lemmatizer.lemmatize(in_list[0].lower(),
                                                   pos=tmp_pos)
        else:
            lem_str = in_list[0].lower()
        self.word_annot_list.append(WalWord(in_list[0], pos=in_list[1],
                        NE=ne_bool, lem=lem_str, snt_idx=sentence_index))

    def add_acro(self,acro_el,principal=False):
        # adds an acro element to the acro elements in .acro_list_pre
        # if boolean 'principal' is false, acro_list_principal otherwise;
        # it also processes the acro element, to include its phrases
        if not principal:
            self.acro_list.append(acro_el)
        else:
            self.acro_list_principal.append(acro_el)
        self.acro_proc(acro_el)

    def acro_proc(self, acro_el):
        # processes an acronym element for where the first element in its phrase
        # list, and plural and singular forms of its acronyms occur in the text
        # (word_annot_list)
        # changes in place: acro_el (phrase_checked, .nen); phrases_checked
        phrase_dict = {'acronym_sing': acro_el.get_acro_sing(),
                       'acronym-plur': acro_el.get_acro_sing() + 's',
                       'phrase': acro_el.get_phrases()[0].get_phrase()}
        for phr_key in phrase_dict:
            tmp_ix = set()  # for holding tuples of any found matches between
            # new_anc_elm.phrase and phrases in word_annot_list; tuples of form
            # (start_index,len(phrase),sentence#,NE_bool)
            phrase = phrase_dict[phr_key]
            phrase_tok = nltk.word_tokenize(phrase)
            cpw_lst = self.check_phrase_wordlist(phrase_tok,tmp_ix)
            # c.p.w. returns [found_a_match,gen_NEN_bool]
            acro_el.set_phrase_checked(phrase)
            if phr_key == 'phrase':
                if not cpw_lst[0]:  # the .phrase should be found at least
                    # once (it came from the text after all...)
                    # (update, this can give ~false warnings for eg
                    # definitions as sentences (tho not used))
                    #logger.warning("acronym phrase %s not found at all in "
                    #               "text(?)" % phrase)
                    pass
                else:
                    acro_el.set_nen(cpw_lst[1])  # is the phrase assoc w/
                    # acronym detected as NEN?
            if phrase not in self.phrases_checked:
                self.phrases_checked[phrase] = tmp_ix

    def add_anch(self,anch_el,principal=False):
        # adds an anchor element to the anchor elements in .anchor_list_pre
        # if boolean 'principal' is false, anchor_list_principal otherwise;
        # it also processes the anchor element, to include its phrases
        if not principal:
            self.anchor_list.append(anch_el)
        else:
            self.anchor_list_principal.append(anch_el)
        self.anch_proc(anch_el)

    def anch_proc(self, anch_el):
        # processes an anchor element by searching its list of phrase elements
        # for phrases in the text (via word_annot_list)
        # changes made in place: anch_el's phrase el "phrase checked" are set;
        # phrases_checked is updated with indices in w.a.l. where .phrase is
        # found; anch_el's .nen set accordingly too
        for phrase_el in anch_el.get_phrases():
            tmp_ix = set()  # for holding tuples of any found matches between
            # new_anc_elm.phrase and phrases in word_annot_list; tuples of form
            # (start_index,len(phrase),sentence#,NE_bool)
            phrase = phrase_el.get_phrase()
            phrase_tok = nltk.word_tokenize(phrase)
            cpw_lst = self.check_phrase_wordlist(phrase_tok, tmp_ix)
            # c.p.w. returns [found_a_match,gen_bool]
            anch_el.set_phrase_checked(phrase)
            if not cpw_lst[0]:  # the .phrase should be found at least once (it
                # came from the text after all...)
                pass
                # note: ~normal exceptions include anything in square brackets
                #logger.debug(
                #    "anchor tag phrase \'%s\' not found at all in text(?)"
                #    % phrase)
            else:
                anch_el.set_nen(cpw_lst[1])
            if phrase not in self.phrases_checked:
                self.phrases_checked[phrase] = tmp_ix

    def make_pre_acro_list(self):
        # generates preliminary acronym list, from acro_dict
        # acro_dict--dictionary with key=acronym (eg CPU) and
        # value=(<acro phrase>,<plural boolean>) (w/ boolean for
        # initialization--eg CPUs)
        # create a list of acro element objects from acro_dict, and upload them
        # into list
        for acro_literal in self.acro_dict:
            new_acro_el = AcronymElement(acro_literal, self.acro_dict[
                acro_literal])
            self.add_acro(new_acro_el)  # processes and appends to acro_list_pre
        # pare acronym list for only elements not associated with NENs

    def make_pre_anch_list(self):
        # creation of a preliminary list of AnchorElement objects; this
        # simply uses the article's anchor list dictionary; main purpose is
        # to determine if anchor object (via its phrase(s)) is a named entity
        # or proper noun (NEN)
        # --phrases_checked is changed in-place; note
        # this does *not* check readable text form of URL suffix (if it's
        # different from in-text anchor phrase)--it only checks the literal
        # text phrase within resp. anchor tags
        for url_suffix in self.anch_dict:   # url_suffix is print friendly (
            # no URL %26 etc)
            url_obj = WikiUrlElement(url_suffix)
            phrase = self.anch_dict[url_suffix]
            new_anch_el = AnchorElement(strs=[phrase],links=[url_obj])
            self.add_anch(new_anch_el)

    def make_rel_anch_list(self):
        # create list of anchor tags that are either not NENs or are associated
        # with whitelisted acronyms (acronym literals are likely tagged as
        # NENs--so don't want to eliminate those)
        # --note, this assumes acronym list has been pared of NEN phrases at
        # this point--ie call this after running make_rel_acro_list
        ok_acros = [phr_el.get_phrase() for acr_el in self.acro_list for
                    phr_el in acr_el.get_acro()]  # eg ['CPU','CPUs',...]
        self.anchor_list = self.anchor_list_principal+[x for x in
                   self.anchor_list if not x.get_nen() or
                   (x.get_nen() and len([phr_el for phr_el in
                   x.get_phrases() if phr_el.get_phrase() in ok_acros]) > 0)]
        # anchor_list will contain only 'relevant' anchors, those not
        # associated with proper nouns or named entities (NE/NN(S) / NENs)

    def anch_ps(self):
        # operates on list of AnchorElement objects, anchor_list
        # this completes the search of anchor object phrase occurrences in the
        # text (word_annot_list), by checking singular / plural complements of
        # phrases in the anchor elements' .phrase list of phrase objects
        # phrase element objects are updated in place, in each anchor_list
        # object
        for anch_el in self.anchor_list:
            self.check_ps_phrase_anch(anch_el)

    def check_ps_phrase_anch(self, anch_el):
        # handles anchor element's .phrase(s), checking for validity of ps
        # complement, then searching for them in the text
        # NB: this handles phrases of more than one word by taking the last word
        # of the phrase, lowercasing it, then looking it up via wiktionary
        # module to take ps complement; case is restored for the ps
        # complement phrase
        # NB: quite a bit of overlap between the 2 check_ps_phrase functions (
        # _acro and _anch)

        pe_list = []  # list of new phrase element objects generated
        for phrase_el in anch_el.get_phrases():
            cp_list = []  # list of ps complement phrases (strings)
            phrase = phrase_el.get_phrase()
            phrase_tok = nltk.word_tokenize(phrase)
            phrase_last_word = phrase_tok[-1]
            if re.match(r'[A-Z]', phrase_last_word[0]):  # capitalized?
                cap = True
            else:
                cap = False
            wik_res = wiktionary_ref.check_ps_comp(phrase_last_word.lower())
            # returns tuple, (NIW/NNF/FNF,singulars_list,plurals_list)
            if wik_res[0] != 'FNF':  # either word not in wiktionary, or doesn't
                # have a noun form
                phrase_el.set_ps_comp(None)
            elif not wik_res[1] and not wik_res[
                2]:  # both pl/sn lists are empty
                phrase_el.set_ps_comp(None)
            elif len(phrase_tok) == 1 and len(phrase) > 1 and re.match(
                    r'^[A-Z0-9]+$', phrase):  # all caps; probably an
                # acronym-as-anchor-tag-phrase
                phrase_el.set_ps_comp(None)
            else:
                ps_list = [x for x in wik_res[1] + wik_res[2] if
                           x != phrase_last_word.lower()]  # 'generous'
                # goes through all plural / singular variants found
                for word in ps_list:
                    if cap:
                        phrase_tok[-1] = word[0].upper() + word[1:]
                    else:
                        phrase_tok[-1] = word
                    tmp_regex = r'' + re.escape(phrase_last_word) + r'$'
                    ps_phrase = re.sub(tmp_regex, phrase_tok[-1], phrase)
                    cp_list.append(ps_phrase)
                    tmp_ix = set()
                    if ps_phrase not in self.phrases_checked:
                        src_res = self.check_phrase_wordlist(phrase_tok,tmp_ix)
                        # src_res = [found_a_match,gen_bool]
                        self.phrases_checked[ps_phrase] = tmp_ix
                    new_phrase_el = PhraseElement(ps_phrase)
                    new_phrase_el.set_ph_checked()
                    new_phrase_el.set_ps_comp(phrase)
                    pe_list.append(new_phrase_el)
            phrase_el.set_ps_comp(cp_list)

        for phrase_el in pe_list:
            ret_bool = anch_el.add_phrase_element(phrase_el)
            if not ret_bool:
                logger.debug("phrase %s already in anchor list?" %
                             phrase_el.phrase)

    def make_rel_acro_list(self):
        # remove acronym phrases deemed NENs (proper nouns or named entities)
        self.acro_list = self.acro_list_principal + [acro_el for acro_el in
                                self.acro_list if not acro_el.get_nen()]

    def acro_ps(self):
        # operates on member attribute acro_list, list of AcronymElement
        # objects, finding singular / plural phrse complements and searching
        # for them in word_annot_list
        # (the literal acronyms' plural complements should have already been
        # done--eg CPU/CPUs)
        for acro_el in self.acro_list:
            self.check_ps_phrase_acro(acro_el)

    def check_ps_phrase_acro(self, acro_el):
        # handles acro element's .phrase(s), checking for validity of ps
        # complement, then searching for them in the text and updating .wix in
        # phrase element if necessary
        # note, only the acronym phrases are checked for p/s complements;
        # earlier, the literal acronyms should have already been checked re
        # complements (eg CPU and CPUs)
        # NB: this handles phrases of more than one word by taking the last word
        # of the phrase, lowercasing it, then looking it up via wiktionary
        # module to take ps complement; case is restored for the ps
        # complement phrase
        # NB: quite a bit of overlap between the 2 check_ps_phrase functions (
        # _acro and _anch)

        pe_list = []  # list of new phrase element objects generated
        for phrase_el in acro_el.get_phrases():
            cp_list = []  # list of ps complement phrases (strings)
            phrase = phrase_el.get_phrase()
            phrase_tok = nltk.word_tokenize(phrase)
            phrase_last_word = phrase_tok[-1]
            if re.match(r'[A-Z]', phrase_last_word[0]):  # capitalized?
                cap = True
            else:
                cap = False
            wik_res = wiktionary_ref.check_ps_comp(phrase_last_word.lower())
            # returns tuple, (NIW/NNF/FNF,singulars_list,plurals_list)
            if wik_res[0] != 'FNF':  # either word not in wiktionary, or doesn't
                # have a noun form
                phrase_el.set_ps_comp(None)
            elif not wik_res[1] and not wik_res[
                2]:  # both pl/sn lists are empty
                phrase_el.set_ps_comp(None)
            else:
                ps_list = [x for x in wik_res[1] + wik_res[2] if
                           x != phrase_last_word.lower()]  # 'generous' goes through
                # all plural / singular variants found
                for word in ps_list:
                    if cap:
                        phrase_tok[-1] = word[0].upper() + word[1:]
                    else:
                        phrase_tok[-1] = word
                    tmp_regex = r'' + re.escape(phrase_last_word) + r'$'
                    ps_phrase = re.sub(tmp_regex, phrase_tok[-1], phrase)
                    cp_list.append(ps_phrase)
                    tmp_ix = set()
                    if ps_phrase not in self.phrases_checked:
                        src_res = self.check_phrase_wordlist(phrase_tok,tmp_ix)
                        # (src_res is [found_a_match,gen_bool])
                        self.phrases_checked[ps_phrase] = tmp_ix
                    new_phrase_el = PhraseElement(ps_phrase)
                    new_phrase_el.set_ph_checked()
                    new_phrase_el.set_ps_comp(phrase)
                    pe_list.append(new_phrase_el)
            phrase_el.set_ps_comp(cp_list)

        for phrase_el in pe_list:
            ret_bool = acro_el.add_phrase_element(phrase_el)
            if not ret_bool:
                logger.debug("phrase %s already in anchor list?" %
                             phrase_el.phrase)

    def merge_common_url_anchors(self):
        # look for and merge any anchor tags with URLs in common
        tmp_anchor_list = []
        alu_dct = {}  # keyed by url, values a set of indices of anchor tags in
        # anchor_list
        for ii, anch_el in enumerate(self.anchor_list):
            tmp_lst = anch_el.get_urls()
            for url_obj in tmp_lst:
                tmp_url = url_obj.get_url()
                if tmp_url not in alu_dct:
                    alu_dct[tmp_url] = {ii}
                else:
                    alu_dct[tmp_url].update({ii})
        idx_set_lst = []  # holds anchor_list_rel index sets grouped by key of
        # url in common
        for url in alu_dct:
            idx_set_lst.append(alu_dct[url])  # list of sets
        glom_idx = self.anc_set_glom(idx_set_lst)  # create list of glommed sets
        # from idx_set_lst; this should be a disjoint cover of all indices in
        # anchor_list_rel
        for st in glom_idx:
            tmp_sl = list(st)
            if len(tmp_sl) > 1:
                # DEBUG
                #logger.debug("merging anchor indices %s" % st)
                mrg_anc = self.anchor_list[tmp_sl[0]]
                for ix in tmp_sl[1:]:
                    self.anch_merge(mrg_anc, self.anchor_list[ix])
                tmp_anchor_list.append(mrg_anc)
            else:
                tmp_anchor_list.append(self.anchor_list[tmp_sl[0]])
        self.anchor_list = tmp_anchor_list

    def anch_merge(self, anch_el_1, anch_el_2):
        # creates a new anchor element, merges the two anchor el arguments into
        # the new anchor element and returns it
        # --note: phrase elements within any given anchor element will be unique
        # by .phrase--this helps simplify the phrase-merge process
        out_anch_el = copy.deepcopy(anch_el_1)
        # .phrases lists merge
        phr_lst = [ph_el.get_phrase() for ph_el in out_anch_el.get_phrases()]
        for ph_el in anch_el_2.get_phrases():
            try:
                ix = phr_lst.index(ph_el.get_phrase())
            except ValueError:
                ix = None
            if ix:  # merge needed
                self.phrase_merge(out_anch_el.get_phrases()[ix], sub=ph_el)
            else:
                out_anch_el.add_phrase_element(copy.deepcopy(ph_el))
        url_lst = [ur_el.get_url() for ur_el in out_anch_el.get_urls()]
        for ur_el in anch_el_2.get_urls():
            try:
                ix = url_lst.index(ur_el.get_url())
            except ValueError:
                ix = None
            if ix:  # merge needed
                self.url_merge(out_anch_el.get_urls()[ix], sub=ur_el)
            else:
                out_anch_el.add_url(copy.deepcopy(ur_el))
        if anch_el_1.get_status() == 'principal' or anch_el_2.get_status(
        ) == 'principal':
            out_anch_el.set_status('principal')

    def anc_set_glom(self, idx_set_lst, ret_lst=None):
        # gloms sets in list of sets idx_set_lst if there are any elements in
        # common; called recursively
        if not ret_lst:
            ret_lst = []
        st_tt = copy.copy(idx_set_lst[0])
        ix_ls = [0]
        for ii, st in enumerate(idx_set_lst[1:]):
            if len(st_tt & idx_set_lst[ii + 1]) > 0:
                ix_ls.append(ii + 1)
                st_tt.update(idx_set_lst[ii + 1])
        ret_lst.append(st_tt)
        out_lst = [x for ii, x in enumerate(idx_set_lst) if ii not in ix_ls]
        if len(out_lst) > 1:
            ret_lst = self.anc_set_glom(out_lst, ret_lst)
        else:
            ret_lst = ret_lst + out_lst
        return ret_lst

    def up_merge_acro_anch(self):
        phs_dct = {}
        for ii, anch_el in enumerate(self.anchor_list):
            for ph_el in anch_el.get_phrases():
                if ph_el not in phs_dct:
                    phs_dct[ph_el.get_phrase()] = [ii]
                else:
                    phs_dct[ph_el.get_phrase()].append(ii)
        for ii, acro_el in enumerate(self.acro_list):
            for ph_el in acro_el.get_phrases() + acro_el.get_acro():
                if ph_el.get_phrase() in phs_dct:
                    ix_ls = phs_dct[ph_el.get_phrase()]
                    # logger.debug("up-merging acronym %s into anchor
                    # list's %s" % (ph_el.get_phrase(), ix_ls))
                    for ix in ix_ls:  # go through match indices in
                        # anchor_list_mrg
                        self.up_merge_an_ac(self.anchor_list[ix], acro_el)
                        acro_el.add_anch(ix)
                        self.anchor_list[ix].add_acro(ii)

        # revise list of acronyms to those not ~up-merged with anchor tags
        self.acro_list = [acro_el for acro_el in self.acro_list if len(
            acro_el.get_anch()) == 0]

    def up_merge_an_ac(self, anch_el, acro_el):
        # copies and places relevant acronym element acro_el data into anchor
        # element anch_el; change is made in-place to anch_el
        ac_pe_lst = acro_el.get_phrases() + acro_el.get_acro()
        an_phrs = [ph_el.get_phrase() for ph_el in anch_el.get_phrases()]
        for ph_el in ac_pe_lst:
            try:
                ix = an_phrs.index(ph_el.get_phrase())
            except ValueError:
                anch_el.add_phrase_element(copy.deepcopy(ph_el))
            else:  # phrase merge needed
                self.phrase_merge(anch_el.get_phrases()[ix], sub=ph_el)
        if acro_el.get_status() == ('principal' and anch_el.get_status() !=
                                    'principal'):
            anch_el.set_status('principal')

    def phrase_merge(self, ph_el_1, *, sub):
        # updates phrase element w/ ~info from "sub" phrase element--ie change
        # in place
        # assumes ps_check, ph_check, and ps_comp (and freq) are the same for
        # both phrase element inputs
        # nb: this initially did things with wal_ix w.a.l. indices; with those
        # being in phrases_checked, there's nothing to do here (unless want
        # logic w/ ps_check etc.)
        pass

    def url_merge(self, ur_el_1, *, sub):
        # updates url element ur_el_1 with contents of url element "sub"; change
        # is made in place
        # as of latest, there's nothing really to merge, so this just returns
        # ur_el_1
        pass

    def gen_freq(self):
        # assign anchor tags individual phrase elements their frequency counts
        # from phrases_checked dictionary
        for anch_el in self.anchor_list:
            for phrase_el in anch_el.get_phrases():
                phrase_el.set_freq(len(self.phrases_checked[
                                           phrase_el.get_phrase()]))
        # assign acro elements' phrase elements, both acro literals and
        # associated phrases, their frequencies:
        for acro_el in self.acro_list:
            for phrase_el in acro_el.get_phrases() + acro_el.get_acro():
                phrase_el.set_freq(len(self.phrases_checked[
                                           phrase_el.get_phrase()]))
        # tally frequencies in anchor_list_mrg and acro_list_par:
        for anch_el in self.anchor_list:
            anch_el.update_freq()
        for acro_el in self.acro_list:
            acro_el.update_freq()

    def gen_word_lift(self, corpus_freq_dict):
        # individual word lift / ~TF/IDF calculations; words "left over" after
        # removing anchor element phrases, acronym element phrases, stop words,
        # NEN's, etc.
        wal_filt, self.article_word_count = self.pare_wal()  #
        # remove all but potentially relevant words
        self.create_word_freq_dict(wal_filt)  # get frequencies by lemma
        self.create_lift_dict(corpus_freq_dict)  # a TF/IDF-type score,
        # using corpus frequencies from freq_dict

    def pare_wal(self):
        # filter word_annot_list of punctuation, stopwords, NENs, n-grams from
        # anchor elements and acronym elements, etc.
        # create total list of indices of anchor elements and acronyms to be
        # removed from w.a.l., and remove:
        ixs_rem = set([y for phr in self.phrases_checked for el in
            self.phrases_checked[phr] for y in [x + el[0] for x in
                                                range(el[1])]])
        tmp_wl = [tok for ii, tok in enumerate(self.word_annot_list) if
                  ii not in ixs_rem]
        # remove punctuation
        tmp_wl = [tok for tok in tmp_wl if not re.search(r'[^\w\s]', tok.wrd)]
        # estimate article word count
        article_word_count = len(tmp_wl) + len(ixs_rem)  # approximate--
        # punctuation could still be within phrases_checked
        # remove tokens / words with unwanted parts of speech:
        tmp_wl = [tok for tok in tmp_wl if not tok.NE and tok.pos != 'NNP' and
                  tok.pos != 'NNPS' and tok.pos[0] != 'V']
        # remove stopwords
        tmp_wl = [tok for tok in tmp_wl if tok.wrd.lower() not in stop_words]
        return tmp_wl, article_word_count

    def create_word_freq_dict(self, wal_filt):
        # grouping by lemma form of words
        # this will receive wal_filt, a list pared from word_annot_list (so of
        # word_list_obj objects), w/ stopwords and punctuation etc. removed;
        # the word_list_obj objects already have their lemmas included
        # (.lem attribute)
        # wf_dict is a dictionary of dictionaries; the main keys are the lemma
        # 'roots'--wf_dict[<lemma>] = [#,{'<a1>':#, '<a2>':#, ...}] where the
        # outer # is total frequency of lemma's antecedents, and inner #s are
        # frequencies of individual antecedents <ai> (should add up to outer #)
        for el in wal_filt:  # re case sensitivity, WalWord objects are formed
            # with their lemmatized form (.lem) in lower case (or should be)
            wrd = el.get_wrd()
            lem = el.get_lem()
            if lem not in self.word_freq_dict:
                self.word_freq_dict[lem] = [1, {wrd: 1}]
            else:
                self.word_freq_dict[lem][0] += 1
                if wrd not in self.word_freq_dict[lem][1]:
                    self.word_freq_dict[lem][1][wrd] = 1
                else:
                    self.word_freq_dict[lem][1][wrd] += 1

    def create_lift_dict(self, freq_dict):
        # performs a TF/IDF-type "lift" calculation on lemmas in wf_dict
        # this compares a word's article frequency to a weighted average of
        # corpus frequencies over all antecedents of the lemma
        self.word_lift_dict = {key: -1.0 for key in self.word_freq_dict}
        msd_wrds = 0
        for key in self.word_freq_dict:
            if self.word_freq_dict[key][0] > LEM_FREQ_CUTOFF:  # won't
                # ~bother with lift unless lemma's antecedents occur
                # sufficient # of times
                frq = 0.0
                wrd_tot = 0
                for wrd in self.word_freq_dict[key][1]:
                    try:
                        frq += freq_dict[wrd] * self.word_freq_dict[key][1][
                            wrd]  # corpus freq
                        # * weight
                        wrd_tot += self.word_freq_dict[key][1][wrd]
                    except KeyError:
                        msd_wrds += self.word_freq_dict[key][1][wrd]
                if wrd_tot > 0:  # ie found at least one "wrd" in the corpus
                    self.word_lift_dict[key] = (self.word_freq_dict[key][0] /
                             self.article_word_count) / (frq/wrd_tot)

    def acro_search(self, words_list):
        # called per (tokenized) sentence or phrase, eg 'I','did','n't','see',
        # 'the','train','.'
        # locate and register any likely acronyms, [name] offset w/
        # parenthetical
        # eg "central processing unit (CPU)"; can also accommodate extra text
        # within the parenthesis, eg "(CPU, in common usage)"
        # this is designed to handle hyphens (eg Central-City Utilities (CCU))
        # --returns dictionary with key=acronym (eg CPU) and value=<acro phrase>
        # (note if acro is plural, will key off singular form of acro,
        # but phrase might be pluralized--CPU:'central processing units' (
        # from CPUs)
        out_dict = {}
        for ix, tmp_wrd in enumerate(words_list):
            if tmp_wrd == '(':
                ct = 1
                while ix + ct < len(words_list) and words_list[ix + ct] != ')':
                    ca_res = check_acro(words_list[ix + ct])  # returns singular
                    # form of acronym, if it thinks it's found one
                    if ca_res:  # found likely acronym
                        tmp_acr = ca_res
                        num_wrds = len(tmp_acr)
                        ck_str = ''  # will hold first letters of prior words
                        wp_ct = 0
                        phrs_lst = []  # holds words and hyphenated phrase
                        # elements
                        while len(ck_str) < num_wrds:
                            if ix - (wp_ct + 1) >= 0:
                                wrd_phrs = words_list[ix - (wp_ct + 1)]
                                wp_ct += 1
                            else:
                                break
                            wp_splt = re.sub(u'(-|\u2013|\u2014)', ' ',
                                             wrd_phrs)
                            # most types of hyphens
                            tmp_lst = wp_splt.split()  # split on resultant
                            # spaces
                            for word in tmp_lst[-1::-1]:
                                ck_str = word[0].lower() + ck_str
                            phrs_lst.insert(0, wrd_phrs)
                        if ck_str == tmp_acr.lower():
                            # if tmp_acr not in acro_dict: # ? why?
                            tmp_elm = (' ').join(phrs_lst)
                            # acronym phrase will be de-hyphenated if it had
                            # hyphens
                            out_dict.update({tmp_acr: tmp_elm})
                            # records singular form of
                            # acronym, if it was plural (like CPUs)
                        break  # stops after first acronym in parenths it thinks
                        # it's found
                    ct += 1
        return out_dict

    def check_phrase_wordlist(self, phrase_tok, index_set):
        # --searches for tokenized phrase_tok (eg ["I","ate","lunch","."])
        # in word_annot_list; search allows variable case sensitivity
        # --updates index_set in place; index set is set of tuples, of type
        # (start_index (in word_annot_list),len(phrase),sentence#,bool)
        if len(phrase_tok) > 1:
            sng_wrd = False  # do case insensitive if phrase is longer than 1
            # word
        else:
            sng_wrd = True  # single "word" (could still be hyphenated)
        tmp_fsi, gen_bool = self.find_sublist_ixs(phrase_tok, sng_wrd)
        # gen_bool is True only if (a) at least one occurrence of phrase in
        # phrase_tok was found in word_annot_list, and (b) all occurrences of
        # the phrase in word...list are NEN's
        found_a_match = False
        if tmp_fsi:
            found_a_match = True
            index_set.update(set([(x[0], len(phrase_tok), x[2], x[1])
                                  for x in tmp_fsi]))
        return [found_a_match, gen_bool]

    def find_sublist_ixs(self, srch_lst, sng_wrd=False):
        # a principal search function for sequences of tokens in word_annot_list
        # finds and returns all indices where ordered srch_lst [el1,el2,...,elk]
        # matches ordered elements in word_annot_list; search is case-sensitive
        # --indices returned are at start of match (eg ['I','am'] searched in [
        # 'Sam','I','am'] returns [1])
        # --also returns NE/NNP/NNP(S) ~status, and sentence # of phrase (if
        # phrase is completely within a single sentence--so ix_list is a list of
        # tuples of type (<index>,<bool>,<sen#>), where sen# is None if either
        # one or more matching elements in word_annot_list had None for
        # sent_num, or if all sentence numbers were not the same over the
        # whole phrase
        # --flt_bool is only True if all search hits have all results turn up
        # as NEN
        # --note: this figures NEN boolean of each phrase occurrence in the
        # test, storing it in an ix_lst tuple; this also returns a ~net boolean,
        # indicating whether all phrase matches in the text all returned an
        # NEN (a quick way for determining by the caller whether the phrase (in
        # srch_lst) should ~qualify as a NEN or not)
        # --sng_wrd=True will make the search case sensitive; False will turn
        # off case sensitivity
        ix_lst = []
        flt_bool = True
        if sng_wrd == True:
            # this creates an exception for case-sensitive match if the search
            # phrase is a single, lowercased word, and the word in w.a.l. is the
            # 1st word of a sentence, not tagged as NEN
            # remove "if" portion of "if/else" (and remove "else") to restore
            # original approach
            if re.match(r'[a-z]', srch_lst[0][0]):  #
                # single word, 1st letter lowercase
                # sentence-leading list:
                if len(self.word_annot_list) > self.fw_indices[-1]:
                    end_ix = len(self.word_annot_list)
                else:  # the ~strange case of the last sentence
                    # has only one word, at the end of w.a.l.
                    end_ix = len(self.word_annot_list) - 1
                fw_ix = [[ii, pos_filt(self.word_annot_list[ii])] for
                         ii, st in enumerate(self.word_annot_list[0:end_ix])
                         if st.wrd == srch_lst[0] or
                         (st.wrd.lower() == srch_lst[
                             0].lower() and ii in self.fw_indices
                          and st.pos != 'NNP' and st.pos != 'NNPS'
                          and not st.NE) or
                         (st.wrd.lower() == srch_lst[
                             0].lower() and ii in self.fw_indices
                          and self.word_annot_list[ii + 1].pos != 'NNP'
                          and self.word_annot_list[ii + 1].pos != 'NNPS'
                          and not self.word_annot_list[ii + 1].NE)]
            else:   # 1-word phrase is capitalized, do case sensitive search
                fw_ix = [[ii, pos_filt(self.word_annot_list[ii])] for
                         ii, st in enumerate(self.word_annot_list) if
                         st.wrd == srch_lst[0]]
        else:
            fw_ix = [[ii, pos_filt(self.word_annot_list[ii])] for
                     ii, st in enumerate(self.word_annot_list) if
                     st.wrd.lower() == srch_lst[
                         0].lower()]
        for el in [tup for tup in fw_ix if tup[0]+len(srch_lst)<=len(
                self.word_annot_list)]:
            tmp_bool = el[1]
            sn_nm = self.word_annot_list[el[0]].sent_num
            for ii, st in enumerate(srch_lst[1:]):
                if sng_wrd == True:
                    cmp_a = st
                    cmp_b = self.word_annot_list[el[0] + ii + 1].wrd
                else:
                    cmp_a = st.lower()
                    cmp_b = self.word_annot_list[el[0] + ii + 1].wrd.lower()
                if cmp_a != cmp_b:
                    break  # bail out; not a match starting at index el[0]
                else:
                    # tmp_bool is compounded (and-ed), to only be True (= found
                    # an NEN) if all words in the phrase are tagged NEN
                    tmp_bool = tmp_bool & pos_filt(
                        self.word_annot_list[el[0] + ii + 1])
                    if sn_nm:
                        if sn_nm != self.word_annot_list[el[0] + ii +
                                                        1].sent_num:
                            sn_nm = None
            else:  # made it through loop w/o bailout
                ix_lst.append((el[0], tmp_bool, sn_nm))
                flt_bool = flt_bool & tmp_bool  # False if at least one
                # instance is False (not NEN)
        return ix_lst, (flt_bool and ix_lst is not None)  # ix_lst is
        # list of tuples, (<index>, <NE/NNP flag>, <sent_num>)
        # where <index> is the index in word_annot_list of start of match;
        # 2nd element in tuple (beside ix_lst) is the net NEN boolean

    def get_anch_list(self):
        return self.anchor_list

    def get_acro_list(self):
        return self.acro_list

    def get_lift_dict(self):
        return self.word_lift_dict

    def get_wal(self):
        return self.word_annot_list

    def get_sent_list(self):
        return self.sentence_list

    def get_phrases_checked(self):
        return self.phrases_checked

    def get_fw_indices(self):
        return self.fw_indices

    def get_principal_anchors(self):
        return self.anchor_list_principal

    def get_anch_dict(self):
        return self.anch_dict


class SentenceTagger:

    # class to handle sentence tagging tasks

    def __init__(self, sentence_list, phrases_checked, fw_indices,
                 word_annot_list):
        # initialize indexing
        self.snt_lst = sentence_list
        self.fwrd_inds = fw_indices
        tmp_phr_lst = [phr for phr in
                       phrases_checked]  # list for phrase indexing
        self.word_annot_list = word_annot_list
        # p.l: index:phrase dictionary: 1:<phrase 1>, 2:<phrase 2>, ...:
        self.phrase_lookup = dict(zip(range(len(tmp_phr_lst)), tmp_phr_lst))
        # p.i.: phrase:index dictionary: (phrases_checked is filtered for unique
        # phrases):
        self.phrase_index = {val: key for key, val in
                             self.phrase_lookup.items()}
        # generate a list of lists, the nth corresponding to the nth sentence,
        # the inner list containing tuples of 2 values, the 1st, the index of a
        # phrase appearing in that sentence, and 2nd, the tuple corresponding to
        # that instance (phrase, and tuple) in phrases_checked:
        # (<phrase index>, <tuple from phrases_checked[phrase]>)
        self.phrs_sent_index_list = [[] for _ in range(len(self.snt_lst))]
        for phr in phrases_checked:
            for tup in phrases_checked[phr]:
                snt_num = tup[2]
                if snt_num is not None:
                    self.phrs_sent_index_list[snt_num].append(
                        (self.phrase_index[phr], tup))
        # dictionary of form <phrase index>: {<sn1>,<sn2>,...} for quick lookup
        # of all sentences a phrase is in, by its index
        #self.phr_ix_sen_lookup = {self.phrase_index[key]: set([tup[2] for
                    # tup in
                         #phrases_checked[key]]) for key in phrases_checked}
        self.phr_ix_sen_lookup = {self.phrase_index[key]: set([tup[2] for
                      tup in phrases_checked[key] if tup[2] is not None])
                                  for key in phrases_checked}
        # filter for overlaps; this creates attribute filt_sent_index_list
        self._filter_overlaps()
        # for optional tracking to prevent tagging sentences redundantly for
        # a phrase pair
        self.phs_tup_chk = []

    def check_sent_tag_indexing(self):
        # displays some example sentences and their key phrases, both before
        # and after taking care of phrase overlaps (if any)
        # this is largely for debugging
        ct = 0
        for ii, sent_tups in enumerate(self.phrs_sent_index_list):
            if sent_tups:  # if it has at least one phrase in the sentence
                print("For sentence: \'%s\':" % self.snt_lst[ii])
                print("the following key phrases appear:")
                for ix in [x[0] for x in sent_tups]:
                    print("\'%s\'" % self.phrase_lookup[ix])
                print("removed of overlaps and sorted:")
                for ix in [x[0] for x in self.filt_sent_index_list[ii]]:
                    print("\'%s\'" % self.phrase_lookup[ix])
                ct += 1
                if ct > 5:
                    break

    def phrase_elements_process(self, obj_1, obj_2, track=False):
        # receives a pair of objects with get_all_phrases methods
        # (which method returns a list of phrase elements); objects can be eg
        # acronym elements or anchor elements
        # --returns a list of sentences containing both a phrase from obj_1 and
        # a phrase from obj_2, with phrases marked by E1 / E2 tags
        # --track boolean is to turn on or off redundancy tracking--so eg
        # once the sentence index lists are searched for an ordered pair of
        # phrases, the tagging will not be performed again on that same
        # ordered pair
        tagged_sent_list = []
        for phrase_el_1 in obj_1.get_all_phrases():
            for phrase_el_2 in obj_2.get_all_phrases():
                phs_tup = (phrase_el_1.get_phrase(),phrase_el_2.get_phrase())
                if track:
                    if phs_tup in self.phs_tup_chk:
                        continue
                    else:
                        self.phs_tup_chk.append(phs_tup)
                tagged_sent_list += self.gen_tagged_sents(*phs_tup)
        return tagged_sent_list

    def gen_tagged_sents(self, e_1, e_2):
        # generates and returns a list of sentences with both phrases e_1 and
        # e_2 occurring in them, w/ phrases marked
        # --if no phrases found (or perhaps the phrases are not valid),
        # returns an empty list
        out_sent_list = []
        if e_1 not in self.phrase_index or e_2 not in self.phrase_index:
            logger.debug('problem with phrase lookup in tagger')
            return out_sent_list
        elif e_1==e_2:
            logging.debug("phrase equality in gen_tagged_sents: %s" % e_1)
            return out_sent_list
        ix_e_1 = self.phrase_index[e_1]
        ix_e_2 = self.phrase_index[e_2]
        # search phr_ix_sen_lookup for any sentences which contain both phrases:
        common_sents = self.phr_ix_sen_lookup[ix_e_1] & self.phr_ix_sen_lookup[
            ix_e_2]
        if common_sents:
            for ii in common_sents:
                sent = self.snt_lst[ii] # sentence to be tagged (w/ both
                # phrases)
                if '\n' in sent:
                    logger.debug('possible problem with sentence tokenizing: %s'
                                 % sent)
                    continue
                tups = self.filt_sent_index_list[ii]  # (<phrs ix>,<wal tup>),
                # ... pairs
                ph_1_lst = [tup for tup in tups if tup[0] == ix_e_1]
                ph_2_lst = [tup for tup in tups if tup[0] == ix_e_2]
                if not ph_1_lst or not ph_2_lst:
                    # the ~difficulty here is the common sentences list is from
                    # phrases_checked, which does not preclude ~phrase
                    # inclusions (eg "integrated circuit" embedded in
                    # "digital integrated circuit"); so a phrase that is
                    # actually in a sentence, but is ~covered by a bigger
                    # phrase will stop the procedure
                    # (so) skip this sentence, go to the next one
                    continue

                wal_snt_st = self.fwrd_inds[ii]  # wal index for the start of
                # this sentence
                for tup_1 in ph_1_lst:
                    wal_ix_1 = tup_1[1][0]  # wal index for particular
                    # found instance of e_1
                    e1_ix = self.sent_seek(sent,self.word_annot_list[
                                           wal_snt_st:wal_ix_1])
                    if e1_ix<0:
                        logger.debug('problem in sentence seek %s' % sent)
                        continue
                    for tup_2 in ph_2_lst:
                        wal_ix_2 = tup_2[1][0]
                        e2_ix = self.sent_seek(sent,self.word_annot_list[
                                           wal_snt_st:wal_ix_2])
                        if e2_ix<0:
                            logger.debug('problem in sentence seek %s' % sent)
                            continue
                        if e1_ix<e2_ix:
                            pos_1 = e1_ix; pos_2 = e1_ix+len(e_1)
                            pos_3 = e2_ix; pos_4 = e2_ix+len(e_2)
                            sent_out = (sent[0:pos_1]+'[E1]'+sent[
                                  pos_1:pos_2]+'[/E1]'+sent[pos_2:pos_3]+
                            '[E2]'+sent[pos_3:pos_4]+'[/E2]'+sent[pos_4:])
                        else:
                            pos_1 = e2_ix; pos_2 = e2_ix + len(e_2)
                            pos_3 = e1_ix; pos_4 = e1_ix + len(e_1)
                            sent_out = (sent[0:pos_1]+'[E2]'+sent[
                                   pos_1:pos_2]+'[/E2]'+sent[pos_2:pos_3]+
                            '[E1]'+sent[pos_3:pos_4]+'[/E1]'+sent[pos_4:])
                        out_sent_list.append(sent_out)
        return out_sent_list

    def sent_seek(self, sent, wal_sublist):
        # returns index in sentence (string) corresponding to prefix determined
        # from list of string tokens in wal_sublist; eg "The orange is ripe."
        # and ["The","orange","is"] should return the index of the 'r' in 'ripe'
        # returns -1 if can't process the sentence
        seek_ix = 0
        for tok in [wrd_el.wrd for wrd_el in wal_sublist]:
            mtch = re.match(r'[\n\t\s]*' + re.escape(tok) + r'[\n\t\s]*',
                            sent[seek_ix:])
            if mtch:
                seek_ix += mtch.end()
            else:
                # check for tokenizer's converting " to `` or ''
                if re.match(r'[\n\t\s]*\"',sent[seek_ix]):
                    if re.escape(tok)=='``' or re.escape(tok)=='\'\'':
                        seek_ix += 1
                else:
                    return -1  # problem in seek
        return seek_ix

    def user_phrase_test(self, e_1, e_2):
        # allows user-entered phrases (have to be in the phrases_checked list
        # dictionary), returning all tagged sentences with both phrases in them
        # --search filt_sent_index_list for occasions where e_1 and e_2 both
        # occur in the same sentence
        # --recall, filt_sent_index_list's ii'th entry corresponds to ii'th
        # sentence in sentence_list, w/ entries eg [(0,(1, 1, 0, False)),
        # (140, (4, 2, 0, False)), ...]
        # i.e. tuples of type (<phrase ix>, (<w.a.l. ix>, <phr len>,..))
        # --this assumes the phrases have been checked for legitimacey
        # beforehand

        ix_e_1 = self.phrase_index[e_1]
        ix_e_2 = self.phrase_index[e_2]

        # search phr_ix_sen_lookup for any sentences which contain both phrases
        common_sents = self.phr_ix_sen_lookup[ix_e_1] & self.phr_ix_sen_lookup[
            ix_e_2]
        if common_sents:
            print("found sentences:")
            for ii in common_sents:
                print(self.snt_lst[ii])
                sent = self.snt_lst[ii]
                tups = self.filt_sent_index_list[
                    ii]  # (<phrs ix>,<wal tup>), ... pairs
                ph_1_lst = [tup for tup in tups if tup[0] == ix_e_1]
                ph_2_lst = [tup for tup in tups if tup[0] == ix_e_2]
                if not ph_1_lst or not ph_2_lst:
                    # the ~difficulty here is the common sentences list is from
                    # phrases_checked, which does not preclude ~phrase
                    # inclusions (eg "integrated circuit" embedded in
                    # "digital integrated circuit"); so a phrase that is
                    # actually in a sentence, but is ~covered by a bigger
                    # phrase will stop the procedure
                    print("(none)")
                    break
                last_ix_1 = 0
                for ph_1_tup in ph_1_lst:
                    sent_out = sent
                    phr_1 = e_1
                    if ph_1_tup[1][0] in self.fwrd_inds:
                        phr_1 = phr_1[0].upper() + phr_1[1:]
                    tmp_ix_1 = sent_out.find(phr_1, last_ix_1)
                    if not tmp_ix_1 >= 0:
                        logger.debug("problem with sentence phrases")
                        raise LookupError
                    last_ix_1 = tmp_ix_1 + len(e_1)
                    sent_out = sent_out[0:last_ix_1].replace(phr_1,
                              '[E1]' + phr_1 + '[/E1]') + sent_out[
                              last_ix_1:]
                    last_ix_1 += 9  # 9 for tags [E1] [/E1]
                    last_ix_2 = 0
                    for ph_2_tup in ph_2_lst:
                        phr_2 = e_2
                        if ph_2_tup[1][0] in self.fwrd_inds:
                            phr_2 = phr_2[0].upper() + phr_2[1:]
                        tmp_ix_2 = sent_out.find(phr_2, last_ix_2)
                        if not tmp_ix_2 >= 0:
                            logger.debug("problem with sentence phrases")
                            raise LookupError
                        last_ix_2 = tmp_ix_2 + len(e_2)
                        sent_out = sent_out[0:last_ix_2].replace(phr_2,
                                  '[E2]' + phr_2 + '[/E2]') + sent_out[
                                  last_ix_2:]
                        last_ix_2 += 9  # 9 for tags [E1] [/E1]
                        print(sent_out)
        else:
            print("no sentences in common found")

    def _filter_overlaps(self):
        # for each sentence, filter phrase tuples to remove overlaps; eg if a
        # phrase tuple wholly contains another, only retain the outermost
        # phrase--"computer peripheral" and "computer" becomes "computer
        # peripheral"
        self.filt_sent_index_list = [[] for _ in range(len(self.snt_lst))]
        for ii, sent_list in enumerate(self.phrs_sent_index_list):
            if sent_list:
                tup_srt = sorted(sent_list, key=lambda x: x[1][0])  # sort by
                # w.a.l. index number (order in sentence)
                # eg [(0, (1, 1, 0, False)), (140, (4, 2, 0, False)),
                # (392, (4, 1, 0, False)), (67, (18, 2, 0, False)),
                # (125, (26, 2, 0, False))]
                rng_lst = [(tup[1][0], tup[1][0] + tup[1][1] - 1) for tup in
                           tup_srt]
                # tup = (phrase_lookup #,(<phrases_checked tuple>))
                # rng_list is a list of ordered pairs (tuples), each showing
                # start and end of range of tokens occupied; eg [4,6],[4]
                # indicates 1st phrase takes up tokens 4,5, and 6, while 2nd
                # phrase takes up token 4
                valid_ranges = tr_consolidate(
                    rng_lst)  # will be a list of pairs
                # with no overlap between them, eg [(1,2),(4,7),(8,11)]
                valid_tuples = [tup_srt[ii] for ii, tup in enumerate(rng_lst) if
                                tup in valid_ranges]
                # so filt_sent_index_list has its ii'th entry corresponding to
                # ii'th sentence in sentence_list; the tuple pairs in this list
                # are of form (<phrase index>, phrases_checked tuple), p_c. t.
                # is (w.a.l. start_index,len(phrase),sentence#, NE_bool)
                # each tuple pairs corresponds to ~valid (non-overlapping)
                # phrases in the sentence, in the order the phrases appear
                self.filt_sent_index_list[ii] = valid_tuples

    def get_phrase_index(self):
        return self.phrase_index





###
# supporting functions; minor or eg secondary and 'lower' calls; these could
# arguably be ~absorbed into the class, as ~static methods
###


def pos_convert(tag_in):
    if tag_in.startswith('J'):
        return 'a'  #wordnet.ADJ
    elif tag_in.startswith('V'):
        return 'v'  #wordnet.VERB
    elif tag_in.startswith('N'):
        return 'n'  #wordnet.NOUN
    elif tag_in.startswith('R'):
        return 'r'  #wordnet.ADV
    else:
        return None


def check_acro(wrd):
    # checks word 'wrd' for if likely acronym (eg CPU); can also handle
    # suspected plural acronyms (eg CPUs); returns the singular version
    # of the acronym
    # suspected acro should be at least 2 letters long
    if re.fullmatch(r'[A-Z]+',wrd) and len(wrd)>=2: # all caps?
        return wrd
    elif re.fullmatch(r'[A-Z]+s',wrd) and len(wrd)>=3: # all caps then
        # plural 's' suffix?
        return wrd[0:-1]


def pos_filt(w_lst):
    # this receives a word_list_obj from word_annot_list
    # this returns a boolean: True if the pos/ne_chunk thinks the word is a
    # proper noun or named entity (or if it's a filter word); False otherwise
    filt_wrds = ['of','the','de','and',',','&'] # for eg "Charles de Gaulle" or
        # "Groton, Connecticut"
    if w_lst.wrd not in filt_wrds:
        return w_lst.NE or w_lst.pos == 'NNP' or w_lst.pos == 'NNPS'
    else:
        return True # filtered / minor words don't count toward NE / NNP
        # test; ie they intentionally default to True (as NEN) to preserve
        # the test condition (requiring all elements to be NEN's before whole
        # phrase is declared a NEN)


def max_asym_over(rng_lst):
    # checks for ~linked asymmetric overlap, starting from first 2-tuple in
    # rng_lst, and returns maximum extent of overlap's "cover"
    # expects rng_list a set of 2-tuples, sorted by 1st element in increasing
    # order--eg [(1,2),(2,5),(2,3),(4,7),(4,4),(8,12)] (returns 7)
    # (dev construction note: based on tup_range_1.py in python_code)
    # --output is interpreted by tr_consolidate function
    tup_tst = rng_lst[0]
    max_rng = tup_tst[1]
    max_tup = tup_tst
    for tup in rng_lst[1:]:
        if tup[0] > tup_tst[1]:  # 1st values of rng_lst tuple pairs are sorted
            break
        if tup[0] <= tup_tst[1] and tup[1] > tup_tst[1] and tup[
            1] > max_rng:
            max_rng = tup[1]
            max_tup = tup
    if tup_tst == max_tup:  # rest of rng_lst is "disconnected" from initial
        # tuple, tup_tst; return
        return max_rng
    rng_lst = [max_tup] + [tup for tup in rng_lst if tup[0] > tup_tst[1]]
    if len(rng_lst) > 1:
        return max_rng
    else:  # if it's only one tuple, can't have any asymmetric
        # overlap with another, so return
        return max_rng


def tr_consolidate(rng_lst):
    # driver for max_asym_over, consolidates ranges represented by 2-tuples and
    # returns valid tuples/ranges
    # eg rng_lst = [(1,2),(3,7),(4,5),(5,6),(10,12)] produces the list
    # [(1,2),(3,7),(10,12)]
    # rng_list must be sorted in increasing order by 1st values in each tuple
    # (so eg [(3,4),(1,4)] is invalid)
    ok_tup = []
    while len(rng_lst) > 0:
        max_rng = max_asym_over(rng_lst)
        lw_rv = rng_lst[0][
            0]  # lowest value of low end of tuples in rng_lst
        if (lw_rv, max_rng) not in rng_lst:
            pass
        else:
            ok_tup.append((lw_rv, max_rng))
        rng_lst = [tup for tup in rng_lst if tup[0] > max_rng]
    return ok_tup


def anchor_inspect(anchor_list_rel,phrases_checked):
    # helper / debugger function
    while True:
        in_str = input("inspect anchor element(y/Y)? ")
        if in_str=='Y' or in_str=='y':
            el_num = input("which index? ")
            anchor_list_rel[int(el_num)].inspect_anchor(phrases_checked)
        else:
            break


def gen_cor_frq_dct():
    # create corpus frequency dictionary
    # pull words from corpus and record frequency counts (et al info)
    # option to just read first N entries from csv file
    freq_dict = {}
    # note, the Kaggle unigram_freq.csv file has one line for each word, each
    # line of format [word],[#occur]; the total # of occurrences of all words
    # in the file (total of 2nd ~column) is 588124220187
    fp = importlib.resources.open_text("wiki_scrape_src.data",
                                       "unigram_freq.csv")
    #with open("./data/unigram_freq.csv") as fp:
    for ii, line in enumerate(csv.reader(fp)):
        freq_dict[line[0]] = float(line[1]) / 588124220187.0
        if ii >= FREQ_DICT_MAX:
            break
    return freq_dict



def article_process(article):

    # main function for tokenizing clean wikipedia article text, and extra text
    # (like caption phrases); return is a list of wikipedia URL suffixes

    # generate frequency dictionary from existing corpus frequencies (for
    # lone-word lift calculations; these though may not be used)
    corpus_freq_dict = gen_cor_frq_dct()

    text_article_processor = TextArticleProcess(article)

    # create ~special anchor elements and acronym elements
    # by default, the article being searched is given its own anchor el (and
    # possibly acro el), w/ 'principal' status
    principal_url_el = WikiUrlElement(article.get_url_suffix())
    canonical_url = article.get_can_url()   # post-/wiki/ canonical URL; may
    # be different from url suffix (redirects)
    # NB: case sensitivity of the article name here becomes important--url
    # object's "get_phrase" is "smart" in it looks to article / url suffix
    # case to try to figure out if assoc phrase is proper noun or not
    tmp_anch_el = AnchorElement([principal_url_el.get_phrase()],
                [principal_url_el],can_url=canonical_url,status='principal')
    principal_anchors = [tmp_anch_el]
    for anch_el in principal_anchors:
        text_article_processor.add_anch(anch_el,principal=True)
    principal_acros = []    # a list of 'special' acronym elements; eg if
    # this article was the result of a prior link (eg 'MOSFET') which had
    # acronym associations, then the acronym elements could be held here;
    # their .status attributes would need to be set accordingly (not
    # 'normal'; usually 'principal') (this feature isn't really used)
    for acro_el in principal_acros:
        text_article_processor.add_acro(acro_el,principal=True)

    # process acronym elements
    text_article_processor.make_pre_acro_list() # process acronym dictionary
    text_article_processor.make_rel_acro_list() # pare any NENs
    text_article_processor.acro_ps()    # check ps complements of acronym
    # phrases

    # process anchor elements
    text_article_processor.make_pre_anch_list() # process anchor tag dictionary
    text_article_processor.make_rel_anch_list() # pare any NENs
    text_article_processor.anch_ps()    # check ps complements of anchor phrases

    # glom any anchor tags with URLs in common
    text_article_processor.merge_common_url_anchors()

    # up-merge any acronym elements associated with an anchor element into
    # that anchor element
    text_article_processor.up_merge_acro_anch()

    # generate frequency counts for post-processed anchor elements and
    # acronym elements
    text_article_processor.gen_freq()

    # generate word lift for words (1-grams) remaining after anchor element
    # phrases, acronym element phrases, NENs, stopwords (etc) have been removed
    # (this is another feature that may not really be used, but nice to know)
    text_article_processor.gen_word_lift(corpus_freq_dict)

    ###
    # acronym, anchor, and 1-gram lift summaries
    ###

    alm_sorted = sorted(text_article_processor.get_anch_list(),
                key=lambda x: x.get_freq(), reverse=True)
    alp_sorted = sorted(text_article_processor.get_acro_list(),
                key=lambda x: x.get_freq(), reverse=True)

    logger.info("top anchors:")
    for anch_el in alm_sorted[0:20]:
        logger.info("%s: %s @ %s" % (anch_el.get_url(),anch_el.get_rep_phrase(),
                              anch_el.get_freq()))

    logger.info("top remaining acros:")
    for acro_el in alp_sorted[0:5]:
        logger.info("%s @ %s" % (acro_el.get_rep_phrase(),acro_el.get_freq()))

    # flatten lift dictionary and display by lemma frequency
    logger.info("1-gram lemmas with highest lift scores:")
    lft_dct = text_article_processor.get_lift_dict()
    tmp_lfs = [[key,lft_dct[key]] for key in lft_dct]
    lf_sort = sorted(tmp_lfs,key = lambda x: x[1],reverse=True)
    for ii in range(5):
        logger.info("%s: %s" % tuple(lf_sort[ii]))

    top_anchs = [anch_el for anch_el in alm_sorted if anch_el.get_freq() >=
                 ACR_ANC_CUTOFF and anch_el.get_status()!='principal']

    top_acros = [acro_el for acro_el in alp_sorted if acro_el.get_freq()
                 >= ACR_ANC_CUTOFF and acro_el.get_status()!='principal']

    top_lems = [lem for lem in lf_sort if lem[1] > SNG_WRD_LIFT_CUTOFF]


    ###
    # create tagged sentences from main article, and apply BERT
    ###

    recursion_links = set() # for storing wiki recursion URL suffixes (ie URL
    # text after /wiki/--this is what article_create expects to receive for a
    # valid article)

    sentence_tagger = SentenceTagger(text_article_processor.get_sent_list(),
                    text_article_processor.get_phrases_checked(),
                    text_article_processor.get_fw_indices(),
                    text_article_processor.get_wal())

    # DEBUG
    #sentence_tagger.check_sent_tag_indexing()

    # fetch principal anchor object
    principal_anchors = text_article_processor.get_principal_anchors()
    princ_anch = principal_anchors[0]    # this regards 1st occurrence of an
    # anchor of 'principal' type to be the principal anchor

    logger.info("relations derived from main text:")
    for anch_el in top_anchs: # fetch an anchor element from sorted anchor
        # list
        tagged_list = sentence_tagger.phrase_elements_process(princ_anch,
                        anch_el,track=True)
        rel_lst = []
        for snt in tagged_list:
            #print(snt)
            rel = ask_bert.get_relation(snt,text_form=False)
            rel_lst.append(rel)
            #print(rel)
        #print("consensus relation:")
        if tagged_list:
            cons = ask_bert.consensus_relation_new(rel_lst) # will be
            # 'recurse' or 'pass'
            if cons=='recurse':
                url_tmp = anch_el.get_canonical()
                if not url_tmp:  # ie if canonical URL does not exist
                    url_tmp = anch_el.get_url()  # gets 1st URL element in
                    # anch_el's url_list; this auto-fetches the first element
                    # in the list; not sure what to do about multiple URL's
                    # assoc w/ a given anchor; the idea was prlly to have the
                    # article's URL, and the canonical URL (if any); so if
                    # there's no canonical, there should only be one URL assoc
                    # w/ anchor
                recursion_links.add(url_tmp)
            logger.info("%s, %s: %s, %s" % (princ_anch.get_rep_phrase(),
                anch_el.get_rep_phrase(),cons,rel_lst))
        else:
            # nb: tagged_list could be empty because no sentences were found,
            # or, if enabled, because all sentences containing paired phrases
            # of interest were already tagged
            pass
            #print("no sentences found for element %s" %
            #      anch_el.get_rep_phrase())

    return recursion_links


if __name__=='__main__':

    #article = article_create.WikiArticle('microprocessor')
    #article = article_create.WikiArticle('brick')
    #article = article_create.WikiArticle('steel')
    #article = article_create.WikiArticle('forge')
    #article = article_create.WikiArticle('semiconductor')
    #article = article_create.WikiArticle('integrated_circuit')
    #article = article_create.WikiArticle('transistor')
    #article = article_create.WikiArticle('foundry')
    article = article_create.WikiArticle('paper_clip')
    article_process(article)
