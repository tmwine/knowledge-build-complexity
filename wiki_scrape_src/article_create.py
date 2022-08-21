'''
file for initial article class creation

nb: to get this to work without going online, need:
    a folder in wiki_cache in the exact name (lowercase) of the article
    in the folder:
        a can_url file w/ the (canonical if want) URL
        an html file w/ raw html code
        a wtext file w/ wikipedia module's output (not including captions; 
        these are extracted via routines here)
'''

import wikipedia
import wikipediaapi
import logging
import os
import requests # for web page ~raw html fetches
import re
from bs4 import BeautifulSoup
import urllib.parse # for handling %.. URL syntax in wiki internal links

wiki_API = wikipediaapi.Wikipedia('en')
#logging.basicConfig(level=logging.DEBUG)   # w/o further config, this shows
# all logging statements, even by stdlib Python modules
logger = logging.getLogger(__name__)
hdlr = logging.StreamHandler()
frmtr = logging.Formatter(fmt=' %(name)s :: %(levelname)-8s :: %(message)s')
hdlr.setFormatter(frmtr)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)    # set to DEBUG for most/all info
logger.propagate = False


class WikiArticle:

    ''' major data variables:
        art_url_suffix--the url suffix we're handed for article creation;
        this may differ from canonical url (eg redirects); should be in
        printable form; this should be case sensitive (for use extracting
        ~case-sensitive phrase(s) for later use of the article object)
        wiki_text--'clean' wikipedia module text; one long string
        raw_html--straight html text from the page
        extra_text--eg figure captions; each caption gets its own line (most
        do not have sentence structure); one long string
        base_anch_dict--place for all anchor tags and assoc phrases;
        <wiki url suffix (no /wiki/)>: <anchor tag text>
        canonical_url--the canonical url to the wikipedia page (from API);
        should be stored in suffix (post-/wiki/) form; may differ from
        art_url_suffix
        cmp_txt--~helper, of all soup's paragraphs, for sentence run-on problem
        '''

    # constructor; this expects a standard, valid wikipedia url suffix in
    # printable form--everything after the '/wiki/' portion; will not
    # expect the case to be accurate (so can't rely on capitalization); both
    # dashes and underscores are possible (eg 'Mercedes-Benz_Group')
    # --the constructor looks for a canonical URL file (/can_url), and if
    # there isn't one, it queries wikipedia API to fetch the canonical URL,
    # which it stores in self.canonical_url
    # nb: caching files in general was to relieve web scraping burdens; so
    # caches are generally only for web related items (html, or wikipedia
    # module calls)
    def __init__(self,art_url_suffix,cache_dir):
        logger.info("creating %s article object" % art_url_suffix)
        self.art_url_suffix = art_url_suffix # should be in printable form;
        # all text after /wiki/; can also use urllib.parse.unquote() if unsure
        self.base_dir = cache_dir+'/'+self.art_url_suffix.lower()    # article
        # filename folders will always be lower cased
        # check if cache exists; if so, no need to use wikipediaapi in
        # constructor
        if os.path.exists(self.base_dir):
            fn = self.base_dir+'/can_url'
            try:
                fp = open(fn,'r')
            except FileNotFoundError:   # dir present, but no url file
                self.article_test() # use wikipediaapi
            else:
                logger.debug("article known to exist; url from cache")
                self.canonical_url = re.sub(r'.*/wiki/','',fp.read())
                fp.close()
        else:   # no directory at all
            os.makedirs(self.base_dir)
            self.article_test() # use wikipediaapi
        self.base_anch_dict = {}
        self.cmp_txt = ""
        self.KBC_score = 0  # KBC value assigned to the article
        self.wikitext_gen() # possible use of cache; note, this checks for
        # disambiguation page, as well as checks if page exists--so this
        # should be run before html fetch
        self.html_gen() # possible use of cache
        self.soup_process() # ~major constructor function, calls several
        # subfunctions

    # get 'clean' wikipedia article text via cache or wikipedia module
    def wikitext_gen(self):
        # this will look for appropriately named file on local cache;
        # if not found, will use web fetch
        fn = self.base_dir + '/' + 'wtext'
        try:
            fp = open(fn, 'r')
        except FileNotFoundError:  # no file in cache; fetch from web
            page = wiki_page_lookup(self.art_url_suffix) # this is the part
            # that checks for disambiguation page
            if not page:  # page fetch failed; DNE or disambiguation page
                logger.warning(
                    '%s article object creation failed' % self.art_url_suffix)
                # TODO: if want the article class to ~continue even after
                #  reaching a ~bad page (page not found, or disambiguation
                #  page), can alter the error handling here; the whole
                #  article fetch should stop at this point nonetheless
                raise ArticleFailure(self.art_url_suffix)
            else:
                logger.debug("fetched wiki text from web")
                self.wiki_text = page.content
                with open(fn, 'w') as fp:   # cache result
                    fp.write(self.wiki_text)
        else:  # file is found in cache
            logger.debug("wiki text found in cache")
            self.wiki_text = fp.read()
            fp.close()

    def html_gen(self):
        # to be called after wikitext_gen has defined url
        fn = self.base_dir + '/' + 'html'
        try:
            fp = open(fn, 'r')
        except FileNotFoundError:  # no file in cache; fetch from web
            logger.debug("fetched html from web")
            full_url = self.canonical_url
            if "/wiki/" not in self.canonical_url: # canonical_url should be
                # stored as just a post-/wiki/ suffix; make full URL for fetch
                full_url = 'https://en.wikipedia.org/wiki/'+self.canonical_url
            self.raw_html = requests.get(full_url).text
            with open(fn, 'w') as fp:
                fp.write(self.raw_html)
        else:  # file is found in cache
            logger.debug("html text found in cache")
            self.raw_html = fp.read()
            fp.close()

    def soup_process(self):
        [before_refs,refs] = self.split_html_on_refs() # note this can return
        # None for refs if it doesn't find any (which is possible)
        soup_bef = BeautifulSoup(before_refs,'lxml')  # note this uses 'lxml'
        # which is OK for html too(?)
        for tag in soup_bef.find_all('style'):
            # removes unwanted in-body style designations (like "emu" article)
            tag.replaceWith('')
        self.extra_gen(soup_bef) # generate extra text (figure captions)
        self.anchor_gen(soup_bef)    # generate anchor tag list
        self.clean_wik_txt() # 'fix' wikipedia modules' end of sentence
        # runons, etc.
        if refs:
            soup_aft = BeautifulSoup(refs,'lxml')
            for tag in soup_aft.find_all('style'):
                tag.replaceWith('')
        else:
            soup_aft = None
        self.gen_KBC(soup_aft) # score knowledge complexity of references

    def extra_gen(self,soup):
        # for generating extra text, like figure captions
        tmp_et = ''
        for image in soup.find_all('div', {'class': 'thumbcaption'}):
            tmp_et += "\n" + image.text
        for image in soup.find_all('div', {'class': 'gallerytext'}):
            tmp_et += "\n" + image.text
        # clean extra spaces and newlines
        tmp_et = re.sub(r'\n +', '\n',tmp_et)
        tmp_et = re.sub(r'\n\n+','\n',tmp_et)
        self.extra_text = tmp_et

    def anchor_gen(self,soup):
        # pull paragraphs, to look for intra-wikipedia hyperlinks, and assoc
        # text
        # nb: in certain cases, the anchor tag may be empty of text--ie
        # <a></a>--have to field this exception
        for paragraph in soup.find_all('p'):
            self.cmp_txt += paragraph.text  # used later in clean_wik_txt()
            # look for wikipedia hyperlinks; note this does not deal with the
            # case where same urls appear two or more times, associated with
            # different anchor text
            for link in paragraph.find_all('a', href=True):
                if not link.text:
                    continue    # empty anchor tag--no associated phrase in text
                tmp_hr = link['href']  # should be a string
                if self.test_wiki_link(tmp_hr):
                    tmp_hr = re.sub(r'.*/wiki/','',tmp_hr)
                    tmp_hr = urllib.parse.unquote(tmp_hr)   # convert url
                    # characters (like 'Bradley_%26_Craven_Ltd')
                    if tmp_hr not in self.base_anch_dict:
                        self.base_anch_dict[tmp_hr] = link.text

    def clean_wik_txt(self):
        # fixes sentence run-ons in wiki_text
        # --also, this pulls anything it identifies as headings--eg
        # '== John Smith =='--these can ~confuse the sentence tokenizer,
        # which tends to include the heading in the 1st sentence under the
        # heading
        # --also, this tries to find collections of phrases, as from lists,
        # eg single-line strings without a period at the end--these mess up
        # the sentence tokenizer

        # ~kludge, to 'fix' wikipedia module's sentence run-ons: eg See spot
        # run.Spot is fast. (lack of space confuses tokenizers)
        may_err = re.findall(r' [^ .]+\.[A-Z][^ .]* ',
                             self.wiki_text)  # returns list of strings
        # of type " this.She " (w/ a space before and after)
        logger.info("%s potential wikipedia module sentence period parse "
                    "errors found"
              % len(may_err))
        self.cmp_txt = self.cmp_txt.replace('\n', ' ')
        self.cmp_txt = re.sub(r'\[.*?\]+','',self.cmp_txt)    # removes from
        # soup's paragraph text anything in square brackets--this is just for
        # the kludge re fixing "runons"; wikipedia module text has reference
        # brackets already removed
        #self.cmp_txt = re.sub(r'\[[0-9]*\]', '',
        #                  self.cmp_txt)  # remove reference [#] markings
        for err in may_err:
            wrd_pr = err.split('.')
            tmp_phr = wrd_pr[0] + '. ' + wrd_pr[1]
            if tmp_phr in self.cmp_txt:
                self.wiki_text = self.wiki_text.replace(err, tmp_phr)

        # remove everything below a "references" or "for further reading"
        # heading
        self.wiki_text = re.sub(r'===* *[Rr]eferences *===* *\n.*','',
                self.wiki_text,flags=re.DOTALL)
        self.wiki_text = re.sub(r'===* *[Ff]urther [Rr]eading *===* *\n.*',
                                '',self.wiki_text,flags=re.DOTALL)
        self.wiki_text = re.sub(r'===* *[Ff]urther [Rr]eading *===* *\n.*',
                                '',self.wiki_text,flags=re.DOTALL)

        # for headings removal, a single = is used in wikitext for the article
        # title (it should not appear in the text body); all other headings
        # should have at least two equals signs adjacent (==, etc.)
        self.wiki_text = re.sub(r'===*.*?===* *\n','',self.wiki_text)

        # find likely one-line phrases and move them to extra_text
        tmp_txt = ''
        for line in self.wiki_text.splitlines():
            if not re.search(r'\.',line) and line.strip():
                self.extra_text += line + '\n'
            else:
                if line.strip():
                    tmp_txt += line + '\n'
        self.wiki_text = tmp_txt

    def split_html_on_refs(self):
        # helper function, splits html text by references section
        tmp_ref = None
        tmp_rab = re.search(r'\n[^\n]*(id=[\'\"][Rr]eferences[\'\"]|id=['
                            r'\'\"][Bb]ibliography[\'\"]|'
                            r'id=[\'\"][Nn]otes[\'\"]).*',
                            self.raw_html,
                            flags=re.DOTALL)
        if tmp_rab:
            tmp_ref = re.search(r'(\n[^\n]*class=[\'\"][Rr]eflist .*|\n['
                            r'^\n]*class=[\'\"][Rr]eferences.*)',
                            tmp_rab.group(),flags=re.DOTALL)
        if not tmp_ref:  # want a reflist or references class shortly after
            # ref/bib/notes heading; note not all articles (a) have a
            # references section or (b) have a references section with
            # something in it
            logger.warning("possible problem with references section in %s "
                           "html" % self.art_url_suffix)
            if tmp_rab: # Ref / Biblio / Notes section found, but it's empty
                # of expected class=reflist/references
                before_refs = self.raw_html.replace(tmp_rab.group(),'')
                refs = None
                return [before_refs, refs]
            else:   # no Ref / Biblio / Notes section found
                tmp_rem = re.search(r'\n[^\n]*(id=[\'\"][Ff]urther[_ ]reading['
                                    r'\'\"]|id=[\'\"][Ee]xternal[_ ]links['
                                    r'\'\"]).*',
                                    self.raw_html, flags=re.DOTALL)
                if tmp_rem:
                    before_refs = self.raw_html.replace(tmp_rem.group(),'')
                else:
                    before_refs = self.raw_html
                refs = None
                return [before_refs, refs]
        elif tmp_ref.start()>1500:
            logger.warning("potential disparity in references section in %s "
                           "html" % self.art_url_suffix)

        refs_and_below = tmp_ref.group()
        # this will remove anything at or below Further reading or External
        # links:
        tmp_rem = re.search(r'\n[^\n]*(id=[\'\"][Ff]urther[_ ]reading['
                            r'\'\"]|id=[\'\"][Ee]xternal[_ ]links['
                            r'\'\"]).*',
                            refs_and_below,flags=re.DOTALL)
        if tmp_rem:
            refs = refs_and_below.replace(tmp_rem.group(),'')
        else:
            refs = refs_and_below
        before_refs = self.raw_html.replace(refs_and_below,'')
        return [before_refs,refs]

    def test_wiki_link(self,in_str):
        # helper function; tests an anchor tag link for wikipedia article
        # reference will exclude things like "citation needed"; expects a string
        tmp_lc = in_str.lower()
        tst1 = re.search(r'/wiki/', tmp_lc)
        tst2 = re.search(r'citation_needed', tmp_lc)
        tst3 = re.search(r'accuracy_dispute', tmp_lc)
        tst4 = re.search(r'disputed_statement', tmp_lc)
        tst5 = re.search(r'/wiki/talk:', tmp_lc)
        return tst1 and not (tst2 or tst3 or tst4 or tst5)

    def article_test(self):
        # helper function for constructor; use wikipediaapi to check if
        # article exists, and if so, get (and cache) canonical URL
        logger.debug('checking article existence')
        page_check = wiki_API.page(self.art_url_suffix)
        if not page_check.exists():
            logger.warning(
                '%s article object creation failed; page DNE')
            raise ArticleFailure(self.art_url_suffix)
        else:
            self.canonical_url = re.sub(r'.*/wiki/','',page_check.canonicalurl)
            fn = self.base_dir + '/can_url'
            with open(fn,'w') as fp:
                fp.write(self.canonical_url)

    def gen_KBC(self,soup):
        # estimates the KBC score, based on article references
        if soup is None:    # it's possible we get None, eg article has no
            # references section
            self.KBC_score = 0
            return
        span_cites_rtx = soup.find_all('span', attrs={'class':
                                                          'reference-text'})
        # including 'reference-text' ensures citations that don't have a
        # "citation" tag get counted too
        span_cites_cit = soup.select('cite[class*="citation"]')
        span_cites = list(set([x.text for x in span_cites_rtx+span_cites_cit]))
        cite_tags = []
        for cit in span_cites:  # go through text of each citation
            # determine all qualifying types:
            bc,ac,pc = (False,False,False)
            bc = self.book_check(cit)
            ac = self.jarticle_check(cit)
            pc = self.patent_check(cit)
            # resolving ~conflict between multiple types:
            if pc:
                cite_tags.append(
                    "patent")  # any ref to a patent, it's prlly a patent
            elif bc and ac:
                # resolving if both book and article indicated
                cite_tags.append("book")
            elif bc:
                cite_tags.append("book")
            elif ac:
                cite_tags.append("article")
            else:
                cite_tags.append("other")
        ref_ct = {"book": 0, "article": 0, "patent": 0, "other": 0}
        for ii in range(len(span_cites)):
            ref_ct[cite_tags[ii]] += 1
            #print(cite_tags[ii])
            #print(entry.text.strip())  # this would be the actual cite text
        # eg simple scoring, just total over all relevant ref types
        self.KBC_score = ref_ct['book'] + ref_ct['article'] + ref_ct[
            'patent']

    def book_check(self,in_str):
        # helper for gen_KBC
        # routine to check if reference is ~likely a book
        return "ISBN" in in_str  # nb: journal papers can have ISBNs too(?)

    def jarticle_check(self,in_str):
        # helper for gen_KBC
        # routine to check if reference is ~likely a journal article
        tst1 = re.search(r'[A-Z][a-z]+, *[A-Z][a-z.]',
                         in_str)  # capitalized word then comma-
        # space, then another capital letter w/ period, or lc's
        # --probably an author name
        tst2 = re.search(r'(\(.*[^0-9]|\()[0-9]{4}([^0-9].*\)|\))', in_str)
        # looks for 4 digits bordered by non-digits, surrounded by
        # parenths (year, eg (2009))
        tst3 = re.search(u'(p\.|pp\.|[0-9]+ *(\u2013|\u2014|\-) *[0-9]+)'
                         , in_str)
        # looks for 'p.' or 'pp.' or #-# (eg 92-105) w/ spaces allowed
        # note concerns over hyphens, em, en dashes
        # \u2013 \u2014
        tst4 = re.search(r'(archived|retrieved)', in_str.lower())
        # eg wikip microprocessor refs have a lot of "archived" links;
        # these are a mixed bag (may be an article, or an interview,
        # etc...) so it's easiest just to preclude them (they can
        # otherwise ~fool the article syntax checker--they often
        # have a retrieval date included, eg 2009-12-26)
        # tst5 = re.search(r'ISSN',in_str) # might use, but ISSN's also
        # apply to regular magazines
        return tst1 and tst2 and tst3 and not tst4

    def patent_check(self,in_str):
        # helper for gen_KBC
        return "patent" in in_str.lower()

    def plur_sing_complement(self,in_str,pos):
        # given a string and part-of-speech tag, does its best to determine
        pass

    def get_can_url(self):
        # returns suffix form of canonical, in printable form
        tmp_url = re.sub(r'.*/wiki/','',self.canonical_url)
        return urllib.parse.unquote(tmp_url)

    def get_url_suffix(self):
        return self.art_url_suffix # this should be in printable form,
        # everything past the /wiki/ phrase

    def get_KBC(self):
        return self.KBC_score


def wiki_page_lookup(phrase):
    # looks for a wikipedia page handle corresponding to phrase (
    # can wikipedia URL suffix; there are options how to do this--(a) try raw
    # url (en.wikipedia/wiki/[phrase w/ underscores]), (b) use wikipediaapi API
    # module, (c) use wikipedia module
    # NOTE, for big scraping jobs, the wikipedia module (and perhaps API) are
    # ~discouraged (see wikipedia module's documentation)
    # note, this uses wikipedia module (of course) to fetch the ~cleaned wiki
    # text, which module will also return a disambiguation error if it hits
    # a disambiguation page
    try:
        page = wikipedia.page(phrase, auto_suggest=False)
    except wikipedia.exceptions.PageError:  # page doesn't exist
        return None
    except wikipedia.exceptions.DisambiguationError:  # the page is a
        # disambiguation page
        return None
    return page


class ArticleFailure(Exception):
    def __str__(self):
        return 'Problem creating article object'


if __name__=='__main__':
    pass
