'''
controller for overseeing article recursion

note, of article name / url suffix case sensitivity:
    the article URL suffix should be case sensitive (eg Paper_clip or
    French_Revolution)
    recurse_on_article may pull an article's ~phrases via url objects
    .get_phrase(), which does a case sensitive examination of the url suffix,
    trying to return the correct case
'''

import time
import wiki_scrape_src.article_create as article_create   # for WikiArticle
# class
import wiki_scrape_src.recurse_on_article as recurse_on_article

art_tree = []   # a list of dictionaries, one dictionary at each recursion
# level, for tracking tree structure, and recording "leaf" complexity

'''
some article name examples:
Paper_clip, Microprocessor, Brick, Kiln, Molding_(process), Steel
'''
art_name = input("Enter article name string (the string should match a "
                 "legitimate wikipedia page URL suffix; for instance "
                 "\'Paper_clip\' from "
                 "https://en.wikipedia.org/wiki/Paper_clip): ")

cache_dir = input("Enter a directory for caching Wikipedia html and text ("
                  "default is ./data/wiki_cache; but to ensure this relative "
                  "path, launch this script from its parent folder): ")
if cache_dir=="":
    cache_dir = './data/wiki_cache'
    with open(cache_dir+"/DO_NOT_REMOVE.txt",'r') as fp:
        tmp_txt = fp.read()
    if '389073500377790555162751634721' not in tmp_txt:
        print("problem with default ./data/wiki_cache directory")
        raise FileNotFoundError

print("initiating on %s" % art_name)
article = article_create.WikiArticle(art_name,cache_dir)
# recursion_links is a list of printable wikipedia URL suffixes; these all
# should lead to the correct wikipedia article
print("initial article complexity: %s" % article.get_KBC())
art_tree.append({})
art_tree[0] = {article.get_can_url():article.get_KBC()}
art_names = {article.get_can_url():{art_name}}    # total dictionary keyed on
# canonical url, for all article names encountered
recursion_out_1 = recurse_on_article.article_process(article)
print(recursion_out_1)

#sys.exit("stopping")

KBC_score = article.get_KBC()


timer = time.time() # timer (in seconds) for throttling wikipedia fetch requests
FETCH_FREQ = 20 # frequency of fetch requests, in seconds

ct = 0
tmp_kbc = 0
recursion_out_2 = set()    # for storing all forward article recursion links
art_tree.append({})
epoch = 1
for link in recursion_out_1:
    if link in [x for y in [val for key,val in art_names.items()] for x in y]:
        continue
    if time.time()-timer < FETCH_FREQ:
        time.sleep(FETCH_FREQ-(time.time()-timer))
    timer = time.time()
    print("recursing on %s" % link)
    article = article_create.WikiArticle(link, cache_dir)
    cn_ur = article.get_can_url()
    if cn_ur in art_names:
        "already included"
        art_names[cn_ur].add(link)
    else:
        art_names[cn_ur] = {link}
        print("article complexity: %s" % article.get_KBC())
        art_tree[epoch][cn_ur] = article.get_KBC()
        tmp_kbc += article.get_KBC()
        recursion_links = recurse_on_article.article_process(article)
        recursion_out_2.update(recursion_links)
        if recursion_links:
            print(recursion_links)
        else:
            print("no recursion links found to meet threshold")
    ct += 1
    #if ct > 2:
    #    break

print("total links to recurse on:")
print(recursion_out_2)
print("pared (non-repeated) links to recurse on:")
print(recursion_out_2-set([x for y in [val for key,val in art_names.items()]
                           for x in y]))

KBC_score = KBC_score + tmp_kbc
print("total KBC after 1st recursion level: %s" % KBC_score)

#sys.exit("stopping")

FETCH_FREQ = 20 # frequency of fetch requests, in seconds

ct = 0
tmp_kbc = 0
recursion_out_3 = set()    # for storing all forward article recursion links
art_tree.append({})
epoch = 2
for link in recursion_out_2:
    if link in [x for y in [val for key,val in art_names.items()] for x in y]:
        continue
    if time.time()-timer < FETCH_FREQ:
        time.sleep(FETCH_FREQ-(time.time()-timer))
    timer = time.time()
    print("recursing on %s" % link)
    article = article_create.WikiArticle(link, cache_dir)
    cn_ur = article.get_can_url()
    if cn_ur in art_names:
        "already included"
        art_names[cn_ur].add(link)
    else:
        art_names[cn_ur] = {link}
        print("article complexity: %s" % article.get_KBC())
        art_tree[epoch][cn_ur] = article.get_KBC()
        tmp_kbc += article.get_KBC()
        recursion_links = recurse_on_article.article_process(article)
        recursion_out_3.update(recursion_links)
        if recursion_links:
            print(recursion_links)
        else:
            print("no recursion links found to meet threshold")
    ct += 1
    #if ct > 2:
    #    break


print("total links to recurse on:")
print(recursion_out_3)
print("pared (non-repeated) links to recurse on:")
print(recursion_out_3-set([x for y in [val for key,val in art_names.items()]
                           for x in y]))

KBC_score = KBC_score + tmp_kbc
print("total KBC after 2nd recursion level: %s" % KBC_score)




