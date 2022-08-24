This is a GitHub repository for the Python project described in the article, ["KBC: The Knowledge Build Complexity"](link). The purpose of the code is to provide a rough, automated method for estimating the complexity of an arbitrary input entity, based on human knowledge inputs.


## Code Overview

The code primarily consists of a Wikipedia web scraper.

Given an entity name in Wikipedia URL suffix format (e.g. "Paper_clip") the code will return a rough componentization tree, and complexity score, for the entity. For each component, starting with the principal entity, the Wikipedia page is scraped. Anchor tags not corresponding to proper nouns or named entities are recorded. Each anchor tag has a list of phrases associated with it, which phrases are searched for in the article text. The frequency of each phrase group determines the frequency of the respective anchor tag. The most frequent anchor tags--those above some cutoff--are retained. For each remaining anchor tag, its phrase group is once again searched for in the text, looking for sentences which contain both a phrase from that anchor tag, and a phrase corresponding to the principal entity.

The resulting sentence has the respective phrase groups marked with delimiters ([E1][/E1] and [E2][/E2] tags), and are run through a neural network relation extractor model based on the [matching-the-blanks (MTB) process](https://arxiv.org/pdf/1906.03158.pdf). The github repo used for that is [here](https://github.com/plkmo/BERT-Relation-Extraction).


## Implementation Details

First, the relation extractor needs to be downloaded and trained. Instructions for that are on the linked [GitHub site](https://github.com/plkmo/BERT-Relation-Extraction). For demonstration purposes in the article, the extractor had only fine-tuning training on the SemEval2010 Task 8 data (see the corresponding section in the linked repo). Full pretraining on the CNN dataset was abandoned because it used too many processing resources (40 hours per epoch on a Pentium 5 with all 4 cores; a GPU was not available). Assume you've cloned the BERT relation extractor repo to local folder BERT-Relation-Extraction-folder.

Some tips for the BERT model / WTS:
- the BERT repo code uses unadorned relative paths to the data folder (./data ref's in both train_funcs.py (base_path), and init.py (completeName))--so unless you're invoking the script from its own directory, or using an IDE that does so, you'll have to modify (eg give absolute paths) the ./data path accordingly
- the routine had some problems creating the relations.pkl file, what appears to be created in the function preprocess_semeval2010_8 of /src/tasks/preprocessing_funcs.py, called from that same script file's load_dataloaders function (the errors may have been mine, in calling / invoking the script somehow), so if this is a problem, just install the relations pickle file in this repo (it's a small file; just a short dictionary of the 19 extraction relations, Component-Whole, etc.) locally in your BERT-Relation-Extraction-folder/data/ directory
- ultimately, for full functionality you want the relation extractor working as demonstrated on the [BERT relation extractor repo](https://github.com/plkmo/BERT-Relation-Extraction); here's a little demo script to check:
```
import sys
import pickle
from src.tasks.infer import infer_from_trained	# BERT relation extractor's "src" package needs to be in sys.path

class args:	# dummy class
	pass
args.model_no = 0; args.model_size='bert-base-uncased'; args.num_classes=19
inferer = infer_from_trained(args,detect_entities=False)
x = inferer.infer_sentence("A [E2]brickworks[/E2] is a place where [E1]bricks[/E1] are manufactured.",detect_entities=False)

# and if wanting to resolve the code returned from infer_sentence:
with open(<file path for relations.pkl>,'rb') as fp:
	Rel_Map = pickle.load(fp)
relations_dict = Rel_Map.idx2rel
print(relations_dict[x])	# should show "Product-Producer(e1,e2)"
```

Once the relation extractor is working, clone this repo to your local drive. Assume you've named the folder knowledge-build-complexity-folder. You need to provide a path to the relation extractor in this repo's BERT.py script:
```
BERT_repo_path = .../knowledge-build-complexity-folder/wiki_scrape_src/bert/
```
You'll also need to decompress the wiktionary plural / singular lookup json file in this repo's .../knowledge-build-complexity-folder/wiki_scrape_src/data folder, using your favorite 7zip utility. You should end up with a json file with name "ps_wiktionary.json" with a size of around 40MB in the folder.

Now you should be ready to go. Run the script main_scraper.py.

Note that Wikipedia does allow web scraping, but discourages high frequency requests--["Please do not use a web crawler to download large numbers of articles. Aggressive crawling of the server can cause a dramatic slow-down of Wikipedia."](https://en.wikipedia.org/wiki/Wikipedia:Database_download). Generally, Wikipedia servers prefer page requests to be more than one second apart. The code in main_scraper.py has a timer throttle to limit requests to once every 20 or so seconds. For repeated use of the code over the same article types, caching is recommended. The program has built-in caching capability. The script begins with a prompt for the location of a cache folder (the default is a relative path, ./data/wiki_cache, for a data folder at the outer level in this repo (not to be confused with the /data folder in this repo's wiki_scrape_src folder) that will only work if the script is launched from the script's parent directory, or called from an IDE with that feature).

If the script will be used heavily, Wikipedia encourages using one of their data dumps--you can enjoy scraping offline without throttling and avoid needlessly taxing Wikipedia's servers. For tips on obtaining a data dump of all of Wikipedia, see [here](https://www.online-tech-tips.com/computer-tips/how-to-download-wikipedia/) and [here](https://www.howtogeek.com/260023/how-to-download-wikipedia-for-offline-at-your-fingertips-reading/).


## Dependencies

There are a certain number of dependencies for this project. Some of the largest are the natural language processing toolkit (nltk), PyTorch, and Spacy. Here is a list of the most prominent:

for the linked [BERT](https://github.com/plkmo/BERT-Relation-Extraction) repository:
- PyTorch
- Spacy
- tqdm

for this repository:
- wikipedia
- wikipediaapi
- logging
- BeautifulSoup
- nltk


## Licenses

This repository's code is issued under the MIT license. However there are two integrated files with their own licenses:

*ps_wiktionary.json*:
- this is a pared down version of a full wiktionary dump obtained via [kaikki.org](https://kaikki.org/dictionary/index.html); this highly reduced file contains plural / singular complementation pairs for all words in wiktionary
- the official citation for the kaikki data is, Tatu Ylonen: Wiktextract: Wiktionary as Machine-Readable Structured Data, Proceedings of the 13th Conference on Language Resources and Evaluation (LREC), pp. 1317-1325, Marseille, 20-25 June 2022
- the kaikki file, and the wiktionary data it contains, are both licensed under CC-BY-SA and GFDL

*unigram_freq.csv*:
- this is a list of the most common 333,333 words in the English language, and their frequency of use, obtained from [kaggle.com](https://www.kaggle.com/rtatman/english-word-frequency)
- the kaggle page, and its reference, to [this site holding the actual word data](https://norvig.com/ngrams/) are both licensed under the MIT license



