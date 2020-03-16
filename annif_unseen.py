#!/usr/bin/env python
from pprint import pprint
import xml.etree.cElementTree as ET
from types import SimpleNamespace

import annif
import annif.corpus
import annif.eval
import annif.project
from annif.project import Access
from annif.corpus.skos import SubjectFileSKOS
from annif.suggestion import SubjectSuggestion, SuggestionResult, \
    LazySuggestionResult, ListSuggestionResult, SuggestionFilter
from annif.corpus import SubjectIndex
import annif.backend
from sklearn.feature_extraction.text import TfidfVectorizer
import xml.etree.cElementTree as ET
from xml.dom import minidom

# text = open('data/TXT_PROC/1975/dds-89398-page-8-article-01.txt', "r").read().lower()
text = open('data/TXT_PROC/1975/dds-89407-page-8-article-10.txt', "r").read()

class DotDict(dict):
    pass

project = DotDict()
project.name = 'Letters Omikuji Parabel English'
project.language= 'en'
project.backend='letters-omikuji-bonsai-en'
project.analyzer = annif.analyzer.get_analyzer('snowball(english)')
project.limit=10
project.vocab='letters-unesco'
project.subjects = SubjectIndex.load('./data/vocabs/letters-unesco/subjects')
project.datadir = str('./data/projects/letters-omikuji-bonsai-en')

backend_type = annif.backend.get_backend("omikuji")
backend = backend_type(
    backend_id='letters-omikuji-bonsai-en',
    config_params={'limit': 5},
    project=project
)

results = backend.suggest(text, project)

# https://stackoverflow.com/questions/28782864/modify-xml-file-using-elementtree
tree = ET.ElementTree(file='dds-89407-page-8-article-10.xml')
root = tree.getroot()

#subjects = ET.SubElement(root, "subjects")
subjects = ET.Element('subjects')
for result in results:
    pprint(result, indent=2)
    ET.SubElement(subjects, "subject", score=str(result.score), uri=str(result.uri)).text = result.label

root.insert(1, subjects)

xmlstr = ET.tostring(root).decode()
xmlstr = minidom.parseString(xmlstr).toprettyxml(indent="", newl="")

with open("filename.xml",'w') as outfile:
    outfile.write(xmlstr)
