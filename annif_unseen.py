#!/usr/bin/env python
import os
import re
import sys
import glob
import argparse
from pathlib import Path
from pprint import pprint
from xml.dom import minidom
import xml.etree.cElementTree as ET

import annif
import annif.eval
import annif.corpus
import annif.project
import annif.backend
from annif.project import Access
from annif.corpus.skos import SubjectFileSKOS
from annif.suggestion import SubjectSuggestion, SuggestionResult, \
    LazySuggestionResult, ListSuggestionResult, SuggestionFilter
from annif.corpus import SubjectIndex

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
    
def append_suggestions_node(results, txt_xml_path):
    # https://stackoverflow.com/questions/28782864/modify-xml-file-using-elementtree
    tree = ET.ElementTree(file=txt_xml_path)
    root = tree.getroot()
    
    #subjects = ET.SubElement(root, "subjects")
    subjects = ET.Element('subjects')
    for result in results:
        pprint(result, indent=2)
        ET.SubElement(subjects, "subject", score=str(result.score), uri=str(result.uri)).text = result.label

    root.insert(1, subjects)

    xmlstr = ET.tostring(root).decode()
    xmlstr = minidom.parseString(xmlstr).toprettyxml(indent="", newl="")
    return xmlstr
                
def main(args):
    # get params
    proc_year = args.year
    
    prj_root = os.getcwd()
    #prj_root = os.path.join(current_directory, rf'{output_name}')
    #prj_root = os.path.dirname(current_directory)
    data_dir = f'{prj_root}/data'
    txt_proc_dir = f'{data_dir}/TXT_PROC'
    txt_xml_dir = f'{data_dir}/TXT_XML'
    xml_dir = f'{data_dir}/XML'    
    
    path_list = []
    for f in sorted(Path(txt_proc_dir).glob(f'{proc_year}/*.txt')):
        txt_path = str(f) # cast PosixPath to str
        txt_name = os.path.basename(txt_path)
        path_list.append(txt_name)

    if not os.path.exists(f"{xml_dir}/{proc_year}"):
        os.makedirs(f"{xml_dir}/{proc_year}")

    for idx, path in enumerate(path_list):
        txt_name = os.path.basename(path)
        txt_sans_ext = os.path.splitext(txt_name)[0]
        txt_proc_path = f"{txt_proc_dir}/{proc_year}/{txt_name}"
        txt_xml_path = f"{txt_xml_dir}/{proc_year}/{txt_sans_ext}.xml"
        xml_path = f"{xml_dir}/{proc_year}/{txt_sans_ext}.xml"

        # only handle valid files i.e. files with useful size of text
        if os.stat(txt_proc_path).st_size > 50: # more 50 bytes at least
            with open(txt_proc_path, 'r', encoding='utf-8') as infile:
                whole_text = infile.read()

                results = backend.suggest(whole_text, project)

                xmlstr = append_suggestions_node(results, txt_xml_path)

                with open(xml_path,'w') as outfile:
                    outfile.write(xmlstr)
                    print(f"{xml_path} DONE!")
                

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(prog="Annifier", description='Suggest Subject and Append Subject into Custom XML')
    parser.add_argument('--year', type=str, help='Year to process') # Optional argument
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()

    if not args.year:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # execute only if run as the entry point into the program
    main(args)                
