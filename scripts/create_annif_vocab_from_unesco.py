#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create Vocabulary Reference File From UNESCO Thesaurus

This is a script file.
"""
import sys
import json
from pprint import pprint
from rdflib import Graph
from rdflib import Literal, Namespace
from rdflib.namespace import SKOS, OWL

UNESCO = Namespace('http://vocabularies.unesco.org/thesaurus/') 

def main(rdfxml_in):
    """Prints the URI and Term contained in the RDF/XML input file."""
    unesco = Graph()
    #print(len(unesco))  # show number of triples
    unesco.parse(rdfxml_in, format='xml') 
    lang = 'en'   
    for i, (subject, predicate, obj) in enumerate(unesco):
        if not (subject,predicate,obj) in unesco:
            raise Exception("Iterator / Container Protocols are Broken!!")

        try:
            if obj.language == lang:
                pLabel = unesco.preferredLabel(subject, lang=lang, default=None, labelProperties=(SKOS.prefLabel,))
                if pLabel:
                    print(f"<{str(subject)}>\t{pLabel[0][1]}") # specify that we want preflabel not scopeNote etc
                    # pprint((str(subject),predicate,str(obj), obj.language), indent=2)
                    # pprint((subject,predicate,obj), indent=2) 
        except AttributeError as e:
            pass
   
if __name__ == '__main__':
    # QUESTION: Which way stdin is expected?
    if sys.stdin.isatty():  # When no redirected input from file
        # Print instructions and wait for input if not redirected input?
        print('No redirected stdin data from file found. '
              'Example usage with input and output data files:\n'
              '\t$ ./create_annif_vocab_from_unesco.py < ../data/thesauri/unesco/unesco-thesaurus.rdf > ../data/vocabs/unesco-en.tsv'
              '\n\nWaiting for input:')
        sys.exit()
    main(sys.stdin)
