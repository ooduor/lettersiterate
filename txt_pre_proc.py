#!/usr/bin/env python
import os
import sys
import glob
import logging
from pathlib import Path
from pprint import pprint
import math
import argparse
from spellchecker import SpellChecker
from functools import reduce
import re
import sre_constants
from collections import Counter

from utils import lines_extraction

# def words(text): return re.findall(r'\w+', text)
def words(text): return re.findall(r'[^-,\.\n\r\s]+', text, flags=re.ASCII) # [^,\s]+ match any text that is not a tab and not a whitespace.

def process_txt(path_to_txt, pre_text_path, empty_output):
    # check if file exists here and exist if not
    try:
        f = open(path_to_txt)
        f.close()
    except FileNotFoundError:
        logging.critical('Given text file does not exist')
        sys.exit(0)

    logging.info(f"Processing {path_to_txt}")

    # nested withs https://stackoverflow.com/a/9283052/754432
    with open(pre_text_path, 'w') as outfile, open(path_to_txt, 'r', encoding='utf-8') as infile:
        infile = infile.read().lower()

        infile = re.sub(r'-\n', "", infile) # conjoin hypehnated newline words
        # infile = re.sub(r'\n\r-,\.', "", infile)
        infile = re.sub(r'\n', " ", infile)
        infile = re.sub(r'[\:\(\)\|\]\[\*©]', "", infile) # discard non-spellable items, don't carry any significance towards misspelling
        infile = re.sub(r'(\w)\.(\w)', "i", infile) # period character inside a word is often closer to 'i' than any other alphabet character for purposes of spelling correction

        WORDS = Counter(words(infile))

        spell = SpellChecker()  # loads default word frequency list

        # find those words that may be misspelled
        misspelled = spell.unknown(WORDS)

        auto_corrects  = {}
        for i, word in enumerate(misspelled):
            auto_corrects[word] = spell.correction(word)
            # Get the one `most likely` answer
            # if logging.getLogger().level == logging.DEBUG: print("#{}".format(i), word, spell.correction(word))
            # Get a list of `likely` options
            # if logging.getLogger().level == logging.DEBUG: print(spell.candidates(word))
        if logging.getLogger().level == logging.DEBUG: pprint(auto_corrects, indent=2)

        try:
            str_out = re.sub(r'\b(%s)\b' % '|'.join(auto_corrects.keys()), lambda m:auto_corrects.get(m.group(1), m.group(1)), infile)
        except sre_constants.error as err:
            print("SRE ERROR:", err)
        except Exception as err:
            print("GENERAL ERROR:", err, type(err))
        finally:
            # SIGH! Just return the lightly cleaned text for this one
            str_out = infile

        #  str_out = re.sub(r'\b(\w+)\b', lambda m:auto_corrects.get(m.group(1), m.group(1)), f_read)
        if logging.getLogger().level == logging.DEBUG: print(str_out)
        outfile.write(str_out)
    logging.info(f"Processed {pre_text_path}")
    return True

def main(args):
    # get params
    path_to_txt = args.txt
    path_to_dir = args.dir
    empty_output = args.empty

    # create out dir
    current_directory = os.getcwd()

    if path_to_dir:
        last_folder_name = os.path.basename(path_to_dir)
        final_directory = os.path.join(current_directory, f'TXT_PRE/{last_folder_name}')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        
        for f in sorted(Path(path_to_dir).glob('**/*.txt')):
            txt_path = str(f) # cast PosixPath to str
            txt_name = os.path.basename(txt_path)
            txt_sans_ext = os.path.splitext(txt_name)[0]

            pre_text_path = os.path.join(final_directory, f'{txt_sans_ext}-PRE.txt')

            if os.path.exists(pre_text_path):
                logging.info(f"FILE EXISTS: {pre_text_path}")
                continue

            process_txt(txt_path, pre_text_path, empty_output)
    elif path_to_txt:
        output_path = os.path.dirname(path_to_txt)
        last_folder_name = os.path.basename(output_path)
        txt_name = os.path.basename(path_to_txt)
        txt_sans_ext = os.path.splitext(txt_name)[0]

        final_directory = os.path.join(current_directory, f'TXT_PRE/{last_folder_name}')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        pre_text_path = os.path.join(final_directory, f'{txt_sans_ext}-PRE.txt')

        process_txt(path_to_txt, pre_text_path, empty_output)

    print('Main code {}'.format(args))

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(prog="LettersIterate", description="Pre-processing to correct 'misspelled' words in the corpora")
    parser.add_argument('--txt', type=str, help='Path to the .txt file')
    parser.add_argument('--dir', type=str, help='An optional path to directory with letter .txt files') # Optional argument
    parser.add_argument('--empty', dest='empty', action='store_true', help='An optional boolean argument to empty output folder before each processing')
    parser.add_argument('--no-empty', dest='empty', action='store_false', help='An optional boolean argument to NOT empty output folder before each processing')
    parser.add_argument('--debug', dest="loglevel", action='store_true', help='print debug messages to stderr')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()

    if not args.txt and not args.dir:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.loglevel:
        logging.basicConfig(level=logging.DEBUG) # 10
    else:
        logging.basicConfig(level=logging.INFO) # 20

    # execute only if run as the entry point into the program
    main(args)
