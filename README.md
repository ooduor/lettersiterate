# lettersiterate
Newspaper Images . Letters to the editor . Repeat

## Setup
Check out source code.

`git clone https://github.com/ooduor/lettersiterate.git`

## Pre-requisites

### Create virtual env (recommended)
`virtualenv -p python3.6 venv`

### Install dependent python libraries
`pip install -r requirements.txt`

## Usage

Image quality and size affects the output. Minimum recommended width for best results is 2048 pixels.

`./main_steps.py --image ../DNIssues/dds-89412-page-8.png --debug`

`./main_steps.py --image test/dds-89491-page-8.png --debug --empty`

`./main.py --dir /PNG/1974/dds-8939 --no-empty`

`./main.py --image test/dds-89475-page-8.png --empty --debug`

## Text Pre-processing and cleaning step

`./txt_pre_proc.py  --txt TXT/1975/dds-89458-page-8-article-3.txt --debug`

`./txt_pre_proc.py  --dir TXT/1975 --debug`

## Gotchas

OSError: [Errno 24] Too many open files

`ulimit -n 7200`

# ANNIF
`annif loadvoc tfidf-en Annif-corpora/vocab/yso-en.tsv`

`annif train tfidf-en Annif-corpora/training/yso-finna-en.tsv.gz`

`cat ../lettersiterate/output/dds-89491-page-8-article-17.txt | annif suggest tfidf-en`


