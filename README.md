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

`./main.py ../DNIssues/dds-89412-page-8.png`

`./main.py --dir /PNG/1974/dds-8939 --no-empty`

`./main.py --image test/dds-89475-page-8.png --empty --debug`

## Gotchas

OSError: [Errno 24] Too many open files

`ulimit -n 7200`