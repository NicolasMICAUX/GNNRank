#!/bin/bash
source ../../venv/bin/activate
cd ../src
python ./train.py --dataset imdb_wiki_sbs --all_methods all_methods_shorter
