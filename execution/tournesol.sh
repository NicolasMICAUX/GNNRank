#!/bin/bash
source ../../venv/bin/activate
cd ../src
python ./train.py -D --dataset tournesol
