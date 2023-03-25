#!/bin/bash
source ../../venv/bin/activate
cd ../src
python ./train.py --dataset tournesol --all_methods all_methods_shorter
