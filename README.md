# GNNRank on custom datasets

## Environment Setup
### Overview
<!-- The underlying project environment composes of following componenets: -->
The project has been tested on the following environment specification:
1. Ubuntu 18.04.6 LTS (Other x86_64 based Linux distributions should also be fine, such as Fedora 32)
2. Nvidia Graphic Card (NVIDIA Tesla T4 with driver version 450.142.00) and CPU (Intel Core i7-10700 CPU @ 2.90GHz)
3. Python 3.7 (and Python 3.6.12)
4. CUDA 11.0 (and CUDA 9.2)
5. Pytorch 1.10.1 (built against CUDA 11.0) and Pytorch 1.8.0 (build against CUDA 10.2)
6. Other libraries and python packages (See below)

### Installation method 2 (manual installation)
The codebase is implemented in Python 3.6.12. package versions used for development are below.
```
networkx           2.6.3
tqdm               4.62.3
numpy              1.20.3
pandas             1.3.4
texttable          1.6.4
latextable         0.2.1
scipy              1.7.1
argparse           1.1.0
scikit-learn       1.0.1
stellargraph       1.2.1 (for link direction prediction: conda install -c stellargraph stellargraph)
torch              1.10.1
torch-scatter      2.0.9
pyg                2.0.3 (follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
sparse             0.13.0
```

### Execution checks
When installation is done, you could check you enviroment via:
```
cd execution
bash setup_test.sh
```

## Folder structure
- ./execution/ stores files that can be executed to generate outputs. For vast number of experiments, we use [GNU parallel](https://www.gnu.org/software/parallel/), which can be downloaded in command line and make it executable via:
```
wget http://git.savannah.gnu.org/cgit/parallel.git/plain/src/parallel
chmod 755 ./parallel
```

- ./joblog/ stores job logs from parallel. 
You might need to create it by 
```
mkdir joblog
```

- ./Output/ stores raw outputs (ignored by Git) from parallel.
You might need to create it by 
```
mkdir Output
```

- ./data/ stores processed data sets.

- ./src/ stores files to train various models, utils and metrics.

- ./result_arrays/ stores results for different data sets. Each data set has a separate subfolder.

- ./result_anlysis/ stores notebooks for generating result plots or tables.

- ./logs/ stores trained models and logs, as well as predicted clusters (optional). When you are in debug mode (see below), your logs will be stored in ./debug_logs/ folder.

## Options
<p align="justify">
GNNRank provides various command line arguments, which can be viewed in the ./src/param_parser.py. Some examples are:
</p>

```
  --epochs                INT         Number of GNNRank (maximum) training epochs.              Default is 1000. 
  --early_stopping        INT         Number of GNNRank early stopping epochs.                  Default is 200. 
  --num_trials            INT         Number of trials to generate results.                     Default is 10.
  --lr                    FLOAT       Initial learning rate.                                    Default is 0.01.  
  --weight_decay          FLOAT       Weight decay (L2 loss on parameters).                     Default is 5^-4. 
  --dropout               FLOAT       Dropout rate (1 - keep probability).                      Default is 0.5.
  --hidden                INT         Number of embedding dimension divided by 2.               Default is 32. 
  --seed                  INT         Random seed.                                              Default is 31.
  --no-cuda               BOOL        Disables CUDA training.                                   Default is False.
  --debug, -D             BOOL        Debug with minimal training setting, not to get results.  Default is False.
  -AllTrain, -All         BOOL        Whether to use all data to do gradient descent.           Default is False.
  --SavePred, -SP         BOOL        Whether to save predicted results.                        Default is False.
  --dataset               STR         Data set to consider.                                     Default is 'ERO/'.
  --all_methods           LST         Methods to use to generate results.                       Default is ['btl','DIGRAC'].
```

## Direct execution with training files

First, get into the ./src/ folder:
```
cd src
```

Then, below are various options to try:

Creating a GNNRank model for animal data using DIGRAC as GNN, also produce results on syncRank.
```
python ./train.py --all_methods DIGRAC syncRank --dataset animal
```
Creating a GNNRank model for ERO data using both DIGRAC and ib as GNN with 350 nodes, using 0.05 as learning rate.
```
python ./train.py --N 350 --all_methods DIGRAC ib --lr 0.05
```
Creating a GNNRank model for basketball data in season 2010 using all baselines excluding mvr, also save predicted results.
```
python ./train.py --dataset basketball --season 2010 -SP --all_methods baselines_shorter
```
Creating a model for HeadToHead data set with specific number of trials, hidden units and use CPU.
```
python ./train.py --dataset HeadToHead --no-cuda --num_trials 5 --hidden 8
```
--------------------------------------------------------------------------------

## Notes
- For certain applications such as financial data sets, the original adjacency matrices might be skew-symmetric with negative edge weights. For our models here, however, we need to preprocess the data so that we only keep the positive edge weights, as our current pipeline, including the loss functions, are restricted to directed unsigned networks as inputs.

# Credits
```bibtex
@inproceedings{he2022gnnrank,
  title={GNNRank: Learning Global Rankings from Pairwise Comparisons via Directed Graph Neural Networks},
  author={He, Yixuan and Gan, Quan and Wipf, David and Reinert, Gesine D and Yan, Junchi and Cucuringu, Mihai},
  booktitle={International Conference on Machine Learning},
  pages={8581--8612},
  year={2022},
  organization={PMLR}
}
```