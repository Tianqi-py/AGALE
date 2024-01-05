# Project Title

A Graph-Aware Continual Learning Evaluation Framework

## Overview

This repository contains the code implementation for the AGALE: A Graph-Aware Continual Learning Evaluation Framework. The paper introduces a framework that accommodates both single-labeled and multi-labeled nodes, addressing the limitations of previous evaluation frameworks in CGL, and the code here is intended to reproduce the experimental results presented in the paper.

## Getting Started

### Prerequisites

- dgl==1.0.2
- matplotlib==3.7.1
- numpy==1.23.5
- pandas==1.3.5
- pipeline==0.1
- scikit_learn==1.2.2
- scipy==1.10.1
- seaborn==0.13.1
- torch==1.12.1
- torch_geometric==2.3.0
- torch_geometric_temporal==0.54.0

### Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:Tianqi-py/AGALE.git
    cd AGALE
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

All the baselines used in this paper are implemented in the baselines/. The backbone GNNs plugged into the baselines are in backbones/. 

The folder baselines/models/ contains all the baseline models reimplemented in the two incremental settings proposed in this paper with the expandable output units for the unknown number of classes coming in the future time steps.

The name of the Python file in baselines/XX.py indicates the name of the baseline and the incremental setting in which it is implemented. For example, LwF_TaskIL is the script to run LwF in TaskIL.

Example 1: To run a baseline model in an incremental setting, run the Python file in baselines/.

Run LwF in TaskIL:

```bash
python3 LwF_TaskIL.py --backbone GCN --n_cls_per_t 2 --lr 0.01
```

Run LwF in ClassIL:

```bash
python3 LwF_ClassIL.py --backbone GCN --n_cls_per_t 2 --lr 0.01
```



