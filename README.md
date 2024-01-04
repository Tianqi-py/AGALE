# Project Title

A Graph-Aware Continual Learning Evaluation Framework

## Overview

This repository contains the code implementation for the AGALE: A Graph-Aware Continual Learning Evaluation Framework. The paper introduces a framework that accommodates both single-labeled and multi-labeled nodes, addressing the limitations of previous evaluation frameworks in CGL, and the code here is intended to reproduce the experimental results presented in the paper.

## Paper Abstract
In recent years, continual learning (CL) techniques have made significant progress in learning from  streaming data while preserving knowledge across sequential tasks, particularly in the realm of Euclidean data. To foster fair evaluation and recognize challenges in CL settings, several evaluation frameworks have been proposed, focusing mainly on the single- and multi-label classification task on Euclidean data. However, these evaluation frameworks are not trivially applicable when the input data is graph-structured, as they do not consider the topological structure inherent in graphs. We develop a graph-aware evaluation (\agale) framework that accommodates both single-labeled and multi-labeled nodes, addressing the limitations of previous evaluation frameworks. In particular, we define new incremental settings and devise data partitioning algorithms tailored to CGL datasets. We perform extensive experiments comparing methods from the domains of continual learning, continual graph learning, and dynamic graph learning (DGL). We theoretically analyze \agale and provide new insights about the role of graph homophily in the performance of the compared method.


## Getting Started

### Prerequisites

- Python 3.9
- Pytorch
- Pytorch_geometric
- Numpy
- Pandas

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



