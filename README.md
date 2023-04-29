# CMU 10708 Final Project 

## Introduction
We propose an no-acyclic constrain learning paradigm for DAG discovery. We also benchmark established algorithms on synthetic data.

## Requirements

please install

- [NOTEARS](https://github.com/xunzheng/notears)
- [DAG-GNN](https://github.com/fishmoon1234/DAG-GNN)
- [DAGMA](https://github.com/kevinsbello/dagma)
- [GOLEM](https://github.com/ignavierng/golem)

## Synthetic Data

The constructed synthetic data on 10, 20, 50, 100 nodes Erdos-Renyi graph are at:

```bash
$ project-708/data_p10_e10_n1000_GP
$ project-708/data_p20_e20_n1000_GP
$ project-708/data_p50_e50_n1000_GP
$ project-708/data_p100_e100_n1000_GP
```

## To Test

NOTEARS

```bash
$ cd notears/
$ python notears/linear.py
```

AutoPerm (ours)

```bash
$ cd notears/
$ python notears/perm_linear.py
```

DAGMA

```bash
$ cd dagma/
$ python dagma/dagma_linear.py
```

GOLEM

```bash
$ cd golem/
$ python src/golem.py
```

DAG-GNN

```bash
$ cd DAG-GNN/
$ python src/train.py
```


