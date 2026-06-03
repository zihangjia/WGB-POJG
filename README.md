# WGB-POJG: An Optimization-Driven Clustering Framework with Weighted Granular Balls Based on the Principle of Justifiable Granularity

This repository contains the code and datasets used in the paper submitted to **European Journal of Operational Research (EJOR)**:

**WGB-POJG: An Optimization-Driven Clustering Framework with Weighted Granular Balls Based on the Principle of Justifiable Granularity**

---

## Overview

The **WGB-POJG** framework is designed for clustering tasks on **complex and high-dimensional datasets**. It adopts **weighted granular balls** as the fundamental clustering units and constructs an optimization-driven clustering framework based on the **principle of justifiable granularity**.

This repository provides the MATLAB implementations of the proposed methods and all publicly available datasets used in the experiments on datasets **D1–D22**.

---

## Contents

* `WGB-POJG code/` : MATLAB implementations of **WGB-SC**, **WGB-DPC**, and **WGB-USC**.
* `dataset/` : All publicly available datasets used in the experiments.
* `README.md` : This file.
* `experiments` : All datasets, detailed experimental results, and source code used in the experiments.

---

## Requirements

* **MATLAB 2025b** : Required to run the **WGB-SC**, **WGB-DPC**, and **WGB-USC** algorithms.

---

## Usage

1. Open MATLAB and navigate to one of the following folders:

   * `WGB-SC`
   * `WGB-DPC`
   * `WGB-USC`

2. Run the `reproduce.m` script to execute the algorithm on datasets **D1–D18**:

```matlab
reproduce
```
