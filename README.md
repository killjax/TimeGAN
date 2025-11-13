# Codebase for "Stress Testing Trading Strategy Framework with TimeGAN Project"

## Project group Members

- Nishvaan Sai H - 200648
- Rohit Raj - 210874
- Aman Kashyap - 210108
- Jayant Malik - 242110401

## Project Report and Project GitHUB link

- Project Report: [CS787_Project_Report.pdf](CS787_Project_Report.pdf)
- GitHUB link: <https://github.com/killjax/Stress_Testing_using_TimeGAN>

## Project Framework

This project is about creating a framework for Stress Testing of Trading Strategies using TimeGAN. This framework uses market condition of a Time-series data as static features.

This directory contains implementations of Stress Testing trading strategies using TimeGAN framework for synthetic time-series data generation of different market scenarios using real-world financial Time-Series data loaded from Yahoo Finance API, using yfinance library

- ^GSPC: <https://finance.yahoo.com/quote/%5EGSPC/history/?p=^GSPC>
- AAPL: <https://finance.yahoo.com/quote/AAPL/history/?p=AAPL>
- GOOG: <https://finance.yahoo.com/quote/GOOG/history?p=GOOG>
- MSFT: <https://finance.yahoo.com/quote/MSFT/history/?p=MSFT>
- AMZN: <https://finance.yahoo.com/quote/AMZN/history/?p=AMZN>

To run the pipeline for training, generating and evaluating synthetic datasets of Stress testing Trading Strategy Framework with TimeGAN, simply see jupyter-notebook tutorial of Stress Testing Trading Stratesgies Framework with TimeGAN in stress_testing_timegan.ipynb.

Note that any model architecture can be used as the generator and
discriminator model such as RNNs or Transformers.

## Paper the project is based on

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper Link: <https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks>

## Code explanation

### (1) data_loading.py

- Loading Financial time series dataset and adding features
- Pre-processing raw time series dataset
- Labelling time series dataset

### (2) Metrics directory

(a) visualization_metrics.py

- PCA and t-SNE analysis between Original data and Synthetic data

(b) discriminative_metrics.py

- Use Post-hoc RNN to classify Original data and Synthetic data

(c) predictive_metrics.py

- Use Post-hoc RNN to predict one-step ahead (last feature)

### (3) timegan.py

- Use original time-series data along with their label as static features as training set to train TimeGAN model and save the networks
- Generate synthetic time series data, along with their synthetic static features

### (4) utils.py

- Some utility functions for metrics and timeGAN.

## Command inputs

- data_name: ^GSPC, AAPL, GOOG, MSFT, AMZN
- seq_len: sequence length
- module: gru, lstm, or lstmLN
- hidden_dim: hidden dimensions
- num_layers: number of layers
- iterations: number of training iterations
- batch_size: the number of samples in each batch
- metric_iterations: number of iterations for metric computation

Note that network parameters should be optimized for different datasets.

## Outputs

- ori_data_x: temporal fetaures of original data
- ori_data_s: static features of original data
- generated_data_x: temporal fetaures of generated synthetic data
- generated_data_s: static features of generated synthetic data
- metric_results: discriminative and predictive scores
- visualization: PCA and tSNE analysis

## Instructions

```shell
$ python3 -m venv venv
source env/bin/activate
pip install -r requirements.txt
```
