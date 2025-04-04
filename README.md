# Project Mercury: Sparse Data Imputation for Crypto Signals

![Crypto Data Analytics](url-to-your-banner-if-you-have-one)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 🎯 Overview
A hybrid data imputation solution for handling sparse order-book data in cryptocurrency trading signals, specifically focused on BTC/USD pairs.

## 🔍 Problem Statement
- Sparse order-book data with 40% missing values
- Degraded ML model accuracy for BTC/USD trading signals

## 💡 Solution
Implemented a hybrid approach combining:
- k-NN imputation for short-term gaps
- GAN-based synthetic data generation for structural holes
- Real-time processing with Kafka integration
- Multi-pair trading support

## 🚀 Performance Metrics
- **Model Accuracy Improvement:** 27% increase
  - Initial F1-score: 0.62
  - Final F1-score: 0.79
- **Latency:** Less than 100ms per 10k rows

## 🛠 Technical Stack
- Python (pandas, TensorFlow)
- FastAPI for REST endpoints
- Kafka for real-time processing
- Redis for caching
- Prometheus for metrics
- Docker for containerization

## 📁 Project Structure 