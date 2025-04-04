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


## 📋 Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8+
- Docker and Docker Compose
- Kafka
- Redis

## 🔧 Installation

bash
Clone the repository
git clone https://github.com/somtonweke1/Mercury.git
Navigate to the project directory
cd project-mercury
Install dependencies
pip install -r requirements.txt
Start the services
docker-compose up -d


## 💻 Usage

### REST API Endpoints

bash
Impute data
curl -X POST "http://localhost:8000/api/v1/impute" \
-H "Content-Type: application/json" \
-d '{"trading_pair": "BTC/USD", "timeframe": "1m", "data": [1000.0, 1001.0]}'



### Supported Trading Pairs
- BTC/USD (depth: 10, min_volume: 1.0)
- ETH/USD (depth: 10, min_volume: 5.0)
- SOL/USD (depth: 10, min_volume: 2.0)
- AVAX/USD (depth: 10, min_volume: 3.0)

## 🧪 Testing

bash
Run tests
pytest tests/

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

- [x] Initial k-NN implementation
- [x] GAN integration
- [x] Kafka real-time processing
- [x] Multi-pair trading support
- [x] REST API implementation
- [ ] WebSocket support
- [ ] Advanced monitoring dashboards
- [ ] Auto-scaling configuration

## 📊 Project Status

Project is: _in active development_
