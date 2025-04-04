from mercury import MercuryDataLoader, DataProcessor
from mercury.config import Config
from sklearn.metrics import f1_score
import xgboost as xgb

# Initialize components
config = Config(
    project="crypto-data",
    dataset="market_data",
    enable_anomaly_detection=True,
    enable_gap_filling=True
)

loader = MercuryDataLoader(config)
processor = DataProcessor(config)

# Load sparse data
raw_data = loader.load_market_data(
    symbol="BTC/USD",
    start_time="2024-03-20T00:00:00",
    interval="1m"
)

# Apply hybrid imputation
clean_data = processor.process(raw_data, target_col='price')

# Evaluate improvement
def evaluate_model(data, target='price_direction'):
    model = xgb.XGBClassifier()
    X = data[['price', 'volume', 'bid_ask_spread']]
    y = data[target]
    
    # Simple train/test split for demonstration
    train_idx = int(len(data) * 0.8)
    model.fit(X[:train_idx], y[:train_idx])
    preds = model.predict(X[train_idx:])
    return f1_score(y[train_idx:], preds)

# Compare results
baseline_score = evaluate_model(raw_data.fillna(method='ffill'))
mercury_score = evaluate_model(clean_data)

print(f"Baseline F1 Score: {baseline_score:.2f}")
print(f"Mercury F1 Score: {mercury_score:.2f}")
print(f"Improvement: {((mercury_score - baseline_score) / baseline_score) * 100:.1f}%")

# Analyze imputation quality
imputation_stats = clean_data['imputation_type'].value_counts()
print("\nImputation Statistics:")
print(f"Original values: {imputation_stats['original']}")
print(f"k-NN filled: {imputation_stats['knn']}")
print(f"GAN synthesized: {imputation_stats['gan']}") 