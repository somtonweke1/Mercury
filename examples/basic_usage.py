from mercury import MercuryDataLoader
from mercury.config import Config

# Initialize with custom config
config = Config(
    project="crypto-data",
    dataset="market_data",
    enable_anomaly_detection=True
)

# Create loader
loader = MercuryDataLoader(config)

# Load BTC/USD data for last 24 hours
df = loader.load_market_data(
    symbol="BTC/USD",
    start_time="2024-03-20T00:00:00",
    interval="1m"
)

# Print available symbols
print("Available trading pairs:", loader.get_symbols())

# Access the data
print("\nFirst few rows:")
print(df.head())

# Check data quality
print("\nMissing data summary:")
print(df.isna().sum()) 