import pytest
from src.processors.real_time_processor import RealTimeProcessor

@pytest.mark.asyncio
async def test_realtime_processing():
    processor = RealTimeProcessor()
    test_data = {
        "trading_pair": "BTC/USD",
        "timeframe": "1m",
        "data": [1000.0, 1001.0, 1002.0]
    }
    
    result = await processor.process_stream(test_data)
    assert result["status"] == "success"
    assert "data" in result 