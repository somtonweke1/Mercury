from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from ..processors.real_time_processor import RealTimeProcessor
from ..websocket.socket_handler import WebSocketHandler
from ..monitoring.dashboard import MonitoringDashboard
from ..scaling.autoscaler import AutoScaler

app = FastAPI()
processor = RealTimeProcessor()
websocket_handler = WebSocketHandler()
monitoring = MonitoringDashboard()
autoscaler = AutoScaler()

class ImputeRequest(BaseModel):
    trading_pair: str
    timeframe: str
    data: List[float]

@app.post("/api/v1/impute")
async def impute_data(request: ImputeRequest):
    try:
        result = await processor.process_stream({
            "trading_pair": request.trading_pair,
            "timeframe": request.timeframe,
            "data": request.data
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_handler.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await websocket_handler.process_message(websocket, data)
    except Exception as e:
        monitoring.record_error("websocket_error")
    finally:
        websocket_handler.disconnect(websocket) 