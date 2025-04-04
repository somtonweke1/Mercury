from fastapi import WebSocket
from typing import Dict, List, Any
import json
import asyncio
from ..processors.real_time_processor import RealTimeProcessor

class WebSocketHandler:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.processor = RealTimeProcessor()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            await connection.send_json(message)

    async def process_message(self, websocket: WebSocket, data: Dict[str, Any]):
        try:
            result = await self.processor.process_stream(data)
            await websocket.send_json(result)
        except Exception as e:
            await websocket.send_json({"error": str(e)}) 