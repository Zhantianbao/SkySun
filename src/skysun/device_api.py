"""Authenticated device HTTP routes and the companion heartbeat channel."""

from collections.abc import Callable
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from skysun.devices import PostgresDeviceStore
from skysun.user_identity import Actor


class DeviceRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    hostname: str = Field(min_length=1, max_length=255)
    operating_system: str = Field(min_length=1, max_length=100)


class Heartbeat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str = "heartbeat"
    cpu_percent: float = Field(ge=0, le=100)
    memory_percent: float = Field(ge=0, le=100)
    disk_percent: float = Field(ge=0, le=100)


def device_router(store: PostgresDeviceStore, identify: Callable) -> APIRouter:
    router = APIRouter()

    @router.post("/devices", status_code=201)
    def register(request: DeviceRegistration, actor: Actor = Depends(identify)) -> dict:
        device_id, token = store.register(actor.user_id, request.name,
                                          request.hostname, request.operating_system)
        return {"device_id": str(device_id), "device_token": token}

    @router.get("/devices")
    def list_devices(actor: Actor = Depends(identify)) -> list[dict]:
        return [{"device_id": str(row.device_id), "name": row.name,
                 "hostname": row.hostname, "operating_system": row.operating_system,
                 "online": row.online,
                 "last_heartbeat_at": row.last_heartbeat_at.isoformat()
                 if row.last_heartbeat_at else None,
                 "cpu_percent": row.cpu_percent, "memory_percent": row.memory_percent,
                 "disk_percent": row.disk_percent} for row in store.list_for_user(actor.user_id)]

    @router.delete("/devices/{device_id}", status_code=204)
    def delete(device_id: UUID, actor: Actor = Depends(identify)) -> None:
        if not store.delete(actor.user_id, device_id):
            raise HTTPException(status_code=404, detail="Device not found")

    @router.websocket("/api/ws/devices/{device_id}")
    async def channel(websocket: WebSocket, device_id: UUID) -> None:
        authorization = websocket.headers.get("authorization", "")
        if not authorization.startswith("Bearer ") or store.authenticate(
            device_id, authorization[7:]
        ) is None:
            await websocket.close(code=4401)
            return
        session_id = uuid4()
        await websocket.accept()
        try:
            store.connect(device_id, session_id)
            await websocket.send_json({"type": "connected", "device_id": str(device_id)})
            while True:
                try:
                    heartbeat = Heartbeat.model_validate(await websocket.receive_json())
                    if heartbeat.type != "heartbeat":
                        raise ValueError("Unsupported device message")
                except (ValueError, TypeError):
                    await websocket.close(code=1008)
                    break
                if not store.heartbeat(device_id, session_id, heartbeat.cpu_percent,
                                       heartbeat.memory_percent, heartbeat.disk_percent):
                    await websocket.close(code=1008)
                    break
                await websocket.send_json({"type": "heartbeat_ack"})
        except WebSocketDisconnect:
            pass
        finally:
            store.disconnect(device_id, session_id)

    return router
