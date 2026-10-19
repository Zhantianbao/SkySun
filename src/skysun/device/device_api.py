"""Authenticated device HTTP routes and the companion heartbeat channel."""

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from skysun.device.development import MAX_SCREENSHOT_BYTES, PostgresDevelopmentStore
from skysun.device.devices import PostgresDeviceStore
from skysun.user.user_identity import Actor


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
    project_name: str | None = Field(default=None, max_length=100)
    git_branch: str | None = Field(default=None, max_length=255)
    git_modified_count: int | None = Field(default=None, ge=0, le=100000)
    last_activity_at: datetime | None = None
    vscode_running: bool | None = None
    codex_active: bool | None = None
    vscode_workspace: str | None = Field(default=None, max_length=100)


class NoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    content: str = Field(min_length=1, max_length=500)


def device_router(store: PostgresDeviceStore, identify: Callable,
                  development: PostgresDevelopmentStore | None = None) -> APIRouter:
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
                 "disk_percent": row.disk_percent, "project_name": row.project_name,
                 "git_branch": row.git_branch, "git_modified_count": row.git_modified_count,
                 "last_activity_at": row.last_activity_at.isoformat()
                 if row.last_activity_at else None,
                 "vscode_running": row.vscode_running, "codex_active": row.codex_active,
                 "vscode_workspace": row.vscode_workspace}
                for row in store.list_for_user(actor.user_id)]

    @router.delete("/devices/{device_id}", status_code=204)
    def delete(device_id: UUID, actor: Actor = Depends(identify)) -> None:
        if not store.delete(actor.user_id, device_id):
            raise HTTPException(status_code=404, detail="Device not found")

    if development is not None:
        def device_auth(device_id: UUID, authorization: str | None) -> None:
            if not authorization or not authorization.startswith("Bearer ") or store.authenticate(
                device_id, authorization[7:]
            ) is None:
                raise HTTPException(status_code=401, detail="Invalid device authentication")

        @router.post("/devices/{device_id}/screenshots", status_code=201)
        def screenshot_request(device_id: UUID, actor: Actor = Depends(identify)) -> dict:
            try:
                request_id = development.request_screenshot(actor.user_id, device_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"request_id": str(request_id), "status": "PENDING"}

        @router.get("/screenshots")
        def screenshots(actor: Actor = Depends(identify)) -> list[dict]:
            return development.list_screenshots(actor.user_id)

        @router.get("/screenshots/{request_id}/image")
        def screenshot_image(request_id: UUID, actor: Actor = Depends(identify)) -> Response:
            try:
                image = development.image(actor.user_id, request_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Screenshot not found") from exc
            return Response(image, media_type="image/png", headers={"Cache-Control": "no-store"})

        @router.post("/devices/{device_id}/screenshots/{request_id}/upload")
        async def screenshot_upload(device_id: UUID, request_id: UUID, request: Request,
                                    authorization: str | None = Header(default=None)) -> dict:
            device_auth(device_id, authorization)
            content = bytearray()
            async for chunk in request.stream():
                content.extend(chunk)
                if len(content) > MAX_SCREENSHOT_BYTES:
                    raise HTTPException(status_code=413, detail="Screenshot exceeds 5 MiB")
            try:
                development.upload_screenshot(device_id, request_id, bytes(content))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Screenshot request not found") from exc
            return {"status": "COMPLETED"}

        @router.post("/devices/{device_id}/screenshots/{request_id}/fail")
        def screenshot_failed(device_id: UUID, request_id: UUID,
                              authorization: str | None = Header(default=None)) -> dict:
            device_auth(device_id, authorization)
            try:
                development.fail_screenshot(device_id, request_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Screenshot request not found") from exc
            return {"status": "FAILED"}

        @router.post("/devices/{device_id}/notes", status_code=201)
        def create_note(device_id: UUID, request: NoteRequest,
                        actor: Actor = Depends(identify)) -> dict:
            try:
                note_id = development.create_note(actor.user_id, device_id, request.content)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc
            return {"note_id": str(note_id), "status": "PENDING"}

        @router.get("/devices/{device_id}/notes")
        def notes(device_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
            try:
                return development.list_notes(actor.user_id, device_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc

        @router.post("/devices/{device_id}/notes/{note_id}/delivered")
        def note_delivered(device_id: UUID, note_id: UUID,
                           authorization: str | None = Header(default=None)) -> dict:
            device_auth(device_id, authorization)
            try:
                development.deliver_note(device_id, note_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Note not found") from exc
            return {"status": "DELIVERED"}

        @router.get("/devices/{device_id}/events")
        def events(device_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
            try:
                return development.list_events(actor.user_id, device_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc

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
                if development is not None:
                    store.update_development(device_id, session_id, heartbeat.model_dump())
                await websocket.send_json({"type": "heartbeat_ack"})
                if development is not None:
                    for action in development.pending_actions(device_id):
                        await websocket.send_json(action)
        except WebSocketDisconnect:
            pass
        finally:
            store.disconnect(device_id, session_id)

    return router
