"""Authenticated device HTTP routes and the companion heartbeat channel."""
# 本文件负责设备通信中的 device、api 相关流程。

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from skysun.device.development import MAX_SCREENSHOT_BYTES, PostgresDevelopmentStore
from skysun.device.devices import PostgresDeviceStore
from skysun.user.user_identity import Actor


# 以下集中定义 DeviceRegistration：数据结构或组件相关的数据和操作。
class DeviceRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    hostname: str = Field(min_length=1, max_length=255)
    operating_system: str = Field(min_length=1, max_length=100)


# 以下集中定义 Heartbeat：数据结构或组件相关的数据和操作。
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


# 以下集中定义 NoteRequest：请求模型相关的数据和操作。
class NoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    content: str = Field(min_length=1, max_length=500)


# 调用位置：src/skysun/core/api.py 的 build_api 调用。
def device_router(store: PostgresDeviceStore, identify: Callable,
                  development: PostgresDevelopmentStore | None = None) -> APIRouter:
    """组建设备相关的 HTTP 和 WebSocket 路由。"""
    router = APIRouter()

    # 以下普通 HTTP 路由处理设备登记、查询和删除，并按登录用户限定设备范围。
    # 调用位置：FastAPI 的 POST /devices 请求触发。
    @router.post("/devices", status_code=201)
    def register(request: DeviceRegistration, actor: Actor = Depends(identify)) -> dict:
        """登记设备并生成设备凭据。"""
        device_id, token = store.register(actor.user_id, request.name,
                                          request.hostname, request.operating_system)
        return {"device_id": str(device_id), "device_token": token}

    # 调用位置：FastAPI 的 GET /devices 请求触发。
    @router.get("/devices")
    def list_devices(actor: Actor = Depends(identify)) -> list[dict]:
        """返回当前用户的设备列表。"""
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

    # 调用位置：FastAPI 的 DELETE /devices/{device_id} 请求触发。
    @router.delete("/devices/{device_id}", status_code=204)
    def delete(device_id: UUID, actor: Actor = Depends(identify)) -> None:
        """删除指定记录并检查所有权。"""
        if not store.delete(actor.user_id, device_id):
            raise HTTPException(status_code=404, detail="Device not found")

    if development is not None:
        # 调用位置：本路由组的截图上传、失败回报和便笺确认接口调用。
        def device_auth(device_id: UUID, authorization: str | None) -> None:
            """核验设备令牌并返回设备身份。"""
            if not authorization or not authorization.startswith("Bearer ") or store.authenticate(
                device_id, authorization[7:]
            ) is None:
                raise HTTPException(status_code=401, detail="Invalid device authentication")

        # 调用位置：后端框架收到 POST /devices/{device_id}/screenshots 请求时调用。
        @router.post("/devices/{device_id}/screenshots", status_code=201)
        def screenshot_request(device_id: UUID, actor: Actor = Depends(identify)) -> dict:
            """创建发送给设备的截图请求。"""
            try:
                request_id = development.request_screenshot(actor.user_id, device_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"request_id": str(request_id), "status": "PENDING"}

        # 调用位置：后端框架收到 GET /screenshots 请求时调用。
        @router.get("/screenshots")
        def screenshots(actor: Actor = Depends(identify)) -> list[dict]:
            """列出当前用户的截图记录。"""
            return development.list_screenshots(actor.user_id)

        # 调用位置：后端框架收到 GET /screenshots/{request_id}/image 请求时调用。
        @router.get("/screenshots/{request_id}/image")
        def screenshot_image(request_id: UUID, actor: Actor = Depends(identify)) -> Response:
            """读取已上传的截图图片。"""
            try:
                image = development.image(actor.user_id, request_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Screenshot not found") from exc
            return Response(image, media_type="image/png", headers={"Cache-Control": "no-store"})

        # 调用位置：后端框架收到 POST /devices/{device_id}/screenshots/{request_id}/upload 请求时调用。
        @router.post("/devices/{device_id}/screenshots/{request_id}/upload")
        async def screenshot_upload(device_id: UUID, request_id: UUID, request: Request,
                                    authorization: str | None = Header(default=None)) -> dict:
            """接收设备上传的截图。"""
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

        # 调用位置：后端框架收到 POST /devices/{device_id}/screenshots/{request_id}/fail 请求时调用。
        @router.post("/devices/{device_id}/screenshots/{request_id}/fail")
        def screenshot_failed(device_id: UUID, request_id: UUID,
                              authorization: str | None = Header(default=None)) -> dict:
            """记录设备截图失败的结果。"""
            device_auth(device_id, authorization)
            try:
                development.fail_screenshot(device_id, request_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Screenshot request not found") from exc
            return {"status": "FAILED"}

        # 调用位置：后端框架收到 POST /devices/{device_id}/notes 请求时调用。
        @router.post("/devices/{device_id}/notes", status_code=201)
        def create_note(device_id: UUID, request: NoteRequest,
                        actor: Actor = Depends(identify)) -> dict:
            """创建发往设备的便笺。"""
            try:
                note_id = development.create_note(actor.user_id, device_id, request.content)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc
            return {"note_id": str(note_id), "status": "PENDING"}

        # 调用位置：后端框架收到 GET /devices/{device_id}/notes 请求时调用。
        @router.get("/devices/{device_id}/notes")
        def notes(device_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
            """列出当前用户的设备便笺。"""
            try:
                return development.list_notes(actor.user_id, device_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc

        # 调用位置：后端框架收到 POST /devices/{device_id}/notes/{note_id}/delivered 请求时调用。
        @router.post("/devices/{device_id}/notes/{note_id}/delivered")
        def note_delivered(device_id: UUID, note_id: UUID,
                           authorization: str | None = Header(default=None)) -> dict:
            """确认设备便笺已送达。"""
            device_auth(device_id, authorization)
            try:
                development.deliver_note(device_id, note_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Note not found") from exc
            return {"status": "DELIVERED"}

        # 调用位置：后端框架收到 GET /devices/{device_id}/events 请求时调用。
        @router.get("/devices/{device_id}/events")
        def events(device_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
            """列出任务或设备的事件记录。"""
            try:
                return development.list_events(actor.user_id, device_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Device not found") from exc

    # WebSocket 通道在设备鉴权后接收心跳，并下发待处理的截图和便笺动作。
    # 调用位置：FastAPI 的 WEBSOCKET /api/ws/devices/{device_id} 请求触发。
    @router.websocket("/api/ws/devices/{device_id}")
    async def channel(websocket: WebSocket, device_id: UUID) -> None:
        """建立设备 WebSocket 通道并处理消息。"""
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
