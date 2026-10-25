// 本文件负责页面路由与交互中的 Devices 相关逻辑。
import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Camera, Monitor, RefreshCw, Send } from "lucide-react";
import { api } from "../api/client";
import { EmptyState, ErrorNotice, Loading, PageHeader, formatDate } from "../components/Ui";
import { useResource } from "../hooks/useResource";
import { useAuth } from "../stores/auth";
import type { Device } from "../types";

// 调用位置：Devices 页面格式化设备指标时调用。
function metric(value: number | null) {
  /** 将缺失的指标显示为占位内容。 */
  return value === null ? "—" : `${Math.round(value)}%`;
}

// 调用位置：App 中 /devices 路由调用。
export function Devices() {
  /** 展示设备状态并提供截图与便笺操作。 */
  const { token } = useAuth();
  const { data, loading, error, refresh } = useResource<Device[]>("/devices");
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState("");
  const [actionError, setActionError] = useState("");
  // 页面停留期间轮询设备在线状态，卸载时清理定时器。
  useEffect(() => {
    const timer = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  // 调用位置：Devices 的截图按钮调用。
  async function requestScreenshot(deviceId: string) {
    /** 请求指定设备采集一张截图。 */
    if (!token) return;
    setNotice(""); setActionError("");
    try {
      await api(`/devices/${deviceId}/screenshots`, token, { method: "POST" });
      setNotice("截图请求已发送。完成后可在截图页查看，有效期为 24 小时。");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "截图请求失败");
    }
  }

  // 调用位置：Devices 的便笺表单提交时调用。
  async function sendNote(event: FormEvent, deviceId: string) {
    /** 向指定设备发送便笺。 */
    event.preventDefault();
    const content = notes[deviceId]?.trim();
    if (!token || !content) return;
    setNotice(""); setActionError("");
    try {
      await api(`/devices/${deviceId}/notes`, token, {
        method: "POST", body: JSON.stringify({ content }),
      });
      setNotes((current) => ({ ...current, [deviceId]: "" }));
      setNotice("提醒已加入桌面端收件箱队列。");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "提醒发送失败");
    }
  }

  return (
    <div className="page">
      <PageHeader eyebrow="DEVELOPMENT CONTROL CENTER" title="我的设备"
        description="查看自己的电脑、项目与开发状态。"
        action={<button className="ghost-button" onClick={() => void refresh()}>
          <RefreshCw size={16} /> 刷新
        </button>} />
      {notice && <p className="device-notice" role="status">{notice}</p>}
      {actionError && <ErrorNotice message={actionError} />}
      {loading && !data ? <Loading /> : error && !data ? <ErrorNotice message={error} retry={() => void refresh()} />
        : !data?.length ? <EmptyState title="还没有设备" description="在电脑上安装并注册 SkySun Desktop Companion 后，设备会显示在这里。" />
        : <div className="device-grid">{data.map((device) => (
          <article className="device-card" key={device.device_id}>
            <div className="device-heading"><span className="device-icon"><Monitor size={22} /></span>
              <div><h2>{device.name}</h2><small>{device.hostname} · {device.operating_system}</small></div>
              <span className={`device-presence ${device.online ? "online" : ""}`}>
                {device.online ? "在线" : "离线"}</span></div>
            <div className="device-metrics">
              <div><span>CPU</span><strong>{metric(device.cpu_percent)}</strong></div>
              <div><span>内存</span><strong>{metric(device.memory_percent)}</strong></div>
              <div><span>磁盘</span><strong>{metric(device.disk_percent)}</strong></div>
            </div>
            <div className="development-status">
              <div><span>当前项目</span><strong>{device.project_name || "未选择"}</strong></div>
              <div><span>Git 分支</span><strong>{device.git_branch || "—"}</strong></div>
              <div><span>改动文件</span><strong>{device.git_modified_count ?? "—"}</strong></div>
              <div><span>VS Code</span><strong>{device.vscode_running ? "运行中" : "未检测到"}</strong></div>
              <div><span>打开的工作区</span><strong>{device.vscode_workspace || "未确认"}</strong></div>
              <div><span>Codex 任务</span><strong>{device.codex_active ? "用户已确认进行中" : "未确认"}</strong></div>
            </div>
            <p className="device-updated">最近心跳：{device.last_heartbeat_at ? formatDate(device.last_heartbeat_at) : "尚未连接"}</p>
            {device.last_activity_at && <p className="device-updated">最近项目活动：{formatDate(device.last_activity_at)}</p>}
            <div className="device-actions">
              <button className="ghost-button" disabled={!device.online}
                onClick={() => void requestScreenshot(device.device_id)}><Camera size={16} /> 请求截图</button>
              <Link className="text-link" to="/screenshots">查看截图</Link>
            </div>
            <form className="device-note-form" onSubmit={(event) => void sendNote(event, device.device_id)}>
              <input aria-label={`给 ${device.name} 的提醒`} maxLength={500}
                placeholder="给这台电脑发送提醒"
                value={notes[device.device_id] || ""}
                onChange={(event) => setNotes((current) => ({ ...current, [device.device_id]: event.target.value }))} />
              <button type="submit" className="ghost-button" disabled={!notes[device.device_id]?.trim()}><Send size={16} /> 发送</button>
            </form>
          </article>
        ))}</div>}
    </div>
  );
}
