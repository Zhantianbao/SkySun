import { useEffect } from "react";
import { Monitor, RefreshCw } from "lucide-react";
import { EmptyState, ErrorNotice, Loading, PageHeader, formatDate } from "../components/Ui";
import { useResource } from "../hooks/useResource";
import type { Device } from "../types";

function metric(value: number | null) {
  return value === null ? "—" : `${Math.round(value)}%`;
}

export function Devices() {
  const { data, loading, error, refresh } = useResource<Device[]>("/devices");
  useEffect(() => {
    const timer = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  return (
    <div className="page">
      <PageHeader eyebrow="DESKTOP COMPANION" title="我的设备"
        description="查看已配对电脑的连接与资源状态。"
        action={<button className="ghost-button" onClick={() => void refresh()}>
          <RefreshCw size={16} /> 刷新
        </button>} />
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
            <p className="device-updated">最近心跳：{device.last_heartbeat_at ? formatDate(device.last_heartbeat_at) : "尚未连接"}</p>
          </article>
        ))}</div>}
    </div>
  );
}
