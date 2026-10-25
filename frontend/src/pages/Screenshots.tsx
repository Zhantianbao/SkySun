// 本文件负责页面路由与交互中的 Screenshots 相关逻辑。
import { useEffect, useState } from "react";
import { Camera, RefreshCw } from "lucide-react";
import { apiBlob } from "../api/client";
import { EmptyState, ErrorNotice, Loading, PageHeader, formatDate } from "../components/Ui";
import { useResource } from "../hooks/useResource";
import { useAuth } from "../stores/auth";
import type { Screenshot } from "../types";

// 调用位置：App 中 /screenshots 路由调用。
export function Screenshots() {
  /** 加载并展示设备截图记录。 */
  const { token } = useAuth();
  const { data, loading, error, refresh } = useResource<Screenshot[]>("/screenshots");
  const [imageUrl, setImageUrl] = useState("");
  const [imageError, setImageError] = useState("");
  // 定期刷新截图请求状态，页面离开时取消定时器。
  useEffect(() => {
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  // 图片链接由浏览器临时创建，切换图片或离开页面时释放。
  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl); }, [imageUrl]);

  // 调用位置：Screenshots 的查看按钮调用。
  async function openImage(requestId: string) {
    /** 下载并打开所选截图。 */
    if (!token) return;
    setImageError("");
    try {
      const image = await apiBlob(`/screenshots/${requestId}/image`, token);
      setImageUrl(URL.createObjectURL(image));
    } catch (cause) {
      setImageError(cause instanceof Error ? cause.message : "无法读取截图");
    }
  }

  return <div className="page">
    <PageHeader eyebrow="ON-DEMAND SCREENSHOT" title="电脑截图"
      description="只有你主动请求时，桌面端才会截屏；图像在服务器保留 24 小时。"
      action={<button className="ghost-button" onClick={() => void refresh()}><RefreshCw size={16} /> 刷新</button>} />
    {imageError && <ErrorNotice message={imageError} />}
    {imageUrl && <div className="screenshot-preview"><img src={imageUrl} alt="请求的电脑截图" /></div>}
    {loading && !data ? <Loading /> : error && !data ? <ErrorNotice message={error} retry={() => void refresh()} />
      : !data?.length ? <EmptyState title="还没有截图" description="在设备页主动请求一次截图后，记录会显示在这里。" />
      : <div className="card-list">{data.map((shot) => <article className="screenshot-card" key={shot.request_id}>
          <Camera size={19} /><div><strong>{shot.status === "COMPLETED" ? "截图已完成" : shot.status === "PENDING" ? "等待电脑响应" : shot.status === "FAILED" ? "截图失败" : "请求已过期"}</strong>
          <small>{formatDate(shot.requested_at)} · 设备 {shot.device_id.slice(0, 8)}</small></div>
          {shot.status === "COMPLETED" && <button className="ghost-button" onClick={() => void openImage(shot.request_id)}>查看</button>}
        </article>)}</div>}
  </div>;
}
