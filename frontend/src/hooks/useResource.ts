// 本文件负责前端数据与组件中的 useResource 相关逻辑。
import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import { useAuth } from "../stores/auth";

// 调用位置：Tasks、MemoryPage、Knowledge、Admin 等页面调用。
export function useResource<T>(path: string) {
  /** 封装带身份验证的数据加载、错误状态和刷新动作。 */
  const { token } = useAuth();
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  // 资源路径或令牌变化时重新请求，并统一暴露加载和错误状态。
  const refresh = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    setError("");
    try {
      setData(await api<T>(path, token));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法读取数据");
    } finally {
      setLoading(false);
    }
  }, [path, token]);
  // 首次使用钩子和刷新函数变化时自动读取资源。
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { data, loading, error, refresh };
}
