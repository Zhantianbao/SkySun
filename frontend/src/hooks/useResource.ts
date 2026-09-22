import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import { useAuth } from "../stores/auth";

export function useResource<T>(path: string) {
  const { token } = useAuth();
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
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
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { data, loading, error, refresh };
}
