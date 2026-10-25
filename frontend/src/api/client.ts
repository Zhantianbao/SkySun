// 本文件负责前端数据与组件中的 client 相关逻辑。
export class ApiError extends Error {
  // 调用位置：本文件的 api 与 apiBlob 在非成功响应时创建此错误。
  constructor(
    public status: number,
    message: string,
  ) {
    /** 保存 HTTP 状态码和可展示的错误信息。 */
    super(message);
  }
}

const base = import.meta.env.VITE_API_BASE_URL || "/api";

// 调用位置：frontend/src/pages、hooks 和 stores 中的请求调用。
export async function api<T>(
  path: string,
  token: string,
  options: RequestInit = {},
): Promise<T> {
  /** 统一发送 JSON 请求、附加身份令牌并处理接口错误。 */
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!response.ok) {
    let detail = `请求失败 (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* keep the status message */
    }
    if (response.status === 401)
      window.dispatchEvent(new Event("skysun:unauthorized"));
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

// 调用位置：frontend/src/pages/Screenshots.tsx 下载截图时调用。
export async function apiBlob(path: string, token: string): Promise<Blob> {
  /** 携带身份令牌下载二进制截图。 */
  const response = await fetch(`${base}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });
  if (!response.ok) {
    if (response.status === 401)
      window.dispatchEvent(new Event("skysun:unauthorized"));
    throw new ApiError(response.status, `请求失败 (${response.status})`);
  }
  return response.blob();
}
