export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const base = import.meta.env.VITE_API_BASE_URL || "/api";

export async function api<T>(
  path: string,
  token: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      Authorization: `Bearer ${token}`,
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

export async function apiBlob(path: string, token: string): Promise<Blob> {
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
