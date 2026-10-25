// 本文件负责前端数据与组件中的 auth 相关逻辑。
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "../api/client";
import type { User } from "../types";

type AuthState = {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (username: string, token: string) => Promise<void>;
  logout: () => void;
};
const AuthContext = createContext<AuthState | null>(null);
const tokenKey = "skysun_token";
const userKey = "skysun_user";

// 调用位置：App 组件调用。
export function AuthProvider({ children }: { children: ReactNode }) {
  /** 维护登录令牌和用户信息，并向页面提供身份上下文。 */
  const [token, setToken] = useState(() => sessionStorage.getItem(tokenKey));
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(Boolean(token));

  // 调用位置：Layout 的退出按钮和未授权事件调用。
  function logout() {
    /** 清除浏览器会话中的身份信息。 */
    sessionStorage.removeItem(tokenKey);
    sessionStorage.removeItem(userKey);
    setToken(null);
    setUser(null);
    setLoading(false);
  }

  // 接收 API 客户端发出的 401 事件，统一清理失效的浏览器身份。
  useEffect(() => {
    const unauthorized = () => logout();
    window.addEventListener("skysun:unauthorized", unauthorized);
    return () =>
      window.removeEventListener("skysun:unauthorized", unauthorized);
  }, []);

  // 页面刷新后以服务端 /me 结果重新确认缓存的令牌与用户。
  useEffect(() => {
    if (!token) return;
    api<User>("/me", token)
      .then((identity) => {
        setUser(identity);
        sessionStorage.setItem(userKey, JSON.stringify(identity));
      })
      .catch(() => logout())
      .finally(() => setLoading(false));
  }, []);

  // 调用位置：Login.submit 提交登录表单时调用。
  async function login(username: string, candidate: string) {
    /** 调用登录接口、核对用户姓名并保存会话身份。 */
    const identity = await api<User>("/login", "", {
      method: "POST",
      body: JSON.stringify({ username: username.trim(), token: candidate.trim() }),
    });
    if (identity.username.toLowerCase() !== username.trim().toLowerCase()) {
      throw new Error("姓名或密码错误");
    }
    sessionStorage.setItem(tokenKey, candidate.trim());
    sessionStorage.setItem(userKey, JSON.stringify(identity));
    setToken(candidate.trim());
    setUser(identity);
  }

  const value = useMemo(
    () => ({ user, token, loading, login, logout }),
    [user, token, loading],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// 调用位置：页面、布局和 API 数据钩子调用。
export function useAuth() {
  /** 读取当前身份上下文供页面使用。 */
  const context = useContext(AuthContext);
  if (!context) throw new Error("AuthProvider is required");
  return context;
}
