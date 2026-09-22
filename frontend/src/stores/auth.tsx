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

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState(() => sessionStorage.getItem(tokenKey));
  const [user, setUser] = useState<User | null>(() => {
    try {
      return JSON.parse(
        sessionStorage.getItem(userKey) || "null",
      ) as User | null;
    } catch {
      return null;
    }
  });
  const [loading, setLoading] = useState(Boolean(token));

  function logout() {
    sessionStorage.removeItem(tokenKey);
    sessionStorage.removeItem(userKey);
    setToken(null);
    setUser(null);
    setLoading(false);
  }

  useEffect(() => {
    const unauthorized = () => logout();
    window.addEventListener("skysun:unauthorized", unauthorized);
    return () =>
      window.removeEventListener("skysun:unauthorized", unauthorized);
  }, []);

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

  async function login(username: string, candidate: string) {
    const identity = await api<User>("/me", candidate.trim());
    if (identity.username.toLowerCase() !== username.trim().toLowerCase()) {
      throw new Error("用户名与访问令牌不匹配");
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

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("AuthProvider is required");
  return context;
}
