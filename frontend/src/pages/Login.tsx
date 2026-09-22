import { FormEvent, useState } from "react";
import { ArrowRight, KeyRound, Sun } from "lucide-react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../stores/auth";

export function Login() {
  const { user, login } = useAuth();
  const [username, setUsername] = useState("");
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to="/chat" replace />;
  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await login(username, token);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "登录失败");
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="login-page">
      <div className="login-accent" aria-hidden="true" />
      <div className="login-panel">
        <div className="login-brand">
          <span className="brand-icon">
            <Sun size={25} />
          </span>
          SkySun
        </div>
        <div className="login-copy">
          <span className="eyebrow">WELCOME BACK</span>
          <h1>
            你的智能工作台，
            <br />
            <em>从这里开始。</em>
          </h1>
          <p>连接你的知识、记忆与任务，让每一次对话都更有上下文。</p>
        </div>
        <form onSubmit={submit} className="login-form">
          <label htmlFor="username">用户名</label>
          <input
            id="username"
            autoComplete="username"
            required
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="请输入用户名"
          />
          <label htmlFor="token">访问令牌</label>
          <div className="input-icon">
            <KeyRound size={18} />
            <input
              id="token"
              type="password"
              autoComplete="current-password"
              required
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="请输入访问令牌"
            />
          </div>
          {error && (
            <div className="error-notice" role="alert">
              {error}
            </div>
          )}
          <button className="primary-button" disabled={busy}>
            {busy ? "正在验证…" : "进入工作台"}
            <ArrowRight size={18} />
          </button>
        </form>
        <p className="login-help">访问令牌由 SkySun 管理员创建并提供。</p>
      </div>
      <div className="login-visual">
        <div className="visual-orbit orbit-one" />
        <div className="visual-orbit orbit-two" />
        <div className="visual-sun">
          <Sun size={72} strokeWidth={1} />
        </div>
        <div className="visual-caption">知识有迹可循 · 记忆持续生长</div>
      </div>
    </div>
  );
}
