// 本文件负责页面路由与交互中的 Login 相关逻辑。
import { FormEvent, useState } from "react";
import { ArrowRight, KeyRound } from "lucide-react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../stores/auth";
import { BrandLogo } from "../components/BrandLogo";

// 调用位置：App 中 /login 路由调用。
export function Login() {
  /** 显示姓名和密码输入表单及登录反馈。 */
  const { user, login } = useAuth();
  const [username, setUsername] = useState("");
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  if (user) return <Navigate to="/chat" replace />;
  // 调用位置：Login 表单提交时调用。
  async function submit(event: FormEvent) {
    /** 读取姓名和密码并完成登录与页面跳转。 */
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
      <div className="login-panel">
        <div className="login-brand">
          <BrandLogo size={110} />
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
          <label htmlFor="username">姓名</label>
          <input
            id="username"
            autoComplete="username"
            required
            pattern="[\u4e00-\u9fff]{2,4}"
            title="姓名只能包含中文，长度为 2~4 个汉字"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="请输入中文姓名"
          />
          <label htmlFor="token">密码</label>
          <div className="input-icon">
            <KeyRound size={18} />
            <input
              id="token"
              type="password"
              autoComplete="current-password"
              required
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="请输入密码"
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
        <p className="login-help">请使用中文姓名和管理员提供的密码登录。</p>
      </div>
      <div className="login-visual">
        <div className="visual-orbit orbit-one" />
        <div className="visual-orbit orbit-two" />
        <div className="visual-sun">
          <BrandLogo size={220} />
        </div>
        <div className="visual-caption">知识有迹可循 · 记忆持续生长</div>
      </div>
    </div>
  );
}
