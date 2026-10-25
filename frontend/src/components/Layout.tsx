// 本文件负责前端数据与组件中的 Layout 相关逻辑。
import {
  Activity,
  BookOpenText,
  Brain,
  LogOut,
  MessageCircle,
  Monitor,
  Camera,
  Shield,
  Workflow,
} from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../stores/auth";
import { BrandLogo } from "./BrandLogo";

const navigation = [
  { to: "/chat", label: "对话", icon: MessageCircle },
  { to: "/tasks", label: "任务", icon: Workflow },
  { to: "/memory", label: "记忆", icon: Brain },
  { to: "/knowledge", label: "知识", icon: BookOpenText },
  { to: "/devices", label: "设备", icon: Monitor },
  { to: "/screenshots", label: "截图", icon: Camera },
];

// 调用位置：Protected 在验证登录后调用。
export function Layout() {
  /** 绘制工作台导航、用户信息和子页面出口。 */
  const { user, logout } = useAuth();
  const items =
    user?.role === "ADMIN"
      ? [...navigation, { to: "/admin", label: "管理", icon: Activity }]
      : navigation;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink to="/chat" className="brand">
          <BrandLogo size={110} />
        </NavLink>
        <div className="nav-label">工作空间</div>
        <nav className="nav-list" aria-label="主导航">
          {items.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `nav-item ${isActive ? "active" : ""}`
              }
            >
              <Icon size={19} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="identity-card">
            <span className="avatar">{user?.username[0].toUpperCase()}</span>
            <span className="identity-name">
              {user?.username}
              <small>
                {user?.role === "ADMIN" ? (
                  <>
                    <Shield size={12} /> 管理员
                  </>
                ) : (
                  "个人空间"
                )}
              </small>
            </span>
          </div>
          <button className="logout" onClick={logout} title="退出登录">
            <LogOut size={18} />
          </button>
        </div>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
      <nav className="mobile-nav" aria-label="移动导航">
        <NavLink to="/chat" className="mobile-brand" aria-label="SkySun 首页">
          <BrandLogo size={43} />
        </NavLink>
        {items.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `mobile-nav-item ${isActive ? "active" : ""}`
            }
          >
            <Icon size={20} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
