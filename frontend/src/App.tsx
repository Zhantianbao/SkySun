// 本文件负责前端数据与组件中的 App 相关逻辑。
import { Suspense, lazy } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Login } from "./pages/Login";
import { AuthProvider, useAuth } from "./stores/auth";

// 各业务页按路由延迟加载，登录页和身份检查保持在主入口。
const Chat = lazy(() => import("./pages/Chat").then((module) => ({ default: module.Chat })));
const Tasks = lazy(() => import("./pages/Tasks").then((module) => ({ default: module.Tasks })));
const MemoryPage = lazy(() => import("./pages/Memory").then((module) => ({ default: module.MemoryPage })));
const Knowledge = lazy(() => import("./pages/Knowledge").then((module) => ({ default: module.Knowledge })));
const Admin = lazy(() => import("./pages/Admin").then((module) => ({ default: module.Admin })));
const Devices = lazy(() => import("./pages/Devices").then((module) => ({ default: module.Devices })));
const Screenshots = lazy(() => import("./pages/Screenshots").then((module) => ({ default: module.Screenshots })));

// 调用位置：由 App 的受保护路由调用。
function Protected({ admin = false }: { admin?: boolean }) {
  /** 检查登录状态和管理员角色，再选择页面布局。 */
  const { user, loading } = useAuth();
  if (loading) return <div className="app-loading">正在验证身份…</div>;
  if (!user) return <Navigate to="/login" replace />;
  if (admin && user.role !== "ADMIN") return <Navigate to="/chat" replace />;
  return admin ? <Admin /> : <Layout />;
}

// 调用位置：frontend/src/main.tsx 挂载应用时调用。
export function App() {
  /** 组合身份上下文、路由和各业务页面。 */
  // 先提供身份上下文，再由路由决定是否进入受保护的工作台页面。
  return (
    <AuthProvider>
      <BrowserRouter>
        <Suspense fallback={<div className="app-loading">正在加载页面…</div>}>
          <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<Protected />}>
            <Route path="/chat" element={<Chat />} />
            <Route path="/tasks" element={<Tasks />} />
            <Route path="/memory" element={<MemoryPage />} />
            <Route path="/knowledge" element={<Knowledge />} />
            <Route path="/devices" element={<Devices />} />
            <Route path="/screenshots" element={<Screenshots />} />
            <Route path="/admin" element={<Protected admin />} />
          </Route>
          <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </AuthProvider>
  );
}
