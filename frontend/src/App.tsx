import { Suspense, lazy } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Login } from "./pages/Login";
import { AuthProvider, useAuth } from "./stores/auth";

const Chat = lazy(() => import("./pages/Chat").then((module) => ({ default: module.Chat })));
const Tasks = lazy(() => import("./pages/Tasks").then((module) => ({ default: module.Tasks })));
const MemoryPage = lazy(() => import("./pages/Memory").then((module) => ({ default: module.MemoryPage })));
const Knowledge = lazy(() => import("./pages/Knowledge").then((module) => ({ default: module.Knowledge })));
const Admin = lazy(() => import("./pages/Admin").then((module) => ({ default: module.Admin })));

function Protected({ admin = false }: { admin?: boolean }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="app-loading">正在验证身份…</div>;
  if (!user) return <Navigate to="/login" replace />;
  if (admin && user.role !== "ADMIN") return <Navigate to="/chat" replace />;
  return admin ? <Admin /> : <Layout />;
}

export function App() {
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
            <Route path="/admin" element={<Protected admin />} />
          </Route>
          <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </AuthProvider>
  );
}
