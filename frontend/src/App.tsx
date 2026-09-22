import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Login } from "./pages/Login";
import { Chat } from "./pages/Chat";
import { Tasks } from "./pages/Tasks";
import { MemoryPage } from "./pages/Memory";
import { Knowledge } from "./pages/Knowledge";
import { Admin } from "./pages/Admin";
import { AuthProvider, useAuth } from "./stores/auth";

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
      </BrowserRouter>
    </AuthProvider>
  );
}
