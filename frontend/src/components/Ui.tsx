// 本文件负责前端数据与组件中的 Ui 相关逻辑。
import type { ReactNode } from "react";

// 调用位置：各业务页面的页头区域调用。
export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  /** 统一显示页面标题、描述与操作入口。 */
  return (
    <header className="page-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </header>
  );
}
// 调用位置：任务、记忆、知识等页面无数据时调用。
export function EmptyState({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children?: ReactNode;
}) {
  /** 展示空状态说明和引导内容。 */
  return (
    <div className="empty-state">
      <div className="empty-symbol">✦</div>
      <h2>{title}</h2>
      <p>{description}</p>
      {children}
    </div>
  );
}
// 调用位置：业务页面等待接口响应时调用。
export function Loading() {
  /** 显示加载中的提示。 */
  return (
    <div className="loading" role="status">
      正在加载…
    </div>
  );
}
// 调用位置：业务页面请求失败时调用。
export function ErrorNotice({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  /** 展示错误及可选的重试操作。 */
  return (
    <div className="error-notice" role="alert">
      <span>{message}</span>
      {retry && <button onClick={retry}>重试</button>}
    </div>
  );
}
// 调用位置：任务、记忆、知识等页面展示时间时调用。
export function formatDate(value: string) {
  /** 按中文格式显示日期和时间。 */
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}
// 调用位置：任务、管理等页面展示状态时调用。
export function StatusBadge({ status }: { status: string }) {
  /** 为状态值选择对应的视觉样式。 */
  return (
    <span className={`status status-${status.toLowerCase()}`}>{status}</span>
  );
}
