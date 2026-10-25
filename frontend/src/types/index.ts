// 此处集中定义前端与后端接口共享的用户、会话、任务、知识和设备数据形状。
export type Role = "USER" | "ADMIN";
export type User = { user_id: string; username: string; role: Role };
export type ChatResponse = {
  conversation_id: string;
  reply: string;
  task_id: string | null;
  task_status: string | null;
};
export type Message = { role: "user" | "assistant"; content: string };
export type ConversationSummary = {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};
export type TaskActivity = {
  stage: string;
  status: string;
  started_at: string;
  duration_ms: number | null;
};
export type UploadResult = {
  document_id: string;
  source_path: string;
  chunk_count: number;
  status: string;
};
export type TaskStep = { step_id: string; state: string };
export type Task = {
  task_id: string;
  conversation_id: string | null;
  goal: string;
  status: string;
  created_at: string;
  updated_at: string;
  steps: TaskStep[];
};
export type Memory = {
  id: string;
  key: string;
  content: string;
  category: string | null;
  source_conversation_id: string | null;
  created_at: string;
  updated_at: string;
};
export type Document = {
  id: string;
  source_name: string;
  source_path: string;
  scope: "global" | "personal";
  updated_at: string;
};
export type Trace = {
  trace_id: string;
  task_id: string;
  user_id: string;
  stage: string;
  status: string;
  duration_ms: number | null;
  started_at: string;
};
export type Evaluation = {
  task_id: string;
  user_id: string;
  score: number;
  status: string;
  reasons: string[];
  created_at: string | null;
};
export type Metrics = Record<string, unknown>;
export type Device = {
  device_id: string;
  name: string;
  hostname: string;
  operating_system: string;
  online: boolean;
  last_heartbeat_at: string | null;
  cpu_percent: number | null;
  memory_percent: number | null;
  disk_percent: number | null;
  project_name: string | null;
  git_branch: string | null;
  git_modified_count: number | null;
  last_activity_at: string | null;
  vscode_running: boolean | null;
  codex_active: boolean | null;
  vscode_workspace: string | null;
};
export type Screenshot = {
  request_id: string;
  device_id: string;
  status: "PENDING" | "COMPLETED" | "FAILED" | "EXPIRED";
  requested_at: string;
  uploaded_at: string | null;
};
