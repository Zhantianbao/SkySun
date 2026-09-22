export type Role = "USER" | "ADMIN";
export type User = { user_id: string; username: string; role: Role };
export type ChatResponse = {
  conversation_id: string;
  reply: string;
  task_id: string | null;
  task_status: string | null;
};
export type Message = { role: "user" | "assistant"; content: string };
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
