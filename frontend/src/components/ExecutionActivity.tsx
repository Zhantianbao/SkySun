// 本文件负责前端数据与组件中的 ExecutionActivity 相关逻辑。
import { Activity, CheckCircle2, CircleDashed, XCircle } from "lucide-react";
import type { TaskActivity } from "../types";

const labels: Record<string, string> = {
  request_received: "已接收请求",
  planning: "正在规划任务",
  workflow_execution: "执行多步骤任务",
  workflow_step: "执行任务步骤",
  tool_call: "调用工具",
  memory_retrieval: "检索个人记忆",
  knowledge_retrieval: "检索知识文档",
  verification: "核查执行结果",
  response_evaluation: "评估回答",
  final_response: "生成最终回答",
  direct_execution: "处理请求",
};

// 调用位置：Chat 页面展示执行过程时调用。
export function ExecutionActivity({
  activity,
  busy,
}: {
  activity: TaskActivity[];
  busy: boolean;
}) {
  /** 呈现任务步骤、事件和执行状态。 */
  const visible = activity.filter((item) => item.stage in labels).slice(0, 6);
  if (!busy && visible.length === 0) return null;
  const current = visible[0];
  return (
    <div className="execution-activity" aria-live="polite">
      <div className="execution-heading">
        <Activity size={15} />
        {busy ? (current ? labels[current.stage] : "SkySun 正在处理请求") : "执行过程"}
      </div>
      {visible.length > 0 && (
        <div className="execution-steps">
          {visible.map((item, index) => (
            <span className="execution-step" key={`${item.stage}-${index}`}>
              {item.status === "RUNNING" ? (
                <CircleDashed size={13} />
              ) : item.status === "FAILED" ? (
                <XCircle size={13} />
              ) : (
                <CheckCircle2 size={13} />
              )}
              {labels[item.stage]}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
