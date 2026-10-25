// 本文件负责页面路由与交互中的 Tasks 相关逻辑。
import { Link } from "react-router-dom";
import {
  ArrowUpRight,
  CheckCircle2,
  CircleDashed,
  Clock3,
  Workflow,
} from "lucide-react";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  StatusBadge,
  formatDate,
} from "../components/Ui";
import { useResource } from "../hooks/useResource";
import type { Task } from "../types";

// 调用位置：App 中 /tasks 路由调用。
export function Tasks() {
  /** 加载并展示用户任务与步骤状态。 */
  const { data, loading, error, refresh } = useResource<Task[]>("/tasks");
  return (
    <div className="page">
      <PageHeader
        eyebrow="TASK ACTIVITY"
        title="任务进度"
        description="查看 Agent 任务的状态、执行步骤与最近活动。"
        action={
          <button className="ghost-button" onClick={() => void refresh()}>
            刷新列表
          </button>
        }
      />
      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorNotice message={error} retry={() => void refresh()} />
      ) : !data?.length ? (
        <EmptyState
          title="还没有任务"
          description="与 SkySun 对话后，执行记录会出现在这里。"
        >
          <Link className="text-link" to="/chat">
            开始对话 <ArrowUpRight size={15} />
          </Link>
        </EmptyState>
      ) : (
        <div className="card-list">
          {data.map((task) => (
            <article className="task-card" key={task.task_id}>
              <div className="task-card-top">
                <span className="task-symbol">
                  <Workflow size={19} />
                </span>
                <StatusBadge status={task.status} />
              </div>
              <h2>{task.goal}</h2>
              <div className="task-meta">
                <span>
                  <Clock3 size={14} /> {formatDate(task.created_at)}
                </span>
                <span>任务 {task.task_id.slice(0, 8)}</span>
              </div>
              {task.steps.length > 0 && (
                <div className="step-list">
                  {task.steps.map((step) => (
                    <div className="step" key={step.step_id}>
                      {step.state === "SUCCESS" ? (
                        <CheckCircle2 size={16} />
                      ) : (
                        <CircleDashed size={16} />
                      )}
                      <span>{step.step_id}</span>
                      <small>{step.state}</small>
                    </div>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
