import {
  Activity,
  CheckCircle2,
  Gauge,
  ListTree,
  ShieldCheck,
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
import type { Evaluation, Metrics, Trace } from "../types";

export function Admin() {
  const metrics = useResource<Metrics>("/admin/metrics");
  const traces = useResource<Trace[]>("/admin/traces?limit=20");
  const evaluations = useResource<Evaluation[]>("/admin/evaluations?limit=20");
  const busy = metrics.loading || traces.loading || evaluations.loading;
  const error = metrics.error || traces.error || evaluations.error;
  const entries = metrics.data
    ? Object.entries(metrics.data).filter(
        ([, value]) => typeof value === "number",
      )
    : [];
  return (
    <div className="page">
      <PageHeader
        eyebrow="ADMIN OBSERVABILITY"
        title="系统观测"
        description="查看 Agent 的运行指标、执行轨迹与回答评估。"
        action={
          <span className="admin-badge">
            <ShieldCheck size={15} /> 管理员
          </span>
        }
      />
      {busy ? (
        <Loading />
      ) : error ? (
        <ErrorNotice
          message={error}
          retry={() => {
            void metrics.refresh();
            void traces.refresh();
            void evaluations.refresh();
          }}
        />
      ) : (
        <>
          <section className="admin-section">
            <h2>
              <Gauge size={19} /> 运行指标
            </h2>
            <div className="metric-grid">
              {entries.map(([key, value]) => (
                <div className="metric-card" key={key}>
                  <span>{key.replaceAll("_", " ")}</span>
                  <strong>
                    {typeof value === "number" ? Number(value.toFixed(2)) : ""}
                  </strong>
                </div>
              ))}
            </div>
          </section>
          <section className="admin-section">
            <h2>
              <Activity size={19} /> 最近执行轨迹
            </h2>
            {!traces.data?.length ? (
              <EmptyState
                title="暂无轨迹"
                description="执行 Agent 任务后会生成轨迹。"
              />
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>阶段</th>
                      <th>状态</th>
                      <th>耗时</th>
                      <th>任务</th>
                    </tr>
                  </thead>
                  <tbody>
                    {traces.data.map((trace) => (
                      <tr key={trace.trace_id}>
                        <td>{trace.stage}</td>
                        <td>
                          <StatusBadge status={trace.status} />
                        </td>
                        <td>
                          {trace.duration_ms == null
                            ? "—"
                            : `${trace.duration_ms} ms`}
                        </td>
                        <td title={trace.task_id}>
                          {trace.task_id.slice(0, 8)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          <section className="admin-section">
            <h2>
              <ListTree size={19} /> 回答评估
            </h2>
            {!evaluations.data?.length ? (
              <EmptyState
                title="暂无评估"
                description="回答完成后会记录评估结果。"
              />
            ) : (
              <div className="evaluation-list">
                {evaluations.data.map((item) => (
                  <div className="evaluation-row" key={item.task_id}>
                    <span className="evaluation-icon">
                      <CheckCircle2 size={18} />
                    </span>
                    <div>
                      <strong>任务 {item.task_id.slice(0, 8)}</strong>
                      <small>
                        {item.reasons.join(" · ") ||
                          (item.created_at && formatDate(item.created_at))}
                      </small>
                    </div>
                    <strong>{item.score} / 100</strong>
                    <StatusBadge status={item.status} />
                  </div>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
