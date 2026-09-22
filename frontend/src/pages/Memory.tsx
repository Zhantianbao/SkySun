import { Brain, CalendarDays, Tag } from "lucide-react";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  formatDate,
} from "../components/Ui";
import { useResource } from "../hooks/useResource";
import type { Memory } from "../types";

export function MemoryPage() {
  const { data, loading, error, refresh } = useResource<Memory[]>("/memory");
  return (
    <div className="page">
      <PageHeader
        eyebrow="PERSONAL MEMORY"
        title="我的记忆"
        description="SkySun 在对话中确认并保留的个人偏好与长期信息。"
      />
      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorNotice message={error} retry={() => void refresh()} />
      ) : !data?.length ? (
        <EmptyState
          title="记忆还很空"
          description="在对话中分享稳定的偏好或信息，SkySun 会筛选值得保留的内容。"
        />
      ) : (
        <div className="memory-grid">
          {data.map((item) => (
            <article className="memory-card" key={item.id}>
              <div className="memory-card-top">
                <span className="memory-icon">
                  <Brain size={19} />
                </span>
                <span className="memory-category">
                  <Tag size={13} />
                  {item.category || "未分类"}
                </span>
              </div>
              <h2>{item.key.replaceAll("_", " ")}</h2>
              <p>{item.content}</p>
              <div className="card-date">
                <CalendarDays size={14} />
                {formatDate(item.updated_at)}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
