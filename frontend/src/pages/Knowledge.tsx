import {
  BookOpenText,
  CalendarDays,
  FileText,
  Globe2,
  LockKeyhole,
} from "lucide-react";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  formatDate,
} from "../components/Ui";
import { useResource } from "../hooks/useResource";
import type { Document } from "../types";

export function Knowledge() {
  const { data, loading, error, refresh } = useResource<Document[]>(
    "/knowledge/documents",
  );
  return (
    <div className="page">
      <PageHeader
        eyebrow="KNOWLEDGE BASE"
        title="知识文档"
        description="Agent 可检索的文档来源。个人文档只对你可见。"
      />
      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorNotice message={error} retry={() => void refresh()} />
      ) : !data?.length ? (
        <EmptyState
          title="还没有知识文档"
          description="管理员可以使用命令行导入文档，导入后会在这里显示来源。"
        />
      ) : (
        <div className="card-list">
          {data.map((doc) => (
            <article className="document-card" key={doc.id}>
              <span className="document-icon">
                <FileText size={21} />
              </span>
              <div>
                <h2>{doc.source_name}</h2>
                <p className="document-path" title={doc.source_path}>
                  {doc.source_path}
                </p>
                <div className="document-meta">
                  <span>
                    {doc.scope === "global" ? (
                      <>
                        <Globe2 size={14} /> 全局文档
                      </>
                    ) : (
                      <>
                        <LockKeyhole size={14} /> 个人文档
                      </>
                    )}
                  </span>
                  <span>
                    <CalendarDays size={14} />
                    {formatDate(doc.updated_at)}
                  </span>
                </div>
              </div>
              <BookOpenText className="document-end" size={18} />
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
