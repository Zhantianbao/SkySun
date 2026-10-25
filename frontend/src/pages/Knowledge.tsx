// 本文件负责页面路由与交互中的 Knowledge 相关逻辑。
import { ChangeEvent, FormEvent, useState } from "react";
import {
  BookOpenText,
  CalendarDays,
  FileText,
  Globe2,
  LockKeyhole,
  UploadCloud,
} from "lucide-react";
import { api } from "../api/client";
import {
  EmptyState,
  ErrorNotice,
  Loading,
  PageHeader,
  formatDate,
} from "../components/Ui";
import { useResource } from "../hooks/useResource";
import { useAuth } from "../stores/auth";
import type { Document, UploadResult } from "../types";

// 调用位置：App 中 /knowledge 路由调用。
export function Knowledge() {
  /** 展示知识文档并提供上传入口。 */
  const { token } = useAuth();
  const { data, loading, error, refresh } = useResource<Document[]>(
    "/knowledge/documents",
  );
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploaded, setUploaded] = useState<UploadResult | null>(null);

  // 调用位置：Knowledge 文件输入框变化时调用。
  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    /** 记录用户选中的待上传文件。 */
    setFile(event.target.files?.[0] || null);
    setUploadError("");
    setUploaded(null);
  }

  // 调用位置：Knowledge 上传表单提交时调用。
  async function upload(event: FormEvent) {
    /** 将文件提交给知识入库接口并刷新列表。 */
    event.preventDefault();
    if (!file || !token || uploading) return;
    if (file.size > 2 * 1024 * 1024) {
      setUploadError("文件不能超过 2 MiB");
      return;
    }
    setUploading(true);
    setUploadError("");
    setUploaded(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const result = await api<UploadResult>("/knowledge/documents", token, {
        method: "POST",
        body,
      });
      setUploaded(result);
      setFile(null);
      await refresh();
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "上传失败");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="KNOWLEDGE BASE"
        title="知识文档"
        description="上传自己的文档，SkySun 会为你建立可检索的知识来源。"
      />
      <form className="upload-card" onSubmit={(event) => void upload(event)}>
        <div className="upload-icon">
          <UploadCloud size={22} />
        </div>
        <div className="upload-copy">
          <strong>添加个人知识</strong>
          <span>支持 UTF-8 的 .md、.markdown、.txt 文档，最大 2 MiB。仅你可检索。</span>
          <input
            aria-label="选择知识文档"
            type="file"
            accept=".md,.markdown,.txt,text/plain,text/markdown"
            onChange={selectFile}
            disabled={uploading}
          />
          {file && <small>已选择：{file.name}</small>}
        </div>
        <button className="primary-button upload-button" disabled={!file || uploading}>
          {uploading ? "正在上传并建立索引…" : "上传文档"}
        </button>
      </form>
      {uploadError && <ErrorNotice message={uploadError} />}
      {uploaded && (
        <div className="upload-success" role="status">
          上传完成，已建立 {uploaded.chunk_count} 个知识片段。
        </div>
      )}
      <div className="knowledge-list-heading">可用文档</div>
      {loading ? (
        <Loading />
      ) : error ? (
        <ErrorNotice message={error} retry={() => void refresh()} />
      ) : !data?.length ? (
        <EmptyState
          title="还没有知识文档"
          description="上传文档后，SkySun 可以在对话中检索并引用它。"
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
