// 本文件负责前端数据与组件中的 Message 相关逻辑。
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import { BookOpenText, Sparkles } from "lucide-react";
import type { Message as MessageType } from "../types";

// 调用位置：Message 组件及前端测试调用。
export function splitSources(content: string) {
  /** 从回答正文中分离知识来源列表。 */
  const marker = "\n\nSources:\n";
  const index = content.lastIndexOf(marker);
  if (index < 0) return { body: content, sources: [] as string[] };
  return {
    body: content.slice(0, index),
    sources: content
      .slice(index + marker.length)
      .split("\n")
      .map((value) => value.replace(/^-\s*/, "").trim())
      .filter(Boolean),
  };
}

// 调用位置：Chat 页面遍历消息列表时调用。
export function Message({ message }: { message: MessageType }) {
  /** 显示用户或助手消息及可引用来源。 */
  const assistant = message.role === "assistant";
  const { body, sources } = assistant
    ? splitSources(message.content)
    : { body: message.content, sources: [] };
  return (
    <article
      className={`message ${assistant ? "assistant-message" : "user-message"}`}
    >
      <div className="message-icon">
        {assistant ? <Sparkles size={17} /> : "你"}
      </div>
      <div className="message-content">
        <div className="message-author">{assistant ? "SkySun" : "你"}</div>
        <div className="markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>{body}</ReactMarkdown>
        </div>
        {sources.length > 0 && (
          <div className="sources">
            <div className="sources-title">
              <BookOpenText size={15} /> 参考来源
            </div>
            {sources.map((source, index) => (
              <div className="source-item" key={`${source}-${index}`}>
                {source}
              </div>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}
