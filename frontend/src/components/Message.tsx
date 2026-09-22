import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpenText, Sparkles } from "lucide-react";
import type { Message as MessageType } from "../types";

export function splitSources(content: string) {
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

export function Message({ message }: { message: MessageType }) {
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
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
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
