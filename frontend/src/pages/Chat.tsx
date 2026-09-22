import { FormEvent, useEffect, useRef, useState } from "react";
import { ArrowUp, MessageCirclePlus, Sparkles, Workflow } from "lucide-react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../api/client";
import { Message } from "../components/Message";
import { ErrorNotice, PageHeader, StatusBadge } from "../components/Ui";
import { useAuth } from "../stores/auth";
import type { ChatResponse, Message as MessageType } from "../types";

export function Chat() {
  const { token, user } = useAuth();
  const storageKey = `skysun_conversation_${user?.user_id}`;
  const [conversationId, setConversationId] = useState<string | null>(() =>
    localStorage.getItem(storageKey),
  );
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(Boolean(conversationId));
  const [error, setError] = useState("");
  const [lastTaskId, setLastTaskId] = useState<string | null>(null);
  const [lastTaskStatus, setLastTaskStatus] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!conversationId || !token) return;
    let active = true;
    api<MessageType[]>(`/conversations/${conversationId}/messages`, token)
      .then((rows) => {
        if (active) setMessages(rows);
      })
      .catch((cause) => {
        if (!active) return;
        if (cause instanceof ApiError && cause.status === 404) {
          setConversationId(null);
          localStorage.removeItem(storageKey);
        } else {
          setError(cause instanceof Error ? cause.message : "无法加载对话");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  function newConversation() {
    setConversationId(null);
    localStorage.removeItem(storageKey);
    setMessages([]);
    setLastTaskId(null);
    setLastTaskStatus(null);
    setError("");
  }
  async function send(event: FormEvent) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || busy || !token) return;
    const previous = messages;
    setMessages([...previous, { role: "user", content: message }]);
    setDraft("");
    setError("");
    setBusy(true);
    try {
      const result = await api<ChatResponse>("/chat", token, {
        method: "POST",
        body: JSON.stringify({
          message,
          ...(conversationId ? { conversation_id: conversationId } : {}),
        }),
      });
      setConversationId(result.conversation_id);
      localStorage.setItem(storageKey, result.conversation_id);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: result.reply },
      ]);
      setLastTaskId(result.task_id);
      setLastTaskStatus(result.task_status);
    } catch (cause) {
      setMessages(previous);
      setDraft(message);
      setError(cause instanceof Error ? cause.message : "消息发送失败");
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="chat-page">
      <PageHeader
        eyebrow="AGENT CHAT"
        title="和 SkySun 对话"
        description="结合个人记忆与可靠来源，把想法推进到下一步。"
        action={
          <button className="ghost-button" onClick={newConversation}>
            <MessageCirclePlus size={17} /> 新对话
          </button>
        }
      />
      <div className="chat-panel">
        <div className="chat-scroll">
          {loading ? (
            <div className="loading">正在加载对话…</div>
          ) : messages.length === 0 ? (
            <div className="chat-welcome">
              <div className="welcome-icon">
                <Sparkles size={27} />
              </div>
              <h2>今天有什么想一起完成？</h2>
              <p>提问、整理知识，或交给 SkySun 一个多步骤任务。</p>
              <div className="suggestions">
                {[
                  "介绍一下你能做什么",
                  "帮我规划一个多步骤任务",
                  "根据我的偏好给出建议",
                ].map((item) => (
                  <button key={item} onClick={() => setDraft(item)}>
                    {item}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((item, index) => (
              <Message key={index} message={item} />
            ))
          )}
          {busy && (
            <div className="thinking">
              <Sparkles size={16} /> SkySun 正在思考并执行任务…
            </div>
          )}
          <div ref={endRef} />
        </div>
        <div className="chat-composer">
          {lastTaskId && (
            <Link className="task-link" to="/tasks">
              <Workflow size={15} /> 本次任务{" "}
              {lastTaskStatus && <StatusBadge status={lastTaskStatus} />} ·
              查看详情
            </Link>
          )}
          {error && <ErrorNotice message={error} />}
          <form onSubmit={send}>
            <textarea
              aria-label="输入消息"
              placeholder="输入你的问题或任务…"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
              rows={2}
            />
            <button
              type="submit"
              disabled={busy || !draft.trim()}
              aria-label="发送消息"
            >
              <ArrowUp size={21} />
            </button>
          </form>
          <small>SkySun 的回答可能有误，请核对重要信息与引用来源。</small>
        </div>
      </div>
    </div>
  );
}
