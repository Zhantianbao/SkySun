// 本文件负责页面路由与交互中的 Chat 相关逻辑。
import { FormEvent, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  List,
  MessageCircle,
  MessageCirclePlus,
  Sparkles,
  Workflow,
} from "lucide-react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../api/client";
import { ExecutionActivity } from "../components/ExecutionActivity";
import { Message } from "../components/Message";
import { ErrorNotice, PageHeader, StatusBadge } from "../components/Ui";
import { useAuth } from "../stores/auth";
import type {
  ChatResponse,
  ConversationSummary,
  Message as MessageType,
  Task,
  TaskActivity,
} from "../types";

// 调用位置：App 中 /chat 路由调用。
export function Chat() {
  /** 管理会话列表、消息、发送状态和任务活动。 */
  const { token, user } = useAuth();
  const storageKey = `skysun_conversation_${user?.user_id}`;
  const [conversationId, setConversationId] = useState<string | null>(() =>
    localStorage.getItem(storageKey),
  );
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [showConversations, setShowConversations] = useState(false);
  const [messages, setMessages] = useState<MessageType[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(Boolean(conversationId));
  const [error, setError] = useState("");
  const [lastTaskId, setLastTaskId] = useState<string | null>(null);
  const [lastTaskStatus, setLastTaskStatus] = useState<string | null>(null);
  const [activity, setActivity] = useState<TaskActivity[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  // 调用位置：Chat 初次加载及发送消息后调用。
  async function refreshConversations() {
    /** 从接口更新侧边栏的会话列表。 */
    if (!token) return;
    try {
      const rows = await api<ConversationSummary[]>("/conversations", token);
      setConversations(Array.isArray(rows) ? rows : []);
    } catch {
      // Chat remains usable if the sidebar cannot refresh.
    }
  }

  // 进入聊天页后同步会话列表与当前会话历史；卸载时停止写入过期请求结果。
  useEffect(() => {
    void refreshConversations();
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

  // 消息或发送状态变化后，将视图滚动到最新内容。
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  // 调用位置：Chat 的新建会话按钮调用。
  function newConversation() {
    /** 清除当前会话状态并准备新对话。 */
    if (busy) return;
    setConversationId(null);
    localStorage.removeItem(storageKey);
    setMessages([]);
    setLastTaskId(null);
    setLastTaskStatus(null);
    setActivity([]);
    setError("");
    setShowConversations(false);
  }

  // 调用位置：Chat 的会话列表项点击时调用。
  async function switchConversation(id: string) {
    /** 加载所选会话的消息并切换本地状态。 */
    if (busy || id === conversationId || !token) return;
    setLoading(true);
    setError("");
    try {
      const rows = await api<MessageType[]>(`/conversations/${id}/messages`, token);
      setConversationId(id);
      localStorage.setItem(storageKey, id);
      setMessages(rows);
      setLastTaskId(null);
      setLastTaskStatus(null);
      setActivity([]);
      setShowConversations(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法切换对话");
    } finally {
      setLoading(false);
    }
  }

  // 调用位置：Chat 的消息表单提交时调用。
  async function send(event: FormEvent) {
    /** 发送消息、显示回复并同步任务活动。 */
    event.preventDefault();
    const message = draft.trim();
    if (!message || busy || !token) return;
    const previous = messages;
    const started = Date.now();
    let active = true;
    let polling = false;
    setMessages([...previous, { role: "user", content: message }]);
    setDraft("");
    setError("");
    setActivity([]);
    setBusy(true);

    // 调用位置：Chat.send 启动的定时器调用。
    async function pollActivity() {
      /** 查询当前聊天对应任务的最新执行活动。 */
      if (!active || polling || !token) return;
      polling = true;
      try {
        const tasks = await api<Task[]>("/tasks?limit=10", token);
        const current = tasks.find(
          (task) =>
            task.goal === message &&
            new Date(task.created_at).getTime() >= started - 5000 &&
            (!conversationId || task.conversation_id === conversationId),
        );
        if (current) {
          const rows = await api<TaskActivity[]>(
            `/tasks/${current.task_id}/activity`,
            token,
          );
          if (active) {
            setActivity(Array.isArray(rows) ? rows : []);
            setLastTaskId(current.task_id);
            setLastTaskStatus(current.status);
          }
        }
      } catch {
        // The request still completes through the existing /chat contract.
      } finally {
        polling = false;
      }
    }

    // 发起聊天请求期间定时读取任务进度，回复返回后再结束轮询。
    const timer = window.setInterval(() => void pollActivity(), 1200);
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
      if (result.task_id) {
        try {
          const rows = await api<TaskActivity[]>(
            `/tasks/${result.task_id}/activity`,
            token,
          );
          setActivity(Array.isArray(rows) ? rows : []);
        } catch {
          // The answer is still available if activity retrieval fails.
        }
      }
      void refreshConversations();
    } catch (cause) {
      setMessages(previous);
      setDraft(message);
      setError(cause instanceof Error ? cause.message : "消息发送失败");
    } finally {
      active = false;
      window.clearInterval(timer);
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
          <div className="chat-header-actions">
            <button
              className="ghost-button conversation-toggle"
              onClick={() => setShowConversations((value) => !value)}
              aria-label="切换对话"
            >
              <List size={17} />
            </button>
            <button className="ghost-button" onClick={newConversation}>
              <MessageCirclePlus size={17} /> 新对话
            </button>
          </div>
        }
      />
      <div className="chat-workspace">
        <aside
          className={`conversation-sidebar ${showConversations ? "open" : ""}`}
          aria-label="会话列表"
        >
          <div className="conversation-sidebar-header">
            <strong>最近对话</strong>
            <button onClick={newConversation} title="创建新对话" aria-label="创建新对话">
              <MessageCirclePlus size={18} />
            </button>
          </div>
          {conversations.length === 0 ? (
            <p className="conversation-empty">新对话会出现在这里。</p>
          ) : (
            <div className="conversation-list">
              {conversations.map((item) => (
                <button
                  key={item.conversation_id}
                  disabled={busy}
                  className={`conversation-item ${item.conversation_id === conversationId ? "active" : ""}`}
                  onClick={() => void switchConversation(item.conversation_id)}
                >
                  <MessageCircle size={16} />
                  <span>{item.title}</span>
                </button>
              ))}
            </div>
          )}
        </aside>
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
            <div ref={endRef} />
          </div>
          <div className="chat-composer">
            <ExecutionActivity activity={activity} busy={busy} />
            {lastTaskId && !busy && (
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
    </div>
  );
}
