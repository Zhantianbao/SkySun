// 本文件负责前端测试中的 app.test 相关逻辑。
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { App } from "../src/App";
import { splitSources } from "../src/components/Message";

const identity = {
  user_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
  username: "张三",
  role: "USER",
};

beforeEach(() => {
  sessionStorage.clear();
  localStorage.clear();
  vi.restoreAllMocks();
  window.history.replaceState({}, "", "/login");
});
afterEach(() => cleanup());

describe("web application", () => {
  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("validates username and token before entering protected chat", async () => {
    /** 验证姓名和令牌都有效后才进入受保护的聊天页。 */
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue({ ok: true, json: async () => identity } as Response);
    render(<App />);
    fireEvent.change(screen.getByLabelText("姓名"), {
      target: { value: "张三" },
    });
    fireEvent.change(screen.getByLabelText("密码"), {
      target: { value: "test-token" },
    });
    fireEvent.click(screen.getByText("进入工作台"));
    await screen.findByText("和 SkySun 对话");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/login",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ username: "张三", token: "test-token" }),
      }),
    );
    expect(sessionStorage.getItem("skysun_token")).toBe("test-token");
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("rejects a token for a different username", async () => {
    /** 验证令牌对应的姓名不一致时拒绝登录。 */
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => identity,
    } as Response);
    render(<App />);
    fireEvent.change(screen.getByLabelText("姓名"), {
      target: { value: "李四" },
    });
    fireEvent.change(screen.getByLabelText("密码"), {
      target: { value: "test-token" },
    });
    fireEvent.click(screen.getByText("进入工作台"));
    await screen.findByRole("alert");
    expect(sessionStorage.getItem("skysun_token")).toBeNull();
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("logs the built-in administrator into a workspace with admin access", async () => {
    /** 验证内置管理员可进入带管理入口的工作台。 */
    const admin = { ...identity, username: "詹天宝", role: "ADMIN" };
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => admin,
    } as Response);
    render(<App />);
    fireEvent.change(screen.getByLabelText("姓名"), { target: { value: "詹天宝" } });
    fireEvent.change(screen.getByLabelText("密码"), { target: { value: "admin-secret" } });
    fireEvent.click(screen.getByText("进入工作台"));
    const links = await screen.findAllByRole("link", { name: "管理" });
    expect(links[0]).toHaveAttribute("href", "/admin");
    expect(JSON.parse(sessionStorage.getItem("skysun_user") || "null")).toEqual(admin);
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("ignores a cached administrator identity when no credential is present", async () => {
    /** 验证没有令牌时不会信任缓存的管理员身份。 */
    sessionStorage.setItem("skysun_user", JSON.stringify({ ...identity, role: "ADMIN" }));
    window.history.replaceState({}, "", "/admin");
    render(<App />);
    await screen.findByLabelText("密码");
    expect(window.location.pathname).toBe("/login");
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("shows an authentication error for an invalid login token", async () => {
    /** 验证错误令牌会显示身份验证失败信息。 */
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({ detail: "姓名或密码错误" }),
    } as Response);
    render(<App />);
    fireEvent.change(screen.getByLabelText("姓名"), {
      target: { value: "张三" },
    });
    fireEvent.change(screen.getByLabelText("密码"), {
      target: { value: "bad-token" },
    });
    fireEvent.click(screen.getByText("进入工作台"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "姓名或密码错误",
    );
    expect(sessionStorage.getItem("skysun_token")).toBeNull();
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("redirects a normal user away from admin", async () => {
    /** 验证普通用户不能进入管理员页面。 */
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    window.history.replaceState({}, "", "/admin");
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => identity,
    } as Response);
    render(<App />);
    await screen.findByText("和 SkySun 对话");
    await waitFor(() => expect(window.location.pathname).toBe("/chat"));
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("clears auth on an unauthorized API response", async () => {
    /** 验证接口返回 401 后清除浏览器身份。 */
    sessionStorage.setItem("skysun_token", "expired");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    window.history.replaceState({}, "", "/memory");
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({ detail: "Invalid authentication" }),
    } as Response);
    render(<App />);
    await waitFor(() => expect(window.location.pathname).toBe("/login"));
    expect(sessionStorage.getItem("skysun_token")).toBeNull();
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("separates source metadata from answer markdown", () => {
    /** 验证回答正文与来源元数据分开展示。 */
    expect(
      splitSources("Answer\n\nSources:\n- guide.md (/docs/guide.md), chunk 0"),
    ).toEqual({
      body: "Answer",
      sources: ["guide.md (/docs/guide.md), chunk 0"],
    });
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("restores a completed conversation and renders its cited source", async () => {
    /** 验证恢复历史会话时显示原消息及引用来源。 */
    const conversationId = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    localStorage.setItem(
      `skysun_conversation_${identity.user_id}`,
      conversationId,
    );
    window.history.replaceState({}, "", "/chat");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      async (url) =>
        ({
          ok: true,
          json: async () =>
            String(url).endsWith("/me")
              ? identity
              : String(url).endsWith("/conversations")
                ? []
                : [
                  { role: "user", content: "What is the marker?" },
                  {
                    role: "assistant",
                    content:
                      "amber-742\n\nSources:\n- orchid.md (/docs/orchid.md), chunk 0",
                  },
                ],
        }) as Response,
    );
    render(<App />);
    await screen.findByText("amber-742");
    expect(
      screen.getByText("orchid.md (/docs/orchid.md), chunk 0"),
    ).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/conversations/${conversationId}/messages`,
      expect.anything(),
    );
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("sends a chat turn and shows its persisted task status", async () => {
    /** 验证发送消息后呈现持久化任务的状态。 */
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    window.history.replaceState({}, "", "/chat");
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (url) =>
        ({
          ok: true,
          json: async () =>
            String(url).endsWith("/me")
              ? identity
              : String(url).endsWith("/chat")
                ? {
                  conversation_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                  reply: "Done",
                  task_id: "cccccccc-cccc-cccc-cccc-cccccccccccc",
                  task_status: "COMPLETED",
                }
                : [],
        }) as Response,
    );
    render(<App />);
    await screen.findByText("和 SkySun 对话");
    fireEvent.change(screen.getByLabelText("输入消息"), {
      target: { value: "Help me" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送消息" }));
    await screen.findByText("Done");
    expect(screen.getByText("COMPLETED")).toBeInTheDocument();
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("switches between stored conversations", async () => {
    /** 验证用户可以切换已保存的会话。 */
    const first = "11111111-1111-1111-1111-111111111111";
    const second = "22222222-2222-2222-2222-222222222222";
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    localStorage.setItem(`skysun_conversation_${identity.user_id}`, first);
    window.history.replaceState({}, "", "/chat");
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => ({
      ok: true,
      json: async () => {
        const path = String(url);
        if (path.endsWith("/me")) return identity;
        if (path.endsWith("/conversations"))
          return [
            { conversation_id: first, title: "First topic" },
            { conversation_id: second, title: "Second topic" },
          ];
        return [{ role: "assistant", content: path.includes(second) ? "Second answer" : "First answer" }];
      },
    } as Response));
    render(<App />);
    await screen.findByText("First answer");
    fireEvent.click(screen.getByRole("button", { name: /Second topic/ }));
    await screen.findByText("Second answer");
    expect(localStorage.getItem(`skysun_conversation_${identity.user_id}`)).toBe(second);
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("uploads a document as multipart form data", async () => {
    /** 验证文档上传采用 multipart 表单。 */
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    window.history.replaceState({}, "", "/knowledge");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (url, options) => ({
      ok: true,
      json: async () => {
        if (String(url).endsWith("/me")) return identity;
        if (options?.method === "POST")
          return { document_id: "doc", source_path: "upload://doc", chunk_count: 1, status: "ingested" };
        return [];
      },
    } as Response));
    render(<App />);
    await screen.findByText("添加个人知识");
    fireEvent.change(screen.getByLabelText("选择知识文档"), {
      target: { files: [new File(["Knowledge"], "note.md", { type: "text/markdown" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "上传文档" }));
    await screen.findByText(/上传完成，已建立 1 个知识片段/);
    const uploadCall = fetchMock.mock.calls.find(([, options]) => options?.method === "POST");
    expect(uploadCall?.[1]?.body).toBeInstanceOf(FormData);
    expect(uploadCall?.[1]?.headers).not.toHaveProperty("Content-Type");
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("shows the current user's desktop companion status", async () => {
    /** 验证页面显示当前用户的桌面伴随程序状态。 */
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    window.history.replaceState({}, "", "/devices");
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => ({
      ok: true,
      json: async () => String(url).endsWith("/me") ? identity : [{
        device_id: "dddddddd-dddd-dddd-dddd-dddddddddddd",
        name: "My Computer", hostname: "laptop", operating_system: "Linux",
        online: true, last_heartbeat_at: "2026-09-23T10:00:00Z",
        cpu_percent: 35, memory_percent: 60, disk_percent: 42,
      }],
    } as Response));
    render(<App />);
    await screen.findByText("My Computer");
    expect(screen.getByText("在线")).toBeInTheDocument();
    expect(screen.getByText("35%")).toBeInTheDocument();
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("requests a screenshot and queues a desktop reminder", async () => {
    /** 验证截图请求和桌面便笺都进入设备动作队列。 */
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    window.history.replaceState({}, "", "/devices");
    const deviceId = "dddddddd-dddd-dddd-dddd-dddddddddddd";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (url, options) => ({
      ok: true,
      json: async () => String(url).endsWith("/me") ? identity
        : options?.method === "POST" ? { status: "PENDING" }
          : [{ device_id: deviceId, name: "Laptop", hostname: "host",
              operating_system: "Linux", online: true, last_heartbeat_at: null,
              cpu_percent: 1, memory_percent: 2, disk_percent: 3,
              project_name: "SkySun", git_branch: "main", git_modified_count: 2,
              last_activity_at: null, vscode_running: true, codex_active: false }],
    } as Response));
    render(<App />);
    await screen.findByText("Laptop");
    expect(screen.getByText("SkySun")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "请求截图" }));
    await screen.findByText(/截图请求已发送/);
    fireEvent.change(screen.getByLabelText("给 Laptop 的提醒"), { target: { value: "Review this" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));
    await screen.findByText(/提醒已加入/);
    expect(fetchMock).toHaveBeenCalledWith(`/api/devices/${deviceId}/screenshots`,
      expect.objectContaining({ method: "POST" }));
    expect(fetchMock).toHaveBeenCalledWith(`/api/devices/${deviceId}/notes`,
      expect.objectContaining({ body: JSON.stringify({ content: "Review this" }) }));
  });

  // 调用位置：Vitest 执行本条前端交互测试时调用。
  it("shows owner screenshot request status", async () => {
    /** 验证截图页只显示当前所有者的请求状态。 */
    sessionStorage.setItem("skysun_token", "test-token");
    sessionStorage.setItem("skysun_user", JSON.stringify(identity));
    window.history.replaceState({}, "", "/screenshots");
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => ({
      ok: true,
      json: async () => String(url).endsWith("/me") ? identity : [{
        request_id: "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        device_id: "dddddddd-dddd-dddd-dddd-dddddddddddd",
        status: "PENDING", requested_at: "2026-09-23T10:00:00Z", uploaded_at: null,
      }],
    } as Response));
    render(<App />);
    expect(await screen.findByText("等待电脑响应")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "查看" })).not.toBeInTheDocument();
  });
});
