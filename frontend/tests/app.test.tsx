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
  username: "alice",
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
  it("validates username and token before entering protected chat", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue({ ok: true, json: async () => identity } as Response);
    render(<App />);
    fireEvent.change(screen.getByLabelText("用户名"), {
      target: { value: "alice" },
    });
    fireEvent.change(screen.getByLabelText("访问令牌"), {
      target: { value: "test-token" },
    });
    fireEvent.click(screen.getByText("进入工作台"));
    await screen.findByText("和 SkySun 对话");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/me",
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer test-token",
        }),
      }),
    );
    expect(sessionStorage.getItem("skysun_token")).toBe("test-token");
  });

  it("rejects a token for a different username", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => identity,
    } as Response);
    render(<App />);
    fireEvent.change(screen.getByLabelText("用户名"), {
      target: { value: "bob" },
    });
    fireEvent.change(screen.getByLabelText("访问令牌"), {
      target: { value: "test-token" },
    });
    fireEvent.click(screen.getByText("进入工作台"));
    await screen.findByRole("alert");
    expect(sessionStorage.getItem("skysun_token")).toBeNull();
  });

  it("shows an authentication error for an invalid login token", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 401,
      json: async () => ({ detail: "Invalid authentication" }),
    } as Response);
    render(<App />);
    fireEvent.change(screen.getByLabelText("用户名"), {
      target: { value: "alice" },
    });
    fireEvent.change(screen.getByLabelText("访问令牌"), {
      target: { value: "bad-token" },
    });
    fireEvent.click(screen.getByText("进入工作台"));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Invalid authentication",
    );
    expect(sessionStorage.getItem("skysun_token")).toBeNull();
  });

  it("redirects a normal user away from admin", async () => {
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

  it("clears auth on an unauthorized API response", async () => {
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

  it("separates source metadata from answer markdown", () => {
    expect(
      splitSources("Answer\n\nSources:\n- guide.md (/docs/guide.md), chunk 0"),
    ).toEqual({
      body: "Answer",
      sources: ["guide.md (/docs/guide.md), chunk 0"],
    });
  });

  it("restores a completed conversation and renders its cited source", async () => {
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

  it("sends a chat turn and shows its persisted task status", async () => {
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

  it("switches between stored conversations", async () => {
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

  it("uploads a document as multipart form data", async () => {
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

  it("shows the current user's desktop companion status", async () => {
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
});
