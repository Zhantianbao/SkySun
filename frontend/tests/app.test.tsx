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
              : {
                  conversation_id: "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                  reply: "Done",
                  task_id: "cccccccc-cccc-cccc-cccc-cccccccccccc",
                  task_status: "COMPLETED",
                },
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
});
