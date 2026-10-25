"use strict";
// 本文件负责VS Code 工作区状态上报中的 extension 相关逻辑。

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vscode = require("vscode");

const statusPath = path.join(os.homedir(), ".config", "skysun", "vscode-status.json");
let timer;
let lastActivityAt = new Date().toISOString();

// 调用位置：publish 收集 VS Code 状态时调用。
function activeWorkspace() {
  /** 取得当前编辑器所在或默认工作区。 */
  const editor = vscode.window.activeTextEditor;
  const folder = editor && vscode.workspace.getWorkspaceFolder(editor.document.uri);
  return folder || (vscode.workspace.workspaceFolders || [])[0] || null;
}

// 调用位置：activate 的事件监听和定时器调用。
function publish() {
  /** 将当前工作区及活动时间写入伴随程序状态文件。 */
  const folder = activeWorkspace();
  const status = {
    workspace_path: folder && folder.uri.scheme === "file" ? folder.uri.fsPath : null,
    updated_at: new Date().toISOString(),
    activity_at: lastActivityAt,
  };
  fs.mkdirSync(path.dirname(statusPath), { recursive: true });
  const temporary = statusPath + "." + process.pid + ".tmp";
  fs.writeFileSync(temporary, JSON.stringify(status), { mode: 0o600 });
  fs.renameSync(temporary, statusPath);
  fs.chmodSync(statusPath, 0o600);
}

// 调用位置：VS Code 启用扩展时调用。
function activate(context) {
  /** 注册编辑器事件与定时状态发布。 */
  publish();
  // 工作区、活动编辑器和文件保存都表示用户活动，统一更新状态文件。
  const activity = () => { lastActivityAt = new Date().toISOString(); publish(); };
  context.subscriptions.push(vscode.workspace.onDidChangeWorkspaceFolders(activity));
  context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(activity));
  context.subscriptions.push(vscode.workspace.onDidSaveTextDocument(activity));
  timer = setInterval(publish, 30000);
  context.subscriptions.push({ dispose: () => clearInterval(timer) });
}

// 调用位置：VS Code 停用扩展时调用。
function deactivate() {
  /** 停止定时器并标记工作区状态过期。 */
  if (timer) clearInterval(timer);
  try {
    fs.writeFileSync(statusPath, JSON.stringify({ workspace_path: null,
      updated_at: new Date().toISOString() }), { mode: 0o600 });
  } catch (_) {
    // The companion will treat an old status file as stale.
  }
}

module.exports = { activate, deactivate };
