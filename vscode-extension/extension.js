"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vscode = require("vscode");

const statusPath = path.join(os.homedir(), ".config", "skysun", "vscode-status.json");
let timer;
let lastActivityAt = new Date().toISOString();

function activeWorkspace() {
  const editor = vscode.window.activeTextEditor;
  const folder = editor && vscode.workspace.getWorkspaceFolder(editor.document.uri);
  return folder || (vscode.workspace.workspaceFolders || [])[0] || null;
}

function publish() {
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

function activate(context) {
  publish();
  const activity = () => { lastActivityAt = new Date().toISOString(); publish(); };
  context.subscriptions.push(vscode.workspace.onDidChangeWorkspaceFolders(activity));
  context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(activity));
  context.subscriptions.push(vscode.workspace.onDidSaveTextDocument(activity));
  timer = setInterval(publish, 30000);
  context.subscriptions.push({ dispose: () => clearInterval(timer) });
}

function deactivate() {
  if (timer) clearInterval(timer);
  try {
    fs.writeFileSync(statusPath, JSON.stringify({ workspace_path: null,
      updated_at: new Date().toISOString() }), { mode: 0o600 });
  } catch (_) {
    // The companion will treat an old status file as stale.
  }
}

module.exports = { activate, deactivate };
