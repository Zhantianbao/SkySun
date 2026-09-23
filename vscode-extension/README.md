# SkySun Workspace Awareness

This optional local VS Code extension writes the active workspace path and update time to `~/.config/skysun/vscode-status.json`. It does not connect to the network, read file contents, or control Codex. The desktop companion only accepts the reported workspace if it matches the directory the user selected with `skysun_companion context --workspace`.

Package locally with `npx @vscode/vsce package` from this directory, then install the resulting VSIX through **VS Code: Extensions: Install from VSIX**. The extension source has no runtime npm dependencies. Remove the extension to stop reporting; the companion ignores records older than 90 seconds.
