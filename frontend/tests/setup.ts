// 本文件负责前端测试中的 setup 相关逻辑。
import "@testing-library/jest-dom/vitest";
// 调用位置：测试运行时，聊天页滚动消息会调用此 DOM 替身。
Element.prototype.scrollIntoView = () => {};
