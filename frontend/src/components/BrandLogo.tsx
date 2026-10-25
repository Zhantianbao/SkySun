// 本文件负责前端数据与组件中的 BrandLogo 相关逻辑。
import { useState } from "react";
import { Sun } from "lucide-react";

// 调用位置：登录页和侧边栏调用。
export function BrandLogo({ size = 28 }: { size?: number }) {
  /** 显示 SkySun 品牌标识。 */
  const [available, setAvailable] = useState(true);
  return available ? (
    <img
      className="brand-logo"
      src="/assets/skysun-logo.png"
      alt="SkySun"
      width={size}
      height={size}
      onError={() => setAvailable(false)}
    />
  ) : (
    <Sun size={size} strokeWidth={1.8} aria-label="SkySun" />
  );
}
