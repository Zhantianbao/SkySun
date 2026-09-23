import { useState } from "react";
import { Sun } from "lucide-react";

export function BrandLogo({ size = 28 }: { size?: number }) {
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
