import { useEffect, useState } from "react";
import { assetObjectUrl } from "./api";

export function useAsset(url?: string | null) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setSrc(null); setFailed(false);
    if (!url) return;
    let made: string | null = null, dead = false;
    assetObjectUrl(url).then((u) => { if (dead) URL.revokeObjectURL(u); else { made = u; setSrc(u); } }).catch(() => !dead && setFailed(true));
    return () => { dead = true; if (made) URL.revokeObjectURL(made); };
  }, [url]);
  return { src, failed };
}
