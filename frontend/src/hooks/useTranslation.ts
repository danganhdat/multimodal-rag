import { useCallback, useRef } from "react";
import * as api from "../api/client";

export function useTranslation() {
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const reqIdRef = useRef(0);

  const translateDebounced = useCallback(
    (
      text: string,
      source: string,
      target: string,
      onResult: (translated: string) => void,
      delay = 300
    ) => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (!text.trim()) {
        onResult("");
        return;
      }
      timerRef.current = setTimeout(async () => {
        const id = ++reqIdRef.current;
        try {
          const resp = await api.translate(text, source, target);
          if (id === reqIdRef.current) {
            onResult(resp.translated);
          }
        } catch {
          if (id === reqIdRef.current) {
            onResult("");
          }
        }
      }, delay);
    },
    []
  );

  return { translateDebounced };
}
