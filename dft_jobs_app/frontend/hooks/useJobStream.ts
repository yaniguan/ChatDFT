"use client";

import { useEffect, useRef, useState } from "react";

import { WS_BASE } from "@/lib/api";

export type StreamMessage = Record<string, unknown>;

export type ConnectionState = "connecting" | "open" | "closed";

export function useJobStream(
  path: string,
  onMessage: (msg: StreamMessage) => void,
): ConnectionState {
  const [state, setState] = useState<ConnectionState>("connecting");
  const handlerRef = useRef(onMessage);
  handlerRef.current = onMessage;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry = 0;
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = (): void => {
      if (cancelled) return;
      setState("connecting");
      const url = `${WS_BASE}${path}`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        retry = 0;
        setState("open");
      };
      ws.onmessage = (ev: MessageEvent<string>) => {
        try {
          const parsed: StreamMessage = JSON.parse(ev.data);
          handlerRef.current(parsed);
        } catch {
          // ignore invalid frames
        }
      };
      ws.onerror = () => {
        ws?.close();
      };
      ws.onclose = () => {
        setState("closed");
        if (cancelled) return;
        const delay = Math.min(30_000, 500 * 2 ** retry);
        retry += 1;
        retryTimer = setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      ws?.close();
    };
  }, [path]);

  return state;
}
