import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, ApiError } from "../api/client";

const KEY = "adsci-session";

type SessionCtx = { sessionId: string | null; error: string | null; reset: () => Promise<void> };
const Ctx = createContext<SessionCtx>({ sessionId: null, error: null, reset: async () => {} });

/** One analysis session per browser tab; reused across reloads while the backend still has it. */
export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const create = useCallback(async () => {
    try {
      const { session_id } = await api.createSession();
      sessionStorage.setItem(KEY, session_id);
      setSessionId(session_id);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the backend.");
    }
  }, []);

  useEffect(() => {
    const saved = sessionStorage.getItem(KEY);
    if (!saved) {
      void create();
      return;
    }
    api
      .schema(saved)
      .then(() => setSessionId(saved))
      .catch((e) => (e instanceof ApiError && e.status === 404 ? create() : setError(e.message)));
  }, [create]);

  return <Ctx.Provider value={{ sessionId, error, reset: create }}>{children}</Ctx.Provider>;
}

export const useSession = () => useContext(Ctx);
