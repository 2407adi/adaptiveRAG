import { useCallback, useEffect, useState } from "react";
import Landing from "./Landing";
import ChatApp from "./chat/ChatApp";
import { clearKey, getStoredKey, rotateClientId, setUnauthorizedHandler } from "./api";

export default function App() {
  // Returning visitor with a stored key skips the landing page; a 401 later
  // (key rotated / revoked) bounces them straight back here.
  const [view, setView] = useState<"landing" | "app">(getStoredKey() ? "app" : "landing");
  const [authNotice, setAuthNotice] = useState<string | null>(null);

  const logout = useCallback((notice?: string) => {
    clearKey();
    if (!notice) rotateClientId();  // explicit logout = fresh identity; a 401 bounce keeps it
    setAuthNotice(notice ?? null);
    setView("landing");
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(() => logout("That key no longer works — it may have been rotated. Paste a fresh one."));
  }, [logout]);

  return view === "landing" ? (
    <Landing notice={authNotice} onAuthed={() => { setAuthNotice(null); setView("app"); }} />
  ) : (
    <ChatApp onLogout={() => logout()} />
  );
}
