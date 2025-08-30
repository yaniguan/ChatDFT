# client/utils/api.py
import os
import requests
from typing import Any, Dict, List, Optional

# ─── Configure your backend URL (default is port 8080) ─────────────────────────
BASE = os.getenv("CHATDFT_BACKEND", "http://localhost:8080").rstrip("/")

def get(path, params=None):
    r = requests.get(f"{BASE}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def post(ep: str, body: dict, timeout: int = 90) -> dict:
    """
    ep: endpoint path, e.g. "/chat/ask"
    body: JSON body to send
    timeout: request timeout in seconds
    """
    url = f"{BASE}/{ep.lstrip('/')}"
    try:
        resp = requests.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        data.setdefault("ok", True)
        return data

    except requests.HTTPError as http_err:
        # Try to parse JSON error body, else fallback to text
        try:
            err_body = resp.json()
        except ValueError:
            err_body = {"detail": resp.text}
        err_body.update({
            "ok": False,
            "detail": f"{http_err} (status {resp.status_code})",
            "status_code": resp.status_code
        })
        return err_body

    except requests.RequestException as e:
        # Network errors, connection refused, timeouts, etc.
        return {"ok": False, "detail": str(e)}

# ---------- Session ----------
def list_sessions() -> dict:
    return post("/chat/session/list", {})

def create_session(name: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> dict:
    return post("/chat/session/create", {"name": name, "meta": meta or {}})

def update_session(session_id: int, **kwargs) -> dict:
    payload = {"session_id": session_id, **kwargs}
    return post("/chat/session/update", payload)

def delete_session(session_id: int) -> dict:
    return post("/chat/session/delete", {"session_id": session_id})

# ---------- Chat Agent ----------
def ask_chat(session_id: int, query: str) -> dict:
    return post("/chat/ask", {"session_id": session_id, "query": query})

# ---------- Intent Agent ----------
def run_intent(query: str) -> dict:
    """Input: {'query': str} → Output: {'fields': dict, ...}"""
    return post("/chat/intent", {"query": query})

# ---------- Knowledge Agent ----------
def run_knowledge(intent: Dict[str, Any], query: str = "") -> dict:
    """Input: {'intent': dict, 'query': str?} → Output: {'result': dict, ...}"""
    payload = {"intent": intent}
    if query:
        payload["query"] = query
    return post("/chat/knowledge", payload)

# ---------- History Agent ----------
def run_history(session_id: int) -> dict:
    """Input: {'session_id': int} → Output: {'history': list/dict, ...}"""
    return post("/chat/history", {"session_id": session_id})

# ---------- Hypothesis Agent ----------
def run_hypothesis(intent: Dict[str, Any],
                   knowledge: Dict[str, Any],
                   history: List[Dict[str, Any]]) -> dict:
    """
    Input: {'intent': dict, 'knowledge': dict, 'history': list[dict]}
    Output: {'result_md': str, ...}
    """
    return post("/chat/hypothesis", {
        "intent": intent,
        "knowledge": knowledge,
        "history": history,
    })

# ---------- Plan Agent ----------
def run_plan(query: str,
             intent: Dict[str, Any],
             hypothesis: str,
             history: List[Dict[str, Any]]) -> dict:
    """
    Input: {'query': str, 'intent': dict, 'hypothesis': str, 'history': list[dict]}
    Output: {'tasks': list[dict], ...}
    """
    return post("/chat/plan", {
        "query": query,
        "intent": intent,
        "hypothesis": hypothesis,
        "history": history,
    })

def run_execute(all_tasks: List[Dict[str, Any]],
                selected_ids: List[int],
                cluster: str = "hoffman2",
                dry_run: bool = False,
                sync_back: bool = True) -> dict:
    """
    Execute under plan (your server route lives under plan):
    Input: {
      'all_tasks': list[dict],
      'selected_ids': list[int],
      'cluster': str,
      'dry_run': bool,
      'sync_back': bool
    }
    Output: {'workdir': str, 'results': list[dict], ...}
    """
    return post("/chat/plan/execute", {
        "all_tasks": all_tasks,
        "selected_ids": selected_ids,
        "cluster": cluster,
        "dry_run": dry_run,
        "sync_back": sync_back,
    })

def list_runs(session_id=None, limit=50):
    body = {"limit": limit}
    if session_id: body["session_id"] = session_id
    return post("/chat/records/list", body)

def get_run(run_id: int):
    return post("/chat/records/get", {"run_id": run_id})