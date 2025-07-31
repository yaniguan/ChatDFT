import os
import requests

# ─── Configure your backend URL (default is port 8080) ─────────────────────────
BASE = os.getenv("CHATDFT_BACKEND", "http://localhost:8080").rstrip("/")

def post(ep: str, body: dict, timeout: int = 90) -> dict:
    """
    ep: endpoint path, e.g. "/chat/ask"
    body: JSON body to send
    timeout: request timeout in seconds
    """
    # build URL safely
    url = f"{BASE}/{ep.lstrip('/')}"
    try:
        resp = requests.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # ensure front‑end sees ok==True when status < 400
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
        return {
            "ok": False,
            "detail": str(e)
        }
