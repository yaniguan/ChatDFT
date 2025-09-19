# server/chat/intent_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


# ---------- text clipping ----------
def clip(s: Any, max_chars: int) -> str:
    s = "" if s is None else (s if isinstance(s, str) else json.dumps(s, ensure_ascii=False))
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n…[truncated {len(s)-max_chars} chars]"


# ---------- lists & few-shots ----------
def limit_list(xs: List[Any] | None, max_items: int) -> List[Any]:
    xs = xs or []
    return xs[:max_items]


def limit_fewshots(fewshots: List[Dict[str, Any]] | None, max_items: int = 6, max_chars: int = 1200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in limit_list(fewshots, max_items):
        if isinstance(e, dict):
            ee: Dict[str, Any] = {}
            for k, v in e.items():
                ee[k] = clip(v, max_chars) if isinstance(v, str) else v
            out.append(ee)
        else:
            out.append({"example": clip(e, max_chars)})
    return out


# ---------- message trimming ----------
def _messages_len_chars(msgs: List[Dict[str, str]]) -> int:
    try:
        return sum(len(m.get("content") or "") for m in msgs)
    except Exception:
        return 0


def hard_trim_messages(msgs: List[Dict[str, str]], budget: int = 16000) -> List[Dict[str, str]]:
    total = _messages_len_chars(msgs)
    if total <= budget:
        return msgs
    sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
    user_msg = msgs[-1] if msgs else None
    if not user_msg:
        return msgs
    keep_sys = [sys_msg] if sys_msg else []
    clipped_user = {"role": "user", "content": clip(user_msg.get("content", ""), budget - (_messages_len_chars(keep_sys) + 200))}
    return keep_sys + [clipped_user]


# ---------- JSON extraction from LLM ----------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1:]
        if s:
            s = "".join(s)
    return s.strip()


def _first_json_block(s: str) -> str | None:
    if not s:
        return None
    s = s.strip()
    if s.lstrip().startswith("{") and s.rstrip().endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None


def json_from_llm_raw(raw) -> dict | None:
    import json as _json
    if raw is None:
        return None
    if isinstance(raw, dict):
        if "content" in raw and isinstance(raw["content"], str):
            s = _strip_code_fences(raw["content"])  # type: ignore
            block = _first_json_block(s)
            if block:
                try:
                    return _json.loads(block)
                except Exception:
                    pass
        if "choices" in raw:
            try:
                s = raw["choices"][0]["message"]["content"]
                s = _strip_code_fences(s)
                block = _first_json_block(s)
                if block:
                    return _json.loads(block)
            except Exception:
                pass
        return raw if raw else None
    if isinstance(raw, str):
        s = _strip_code_fences(raw)
        block = _first_json_block(s)
        if not block:
            return None
        try:
            return _json.loads(block)
        except Exception:
            return None
    try:
        return _json.loads(str(raw))
    except Exception:
        return None


# ---------- RAG normalizer ----------
def normalize_rag(rag) -> tuple[str, list[dict]]:
    if rag is None:
        return "", []
    if isinstance(rag, str):
        return rag, []
    texts: List[str] = []
    refs: List[dict] = []
    seq = rag if isinstance(rag, list) else [rag]
    for it in seq:
        if isinstance(it, str):
            texts.append(it)
        elif isinstance(it, dict):
            texts.append(str(it.get("text", "")).strip())
            refs.append({
                "title": it.get("title") or it.get("id") or "",
                "source": it.get("source"),
                "url": it.get("url"),
            })
        else:
            texts.append(str(it))
    texts = [t for t in texts if t]
    return "\n\n".join(texts), refs


# ---------- species helpers ----------
def to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def label_of(x) -> str:
    if isinstance(x, dict):
        return str(x.get("label") or x.get("name") or x.get("id") or "")
    if isinstance(x, (list, tuple)):
        return " + ".join(label_of(t) for t in x)
    return "" if x is None else str(x)


def _split_species(s: str) -> List[str]:
    toks = re.split(r"[+,/]|(?<!\() ", s)
    return [t.strip() for t in toks if t and t.strip()]


def _is_species(tok: str) -> bool:
    if not tok or " " in tok:
        return False
    return tok.endswith("*") or tok.endswith("(g)") or tok.endswith("(aq)")


def extract_star(side: Any) -> List[str]:
    if isinstance(side, dict):
        raw = side.get("species") or side.get("reactants") or side.get("lhs") or side.get("side")
        return extract_star(raw)
    seen, out = set(), []
    for item in to_list(side):
        if isinstance(item, dict):
            tok = label_of(item)
            if _is_species(tok) and tok not in seen:
                seen.add(tok)
                out.append(tok)
        elif isinstance(item, str):
            for tok in _split_species(item):
                if _is_species(tok) and tok not in seen:
                    seen.add(tok)
                    out.append(tok)
        else:
            tok = label_of(item)
            if _is_species(tok) and tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


def extract_lhs_rhs(step: Any) -> Tuple[List[str], List[str]]:
    if isinstance(step, dict):
        lhs = extract_star(step.get("reactants") or step.get("lhs") or step.get("from") or [])
        rhs = extract_star(step.get("products") or step.get("rhs") or step.get("to") or [])
        return lhs, rhs
    if isinstance(step, str) and "->" in step:
        L, R = step.split("->", 1)
        return extract_star(L), extract_star(R)
    side = extract_star(step)
    return side, []

