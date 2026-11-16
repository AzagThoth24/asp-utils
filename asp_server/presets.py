"""LoRA 组预设与触发词覆盖的持久化（asp-utils）。

数据文件（原子写入：临时文件 + os.replace）：
  user/lora_presets.json  {"presets": {"<名字>": {"rows": [...], "updated_at": "..."}}, "version": 1}
  user/lora_tags.json     {"tags": {"<lora名>": ["触发词", ...]}, "version": 1}
"""

import json
import os
import re
import threading
import time
from pathlib import Path

USER_DIR = Path(__file__).resolve().parent.parent / "user"
PRESETS_FILE = "lora_presets.json"
TAGS_FILE = "lora_tags.json"

_MAX_PRESETS = 200
_MAX_ROWS = 50
_NAME_MAX = 64
_NAME_BAD = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
_STRENGTH_MIN, _STRENGTH_MAX = -10.0, 10.0

_LOCK = threading.Lock()
_FILE_CACHE = {}  # path -> mtime 防重复读盘


# ---------------------------------------------------------------- 基础读写

def _atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-%d" % int(time.time() * 1000))
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    _FILE_CACHE.pop(str(path), None)


def _read_json(path, default):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return default
    cached = _FILE_CACHE.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return default
    _FILE_CACHE[str(path)] = (mtime, data)
    return data


def _sanitize_name(name):
    name = _NAME_BAD.sub("_", str(name or "").strip()).strip()
    return name[:_NAME_MAX]


def sanitize_rows(rows):
    """把前端传来的 rows 清洗成规范形态，异常返回 None。"""
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_ROWS:
        return None
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        lora = str(row.get("lora") or "").strip()
        if not lora:
            return None
        try:
            sm = float(row.get("strength_model", 1.0))
            sc = float(row.get("strength_clip", row.get("strength_model", 1.0)))
        except (TypeError, ValueError):
            return None
        clean.append({
            "on": bool(row.get("on", True)),
            "lora": lora[:255],
            "strength_model": max(_STRENGTH_MIN, min(_STRENGTH_MAX, sm)),
            "strength_clip": max(_STRENGTH_MIN, min(_STRENGTH_MAX, sc)),
        })
    return clean


# ---------------------------------------------------------------- 预设

def get_presets():
    data = _read_json(USER_DIR / PRESETS_FILE, {})
    presets = data.get("presets") if isinstance(data, dict) else None
    return presets if isinstance(presets, dict) else {}


def save_preset(name, rows):
    """新建或覆盖一个预设；返回 (ok, 消息)。"""
    name = _sanitize_name(name)
    if not name:
        return False, "预设名不能为空"
    rows = sanitize_rows(rows)
    if rows is None:
        return False, "行数据不合法（需要 1-%d 行，且每行有 lora 名与数值强度）" % _MAX_ROWS
    with _LOCK:
        presets = get_presets()
        if name not in presets and len(presets) >= _MAX_PRESETS:
            return False, "预设数量已达上限 %d" % _MAX_PRESETS
        presets[name] = {"rows": rows, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        _atomic_write_json(USER_DIR / PRESETS_FILE,
                           {"version": 1, "presets": presets})
    return True, "已保存预设「%s」" % name


def delete_preset(name):
    name = _sanitize_name(name)
    if not name:
        return False, "预设名不能为空"
    with _LOCK:
        presets = get_presets()
        if name not in presets:
            return False, "预设「%s」不存在" % name
        del presets[name]
        _atomic_write_json(USER_DIR / PRESETS_FILE,
                           {"version": 1, "presets": presets})
    return True, "已删除预设「%s」" % name


# ---------------------------------------------------------------- 触发词覆盖

def get_tags():
    data = _read_json(USER_DIR / TAGS_FILE, {})
    tags = data.get("tags") if isinstance(data, dict) else None
    return tags if isinstance(tags, dict) else {}


def set_tags(lora, words):
    """手动覆盖某个 LoRA 的触发词；words 为空列表表示清除该条。"""
    lora = str(lora or "").strip()[:255]
    if not lora:
        return False, "lora 名不能为空"
    clean = []
    if isinstance(words, str):
        words = [w for w in (x.strip() for x in words.replace("\n", ",").split(","))]
    if not isinstance(words, list):
        return False, "触发词必须是列表"
    seen = set()
    for w in words:
        w = str(w).strip()
        if w and w not in seen:
            seen.add(w)
            clean.append(w)
        if len(clean) >= 32:
            break
    with _LOCK:
        tags = get_tags()
        if clean:
            tags[lora] = clean
        else:
            tags.pop(lora, None)
        _atomic_write_json(USER_DIR / TAGS_FILE, {"version": 1, "tags": tags})
    return True, "已更新「%s」的触发词" % lora
