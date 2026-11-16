"""LoRA 扫描与触发词提取（asp-utils）。

触发词来源优先级（高 → 低）：
  1. user/lora_tags.json 手动覆盖（UI 行内编辑写入）
  2. 同名 .civitai.info / .json 边车文件里的 trainedWords
  3. safetensors 头部 __metadata__ 的 modelspec.trigger_phrase
  4. ss_tag_frequency 高频标签（兜底，仅当上面都缺失）

safetensors 头部解析不依赖 torch：只读文件前 8 字节长度 + header JSON。
扫描结果按 (mtime, size) 缓存，重复请求零 IO。
"""

import json
import logging
import os
import struct
from pathlib import Path

_LOG = logging.getLogger("asp")

# 视为 LoRA 模型的扩展名（与 folder_paths 过滤规则保持一致即可，这里更宽松些）
LORA_EXTS = {".safetensors", ".sft", ".ckpt", ".pt"}
# 单个边车文件 / safetensors header 的最大读取字节数，防异常大文件
_MAX_SIDELOAD = 8 * 1024 * 1024
_MAX_HEADER = 128 * 1024 * 1024
# ss_tag_frequency 兜底：出现次数下限与最多取多少个标签
_TAG_FREQ_MIN_COUNT = 3
_TAG_FREQ_MAX_WORDS = 8

# name -> {"mtime": float, "size": int, "info": {...}}
_CACHE = {}
_TAGS_OVERRIDE_FILE = None  # 由 presets.py 注入路径；避免循环导入，运行时设置


def set_tags_override_path(path):
    """注册 user/lora_tags.json 的位置；路径变化时清空缓存。"""
    global _TAGS_OVERRIDE_FILE
    path = str(path)
    if _TAGS_OVERRIDE_FILE != path:
        _CACHE.clear()
        _TAGS_OVERRIDE_FILE = path


def _load_json(path, cap=_MAX_SIDELOAD):
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > cap:
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def read_safetensors_metadata(path):
    """读取 safetensors 头部的 __metadata__ dict；失败/缺失返回 None。"""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > _MAX_HEADER:
                return None
            header = json.loads(f.read(n))
        meta = header.get("__metadata__") if isinstance(header, dict) else None
        return meta if isinstance(meta, dict) else None
    except Exception:
        return None


def _words_from_trained(value):
    """civitai trainedWords 兼容 str / [str] / [{word|name|triggerWord}] 三种形态。"""
    words = []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return words
    for item in value:
        if isinstance(item, str):
            w = item.strip()
        elif isinstance(item, dict):
            w = str(item.get("word") or item.get("name")
                    or item.get("triggerWord") or "").strip()
        else:
            continue
        if w:
            words.append(w)
    return words


def _words_from_trigger_phrase(value):
    if not isinstance(value, str):
        return []
    return [w.strip() for w in value.replace("\n", ",").split(",") if w.strip()]


def _words_from_tag_frequency(value):
    """ss_tag_frequency: {"<数据集目录>": {"<tag>": count, ...}} 的 JSON 字符串。"""
    try:
        data = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    counts = {}
    if isinstance(data, dict):
        for tags in data.values():
            if not isinstance(tags, dict):
                continue
            for tag, cnt in tags.items():
                tag = str(tag).strip()
                if not tag:
                    continue
                try:
                    cnt = int(cnt)
                except (TypeError, ValueError):
                    cnt = 0
                counts[tag] = counts.get(tag, 0) + cnt
    ranked = sorted((t for t in counts.items() if t[1] >= _TAG_FREQ_MIN_COUNT),
                    key=lambda kv: -kv[1])[:_TAG_FREQ_MAX_WORDS]
    return [t for t, _ in ranked]


def _load_tags_override():
    if not _TAGS_OVERRIDE_FILE:
        return {}
    data = _load_json(_TAGS_OVERRIDE_FILE)
    tags = data.get("tags") if isinstance(data, dict) else None
    return tags if isinstance(tags, dict) else {}


def _sidecar_candidates(path):
    p = Path(path)
    stem_json = p.with_suffix(".json")
    return [Path(str(p) + ".civitai.info"), p.with_suffix(".civitai.info"), stem_json]


def extract_for_name(name, full_path, override_words=None):
    """提取某个 LoRA 的信息。

    name: folder_paths 风格的相对名（可含子目录，正斜杠分隔）
    full_path: 磁盘绝对路径
    override_words: user/lora_tags.json 里的手动覆盖（可 None）
    """
    base_keys = [name, os.path.basename(name),
                 os.path.splitext(os.path.basename(name))[0]]
    if override_words is None:
        override_words = []
    for key in base_keys:
        if key in override_words and isinstance(override_words[key], list):
            words = [str(w).strip() for w in override_words[key] if str(w).strip()]
            if words:
                return words, "manual", _meta_base_model(full_path)

    # 2) civitai 边车
    for cand in _sidecar_candidates(full_path):
        data = _load_json(cand)
        if not isinstance(data, dict):
            continue
        words = _words_from_trained(
            data.get("trainedWords") or data.get("triggerWords"))
        if words:
            base = str(data.get("baseModel") or "") or _meta_base_model(full_path)
            return words, "civitai", base

    # 3/4) safetensors 内嵌 metadata
    meta = read_safetensors_metadata(full_path)
    if meta:
        words = _words_from_trigger_phrase(
            meta.get("modelspec.trigger_phrase")
            or meta.get("trigger_phrase")
            or meta.get("trainedWords"))
        if words:
            return words, "metadata", str(meta.get("modelspec.base_model") or "")
        words = _words_from_tag_frequency(meta.get("ss_tag_frequency"))
        if words:
            return words, "tag_frequency", str(meta.get("modelspec.base_model") or "")
    return [], "", str(meta.get("modelspec.base_model") or "") if meta else ""


def _meta_base_model(path):
    meta = read_safetensors_metadata(path)
    if isinstance(meta, dict):
        return str(meta.get("modelspec.base_model") or "")
    return ""


def scan_loras(force=False):
    """扫描 loras 目录，返回 [{name, trigger_words, source, base_model, mtime, size}]。

    列表来源用 folder_paths.get_filename_list("loras")，与 ComfyUI 下拉列表同源，
    含子目录前缀且随 ComfyUI 刷新（R 键）自动更新。
    """
    import folder_paths  # 延迟导入：tests 通过 AST 提取时不触发

    names = list(folder_paths.get_filename_list("loras"))
    overrides = _load_tags_override()
    result = []
    for name in names:
        if os.path.splitext(name)[1].lower() not in LORA_EXTS:
            continue
        full = folder_paths.get_full_path("loras", name)
        if not full:
            continue
        try:
            st = os.stat(full)
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            continue
        cached = _CACHE.get(name)
        if (not force and cached
                and cached["mtime"] == mtime and cached["size"] == size):
            info = cached["info"]
        else:
            words, source, base = extract_for_name(name, full, overrides)
            info = {"name": name, "trigger_words": words, "source": source,
                    "base_model": base}
            _CACHE[name] = {"mtime": mtime, "size": size, "info": info}
        entry = dict(info)
        entry["mtime"] = mtime
        entry["size"] = size
        result.append(entry)
    return result


def get_trigger_words(name):
    """执行期查询：返回某个 LoRA 的触发词列表（未命中缓存则强制重扫一次）。"""
    for entry in scan_loras():
        if entry["name"] == name:
            return list(entry["trigger_words"])
    for entry in scan_loras(force=True):
        if entry["name"] == name:
            return list(entry["trigger_words"])
    _LOG.warning("[asp] 触发词未找到: %s", name)
    return []
