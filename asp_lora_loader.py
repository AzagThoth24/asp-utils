"""多 LoRA 加载器（asp-utils）。

在一个节点里加载任意多个 LoRA（参考 rgthree Power Lora Loader 的执行机制：
逐行调用官方 LoraLoader 串行叠加），并扩展：

  * trigger_words 输出：已启用 LoRA 的触发词拼接，可直接接入提示词
  * 触发词来源与手动覆盖见 asp_server/lora_info.py
  * 前端 web/js/asp_lora_loader.js 提供 DOM widget 行 UI（旧画布与
    Nodes 2.0 双渲染兼容）、LoRA 组预设（新建/保存/切换/删除）

行数据存在 lora_data 这一个 STRING widget 里（JSON），不依赖动态输入校验
hack，API 格式工作流天然兼容：

  {"rows": [{"on": true, "lora": "xxx.safetensors",
             "strength_model": 1.0, "strength_clip": 1.0}],
   "preset": "预设名"}
"""

import json
import logging
import os

import folder_paths
import nodes

from .asp_server import lora_info

_LOG = logging.getLogger("asp")

NODE_NAME = "AspPowerLoraLoader"
CATEGORY = "asp-utils"

DEFAULT_LORA_DATA = {"rows": [], "preset": ""}


def parse_rows(lora_data):
    """解析 lora_data JSON 为规范 rows；非法输入返回空列表并告警。"""
    if not lora_data or not str(lora_data).strip():
        return []
    try:
        data = json.loads(lora_data)
    except (ValueError, TypeError):
        _LOG.warning("[asp] lora_data 不是合法 JSON，忽略全部行: %.120s", lora_data)
        return []
    if isinstance(data, dict):
        data = data.get("rows")
    if not isinstance(data, list):
        _LOG.warning("[asp] lora_data.rows 不是列表，忽略")
        return []
    rows = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            sm = float(row.get("strength_model", 1.0))
            sc = float(row.get("strength_clip", row.get("strength_model", 1.0)))
        except (TypeError, ValueError):
            continue
        rows.append({
            "on": bool(row.get("on", True)),
            "lora": str(row.get("lora") or "").strip(),
            "strength_model": sm,
            "strength_clip": sc,
        })
    return rows


def resolve_lora_name(name):
    """宽容解析 LoRA 文件名（与 rgthree get_lora_by_filename 同策略）。

    精确 → 忽略大小写 → 去扩展名 → basename（含去扩展名）→ 唯一子串；
    多个候选时告警并取第一个，找不到返回 None。
    """
    name = str(name or "").strip()
    if not name:
        return None
    names = list(folder_paths.get_filename_list("loras"))
    if name in names:
        return name
    low = name.lower()
    for candidate in names:  # 忽略大小写精确
        if candidate.lower() == low:
            return candidate
    for candidate in names:  # 去扩展名（无扩展名的名字）
        if os.path.splitext(candidate)[0].lower() == low:
            return candidate
    base_hits = [c for c in names
                 if os.path.basename(c).lower() == low
                 or os.path.splitext(os.path.basename(c))[0].lower() == low]
    if base_hits:
        if len(base_hits) > 1:
            _LOG.warning("[asp] LoRA 名 '%s' 命中 %d 个文件，取第一个: %s",
                         name, len(base_hits), base_hits[0])
        return base_hits[0]
    sub_hits = [c for c in names if low in c.lower()]
    if len(sub_hits) == 1:
        return sub_hits[0]
    if len(sub_hits) > 1:
        _LOG.warning("[asp] LoRA 名 '%s' 子串命中 %d 个文件，取第一个: %s",
                     name, len(sub_hits), sub_hits[0])
        return sub_hits[0]
    _LOG.warning("[asp] 找不到 LoRA: %s", name)
    return None


class AspPowerLoraLoader:
    """单节点多 LoRA 串行叠加，MODEL/CLIP 均可选，附触发词输出。"""

    NAME = NODE_NAME
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_data": ("STRING", {
                    "default": json.dumps(DEFAULT_LORA_DATA, ensure_ascii=False),
                    "tooltip": "LoRA 行数据 JSON，由前端 UI 维护；也可手写",
                }),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "trigger_words")
    OUTPUT_TOOLTIPS = ("叠加后的模型", "叠加后的 CLIP",
                       "已启用 LoRA 的触发词（逗号分隔），可串入提示词")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("在一个节点里加载多个 LoRA：逐行串行叠加（复用官方 "
                   "LoraLoader），支持每行独立开关与 Model/Clip 双强度；"
                   "第三个输出是已启用 LoRA 的触发词拼接，可直接接入提示词链。")

    def load(self, lora_data, model=None, clip=None):
        rows = parse_rows(lora_data)
        triggers = []
        seen = set()
        loader = nodes.LoraLoader()

        for row in rows:
            if not row["on"]:
                continue
            lora = row["lora"]
            if not lora or lora.lower() == "none":
                continue
            sm, sc = row["strength_model"], row["strength_clip"]
            resolved = resolve_lora_name(lora)
            if resolved is None:
                continue
            if model is None and clip is None:
                _LOG.warning("[asp] model 与 clip 均未连接，跳过全部 LoRA 加载")
                break
            if sm == 0 and sc == 0:
                _LOG.info("[asp] LoRA '%s' 双强度为 0，跳过", resolved)
            else:
                try:
                    model, clip = loader.load_lora(model, clip, resolved, sm, sc)
                except Exception as err:
                    _LOG.warning("[asp] LoRA '%s' 加载失败: %s", resolved, err)
                    continue
            for word in lora_info.get_trigger_words(resolved):
                if word not in seen:
                    seen.add(word)
                    triggers.append(word)

        return (model, clip, ", ".join(triggers))


NODE_CLASS_MAPPINGS = {NODE_NAME: AspPowerLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "多 LoRA 加载器 (asp)"}
