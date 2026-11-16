import json
import logging
import re

import torch

from .krea2_regional import Krea2RegionalPrompt

_LORA_CACHE = {}
MAX_PERSONS = 5


def _load_lora_sd(name):
    if name not in _LORA_CACHE:
        import folder_paths
        import comfy.utils
        path = folder_paths.get_full_path_or_raise("loras", name)
        _LORA_CACHE[name] = comfy.utils.load_torch_file(path, safe_load=True)
        logging.info("[RegionRectCanvas] cached lora sd: %s", name)
    return _LORA_CACHE[name]


def _encode_section(clip, text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _split_prompt_sections(text):
    """\u884c\u9996\u5e26 # \u7684\u884c = \u6bb5\u843d\u5206\u754c\u3002

    sections[0] = base \u6bb5\uff08\u7b2c\u4e00\u4e2a\u6807\u8bb0\u884c + \u5176\u6b63\u6587\uff0c\u542b\u524d\u5bfc\u6587\u672c\uff09\uff0c
    sections[1..] \u4f9d\u6b21\u5bf9\u5e94\u4eba\u72691..N\u3002\u6bb5\u843d\u4fdd\u7559\u81ea\u5df1\u7684 # \u6807\u8bb0\u884c\u3002
    """
    lines = (text or "").splitlines()
    markers = [i for i, l in enumerate(lines) if l.strip().startswith("#")]
    if not markers:
        return [("\n".join(lines)).strip("\n")]
    bounds = markers + [len(lines)]
    sections = ["\n".join(lines[:bounds[1]]).strip("\n")]
    for k in range(1, len(markers)):
        sections.append("\n".join(lines[bounds[k]:bounds[k + 1]]).strip("\n"))
    return sections


# ---------------------------------------------------------------------------
# 自动布局句（沿用旧 Krea2-Regional Builder 的 _layout_hint / _hint_clause /
# _looks_like_trigger / _zone_name）：按每个矩形生成一句自然语言方位提示，
# 注入 base 段。Qwen3-VL 编码器读取布局指引来决定"谁画在哪"，
# 注意力掩码负责"谁的 token 读哪段词"——两者配合是旧版稳定的根源。
# ---------------------------------------------------------------------------

_ZONES = [["upper left", "top", "upper right"],
          ["left side", "center", "right side"],
          ["lower left", "bottom", "lower right"]]

# 自动布局句开关。base 段已有手写定位描述时置 0：双重定位会把角色钉死
# （过拟合感）；提示词里没有任何方位描述时可置 1 让矩形自动补布局句。
LAYOUT_HINTS = 0


def _looks_like_trigger(word):
    """类触发词（含数字/下划线/大写）不能进 base 布局句，会被当画面文字渲染。"""
    if " " in word:
        return False
    return (any(ch.isdigit() for ch in word) or "_" in word
            or any(ch.isupper() for ch in word[1:]))


def _hint_clause(text):
    """取区域段第一个可读的多词短句，作为 base 布局句里的主体指代。"""
    clauses = [c.strip().strip('"') if not c.strip().startswith("the text")
               else c.strip() for c in text.split(",")]
    clauses = [re.sub(r"<[^>]*>", "", c).strip() for c in clauses if c.strip()]
    multi = [c for c in clauses if len(c.split()) >= 2]
    if multi:
        return multi[0]
    for c in clauses:
        if not _looks_like_trigger(c):
            return c
    return "a subject"


def _layout_hint(x, y, w, h, text):
    """按矩形生成一句方位提示（九宫格方位 + 面积比例 + 主体短句）。"""
    cx, cy = x + w / 2.0, y + h / 2.0
    col = 0 if cx < 1 / 3 else (1 if cx < 2 / 3 else 2)
    row = 0 if cy < 1 / 3 else (1 if cy < 2 / 3 else 2)
    zone = _ZONES[row][col]
    size = w * h
    scale = ("dominating the frame" if size > 0.5
             else "prominent" if size > 0.22 else "small")
    short = _hint_clause(text)
    if len(short) > 70:
        short = short[:67].rstrip() + "..."
    if zone == "center":
        return f"in the center of the image, {scale}: {short}"
    return f"in the {zone} of the image, {scale}: {short}"


class Krea2CharacterInfo:
    """\u5355\u4e2a\u4eba\u7269\u7684\u5168\u90e8\u8bbe\u7f6e\uff1aLoRA\u3001\u5f3a\u5ea6\u3001\u4e8c\u91c7\u7ec6\u5316\u9879\uff08\u4eba\u4f53/\u773c\u775b/\u8138\u90e8/\u624b\u90e8\uff09\u4e0e\u7ec6\u5316\u964d\u566a\u3002

    \u8f93\u51fa\u63a5\u5165 RegionRectCanvas \u7684\u4eba\u7269\u4fe1\u606fN \u8f93\u5165\u53e3\uff1b\u4e00\u4e2a\u4eba\u7269\u4e00\u4e2a\u8282\u70b9\u3002
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        choices = ["None"] + folder_paths.get_filename_list("loras")
        return {"required": {
            "lora": (choices,),
            "\u5f3a\u5ea6": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
            "\u7ec6\u5316\u4eba\u4f53": ("BOOLEAN", {"default": False, "label_on": "\u5f00", "label_off": "\u5173"}),
            "\u7ec6\u5316\u773c\u775b": ("BOOLEAN", {"default": True, "label_on": "\u5f00", "label_off": "\u5173"}),
            "\u7ec6\u5316\u8138\u90e8": ("BOOLEAN", {"default": True, "label_on": "\u5f00", "label_off": "\u5173"}),
            "\u7ec6\u5316\u624b\u90e8": ("BOOLEAN", {"default": False, "label_on": "\u5f00", "label_off": "\u5173"}),
            "\u7ec6\u5316\u964d\u566a": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("KREA2_PERSON",)
    RETURN_NAMES = ("\u4eba\u7269\u4fe1\u606f",)
    FUNCTION = "build"
    CATEGORY = "mask/krea2_region"
    DESCRIPTION = "\u4eba\u7269\u4fe1\u606f\uff1aLoRA + \u5f3a\u5ea6 + \u5404\u90e8\u4f4d\u7ec6\u5316\u5f00\u5173\u4e0e\u964d\u566a\u529b\u5ea6\u3002"

    def build(self, **kw):
        info = {
            "lora": kw["lora"],
            "strength": float(kw["\u5f3a\u5ea6"]),
            "refine_body": bool(kw["\u7ec6\u5316\u4eba\u4f53"]),
            "refine_eyes": bool(kw["\u7ec6\u5316\u773c\u775b"]),
            "refine_face": bool(kw["\u7ec6\u5316\u8138\u90e8"]),
            "refine_hands": bool(kw["\u7ec6\u5316\u624b\u90e8"]),
            "denoise": float(kw["\u7ec6\u5316\u964d\u566a"]),
        }
        return (info,)


class RegionRectCanvas:
    """\u591a\u4eba\u4e13\u7528\uff082-5 \u4eba\uff09\uff1a\u753b\u5e03\u62d6\u77e9\u5f62 = \u5404\u4eba\u7269\u533a\u57df\u3002

    v2\uff1a\u603b\u63d0\u793a\u8bcd\uff08#\u6807\u8bb0\u884c\u5206\u8282\uff09+ \u4eba\u7269\u4fe1\u606fN \u8282\u70b9\uff08LoRA/\u5f3a\u5ea6/\u7ec6\u5316\uff09\u5168\u90e8\u4ece\u5916\u90e8\u8f93\u5165\uff0c
    \u6267\u884c\u65f6\u5185\u90e8\u5207\u6bb5\u3001CLIP \u7f16\u7801\u3001\u6253\u5305\u533a\u57df\u94fe\uff08vendored Krea2RegionalPrompt\uff09\uff0c
    \u4e00\u6b21\u8f93\u51fa ApplyRegional / RegionalDetailer \u8981\u7684\u5168\u90e8\u4e1c\u897f\u3002

    rectN = \u4eba\u7269N\uff1b\u77e9\u5f621 = \u4eba\u72691 = \u524d\u666f\uff08\u533a\u57df\u94fe\u9996\uff0cexclusive \u91cd\u53e0\u5e26\u5f52\u5b83\uff09\u3002
    \u672a\u4f7f\u7528\u7684\u4eba\u7269\u69fd\u8f93\u51fa\u96f6\u63a9\u7801 + \u7a7a LoRA \u5305\uff0c\u65e0\u526f\u4f5c\u7528\uff1b
    \u4e0d\u63a5 clip/\u603b\u63d0\u793a\u8bcd\u65f6\u533a\u57df\u94fe\u8f93\u51fa None\uff0c\u884c\u4e3a\u4e0e v1 \u76f8\u540c\u3002
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "width": ("INT", {"default": 2048, "min": 64, "max": 8192,
                              "step": 8}),
            "height": ("INT", {"default": 1136, "min": 64, "max": 8192,
                               "step": 8}),
            # \u753b\u5e03\u62d6\u6846\u6570\u636e\uff0c\u524d\u7aef\u7ef4\u62a4\uff0c\u4e0d\u4f5c\u4e3a\u7528\u6237\u8f93\u5165
            "rects_data": ("STRING",
                           {"default": json.dumps(
                               {"rects": [
                                   {"x": 0.06, "y": 0.02, "w": 0.30, "h": 0.96},
                                   {"x": 0.38, "y": 0.02, "w": 0.30, "h": 0.96},
                                   {"x": 0.70, "y": 0.02, "w": 0.28, "h": 0.96}]}),
                            "multiline": False}),
            # \u653e\u5728\u6700\u540e\uff1a\u65e7\u5de5\u4f5c\u6d41\u6309\u4f4d\u7f6e\u8bfb widgets_values \u65f6\u4e0d\u4e32\u4f4d
            "\u603b\u63d0\u793a\u8bcd": ("STRING", {
                "default": "#masterpiece\nmasterpiece, very aesthetic, ultra detailed.\n\n"
                           "#1girl\n(char_mikazuki_default:1.0), \u4eba\u72691\u7684\u6bb5\u3002\n\n"
                           "#1girl\n\u4eba\u72692\u7684\u6bb5\u3002",
                "multiline": True}),
        }
        optional = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
        }
        for i in range(1, MAX_PERSONS + 1):
            optional[f"\u4eba\u7269\u4fe1\u606f{i}"] = ("KREA2_PERSON",)
        # \u65b0\u8f93\u5165\u53ea\u80fd\u8ffd\u52a0\u5728\u6700\u540e\uff1a\u63d2\u8fdb\u4e2d\u95f4\u4f1a\u628a\u65e7\u5de5\u4f5c\u6d41\u91cc
        # \u6309\u69fd\u4f4d\u8bb0\u5f55\u7684\u8fde\u7ebf\u6574\u4f53\u9876\u540e\u4e00\u4f4d\uff08\u5168\u90e8\u63a5\u9519\u4f4d\uff09
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "KREA2_REGIONS", "CONDITIONING", "STRING",
                    "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model (原样透传)", "regions (区域链)",
                    "base_cond (基础段)", "refines_json (细化配置)",
                    "full_prompt (全文, 勿作采样正词)", "base_prompt",
                    "prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5")
    FUNCTION = "draw"
    CATEGORY = "mask/krea2_region"
    DESCRIPTION = ("\u591a\u4eba\u533a\u57df\uff1a\u753b\u5e03\u62d6\u77e9\u5f62 + \u603b\u63d0\u793a\u8bcd\u81ea\u52a8\u5207\u6bb5\u7f16\u7801 + \u4eba\u7269\u4fe1\u606f\uff08LoRA/\u5f3a\u5ea6/\u7ec6\u5316\uff09\u3002"
                   "\u8f93\u51fa \u5168\u5c40\u6a21\u578b + \u533a\u57df\u94fe + \u57fa\u7840\u6bb5\u6761\u4ef6 + \u7ec6\u5316\u914d\u7f6e + \u5206\u6bb5\u63d0\u793a\u8bcd\u63a5\u51fa\u53e3\u3002")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def draw(self, **kw):
        W, H = int(kw["width"]), int(kw["height"])
        rects = []
        try:
            data = json.loads(kw["rects_data"] or "{}")
            raw = data.get("rects", []) if isinstance(data, dict) else []
            for r in raw[:MAX_PERSONS]:
                x = min(1.0, max(0.0, float(r.get("x", 0.0))))
                y = min(1.0, max(0.0, float(r.get("y", 0.0))))
                w = min(1.0 - x, max(0.0, float(r.get("w", 0.0))))
                h = min(1.0 - y, max(0.0, float(r.get("h", 0.0))))
                rects.append((x, y, w, h))
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            logging.warning("[RegionRectCanvas] rects_data unparsable")
            rects = []
        while len(rects) < MAX_PERSONS:
            rects.append((0.0, 0.0, 0.0, 0.0))

        masks = []
        for i in range(MAX_PERSONS):
            m = torch.zeros((H, W), dtype=torch.float32)
            x, y, w, h = rects[i]
            x0 = max(0, min(int(round(x * W)), W))
            y0 = max(0, min(int(round(y * H)), H))
            x1 = max(0, min(int(round((x + w) * W)), W))
            y1 = max(0, min(int(round((y + h) * H)), H))
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = 1.0
            masks.append(m)

        # \u6bcf\u4eba\u7269\u4fe1\u606f\u5168\u90e8\u6765\u81ea\u4eba\u7269\u4fe1\u606f\u8282\u70b9\uff08lora/\u5f3a\u5ea6/\u7ec6\u5316\uff09\uff0c\u6ca1\u6709\u5219\u8be5\u69fd\u4e3a\u7a7a
        persons = []
        refines = {}
        for i in range(MAX_PERSONS):
            info = kw.get(f"\u4eba\u7269\u4fe1\u606f{i + 1}")
            name, strength, refine = "None", 1.0, None
            if isinstance(info, dict):
                name = info.get("lora") or "None"
                strength = float(info.get("strength", 1.0))
                refine = {k: info[k] for k in
                          ("refine_body", "refine_eyes", "refine_face",
                           "refine_hands", "denoise")}
                refines[str(i + 1)] = refine
            persons.append((name, strength, refine))

        bundles = []
        for i in range(MAX_PERSONS):
            name, strength, refine = persons[i]
            bundle = []
            if name != "None":
                try:
                    bundle = [{"sd": _load_lora_sd(name),
                               "strength": strength, "label": name}]
                except Exception as e:
                    logging.warning("[RegionRectCanvas] lora '%s' failed: %s",
                                    name, e)
            bundles.append(bundle)

        # model 原样透传。人物 LoRA 只走两条正路：一采的区域路由（regions 链 →
        # Krea2ApplyRegional 按 token 掩码生效）和 detailer 的逐区域加载。
        # 绝不全局叠加到 model 上——那会把所有人物特征搅进整张图的每个区域
        # （多人互相串脸的根源）。
        model_out = kw.get("model")

        # \u603b\u63d0\u793a\u8bcd\u5207\u6bb5 \u2192 \u7f16\u7801 \u2192 \u533a\u57df\u94fe\uff08\u590d\u7528 vendored Regional \u5305\u7684\u6253\u5305\u903b\u8f91\uff09
        base_cond = None
        regions = None
        refines_json = "{}"
        full_prompt = None
        base_prompt = None
        prompts = [None] * MAX_PERSONS
        clip = kw.get("clip")
        master = kw.get("\u603b\u63d0\u793a\u8bcd")
        if clip is not None and (master or "").strip():
            sections = _split_prompt_sections(master)
            # \u81ea\u52a8\u5e03\u5c40\u53e5\uff1a\u6309\u5f53\u524d\u77e9\u5f62\u751f\u6210\u65b9\u4f4d\u63d0\u793a\u8ffd\u52a0\u8fdb base \u6bb5
            # \uff08\u65e7 Builder \u7684 layout_in_base=position hints \u884c\u4e3a\uff1a\u77e9\u5f62\u6bcf\u6b21\u53d8\u52a8
            # \u4e0b\u4e00\u6b21\u6267\u884c\u81ea\u52a8\u8ddf\u968f\uff0c\u6c38\u4e0d\u4e0e\u77e9\u5f62\u9519\u4f4d\uff09
            # LAYOUT_HINTS=0\uff1abase \u5df2\u6709\u624b\u5199\u5b9a\u4f4d\u63cf\u8ff0\u65f6\u5173\u95ed\uff0c\u907f\u514d\u53cc\u91cd\u5b9a\u4f4d\u628a\u89d2\u8272\u9489\u6b7b
            hints = []
            if LAYOUT_HINTS:
                for i in range(MAX_PERSONS):
                    if i + 1 >= len(sections) or not sections[i + 1].strip():
                        continue
                    if masks[i].sum().item() == 0:
                        continue
                    x, y, w, h = rects[i]
                    body = re.sub(r"(?m)^\s*#[^\n]*$", "", sections[i + 1]).strip()
                    if body:
                        hints.append(_layout_hint(x, y, w, h, body))
                if hints:
                    head = sections[0].strip().rstrip(".")
                    sections[0] = ((head + ". " + ". ".join(hints)) if head
                                   else ". ".join(hints))
            base_cond = _encode_section(clip, sections[0]) if sections[0] else None
            packer = Krea2RegionalPrompt()

            for i in range(MAX_PERSONS):
                name, strength, refine = persons[i]
                sec = sections[i + 1] if i + 1 < len(sections) else None
                prompts[i] = sec
                if not sec or not sec.strip():
                    continue
                if masks[i].sum().item() == 0:
                    logging.warning("[RegionRectCanvas] \u4eba\u7269%d \u65e0\u77e9\u5f62\uff0c\u8df3\u8fc7\u533a\u57df\u6bb5", i + 1)
                    continue
                regions = packer.build(
                    conditioning=_encode_section(clip, sec),
                    mask=masks[i],
                    loras=bundles[i] or None,
                    prev_regions=regions,
                )[0]
                # \u7ec6\u5316\u914d\u7f6e\u5199\u8fdb\u533a\u57df\u5b57\u5178\uff0cvendored \u7684 Krea2RegionalDetailer \u9010\u533a\u57df\u6d88\u8d39
                if refine and regions:
                    regions[-1]["refine"] = refine
            refines_json = json.dumps(refines, ensure_ascii=False)
            full_prompt = ",".join(sections) if sections else None
            base_prompt = sections[0] if sections and sections[0] else None

        return tuple([model_out, regions, base_cond, refines_json, full_prompt,
                      base_prompt] + prompts)


NODE_CLASS_MAPPINGS = {
    "RegionRectCanvas": RegionRectCanvas,
    "Krea2CharacterInfo": Krea2CharacterInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RegionRectCanvas": "\u591a\u4eba\u533a\u57df\u753b\u5e03\uff082-5 \u77e9\u5f62 + \u63d0\u793a\u8bcd + \u4eba\u7269\u4fe1\u606f\uff09",
    "Krea2CharacterInfo": "\u4eba\u7269\u4fe1\u606f\uff08LoRA+\u5f3a\u5ea6+\u7ec6\u5316\uff09",
}
