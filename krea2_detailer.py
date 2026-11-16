"""Krea2 Regional Detailer \u2014 vendored from ComfyUI-Krea2-Regional + \u6269\u5c55.

\u5728\u539f\u7248\u57fa\u7840\u4e0a\u589e\u52a0 per-person \u7ec6\u5316\uff1a\u533a\u57df\u5b57\u5178\uff08KREA2_REGIONS \u7684\u6bcf\u4e00\u9879\uff09\u53ef\u4ee5\u643a\u5e26
``refine`` \u914d\u7f6e\uff08\u7531 RegionRectCanvas \u4ece\u4eba\u7269\u4fe1\u606f\u8282\u70b9\u5199\u5165\uff09::

    {"refine_body":  bool, "refine_eyes":  bool,
     "refine_face":  bool, "refine_hands": bool,
     "denoise":      float}

\u6709 ``refine`` \u7684\u533a\u57df\u53ea\u7ec6\u5316\u88ab\u6253\u5f00\u7684\u90e8\u4f4d\uff0c\u6bcf\u4e2a\u90e8\u4f4d\u7528\u5404\u81ea\u7684\u68c0\u6d4b\u6e90\u548c\u8be5\u4eba\u7269\u7684
\u7ec6\u5316\u964d\u566a\u529b\u5ea6\uff1b\u56db\u9879\u5168\u5173\u7684\u533a\u57df\u5b8c\u5168\u4e0d\u52a8\u3002\u6ca1\u6709 ``refine`` \u7684\u533a\u57df\u8d70\u539f\u7248\u884c\u4e3a
\uff08detection_masks / segs \u5355\u4e00\u68c0\u6d4b\u6e90 + \u5168\u5c40 denoise\uff09\u3002

\u53e6\u542b\u56db\u4e2a\u72ec\u7acb\u90e8\u4f4d\u7ec6\u5316\u8282\u70b9\uff08\u4eba\u4f53/\u8138\u90e8/\u773c\u775b/\u624b\u90e8\uff09\uff0c\u4f9b\u5de5\u4f5c\u6d41\u4e32\u94fe\u4f7f\u7528\u3002
"""

import copy
import logging

import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.sd
import nodes
import math


# ---------------------------------------------------------------------------
# \u539f caption_utils \u91cc\u7684\u4e24\u4e2a\u5c0f\u5de5\u5177\uff08\u5185\u8054\uff0c\u6446\u8131\u5bf9\u6574\u4e2a caption_utils \u7684\u4f9d\u8d56\uff09
# ---------------------------------------------------------------------------


def _gaussian_kernel1d(sigma: float, device):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()), radius


def _grow_feather(m, grow_px, feather_px):
    """(H, W) float mask -> dilated + gaussian-feathered."""
    m = m[None, None]
    if grow_px > 0:
        k = 2 * int(grow_px) + 1
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
    if feather_px > 0:
        k, r = _gaussian_kernel1d(float(feather_px) / 2.0, m.device)
        m = F.conv2d(m, k.view(1, 1, 1, -1), padding=(0, r))
        m = F.conv2d(m, k.view(1, 1, -1, 1), padding=(r, 0))
    return m[0, 0].clamp(0, 1)


# ---------------------------------------------------------------------------
# \u4ee5\u4e0b\u4e0e\u539f\u7248 krea2_detailer.py \u76f8\u540c
# ---------------------------------------------------------------------------


def _mask_bbox(mask, thresh=0.5):
    ys, xs = torch.where(mask > thresh)
    if xs.numel() == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _round16(v, lo=64):
    return max(lo, int(round(float(v) / 16.0)) * 16)


def _nhwc_resize(img, h, w):
    x = img.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(h, w), mode="bicubic", antialias=True)
    return x.permute(0, 2, 3, 1).clamp(0, 1)


def _to_hw(m, H, W):
    m = torch.as_tensor(m, dtype=torch.float32).cpu()
    while m.dim() > 2:
        m = m[0]
    if tuple(m.shape) != (H, W):
        m = F.interpolate(m[None, None], size=(H, W), mode="bilinear",
                          align_corners=False)[0, 0]
    return m.clamp(0, 1)


def _segs_to_masks(segs, H, W):
    """Duck-typed Impact-Pack SEGS -> list of (H, W) float masks."""
    out = []
    try:
        seg_list = segs[1]
    except (TypeError, IndexError):
        return out
    for s in seg_list or []:
        try:
            cm = getattr(s, "cropped_mask", None)
            cr = getattr(s, "crop_region", None)
            if cm is None or cr is None:
                continue
            m = torch.as_tensor(cm, dtype=torch.float32).cpu()
            while m.dim() > 2:
                m = m[0]
            x1, y1, x2, y2 = [int(v) for v in cr]
            x1 = max(0, min(x1, W)); x2 = max(x1 + 1, min(x2, W))
            y1 = max(0, min(y1, H)); y2 = max(y1 + 1, min(y2, H))
            if m.shape != (y2 - y1, x2 - x1):
                m = F.interpolate(m[None, None], size=(y2 - y1, x2 - x1),
                                  mode="bilinear", align_corners=False)[0, 0]
            full = torch.zeros(H, W)
            full[y1:y2, x1:x2] = m.clamp(0, 1)
            out.append(full)
        except Exception as e:
            logging.warning("[Krea2Detailer] skipped a SEG: %s", e)
    return out


def _apply_crop_shift(model_patcher, w, h,
                      base_shift=0.5, max_shift=1.15,
                      min_resolution=256, max_resolution=1280):
    """Apply Krea 2 RAW's resolution-derived timestep-shift ``mu``."""
    try:
        import comfy.model_sampling as ms

        current = None
        try:
            current = model_patcher.get_model_object("model_sampling")
        except Exception:
            current = getattr(getattr(model_patcher, "model", None),
                              "model_sampling", None)
        if current is None or not isinstance(current, ms.ModelSamplingFlux):
            return model_patcher

        x1 = (min_resolution // 16) ** 2
        x2 = (max_resolution // 16) ** 2
        seq = (max(16, int(w)) // 16) * (max(16, int(h)) // 16)
        seq = max(x1, min(seq, x2))
        mu = base_shift + (seq - x1) * (max_shift - base_shift) / (x2 - x1)

        m = model_patcher.clone()
        sampling = copy.deepcopy(current)
        sampling.set_parameters(shift=float(mu))
        m.add_object_patch("model_sampling", sampling)
        logging.info("[Krea2Detailer] RAW crop shift: %dx%d, tokens=%d, "
                     "mu=%.4f", w, h, seq, mu)
        return m
    except Exception as e:
        logging.info("[Krea2Detailer] per-crop shift unavailable "
                     "(%s) \u2014 using the model's shift as-is.", e)
        return model_patcher


def _center(mask):
    ys, xs = torch.where(mask > 0.5)
    if xs.numel() == 0:
        return None
    return float(xs.float().mean()), float(ys.float().mean())


def _match_detections(dets, region_masks):
    """Greedy 1:1 assignment: region index -> detection mask (or None)."""
    n_r, n_d = len(region_masks), len(dets)
    assign = {i: None for i in range(n_r)}
    if n_d == 0:
        return assign
    rbins = [rm > 0.5 for rm in region_masks]
    dbins = [d > 0.5 for d in dets]
    dareas = [max(int(d.sum().item()), 1) for d in dbins]
    rareas = [max(int(r.sum().item()), 1) for r in rbins]
    scored = []
    for i in range(n_r):
        for j in range(n_d):
            if dareas[j] < 64:
                continue
            inter = int((rbins[i] & dbins[j]).sum().item())
            if inter == 0:
                continue
            scored.append((inter / dareas[j] * (inter / rareas[i]), i, j))
    used_r, used_d = set(), set()
    for score, i, j in sorted(scored, reverse=True):
        if i in used_r or j in used_d:
            continue
        assign[i] = dets[j]
        used_r.add(i); used_d.add(j)
    left_d = [j for j in range(n_d) if j not in used_d and dareas[j] >= 64]
    left_r = [i for i in range(n_r) if i not in used_r]
    if left_d and left_r:
        pairs = []
        for i in left_r:
            rc = _center(region_masks[i])
            rb = _mask_bbox(region_masks[i])
            if rc is None or rb is None:
                continue
            rw, rh = rb[2] - rb[0], rb[3] - rb[1]
            max_dist = max(32.0, 0.75 * (rw * rw + rh * rh) ** 0.5)
            for j in left_d:
                dc = _center(dets[j])
                if dc is None:
                    continue
                dist = ((rc[0] - dc[0]) ** 2 + (rc[1] - dc[1]) ** 2) ** 0.5
                if dist <= max_dist:
                    pairs.append((dist, i, j))
        for dist, i, j in sorted(pairs):
            if i in used_r or j in used_d:
                continue
            assign[i] = dets[j]
            used_r.add(i); used_d.add(j)
            logging.info("[Krea2Detailer] region %d matched a drifted "
                         "subject %.0f px from its box", i + 1, dist)
    return assign


def _build_detail_plan(valid_regions, part, dets, detection_mode, denoise):
    """Assign detections once, then apply each person's detail settings.

    Include disabled people in ownership matching: their detections still
    belong to them and must not be repainted with a neighbour's LoRA.
    """
    region_masks = [rm for _, _, rm in valid_regions]
    assign = (_match_detections(dets, region_masks) if detection_mode else {})
    has_refine_cfg = any(r.get("refine") for _, r, _ in valid_regions)
    plan = []
    for local_ridx in reversed(range(len(valid_regions))):
        ridx, reg, region_mask = valid_regions[local_ridx]
        rc = (reg or {}).get("refine") or {}
        if has_refine_cfg and not rc.get(f"refine_{part}", True):
            continue
        r_denoise = float(rc.get("denoise", denoise)) if has_refine_cfg else denoise
        if r_denoise <= 0:
            continue
        m = assign.get(local_ridx) if detection_mode else region_mask
        if m is None:
            logging.info("[Krea2Detailer] region %d: no %s detection matched - skipped",
                         ridx + 1, part)
            continue
        plan.append((ridx, m, r_denoise, detection_mode))
    return plan


# \u90e8\u4f4d\u987a\u5e8f = \u5408\u6210\u987a\u5e8f\uff1a\u5927\u90e8\u4f4d\u5148\u753b\uff0c\u5c0f\u90e8\u4f4d\uff08\u773c/\u624b\uff09\u540e\u753b\u3001\u540e\u753b\u7684\u8d62
_PART_ORDER = ("body", "face", "eyes", "hands")


def _detail_one_part(image, model, vae, regions, part, dets, detection_mode,
                     seed, steps, cfg, sampler_name, scheduler, denoise,
                     base_dict, negative, crop_padding, detail_size,
                     feather_px, mask_grow_px, shift):
    """\u5bf9\u4e00\u4e2a\u90e8\u4f4d\u6267\u884c\u4e00\u8f6e \u68c0\u6d4b\u2192\u5339\u914d\u2192\u9010\u4eba crop \u91cd\u91c7\u6837\u2192\u5408\u6210\u3002

    \u533a\u57df\u5b57\u5178\u5e26 refine \u914d\u7f6e\u65f6\uff1a\u53ea\u7ec6\u5316 refine_<part> \u6253\u5f00\u7684\u4eba\u7269\uff0c
    \u964d\u566a\u7528\u8be5\u4eba\u7269\u7684\u300e\u7ec6\u5316\u964d\u566a\u300f\uff1b\u672a\u6253\u5f00\u7684\u4eba\u7269\u8df3\u8fc7\u3002
    \u65e0 refine \u914d\u7f6e\uff08\u65e7\u884c\u4e3a\uff09\uff1a\u6240\u6709\u6709\u4eba\u7269\u63a9\u7801\u7684\u533a\u57df\u90fd\u7ec6\u5316\uff0c\u7528\u5168\u5c40\u964d\u566a\u3002
    \u8fd4\u56de (\u8f93\u51fa\u56fe\u50cf, \u5e94\u7528\u4e86\u7ec6\u5316\u7684\u8054\u5408\u63a9\u7801)\u3002
    """
    B, H, W, _ = image.shape
    union = torch.zeros(B, H, W)
    out = image.clone()
    if not regions or denoise <= 0.0:
        return out, union

    valid_regions = [(idx, r, _to_hw(r["mask"], H, W))
                     for idx, r in enumerate(regions)
                     if r and r.get("mask") is not None]
    if not valid_regions:
        return out, union

    plan = _build_detail_plan(valid_regions, part, dets, detection_mode, denoise)
    if not plan:
        return out, union

    for b in range(B):
        # pidx 逐 crop 递增，保证采样种子按 (区域, 部位, crop) 去重
        for pidx, (ridx, m, r_denoise, detected) in enumerate(plan):
            reg = regions[ridx] or {}
            cond = reg.get("cond")
            if cond is None:
                continue
            bbox = _mask_bbox(m)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            pad = int(round(crop_padding * max(x1 - x0, y1 - y0)))
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
            cw, ch = x1 - x0, y1 - y0
            if cw < 16 or ch < 16:
                continue

            scale = max(1.0, detail_size / float(max(cw, ch)))
            tw = _round16(cw * scale)
            th = _round16(tw * ch / cw)
            crop = image[b:b + 1, y0:y1, x0:x1, :3]
            crop_up = _nhwc_resize(crop, th, tw)

            sample_mm = m[y0:y1, x0:x1]
            grow = mask_grow_px if detected else 0
            if grow > 0 or feather_px > 0:
                sample_mm = _grow_feather(sample_mm, grow, feather_px)
            noise_mask = F.interpolate(
                sample_mm[None, None], size=(th, tw), mode="bilinear",
                align_corners=False).clamp(0, 1)
            latent = {"samples": vae.encode(crop_up),
                      "noise_mask": noise_mask}

            m_model = model
            if str(shift).startswith("auto"):
                m_model = _apply_crop_shift(m_model, tw, th)
            for e in reg.get("loras") or []:
                try:
                    m_model = comfy.sd.load_lora_for_models(
                        m_model, None, e["sd"],
                        float(e.get("strength", 1.0)), 0.0)[0]
                except Exception as err:
                    logging.warning(
                        "[Krea2Detailer] lora '%s' failed to apply: %s",
                        e.get("label", "?"), err)

            pos = [[cond, dict(base_dict)]]
            neg = negative
            if neg is None:
                nd = dict(base_dict)
                if isinstance(nd.get("pooled_output"), torch.Tensor):
                    nd["pooled_output"] = torch.zeros_like(
                        nd["pooled_output"])
                neg = [[torch.zeros_like(cond), nd]]

            import time as _time
            _t0 = _time.time()
            samples = nodes.common_ksampler(
                m_model, seed + ridx + _PART_ORDER.index(part) + pidx,
                steps, cfg, sampler_name,
                scheduler, pos, neg, latent, denoise=r_denoise)[0]
            det = vae.decode(samples["samples"])
            if det.dim() == 5:
                det = det.reshape(-1, det.shape[-3], det.shape[-2],
                                  det.shape[-1])[:1]
            if det.shape[1] != th or det.shape[2] != tw:
                det = det[:, :th, :tw, :]
            det = _nhwc_resize(det[:, :, :, :3], ch, cw).to(out)

            mm = sample_mm
            mm4 = mm.clamp(0, 1).to(out)[None, :, :, None]
            out[b:b + 1, y0:y1, x0:x1, :3] = \
                det * mm4 + out[b:b + 1, y0:y1, x0:x1, :3] * (1 - mm4)
            union[b, y0:y1, x0:x1] = torch.maximum(
                union[b, y0:y1, x0:x1], mm.clamp(0, 1).cpu())
            logging.info("[Krea2Detailer] region %d (%s): %dx%d crop "
                         "sampled at %dx%d in %.1fs", ridx + 1, part,
                         cw, ch, tw, th, _time.time() - _t0)

    return out, union


class Krea2RegionalDetailer:
    """Detect -> match -> per-subject resample -> composite\uff08\u591a\u90e8\u4f4d\u4e00\u6b21\u5b8c\u6210\uff09\u3002"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL", {"tooltip": "\u57fa\u7840\u6a21\u578b\uff08\u975e Apply Regional \u8f93\u51fa\uff09\u3002"}),
                "vae": ("VAE",),
                "regions": ("KREA2_REGIONS", {"tooltip": "\u6765\u81ea\u591a\u4eba\u533a\u57df\u753b\u5e03\u7684\u533a\u57df\u94fe\uff0c"
                             "\u643a\u5e26\u6bcf\u4e2a\u4eba\u7269\u7684\u6761\u4ef6\u3001\u63a9\u7801\u3001LoRA \u4e0e\u7ec6\u5316\u914d\u7f6e\u3002"}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0,
                                  "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "\u5168\u5c40\u515c\u5e95\u964d\u566a\u3002\u5e26\u7ec6\u5316\u914d\u7f6e\u7684\u4eba\u7269\u7528\u5176\u300e\u7ec6\u5316\u964d\u566a\u300f\u3002"}),
            },
            "optional": {
                "detection_masks": ("MASK", {"tooltip": "\u65e7\u7248\u517c\u5bb9\uff1a\u4e3b\u4f53\u8f6e\u5ed3\u6279\u91cf\u63a9\u7801\u3002"}),
                "segs": ("SEGS", {"tooltip": "\u65e7\u7248\u517c\u5bb9\uff1a\u8138\u578b\u68c0\u6d4b SEGS\u3002"}),
                "person_segs": ("SEGS", {"tooltip": "\u4eba\u4f53\u68c0\u6d4b\uff08person_yolov8m \u7b49\uff09\u2192 \u7ec6\u5316\u4eba\u4f53\u3002"}),
                "face_segs": ("SEGS", {"tooltip": "\u8138\u90e8\u68c0\u6d4b\uff08face_yolov8m \u7b49\uff09\u2192 \u7ec6\u5316\u8138\u90e8\u3002"}),
                "eye_segs": ("SEGS", {"tooltip": "\u773c\u775b\u68c0\u6d4b\uff08eye_yolov8m \u7b49\uff09\u2192 \u7ec6\u5316\u773c\u775b\u3002"}),
                "hand_segs": ("SEGS", {"tooltip": "\u624b\u90e8\u68c0\u6d4b\uff08hand_yolov8m \u7b49\uff09\u2192 \u7ec6\u5316\u624b\u90e8\u3002"}),
                "base_conditioning": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "crop_padding": ("FLOAT", {"default": 0.25, "min": 0.0,
                    "max": 1.0, "step": 0.05}),
                "detail_size": ("INT", {"default": 1024, "min": 256,
                    "max": 4096, "step": 16}),
                "feather_px": ("INT", {"default": 24, "min": 0, "max": 256}),
                "mask_grow_px": ("INT", {"default": 8, "min": 0, "max": 128}),
                "shift": (["model default", "auto (RAW only, per-crop)"],
                    {"default": "model default"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "applied_masks")
    FUNCTION = "detail"
    CATEGORY = "mask/krea2_region"

    def detail(self, image, model, vae, regions, seed, steps, cfg,
               sampler_name, scheduler, denoise, detection_masks=None,
               segs=None, person_segs=None, face_segs=None, eye_segs=None,
               hand_segs=None, base_conditioning=None, negative=None,
               crop_padding=0.25, detail_size=1024, feather_px=24,
               mask_grow_px=8, shift="model default"):
        B, H, W, _ = image.shape
        union = torch.zeros(B, H, W)
        if not regions or denoise <= 0.0:
            return (image, union)
        out = image

        base_dict = {}
        try:
            if base_conditioning and len(base_conditioning[0]) > 1:
                base_dict = dict(base_conditioning[0][1])
        except (TypeError, IndexError):
            pass

        # \u5404\u90e8\u4f4d\u68c0\u6d4b\u6e90
        part_sources = {}
        if person_segs is not None:
            part_sources["body"] = _segs_to_masks(person_segs, H, W)
        if face_segs is not None:
            part_sources["face"] = _segs_to_masks(face_segs, H, W)
        if eye_segs is not None:
            part_sources["eyes"] = _segs_to_masks(eye_segs, H, W)
        if hand_segs is not None:
            part_sources["hands"] = _segs_to_masks(hand_segs, H, W)

        legacy_dets = []
        detection_mode = False
        if detection_masks is not None:
            dm = detection_masks
            if dm.dim() == 2:
                dm = dm[None]
            legacy_dets = [_to_hw(dm[j], H, W) for j in range(dm.shape[0])]
        elif segs is not None:
            legacy_dets = _segs_to_masks(segs, H, W)
        if legacy_dets:
            detection_mode = True
        elif (detection_masks is not None or segs is not None) and not part_sources:
            logging.warning("[Krea2Detailer] detector input connected but "
                            "yielded no usable masks \u2014 nothing detailed.")
            return (out, union)

        # \u6709\u90e8\u4f4d\u68c0\u6d4b\u6e90\u5c31\u6309\u90e8\u4f4d\u9010\u8f6e\u7ec6\u5316\uff08\u6bcf\u8f6e\u7528 _detail_one_part \u7684\u8ba1\u5212\u903b\u8f91\uff09
        base_dict_pass = base_dict
        neg_pass = negative

        def run_part(part, dets, default_denoise):
            nonlocal out, union
            res = _detail_one_part(
                out, model, vae, regions, part, dets, detection_mode or bool(dets),
                seed, steps, cfg, sampler_name, scheduler, default_denoise,
                base_dict_pass, neg_pass, crop_padding, detail_size,
                feather_px, mask_grow_px, shift)
            out, u = res[0], res[1]
            union[:, :, :] = torch.maximum(union, u)

        if part_sources:
            for part in _PART_ORDER:
                if part in part_sources:
                    run_part(part, part_sources[part], denoise)
        elif legacy_dets:
            run_part("face", legacy_dets, denoise)
        else:
            run_part("face", [], denoise)

        return (out, union)


class _PartDetailerBase:
    PART = "face"
    DEFAULT_DENOISE = 0.30

    @classmethod
    def INPUT_TYPES(cls):
        part_cn = {"body": "\u4eba\u4f53", "face": "\u8138\u90e8",
                   "eyes": "\u773c\u775b", "hands": "\u624b\u90e8"}[cls.PART]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL", {"tooltip": "\u57fa\u7840\u6a21\u578b\uff08\u975e Apply Regional \u8f93\u51fa\uff09\u3002"}),
                "vae": ("VAE",),
                "regions": ("KREA2_REGIONS", {"tooltip": "\u591a\u4eba\u533a\u57df\u753b\u5e03\u7684\u533a\u57df\u94fe\uff0c"
                           "\u643a\u5e26\u6bcf\u4e2a\u4eba\u7269\u7684\u7ec6\u5316\u5f00\u5173\u4e0e\u964d\u566a\u3002"}),
                "seed": ("INT", {"default": 0, "min": 0,
                                 "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0,
                                  "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": cls.DEFAULT_DENOISE, "min": 0.0,
                    "max": 1.0, "step": 0.01,
                    "tooltip": "\u515c\u5e95\u964d\u566a\u3002\u4eba\u7269\u4fe1\u606f\u8282\u70b9\u5e26\u7ec6\u5316\u914d\u7f6e\u7684\u4eba\u7269\uff0c"
                               "\u7528\u8be5\u4eba\u7269\u7684\u300e\u7ec6\u5316\u964d\u566a\u300f\u3002"}),
            },
            "optional": {
                "segs": ("SEGS", {"tooltip": f"{part_cn}\u90e8\u4f4d\u7684\u68c0\u6d4b SEGS"
                           "\uff08UltralyticsDetectorProvider \u5bf9\u5e94\u6a21\u578b\uff09\u3002"
                           "\u4e0d\u63a5 = \u6309\u533a\u57df\u77e9\u5f62\u6574\u4f53\u7ec6\u5316\u3002"}),
                "base_conditioning": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "crop_padding": ("FLOAT", {"default": 0.25, "min": 0.0,
                    "max": 1.0, "step": 0.05}),
                "detail_size": ("INT", {"default": 1024, "min": 256,
                    "max": 4096, "step": 16}),
                "feather_px": ("INT", {"default": 24, "min": 0, "max": 256}),
                "mask_grow_px": ("INT", {"default": 8, "min": 0, "max": 128}),
                "shift": (["model default", "auto (RAW only, per-crop)"],
                    {"default": "model default"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "detail"
    CATEGORY = "mask/krea2_region"

    def detail(self, image, model, vae, regions, seed, steps, cfg,
               sampler_name, scheduler, denoise, segs=None,
               base_conditioning=None, negative=None,
               crop_padding=0.25, detail_size=1024, feather_px=24,
               mask_grow_px=8, shift="model default"):
        return _refine_part(image, model, vae, regions, self.PART, segs,
                            seed, steps, cfg, sampler_name, scheduler,
                            denoise, base_conditioning, negative,
                            crop_padding, detail_size, feather_px,
                            mask_grow_px, shift)


def _refine_part(image, model, vae, regions, part, segs, seed, steps, cfg,
                 sampler_name, scheduler, denoise, base_conditioning,
                 negative, crop_padding, detail_size, feather_px,
                 mask_grow_px, shift):
    """\u5355\u90e8\u4f4d\u7ec6\u5316\uff1a\u533a\u57df\u5e26 refine \u914d\u7f6e \u2192 \u53ea\u7ec6\u5316\u5f00\u5173\u6253\u5f00\u7684\u4eba\u7269\uff1b\u65e0\u914d\u7f6e \u2192 \u5168\u90e8\u7ec6\u5316\u3002"""
    B, H, W, _ = image.shape
    if not regions or denoise <= 0.0:
        return (image,)

    base_dict = {}
    try:
        if base_conditioning and len(base_conditioning[0]) > 1:
            base_dict = dict(base_conditioning[0][1])
    except (TypeError, IndexError):
        pass

    dets = _segs_to_masks(segs, H, W) if segs is not None else []
    detection_mode = segs is not None

    valid_regions = [(idx, r, _to_hw(r["mask"], H, W))
                     for idx, r in enumerate(regions)
                     if r and r.get("mask") is not None]
    if not valid_regions:
        return (image,)

    out = image.clone()
    plan = _build_detail_plan(valid_regions, part, dets, detection_mode, denoise)
    if not plan:
        return (out,)

    for b in range(B):
        for pidx, (ridx, m, r_denoise, detected) in enumerate(plan):
            reg = regions[ridx] or {}
            cond = reg.get("cond")
            if cond is None:
                continue
            bbox = _mask_bbox(m)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            pad = int(round(crop_padding * max(x1 - x0, y1 - y0)))
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
            cw, ch = x1 - x0, y1 - y0
            if cw < 16 or ch < 16:
                continue

            scale = max(1.0, detail_size / float(max(cw, ch)))
            tw = _round16(cw * scale)
            th = _round16(tw * ch / cw)
            crop = image[b:b + 1, y0:y1, x0:x1, :3]
            crop_up = _nhwc_resize(crop, th, tw)

            sample_mm = m[y0:y1, x0:x1]
            grow = mask_grow_px if detected else 0
            if grow > 0 or feather_px > 0:
                sample_mm = _grow_feather(sample_mm, grow, feather_px)
            noise_mask = F.interpolate(
                sample_mm[None, None], size=(th, tw), mode="bilinear",
                align_corners=False).clamp(0, 1)
            latent = {"samples": vae.encode(crop_up),
                      "noise_mask": noise_mask}

            m_model = model
            if str(shift).startswith("auto"):
                m_model = _apply_crop_shift(m_model, tw, th)
            for e in reg.get("loras") or []:
                try:
                    m_model = comfy.sd.load_lora_for_models(
                        m_model, None, e["sd"],
                        float(e.get("strength", 1.0)), 0.0)[0]
                except Exception as err:
                    logging.warning(
                        "[Krea2Detailer] lora '%s' failed to apply: %s",
                        e.get("label", "?"), err)

            pos = [[cond, dict(base_dict)]]
            neg = negative
            if neg is None:
                nd = dict(base_dict)
                if isinstance(nd.get("pooled_output"), torch.Tensor):
                    nd["pooled_output"] = torch.zeros_like(
                        nd["pooled_output"])
                neg = [[torch.zeros_like(cond), nd]]

            import time as _time
            _t0 = _time.time()
            samples = nodes.common_ksampler(
                m_model, seed + ridx * 8 + pidx,
                steps, cfg, sampler_name,
                scheduler, pos, neg, latent, denoise=r_denoise)[0]
            det = vae.decode(samples["samples"])
            if det.dim() == 5:
                det = det.reshape(-1, det.shape[-3], det.shape[-2],
                                  det.shape[-1])[:1]
            if det.shape[1] != th or det.shape[2] != tw:
                det = det[:, :th, :tw, :]
            det = _nhwc_resize(det[:, :, :, :3], ch, cw).to(out)

            mm = sample_mm
            mm4 = mm.clamp(0, 1).to(out)[None, :, :, None]
            out[b:b + 1, y0:y1, x0:x1, :3] = \
                det * mm4 + out[b:b + 1, y0:y1, x0:x1, :3] * (1 - mm4)
            logging.info("[Krea2Detailer] region %d (%s): %dx%d crop "
                         "sampled at %dx%d in %.1fs", ridx + 1, part,
                         cw, ch, tw, th, _time.time() - _t0)

    return (out,)


def _make_part_detailer(part, default_denoise):
    class _Krea2PartDetailer(_PartDetailerBase):
        PART = part
        DEFAULT_DENOISE = default_denoise

    _Krea2PartDetailer.__name__ = "Krea2" + part.capitalize() + "Detailer"
    _Krea2PartDetailer.QUALNAME = _Krea2PartDetailer.__name__
    return _Krea2PartDetailer


_PART_SPECS = (
    ("body", 0.30),
    ("face", 0.30),
    ("eyes", 0.40),
    ("hands", 0.40),
)

PART_DETAILER_NODE_CLASS_MAPPINGS = {}
PART_DETAILER_NODE_DISPLAY_NAME_MAPPINGS = {}
_PART_CN = {"body": "\u4eba\u4f53", "face": "\u8138\u90e8", "eyes": "\u773c\u775b", "hands": "\u624b\u90e8"}
for _part, _dd in _PART_SPECS:
    _cls = _make_part_detailer(_part, _dd)
    _cname = "Krea2" + _part.capitalize() + "Detailer"
    PART_DETAILER_NODE_CLASS_MAPPINGS[_cname] = _cls
    PART_DETAILER_NODE_DISPLAY_NAME_MAPPINGS[_cname] = \
        f"Krea2 \u7ec6\u5316 - {_PART_CN[_part]}"

DETAILER_NODE_CLASS_MAPPINGS = {
    "Krea2RegionalDetailer": Krea2RegionalDetailer,
}
DETAILER_NODE_CLASS_MAPPINGS.update(PART_DETAILER_NODE_CLASS_MAPPINGS)
DETAILER_NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RegionalDetailer": "Krea 2 - Regional Detailer\uff08per-person \u7ec6\u5316\uff09",
}
DETAILER_NODE_DISPLAY_NAME_MAPPINGS.update(PART_DETAILER_NODE_DISPLAY_NAME_MAPPINGS)
