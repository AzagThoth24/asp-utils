"""CPU routing regressions; run with Python + torch, without ComfyUI or models.

Load the production functions from their AST so this small suite does not
need ComfyUI's server, detectors, or GPU model weights.
"""

import ast
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(filename, names, **extra):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    definitions = [n for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in names]
    assert {n.name for n in definitions} == set(names)
    namespace = {"torch": torch, "F": F, "logging": logging, **extra}
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 str(ROOT / filename), "exec"), namespace)
    return namespace


KR = load_definitions("krea2_regional.py", [
    "_Ctx", "_latent_mask", "_exclusive_soft", "_routing_support",
    "_merge_attention_mask", "_grow_tokens", "_finalize_adaptive",
    "_image_region_links", "_build_allow", "_build_token_masks", "_diffusion_wrapper",
], BUNDLE_KEY="krea2_regional", CONSUMED_KEY="krea2_regional_consumed")
KR["CTX"] = KR["_Ctx"]()
KD = load_definitions("krea2_detailer.py", [
    "_mask_bbox", "_center", "_match_detections", "_build_detail_plan",
])


class ApplyRegionalTests(unittest.TestCase):
    def setUp(self):
        self.patched = SimpleNamespace(model_options={}, add_wrapper_with_key=lambda *args: None)
        self.model = SimpleNamespace(clone=lambda: self.patched)
        self.base = [[torch.ones(1, 2, 3), {"pooled_output": None}]]
        self.inject_lora = Mock(return_value=1)
        namespace = load_definitions(
            "krea2_regional.py", ["Krea2ApplyRegional"], BUNDLE_KEY="krea2_regional",
            WrappersMP=SimpleNamespace(DIFFUSION_MODEL="diffusion_model"),
            _diffusion_wrapper=KR["_diffusion_wrapper"], _UID=iter([1]),
            _inject_lora=self.inject_lora)
        self.apply = namespace["Krea2ApplyRegional"]().apply

    def test_empty_regions_preserve_all_base_conditioning(self):
        conditioning = self.base * 2
        for regions in (None, []):
            with self.subTest(regions=regions), self.assertLogs(level="WARNING"):
                model, result = self.apply(self.model, conditioning, regions, True)
                self.assertIs(model, self.model)
                self.assertIs(result, conditioning)

    def test_empty_regions_keep_base_loras(self):
        lora = {"sd": {}, "label": "style", "strength": 0.6}
        with self.assertLogs(level="WARNING"):
            model, result = self.apply(self.model, self.base, None, True,
                                       base_loras=[lora], region_lock_strength=0.5)
        self.inject_lora.assert_called_once_with(model, lora["sd"], "k2r1", 0.6)
        self.assertTrue(torch.equal(result[0][0], self.base[0][0]))
        bundle = model.model_options["transformer_options"]["krea2_regional"]
        self.assertEqual(bundle["segments"], [(0, 2)])
        self.assertEqual(bundle["masks"], [])
        self.assertIsNone(bundle["loras"][0]["mask_idx"])

    def test_missing_base_conditioning_identifies_canvas_inputs(self):
        with self.assertRaisesRegex(ValueError, "base_cond"):
            self.apply(self.model, None, None, False)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.segments = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.soft = torch.tensor([[.3, 0, 0, 0],
                                  [0, .5, 0, 0],
                                  [0, 0, .8, 0]])

    def test_weak_lora_tokens_keep_their_prompt(self):
        support = KR["_routing_support"](self.soft)
        allow, _ = KR["_build_allow"](self.segments, support, 4, False, "cpu")
        specs = [{"uid": str(i), "weight": 1, "seg": i + 1, "mask_idx": i}
                 for i in range(3)]
        masks = KR["_build_token_masks"]({"loras": specs}, self.segments,
                                         self.soft, 4, [True], "cpu")
        for i in range(3):
            image_gate = masks[str(i)][0, 4:, 0] > 0
            self.assertTrue(torch.equal(allow[4:, i + 1], image_gate))
        self.assertTrue(allow[4, 1])  # .3 used to fail the second .5 cutoff
        self.assertTrue(allow[5, 2])  # equality at .5 also used to fail

    def test_restriction_blocks_multilayer_relays(self):
        allow, _ = KR["_build_allow"](self.segments,
                                      KR["_routing_support"](self.soft),
                                      4, True, "cpu")
        # Include a background token: neither it nor base text may relay
        # one subject's identity into another over successive DiT blocks.
        reach = allow.clone()
        for _ in range(5):
            reach |= (reach.float() @ allow.float()) > 0
        for r in range(3):
            for other in range(3):
                if r != other:
                    self.assertFalse(reach[4 + r, 1 + other])
                    self.assertFalse(reach[4 + r, 4 + other])
        self.assertTrue(allow[4:, 0].all())  # shared scene still readable
        self.assertFalse(allow[0, 4:].any())

    def test_release_restores_image_and_scene_attention(self):
        allow, _ = KR["_build_allow"](self.segments,
                                      KR["_routing_support"](self.soft),
                                      4, False, "cpu")
        self.assertTrue(allow[4:, 4:].all())
        self.assertTrue(allow[0, 4:].all())
        self.assertFalse(allow[4, 2])  # text routing remains regional

    def test_no_regions_is_stock_attention(self):
        allow, text = KR["_build_allow"]([(0, 2)], torch.zeros(0, 3).bool(),
                                         3, True, "cpu")
        self.assertTrue(allow.all())
        self.assertTrue(text.all())

    def test_exclusive_masks_preserve_depth_priority(self):
        soft = torch.tensor([[1., .2, 0], [1., .3, 0]])
        result = KR["_exclusive_soft"](soft)
        self.assertTrue(torch.equal(result, torch.tensor([[1., 0, 0], [0, .3, 0]])))

    def test_existing_boolean_mask_and_routing_both_survive(self):
        regional = torch.zeros(1, 1, 3, 3)
        regional[..., 0, 2] = torch.finfo(torch.float32).min
        existing = torch.ones(3, 3, dtype=torch.bool)
        existing[1, 0] = False
        merged = KR["_merge_attention_mask"](regional, existing)
        self.assertLess(merged[0, 0, 0, 2], -1e20)
        self.assertLess(merged[0, 0, 1, 0], -1e20)
        self.assertEqual(merged[0, 0, 2, 2], 0)
        self.assertEqual(regional[0, 0, 1, 0], 0)  # cached mask untouched

    def test_existing_additive_bias_survives(self):
        regional = torch.zeros(1, 1, 3, 3)
        existing = torch.tensor([[0., -2, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(KR["_merge_attention_mask"](regional, existing)[0, 0, 0, 1], -2)

    def test_batched_padding_and_joint_masks(self):
        regional = torch.zeros(1, 1, 3, 3)
        padding = torch.tensor([[True, True, False], [True, False, True]])
        merged = KR["_merge_attention_mask"](regional, padding)
        self.assertEqual(merged.shape, (2, 1, 3, 3))
        self.assertTrue((merged[0, :, :, 2] < -1e20).all())
        joint = torch.ones(2, 3, 3, dtype=torch.bool)
        joint[1, 0, 1] = False
        merged = KR["_merge_attention_mask"](regional, joint)
        self.assertEqual(merged.shape, (2, 1, 3, 3))
        self.assertLess(merged[1, 0, 0, 1], -1e20)

    def test_negative_batch_does_not_receive_lora(self):
        bundle = {"loras": [{"uid": "person", "weight": 1,
                              "seg": 1, "mask_idx": 0}]}
        masks = KR["_build_token_masks"](bundle, self.segments, self.soft,
                                         4, [True, False], "cpu")["person"]
        self.assertGreater(masks[0].sum(), 0)
        self.assertEqual(masks[1].sum(), 0)

    def test_adaptive_empty_discovery_keeps_original_box(self):
        aff = torch.zeros(3, 4)
        refined = KR["_finalize_adaptive"](aff, self.soft, "refine", .45, 2, 2)
        self.assertTrue(torch.equal(refined, self.soft))

    def test_wrapper_merges_masks_releases_and_clears_context(self):
        masks = [torch.tensor([[1., 0], [1., 0]]),
                 torch.tensor([[0., 1], [0., 1]])]
        bundle = {"segments": [(0, 1), (1, 2), (2, 3)], "txt_total": 3,
                  "masks": masks, "loras": [], "restrict_img_attn": True,
                  "restrict_end_sigma": .5, "adaptive": {"mode": "off"}}
        options = {"krea2_regional": bundle}
        captured = []
        class Executor:
            class_obj = SimpleNamespace(patch=2, txtlayers=12, blocks=[None] * 28)
            def __call__(self, x, ts, ctx, *rest, transformer_options, **kw):
                q = torch.zeros(1, 1, 7, 4)
                existing = torch.ones(7, 7, dtype=torch.bool)
                existing[0, 0] = False
                def attention(q, k, v, heads, mask=None, **kw):
                    captured.append(mask.clone())
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                return transformer_options["optimized_attention_override"](
                    attention, q, q, q, 1, mask=existing)
        for timestep in [.8, .2]:
            KR["_diffusion_wrapper"](Executor(), torch.zeros(1, 16, 4, 4),
                                      torch.tensor([timestep]), torch.zeros(1, 3, 2),
                                      transformer_options=options)
            self.assertEqual(KR["CTX"].token_masks, {})
            self.assertNotIn("optimized_attention_override", options)
            self.assertNotIn("krea2_regional_consumed", options)
        self.assertLess(captured[0][0, 0, 3, 4], -1e20)
        self.assertEqual(captured[1][0, 0, 3, 4], 0)
        self.assertLess(captured[1][0, 0, 0, 0], -1e20)


class SoftRoutingTests(unittest.TestCase):
    def capture(self, penalty, *, restrict=False, timesteps=(.8,), rows=(0,),
                existing=None, masks=None):
        if masks is None:
            masks = [torch.tensor([[1., 0], [0, 0]]),
                     torch.tensor([[0., 1], [0, 0]]),
                     torch.tensor([[0., 0], [1., 0]])]
        bundle = {"segments": [(0, 1), (1, 2), (2, 3), (3, 4)],
                  "txt_total": 4, "masks": masks, "loras": [],
                  "restrict_img_attn": restrict, "restrict_end_sigma": .5,
                  "cross_region_penalty": penalty, "adaptive": {"mode": "off"}}
        options = {"krea2_regional": bundle, "cond_or_uncond": list(rows)}
        captured = []
        class Executor:
            class_obj = SimpleNamespace(patch=2, txtlayers=12, blocks=[None] * 28)
            def __call__(self, x, ts, ctx, *rest, transformer_options, **kw):
                q = torch.zeros(len(rows), 1, 8, 4)
                def attention(q, k, v, heads, mask=None, **kw):
                    captured.append(mask.clone())
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                return transformer_options["optimized_attention_override"](
                    attention, q, q, q, 1, mask=existing)
        for timestep in timesteps:
            KR["_diffusion_wrapper"](Executor(), torch.zeros(len(rows), 16, 4, 4),
                                      torch.tensor([timestep]),
                                      torch.zeros(len(rows), 4, 2),
                                      transformer_options=options)
            self.assertEqual(KR["CTX"].token_masks, {})
            self.assertNotIn("optimized_attention_override", options)
        return captured

    def test_soft_bias_reduces_cross_image_odds_but_keeps_links(self):
        bias = self.capture(3.0)[0]
        q = torch.zeros(1, 1, 8, 8)
        weights = F.scaled_dot_product_attention(
            q, q, torch.eye(8)[None, None], attn_mask=bias)
        # Right region can still read middle images; its own images have
        # exp(3) times their attention odds at equal QK scores.
        cross, own = weights[0, 0, 6, 5], weights[0, 0, 6, 6]
        self.assertGreater(cross, 0)
        self.assertAlmostEqual(float(cross / own), math.exp(-3), places=6)
        self.assertEqual(bias[0, 0, 6, 6], 0)
        self.assertLess(bias[0, 0, 6, 2], -1e20)  # other prompt still hidden
        self.assertEqual(bias[0, 0, 6, 3], 0)  # own prompt still readable

    def test_soft_base_cannot_relay_identity(self):
        bias = self.capture(3.0)[0]
        self.assertTrue((bias[0, 0, 0, 4:] < -1e20).all())
        self.assertTrue((bias[0, 0, 4:, 0] == 0).all())

    def test_soft_background_links_are_finite_but_penalized(self):
        bias = self.capture(3.0)[0]
        self.assertEqual(bias[0, 0, 6, 7], -3)
        self.assertEqual(bias[0, 0, 7, 6], -3)
        self.assertEqual(bias[0, 0, 7, 7], 0)

    def test_zero_penalty_preserves_previous_open_behavior(self):
        bias = self.capture(0.0)[0]
        self.assertTrue((bias[0, 0, 4:, 4:] == 0).all())
        self.assertTrue((bias[0, 0, 0, 4:] == 0).all())

    def test_hard_restriction_releases_into_soft_routing(self):
        early, late = self.capture(3.0, restrict=True, timesteps=(.8, .2))
        self.assertLess(early[0, 0, 6, 5], -1e20)
        self.assertEqual(late[0, 0, 6, 5], -3)

    def test_unconditional_rows_stay_clean(self):
        bias = self.capture(3.0, rows=(0, 1))[0]
        self.assertEqual(bias[0, 0, 6, 5], -3)
        self.assertTrue((bias[1] == 0).all())

    def test_upstream_additive_bias_combines_with_soft_penalty(self):
        existing = torch.zeros(8, 8)
        existing[6, 5] = -2
        bias = self.capture(3.0, existing=existing)[0]
        self.assertEqual(bias[0, 0, 6, 5], -5)
        self.assertEqual(existing[6, 5], -2)

    def test_actual_two_layer_identity_relay_is_reduced(self):
        q = torch.zeros(1, 1, 8, 1)
        v = torch.zeros_like(q)
        v[0, 0, 2, 0] = 1  # identity in the middle region's prompt
        relays = []
        for penalty in [0., 3.]:
            bias = self.capture(penalty)[0]
            first = F.scaled_dot_product_attention(q, q, v, attn_mask=bias)
            second = F.scaled_dot_product_attention(q, q, first, attn_mask=bias)
            relays.append(float(second[0, 0, 6, 0]))
        self.assertGreater(relays[1], 0)  # deliberately soft, never a guarantee
        self.assertLess(relays[1], relays[0] * .2)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.regions = []
        for i, (x0, x1) in enumerate([(32, 543), (773, 1305), (1477, 2001)]):
            mask = torch.zeros(832, 2048)
            mask[:, x0:x1] = 1
            self.regions.append((i, {"refine": {"refine_face": True,
                                               "denoise": .2}}, mask))
        self.face = torch.zeros(832, 2048)
        self.face[366:466, 989:1089] = 1

    def test_missing_neighbour_face_is_not_repainted_twice(self):
        # Old per-person calls selected this center face for persons 2 AND 3.
        plan = KD["_build_detail_plan"](self.regions, "face", [self.face], True, .3)
        self.assertEqual([p[0] for p in plan], [1])
        self.assertIs(plan[0][1], self.face)
        self.assertEqual(plan[0][2], .2)

    def test_disabled_person_retains_detection_ownership(self):
        self.regions[1][1]["refine"]["refine_face"] = False
        plan = KD["_build_detail_plan"](self.regions, "face", [self.face], True, .3)
        self.assertEqual(plan, [])

    def test_zero_denoise_does_not_sample(self):
        self.regions[1][1]["refine"]["denoise"] = 0
        self.assertEqual(KD["_build_detail_plan"](
            self.regions, "face", [self.face], True, .3), [])

    def test_no_detector_keeps_reverse_compositing_order(self):
        plan = KD["_build_detail_plan"](self.regions, "face", [], False, .3)
        self.assertEqual([p[0] for p in plan], [2, 1, 0])
        self.assertTrue(all(not p[3] for p in plan))

    def test_legacy_regions_keep_global_denoise(self):
        legacy = [(i, {}, mask) for i, _, mask in self.regions]
        plan = KD["_build_detail_plan"](legacy, "face", [self.face], True, .3)
        self.assertEqual([p[0] for p in plan], [1])
        self.assertEqual(plan[0][2], .3)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
