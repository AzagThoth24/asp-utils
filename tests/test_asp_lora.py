"""asp-utils 新增模块的 CPU 单测；不依赖 ComfyUI / torch。

- asp_lora_loader.py：通过 AST 提取（同 test_regressions.py 的手法），
  注入伪造的 folder_paths / nodes / lora_info。
- asp_server/lora_info.py、asp_server/presets.py：无第三方依赖，
  直接按文件路径 import，测试真实模块。
"""

import ast
import importlib.util
import json
import logging
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 载入工具

def load_definitions(filename, names, **extra):
    """从生产文件里按名字提取函数/类定义并执行（与 test_regressions 相同手法）。"""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    definitions = [n for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in names]
    assert {n.name for n in definitions} == set(names), \
        f"{filename} 缺少定义: {set(names) - {n.name for n in definitions}}"
    namespace = {"__name__": "asp_test", **extra}
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 str(ROOT / filename), "exec"), namespace)
    return namespace


def import_by_path(relpath, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_safetensors(path, metadata, header_extra=None):
    """构造一个只含 __metadata__ 的假 safetensors（头部 JSON 可被纯 Python 解析）。"""
    header = {"__metadata__": metadata}
    if header_extra:
        header.update(header_extra)
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)  # 头部按 8 对齐（解析器只读前 n 字节）
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\x00" * 64)


# ---------------------------------------------------------------- 伪造依赖

class _FakeLoraLoader:
    calls = []

    def load_lora(self, model, clip, lora, strength_model, strength_clip):
        type(self).calls.append((lora, strength_model, strength_clip))
        return ((model or "M") + "+" + lora, (clip or "C") + "+" + lora)


class _FakeFolderPaths:
    FILES = ["a.safetensors", "sub/char_a.safetensors", "sub/char_b.safetensors",
             "StyleLoRA.safetensors"]

    @classmethod
    def get_filename_list(cls, kind):
        assert kind == "loras"
        return list(cls.FILES)

    @classmethod
    def get_full_path(cls, kind, name):
        return os.path.join("/fake/loras", name)


TRIGGER_MAP = {
    "a.safetensors": ["t1"],
    "sub/char_b.safetensors": ["t2", "t1"],
}


def make_loader_namespace():
    return load_definitions(
        "asp_lora_loader.py",
        ["parse_rows", "resolve_lora_name", "AspPowerLoraLoader"],
        json=json, logging=logging, os=os,
        _LOG=logging.getLogger("test"),
        NODE_NAME="AspPowerLoraLoader", CATEGORY="asp-utils",
        DEFAULT_LORA_DATA={"rows": [], "preset": ""},
        folder_paths=_FakeFolderPaths,
        nodes=SimpleNamespace(LoraLoader=_FakeLoraLoader),
        lora_info=SimpleNamespace(
            get_trigger_words=lambda n: list(TRIGGER_MAP.get(n, []))),
    )


# ---------------------------------------------------------------- parse_rows

class ParseRowsTests(unittest.TestCase):
    def setUp(self):
        self.ns = make_loader_namespace()

    def test_valid_dict_and_bare_list(self):
        rows = self.ns["parse_rows"](
            '{"rows": [{"on": true, "lora": "a", "strength_model": 0.5, '
            '"strength_clip": 0.7}]}')
        self.assertEqual(rows, [{"on": True, "lora": "a",
                                 "strength_model": 0.5, "strength_clip": 0.7}])
        self.assertEqual(self.ns["parse_rows"]('[{"lora": "a"}]'),
                         [{"on": True, "lora": "a",
                           "strength_model": 1.0, "strength_clip": 1.0}])

    def test_invalid_json_returns_empty(self):
        self.assertEqual(self.ns["parse_rows"]("{oops"), [])
        self.assertEqual(self.ns["parse_rows"](""), [])
        self.assertEqual(self.ns["parse_rows"](None), [])

    def test_dirty_rows_filtered(self):
        rows = self.ns["parse_rows"](
            '{"rows": ["junk", {"lora": "a"}, {"lora": "b", "strength_model": "x"}]}')
        self.assertEqual([r["lora"] for r in rows], ["a"])

    def test_strength_clip_defaults_to_model(self):
        rows = self.ns["parse_rows"](
            '{"rows": [{"lora": "a", "strength_model": 0.6}]}')
        self.assertEqual(rows[0]["strength_clip"], 0.6)


# ---------------------------------------------------------------- resolve_lora_name

class ResolveLoraNameTests(unittest.TestCase):
    def setUp(self):
        self.ns = make_loader_namespace()
        self.resolve = self.ns["resolve_lora_name"]

    def test_exact_and_case(self):
        self.assertEqual(self.resolve("a.safetensors"), "a.safetensors")
        self.assertEqual(self.resolve("A.SAFETENSORS"), "a.safetensors")

    def test_without_extension(self):
        self.assertEqual(self.resolve("StyleLoRA"), "StyleLoRA.safetensors")

    def test_basename_with_subfolder(self):
        self.assertEqual(self.resolve("char_a"), "sub/char_a.safetensors")

    def test_ambiguous_substring_takes_first(self):
        with self.assertLogs(level=logging.WARNING):
            self.assertEqual(self.resolve("char"), "sub/char_a.safetensors")

    def test_missing(self):
        with self.assertLogs(level=logging.WARNING):
            self.assertIsNone(self.resolve("does-not-exist"))


# ---------------------------------------------------------------- 节点 load()

class LoaderNodeTests(unittest.TestCase):
    def setUp(self):
        self.ns = make_loader_namespace()
        _FakeLoraLoader.calls = []
        self.node = self.ns["AspPowerLoraLoader"]()

    def test_applies_enabled_rows_and_outputs_triggers(self):
        data = json.dumps({"rows": [
            {"on": True, "lora": "a.safetensors", "strength_model": 1, "strength_clip": 1},
            {"on": False, "lora": "StyleLoRA", "strength_model": 1, "strength_clip": 1},
            {"on": True, "lora": "char_b", "strength_model": 0.5, "strength_clip": 0.5},
            {"on": True, "lora": "missing.safetensors", "strength_model": 1, "strength_clip": 1},
        ]})
        with self.assertLogs(level=logging.WARNING):  # missing 那行要告警
            model, clip, triggers = self.node.load(data, "m0", "c0")
        self.assertEqual([c[0] for c in _FakeLoraLoader.calls],
                         ["a.safetensors", "sub/char_b.safetensors"])
        self.assertEqual(_FakeLoraLoader.calls[1],
                         ("sub/char_b.safetensors", 0.5, 0.5))
        self.assertTrue(model.endswith("+a.safetensors+sub/char_b.safetensors"))
        self.assertEqual(triggers, "t1, t2")  # 去重且保持顺序

    def test_both_inputs_missing_breaks_early(self):
        with self.assertLogs(level=logging.WARNING):
            model, clip, triggers = self.node.load(
                '{"rows": [{"on": true, "lora": "a"}]}')
        self.assertEqual(_FakeLoraLoader.calls, [])
        self.assertEqual((model, clip, triggers), (None, None, ""))

    def test_bad_json_is_not_fatal(self):
        model, clip, triggers = self.node.load("not json", "m0", "c0")
        self.assertEqual((model, clip, triggers), ("m0", "c0", ""))


# ---------------------------------------------------------------- lora_info（真实模块）

LORA_INFO = import_by_path("asp_server/lora_info.py", "asp_lora_info_test")


class SafetensorsMetadataTests(unittest.TestCase):
    def test_roundtrip_and_bad_files(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.safetensors"
            make_safetensors(p, {"modelspec.trigger_phrase": "char_x, style"})
            self.assertEqual(LORA_INFO.read_safetensors_metadata(str(p)),
                             {"modelspec.trigger_phrase": "char_x, style"})
            self.assertIsNone(LORA_INFO.read_safetensors_metadata(str(p) + ".nope"))
            bad = Path(td) / "bad.safetensors"
            bad.write_bytes(b"\x01")  # 长度字段都不够
            self.assertIsNone(LORA_INFO.read_safetensors_metadata(str(bad)))


class TriggerWordExtractionTests(unittest.TestCase):
    def test_trained_words_shapes(self):
        w = LORA_INFO._words_from_trained
        self.assertEqual(w("solo"), ["solo"])
        self.assertEqual(w(["a", " b ", ""]), ["a", "b"])
        self.assertEqual(w([{"word": "x"}, {"name": "y"}, {"other": 1}]), ["x", "y"])

    def test_trigger_phrase_split(self):
        self.assertEqual(LORA_INFO._words_from_trigger_phrase("a, b\nc "),
                         ["a", "b", "c"])

    def test_tag_frequency_threshold_and_cap(self):
        freq = {f"t{i}": 10 for i in range(12)}
        freq["small"] = 2
        words = LORA_INFO._words_from_tag_frequency(
            json.dumps({"d1": freq, "d2": {"t0": 5}}))
        self.assertNotIn("small", words)
        self.assertIn("t0", words)  # 跨数据集聚合计数
        self.assertLessEqual(len(words), 8)

    def test_priority_manual_over_civitai_over_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            lora = Path(td) / "lora.safetensors"
            make_safetensors(lora, {"modelspec.trigger_phrase": "meta_w",
                                    "modelspec.base_model": "Krea2"})
            full = str(lora)
            # 无边车：metadata 生效
            words, source, base = LORA_INFO.extract_for_name(
                "lora.safetensors", full, {})
            self.assertEqual((words, source, base), (["meta_w"], "metadata", "Krea2"))
            # 有 civitai 边车：civitai 覆盖 metadata
            (Path(td) / "lora.safetensors.civitai.info").write_text(
                json.dumps({"trainedWords": ["civ_w"], "baseModel": "Flux"}), "utf-8")
            words, source, base = LORA_INFO.extract_for_name(
                "lora.safetensors", full, {})
            self.assertEqual((words, source, base), (["civ_w"], "civitai", "Flux"))
            # 手动覆盖优先级最高
            words, source, _ = LORA_INFO.extract_for_name(
                "lora.safetensors", full, {"lora.safetensors": ["manual_w"]})
            self.assertEqual((words, source), (["manual_w"], "manual"))

    def test_tag_frequency_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            lora = Path(td) / "f.safetensors"
            make_safetensors(lora, {"ss_tag_frequency": json.dumps({"d": {"t_a": 9, "t_b": 1}})})
            words, source, _ = LORA_INFO.extract_for_name("f.safetensors", str(lora), {})
            self.assertEqual((words, source), (["t_a"], "tag_frequency"))


class ScanLorasTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("folder_paths", None)
        LORA_INFO._CACHE.clear()

    def test_scan_reads_sidecar_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            lora = Path(td) / "x.safetensors"
            make_safetensors(lora, {"modelspec.trigger_phrase": "ph"})
            (Path(td) / "x.safetensors.civitai.info").write_text(
                json.dumps({"trainedWords": ["cw"], "baseModel": "Krea2"}), "utf-8")
            sys.modules["folder_paths"] = SimpleNamespace(
                get_filename_list=lambda kind: ["x.safetensors", "readme.txt"],
                get_full_path=lambda kind, name: str(lora))
            LORA_INFO.set_tags_override_path(Path(td) / "nonexistent.json")
            entries = LORA_INFO.scan_loras()
            self.assertEqual(len(entries), 1)  # readme.txt 被扩展名过滤
            self.assertEqual(entries[0]["trigger_words"], ["cw"])
            self.assertEqual(entries[0]["source"], "civitai")
            self.assertEqual(entries[0]["name"], "x.safetensors")


# ---------------------------------------------------------------- presets（真实模块）

PRESETS = import_by_path("asp_server/presets.py", "asp_presets_test")


class PresetsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._orig_dir = PRESETS.USER_DIR
        PRESETS.USER_DIR = Path(self._tmp.name)

    def tearDown(self):
        PRESETS.USER_DIR = self._orig_dir

    def test_save_get_delete_roundtrip(self):
        rows = [{"on": True, "lora": "a.safetensors",
                 "strength_model": 0.8, "strength_clip": 0.8}]
        ok, _ = PRESETS.save_preset("风格组 A", rows)
        self.assertTrue(ok)
        self.assertEqual(PRESETS.get_presets()["风格组 A"]["rows"], rows)
        ok, _ = PRESETS.save_preset("风格组 A", rows + [dict(rows[0], lora="b")])
        self.assertTrue(ok)
        self.assertEqual(len(PRESETS.get_presets()["风格组 A"]["rows"]), 2)
        ok, _ = PRESETS.delete_preset("风格组 A")
        self.assertTrue(ok)
        self.assertEqual(PRESETS.get_presets(), {})
        ok, _ = PRESETS.delete_preset("风格组 A")
        self.assertFalse(ok)

    def test_invalid_rows_and_name_rejected(self):
        ok, _ = PRESETS.save_preset("x", [])
        self.assertFalse(ok)
        ok, _ = PRESETS.save_preset("x", [{"lora": "a", "strength_model": "?"}])
        self.assertFalse(ok)
        ok, _ = PRESETS.save_preset("  ", [{"lora": "a"}])
        self.assertFalse(ok)

    def test_name_sanitized_and_strength_clamped(self):
        ok, _ = PRESETS.save_preset('a<b>/\\:c', [{"lora": "a", "strength_model": 99}])
        self.assertTrue(ok)
        name = next(iter(PRESETS.get_presets()))
        self.assertNotIn("<", name)
        row = PRESETS.get_presets()[name]["rows"][0]
        self.assertEqual(row["strength_model"], 10.0)

    def test_tags_set_and_clear(self):
        ok, _ = PRESETS.set_tags("my_lora", ["w1", "w2", "w1", ""])
        self.assertTrue(ok)
        self.assertEqual(PRESETS.get_tags()["my_lora"], ["w1", "w2"])
        ok, _ = PRESETS.set_tags("my_lora", [])
        self.assertTrue(ok)
        self.assertEqual(PRESETS.get_tags(), {})
        ok, _ = PRESETS.set_tags("", ["x"])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
