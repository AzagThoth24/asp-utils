// asp-utils 多 LoRA 加载器前端
//
// 设计约束（Nodes 2.0 双渲染兼容）：
//  - 仅使用官方稳定 v1 扩展 API（app.registerExtension + addDOMWidget）
//  - 全部交互在 DOM widget 内部完成：不 patch 原型、不自绘 canvas、
//    不用 litegraph ContextMenu、不用 window.prompt
//  - 行数据序列化到隐藏的 lora_data STRING widget（JSON 字符串），
//    DOM widget 自身 serialize:false
//
// 路由约定见 asp_server/__init__.py（前缀 /asp）。

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAME = "AspPowerLoraLoader";
const TAG = "[asp-lora]";
const MIN_NODE_WIDTH = 430;

// ---------------------------------------------------------------- 缓存与请求

const loraCache = { at: 0, list: [] };
const presetCache = { at: 0, map: {} };

async function requestJson(path, body) {
  const opts = body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const resp = await api.fetchApi(path, opts);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);
  return data;
}

async function getLoras(force = false) {
  const fresh = Date.now() - loraCache.at < 60_000;
  if (!force && fresh && loraCache.list.length) return loraCache.list;
  try {
    const data = await requestJson("/asp/loras");
    loraCache.list = Array.isArray(data.loras) ? data.loras : [];
    loraCache.at = Date.now();
  } catch (err) {
    console.warn(TAG, "拉取 LoRA 列表失败:", err?.message || err);
  }
  return loraCache.list;
}

async function getPresets(force = false) {
  if (!force && Date.now() - presetCache.at < 15_000) return presetCache.map;
  try {
    const data = await requestJson("/asp/presets");
    presetCache.map = data.presets && typeof data.presets === "object" ? data.presets : {};
    presetCache.at = Date.now();
  } catch (err) {
    console.warn(TAG, "拉取预设失败:", err?.message || err);
  }
  return presetCache.map;
}

function toast(severity, summary, detail) {
  const add = app?.extensionManager?.toast?.add;
  if (typeof add === "function") {
    add({ severity, summary, detail: detail || undefined, life: 3500 });
  } else {
    console[severity === "error" ? "error" : "log"](TAG, summary, detail || "");
  }
}

function findLoraInfo(name) {
  if (!name) return null;
  const list = loraCache.list;
  return list.find((l) => l.name === name)
      || list.find((l) => l.name?.endsWith("/" + name))
      || list.find((l) => l.name === name + ".safetensors")
      || null;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.cssText = "position:fixed;opacity:0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch (_) {
      return false;
    }
  }
}

// ---------------------------------------------------------------- 弹窗（模态）

function buildBackdrop() {
  const el = document.createElement("div");
  el.className = "asp-backdrop";
  return el;
}

/** 通用文本输入弹窗；resolve(文本) 或 resolve(null) 取消。 */
function askText({ title, value = "", placeholder = "", okText = "确定", multiline = false }) {
  return new Promise((resolve) => {
    const backdrop = buildBackdrop();
    backdrop.innerHTML = `
      <div class="asp-dialog">
        <div class="asp-dialog-title"></div>
        ${multiline
          ? `<textarea class="asp-dialog-input" rows="3"></textarea>`
          : `<input class="asp-dialog-input" type="text"></input>`}
        <div class="asp-dialog-row">
          <button class="asp-btn" data-act="cancel">取消</button>
          <button class="asp-btn asp-primary" data-act="ok"></button>
        </div>
      </div>`;
    const titleEl = backdrop.querySelector(".asp-dialog-title");
    const input = backdrop.querySelector(".asp-dialog-input");
    const okBtn = backdrop.querySelector('[data-act="ok"]');
    titleEl.textContent = title || "";
    okBtn.textContent = okText;
    if (placeholder) input.placeholder = placeholder;
    input.value = value;
    const done = (val) => { backdrop.remove(); document.removeEventListener("keydown", onKey, true); resolve(val); };
    const onKey = (e) => {
      e.stopPropagation();
      if (e.key === "Escape") done(null);
      else if (e.key === "Enter" && (!multiline || e.ctrlKey)) { e.preventDefault(); done(input.value); }
    };
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) done(null); });
    backdrop.querySelector('[data-act="cancel"]').addEventListener("click", () => done(null));
    okBtn.addEventListener("click", () => done(input.value));
    input.addEventListener("keydown", (e) => e.stopPropagation());
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(backdrop);
    input.focus();
    input.select?.();
  });
}

/** LoRA 选择弹窗：可搜索、显示触发词/base model；resolve(名字) 或 null。 */
function pickLora(current = "") {
  return new Promise(async (resolve) => {
    const backdrop = buildBackdrop();
    backdrop.innerHTML = `
      <div class="asp-dialog asp-picker">
        <div class="asp-dialog-title">选择 LoRA</div>
        <input class="asp-dialog-input" type="text" placeholder="搜索名字 / 触发词 / base model…">
        <div class="asp-picker-list"><div class="asp-picker-empty">加载中…</div></div>
        <div class="asp-dialog-row">
          <button class="asp-btn" data-act="refresh">↻ 刷新列表</button>
          <span class="asp-flex"></span>
          <button class="asp-btn" data-act="cancel">取消</button>
        </div>
      </div>`;
    const input = backdrop.querySelector(".asp-dialog-input");
    const listEl = backdrop.querySelector(".asp-picker-list");
    const emptyEl = () => { listEl.innerHTML = `<div class="asp-picker-empty">没有匹配的 LoRA</div>`; };
    document.body.appendChild(backdrop);
    const done = (val) => { backdrop.remove(); document.removeEventListener("keydown", onKey, true); resolve(val); };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); done(null); } };
    document.addEventListener("keydown", onKey, true);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) done(null); });
    backdrop.querySelector('[data-act="cancel"]').addEventListener("click", () => done(null));
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") {
        const first = listEl.querySelector(".asp-picker-item");
        first?.dispatchEvent(new CustomEvent("asp-pick", { detail: first.dataset.name }));
      }
    });
    input.addEventListener("input", () => render(input.value.trim().toLowerCase()));
    backdrop.querySelector('[data-act="refresh"]').addEventListener("click", async () => {
      listEl.innerHTML = `<div class="asp-picker-empty">加载中…</div>`;
      await getLoras(true);
      render(input.value.trim().toLowerCase());
    });

    function matches(item, q) {
      if (!q) return true;
      return (item.name || "").toLowerCase().includes(q)
        || (item.trigger_words || []).some((w) => w.toLowerCase().includes(q))
        || (item.base_model || "").toLowerCase().includes(q);
    }

    function render(q) {
      const items = loraCache.list.filter((l) => matches(l, q)).slice(0, 300);
      listEl.innerHTML = "";
      if (!items.length) return emptyEl();
      for (const item of items) {
        const div = document.createElement("div");
        div.className = "asp-picker-item" + (item.name === current ? " asp-current" : "");
        div.dataset.name = item.name;
        const trig = (item.trigger_words || []).join(", ");
        div.innerHTML = `
          <div class="asp-picker-name"></div>
          <div class="asp-picker-meta"></div>`;
        div.querySelector(".asp-picker-name").textContent = item.name;
        const meta = [item.base_model, trig ? `🏷 ${trig}` : ""].filter(Boolean).join(" · ");
        div.querySelector(".asp-picker-meta").textContent = meta || "（无元数据）";
        div.addEventListener("click", () => done(item.name));
        div.addEventListener("asp-pick", (e) => done(e.detail));
        listEl.appendChild(div);
      }
    }

    await getLoras();
    render("");
    input.focus();
  });
}

// ---------------------------------------------------------------- 样式

function ensureStyle() {
  if (document.getElementById("asp-utils-style")) return;
  const style = document.createElement("style");
  style.id = "asp-utils-style";
  style.textContent = `
.asp-lora { display: flex; flex-direction: column; gap: 4px; font-size: 12px;
  color: var(--fg-color, #d4d4d4); user-select: auto; }
.asp-toolbar { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
.asp-preset { flex: 1 1 120px; min-width: 90px; max-width: 220px; height: 24px;
  background: var(--comfy-input-bg, #222); color: inherit; border: 1px solid var(--border-color, #333);
  border-radius: 4px; padding: 0 4px; font-size: 12px; }
.asp-btn { height: 24px; padding: 0 8px; font-size: 12px; cursor: pointer;
  background: var(--comfy-input-bg, #2a2a2a); color: inherit;
  border: 1px solid var(--border-color, #3a3a3a); border-radius: 4px; white-space: nowrap; }
.asp-btn:hover { border-color: var(--p-button-text-primary-hover-bg, #6a6a6a); }
.asp-btn.asp-primary { border-color: #4f7cff; }
.asp-btn.asp-danger:hover { border-color: #c33; color: #e66; }
.asp-add { border-color: #4f7cff; }
.asp-flex { flex: 1 1 auto; }
.asp-empty { padding: 10px 4px; text-align: center; opacity: 0.55; }
.asp-row { border: 1px solid var(--border-color, #333); border-radius: 6px; padding: 4px 6px;
  display: flex; flex-direction: column; gap: 3px; background: rgba(0,0,0,0.18); }
.asp-row.asp-off { opacity: 0.45; }
.asp-line1 { display: flex; gap: 6px; align-items: center; min-width: 0; }
.asp-toggle { position: relative; display: inline-block; width: 26px; height: 14px; flex: 0 0 auto; margin: 0; }
.asp-toggle input { opacity: 0; width: 0; height: 0; position: absolute; }
.asp-toggle span { position: absolute; inset: 0; border-radius: 8px;
  background: #555; transition: background 0.15s; cursor: pointer; }
.asp-toggle span::after { content: ""; position: absolute; top: 2px; left: 2px; width: 10px; height: 10px;
  border-radius: 50%; background: #ddd; transition: left 0.15s; }
.asp-toggle input:checked + span { background: #4f7cff; }
.asp-toggle input:checked + span::after { left: 14px; }
.asp-name { flex: 1 1 auto; min-width: 60px; text-align: left; height: 22px; padding: 0 6px;
  background: var(--comfy-input-bg, #222); color: inherit;
  border: 1px solid var(--border-color, #333); border-radius: 4px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; font-size: 12px; }
.asp-name:hover { border-color: #4f7cff; }
.asp-str { display: inline-flex; align-items: center; gap: 3px; flex: 0 0 auto; }
.asp-str label { opacity: 0.6; font-size: 11px; }
.asp-str input { width: 52px; height: 22px; padding: 0 4px; font-size: 12px;
  background: var(--comfy-input-bg, #222); color: inherit;
  border: 1px solid var(--border-color, #333); border-radius: 4px; }
.asp-ops { display: inline-flex; gap: 2px; flex: 0 0 auto; }
.asp-ops button { width: 20px; height: 20px; padding: 0; font-size: 11px; line-height: 1;
  background: transparent; color: inherit; border: 1px solid transparent; border-radius: 4px; cursor: pointer; }
.asp-ops button:hover { border-color: var(--border-color, #444); background: rgba(255,255,255,0.06); }
.asp-trig { display: flex; align-items: center; gap: 6px; min-width: 0; opacity: 0.85; }
.asp-trig-text { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  cursor: pointer; font-size: 11px; }
.asp-trig-text:hover { text-decoration: underline dotted; }
.asp-trig-text.asp-dim { opacity: 0.45; font-style: italic; }
.asp-trig-edit { flex: 0 0 auto; }
.asp-backdrop { position: fixed; inset: 0; z-index: 100000; background: rgba(0,0,0,0.55);
  display: flex; align-items: center; justify-content: center; }
.asp-dialog { width: min(480px, 92vw); max-height: 78vh; display: flex; flex-direction: column; gap: 8px;
  background: var(--comfy-menu-bg, #1e1e1e); border: 1px solid var(--border-color, #3a3a3a);
  border-radius: 8px; padding: 12px 14px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); }
.asp-dialog-title { font-size: 13px; font-weight: 600; }
.asp-dialog-input { width: 100%; box-sizing: border-box; height: 26px; padding: 0 6px; font-size: 12px;
  background: var(--comfy-input-bg, #222); color: inherit;
  border: 1px solid var(--border-color, #333); border-radius: 4px; resize: vertical; }
textarea.asp-dialog-input { height: auto; padding: 4px 6px; }
.asp-dialog-row { display: flex; gap: 6px; justify-content: flex-end; align-items: center; }
.asp-picker-list { flex: 1 1 auto; overflow-y: auto; min-height: 200px; max-height: 46vh;
  border: 1px solid var(--border-color, #2e2e2e); border-radius: 4px; background: rgba(0,0,0,0.25); }
.asp-picker-item { padding: 5px 8px; cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.04); }
.asp-picker-item:hover, .asp-picker-item.asp-current { background: rgba(79,124,255,0.16); }
.asp-picker-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
.asp-picker-meta { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; opacity: 0.6; }
.asp-picker-empty { padding: 20px; text-align: center; opacity: 0.55; font-size: 12px; }`;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------- 节点状态与 UI

const nodeState = new WeakMap();

function safeParse(value) {
  let data = null;
  try { data = JSON.parse(value); } catch (_) { /* 容错 */ }
  const rows = Array.isArray(data?.rows) ? data.rows : [];
  const clean = rows.filter((r) => r && typeof r === "object").map((r) => ({
    on: r.on !== false,
    lora: String(r.lora || ""),
    strength_model: Number.isFinite(+r.strength_model) ? +r.strength_model : 1,
    strength_clip: Number.isFinite(+r.strength_clip) ? +r.strength_clip
      : (Number.isFinite(+r.strength_model) ? +r.strength_model : 1),
  }));
  return { rows: clean, preset: typeof data?.preset === "string" ? data.preset : "" };
}

function shortName(name) {
  if (!name) return "＋ 选择 LoRA";
  const base = name.split("/").pop();
  return base.replace(/\.(safetensors|sft|ckpt|pt)$/i, "");
}

function triggerText(info) {
  if (!info) return null;
  const words = info.trigger_words || [];
  return words.length ? `🏷 ${words.join(", ")}` : null;
}

function makeRowEl(row, idx) {
  const el = document.createElement("div");
  el.className = "asp-row" + (row.on ? "" : " asp-off");
  el.dataset.idx = String(idx);
  el.innerHTML = `
    <div class="asp-line1">
      <label class="asp-toggle" title="开关（关闭=该行不生效）">
        <input type="checkbox" data-k="on"><span></span>
      </label>
      <button class="asp-name" title=""></button>
      <span class="asp-str" title="Model 强度"><label>M</label>
        <input type="number" step="0.05" min="-10" max="10" data-k="strength_model"></span>
      <span class="asp-str" title="CLIP 强度"><label>C</label>
        <input type="number" step="0.05" min="-10" max="10" data-k="strength_clip"></span>
      <span class="asp-ops">
        <button data-op="up" title="上移">↑</button>
        <button data-op="down" title="下移">↓</button>
        <button data-op="del" title="删除该行">✕</button>
      </span>
    </div>
    <div class="asp-trig">
      <span class="asp-trig-text asp-dim">触发词加载中…</span>
      <button class="asp-btn asp-trig-edit" data-op="tags" title="手动编辑触发词">✎</button>
    </div>`;
  el.querySelector('[data-k="on"]').checked = !!row.on;
  const nameBtn = el.querySelector(".asp-name");
  nameBtn.textContent = shortName(row.lora);
  nameBtn.title = row.lora || "点击选择 LoRA";
  const sm = el.querySelector('[data-k="strength_model"]');
  const sc = el.querySelector('[data-k="strength_clip"]');
  sm.value = row.strength_model;
  sc.value = row.strength_clip;
  return el;
}

function refreshTriggerLine(st, rowEl) {
  const row = st.rows[Number(rowEl.dataset.idx)];
  const textEl = rowEl.querySelector(".asp-trig-text");
  if (!row || !textEl) return;
  const render = () => {
    const info = findLoraInfo(row.lora);
    const text = triggerText(info);
    textEl.textContent = text || "（无触发词信息 · 点击 ✎ 手动填写）";
    textEl.classList.toggle("asp-dim", !text);
    textEl.title = info?.base_model ? `base: ${info.base_model}` : (text || "");
  };
  if (!loraCache.list.length) {
    getLoras().then(render);
  } else {
    render();
  }
}

function renderRows(st) {
  const wrap = st.el.querySelector(".asp-rows");
  wrap.innerHTML = "";
  st.el.querySelector(".asp-empty").style.display = st.rows.length ? "none" : "";
  st.rows.forEach((row, idx) => {
    const rowEl = makeRowEl(row, idx);
    wrap.appendChild(rowEl);
    refreshTriggerLine(st, rowEl);
  });
}

async function refreshPresetSelect(st) {
  const select = st.el.querySelector(".asp-preset");
  const presets = await getPresets();
  const names = Object.keys(presets).sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = names.length ? "— LoRA 组预设 —" : "— 暂无预设 —";
  select.appendChild(placeholder);
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  }
  select.value = presets[st.presetName] ? st.presetName : "";
}

function commit(st) {
  st.dataWidget.value = JSON.stringify({ rows: st.rows, preset: st.presetName });
  st.node.setDirtyCanvas?.(true, true);
}

function syncIn(node) {
  const st = nodeState.get(node);
  if (!st) return;
  const data = safeParse(st.dataWidget.value);
  st.rows = data.rows;
  st.presetName = data.preset;
  renderRows(st);
  refreshPresetSelect(st);
}

function buildUi(node, dataWidget) {
  const el = document.createElement("div");
  el.className = "asp-lora";
  el.innerHTML = `
    <div class="asp-toolbar">
      <select class="asp-preset" title="切换预设（立即应用到当前行）"></select>
      <button class="asp-btn" data-act="save" title="把当前行覆盖保存到选中预设">保存</button>
      <button class="asp-btn" data-act="saveas" title="用新名字保存当前行">另存为</button>
      <button class="asp-btn asp-danger" data-act="del" title="删除选中预设">删除</button>
      <span class="asp-flex"></span>
      <button class="asp-btn asp-add" data-act="add">＋ 添加 LoRA</button>
    </div>
    <div class="asp-rows"></div>
    <div class="asp-empty">还没有 LoRA，点「＋ 添加 LoRA」开始</div>`;
  return el;
}

function wireEvents(st) {
  const { el } = st;

  el.addEventListener("change", (e) => {
    const k = e.target.dataset?.k;
    const rowEl = e.target.closest(".asp-row");
    if (!rowEl) return;
    const idx = Number(rowEl.dataset.idx);
    const row = st.rows[idx];
    if (!row) return;
    if (k === "on") {
      row.on = e.target.checked;
      rowEl.classList.toggle("asp-off", !row.on);
      commit(st);
    }
  });

  el.addEventListener("input", (e) => {
    const k = e.target.dataset?.k;
    if (!k || !k.startsWith("strength")) return;
    const rowEl = e.target.closest(".asp-row");
    const row = st.rows[Number(rowEl.dataset.idx)];
    if (!row) return;
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v)) {
      row[k] = Math.max(-10, Math.min(10, v));
      commit(st);
    }
  });

  el.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) {
      // 点击触发词文本 = 复制
      const trig = e.target.closest(".asp-trig-text");
      if (trig && !trig.classList.contains("asp-dim")) {
        const ok = await copyText(trig.textContent.replace(/^🏷\s*/, ""));
        toast(ok ? "success" : "error", ok ? "触发词已复制" : "复制失败，请手动选择文本");
      }
      return;
    }

    const act = btn.dataset.act;
    const op = btn.dataset.op;
    const rowEl = btn.closest(".asp-row");
    const idx = rowEl ? Number(rowEl.dataset.idx) : -1;

    if (act === "add") {
      st.rows.push({ on: true, lora: "", strength_model: 1, strength_clip: 1 });
      renderRows(st);
      commit(st);
      const name = await pickLora();
      if (name) {
        st.rows[st.rows.length - 1].lora = name;
        renderRows(st);
        commit(st);
      }
      return;
    }
    if (act === "save") {
      if (!st.rows.length) return toast("warn", "当前没有行可保存");
      if (!st.presetName) return doSaveAs(st);
      const resp = await requestJson("/asp/presets", { name: st.presetName, rows: st.rows });
      toast(resp.ok ? "success" : "error", resp.message || "");
      await getPresets(true);
      return;
    }
    if (act === "saveas") return doSaveAs(st);
    if (act === "del") {
      if (!st.presetName) return toast("warn", "请先在下拉框选中要删除的预设");
      const resp = await requestJson("/asp/presets/delete", { name: st.presetName });
      toast(resp.ok ? "success" : "error", resp.message || "");
      if (resp.ok) st.presetName = "";
      await getPresets(true);
      refreshPresetSelect(st);
      commit(st);
      return;
    }

    if (rowEl && st.rows[idx]) {
      if (op === "up" && idx > 0) {
        [st.rows[idx - 1], st.rows[idx]] = [st.rows[idx], st.rows[idx - 1]];
        renderRows(st); commit(st);
      } else if (op === "down" && idx < st.rows.length - 1) {
        [st.rows[idx + 1], st.rows[idx]] = [st.rows[idx], st.rows[idx + 1]];
        renderRows(st); commit(st);
      } else if (op === "del") {
        st.rows.splice(idx, 1);
        renderRows(st); commit(st);
      } else if (op === "tags") {
        const row = st.rows[idx];
        if (!row.lora) return toast("warn", "请先选择 LoRA");
        const info = findLoraInfo(row.lora);
        const current = (info?.trigger_words || []).join(", ");
        const text = await askText({
          title: `编辑「${shortName(row.lora)}」的触发词（逗号分隔，保存到服务器）`,
          value: current, multiline: true,
        });
        if (text === null) return;
        const words = text.split(/[,，\n]/).map((w) => w.trim()).filter(Boolean);
        const resp = await requestJson("/asp/lora_tags", { lora: row.lora, trigger_words: words });
        toast(resp.ok ? "success" : "error", resp.message || "");
        if (resp.ok) {
          loraCache.at = 0;
          await getLoras(true);
          renderRows(st);
        }
      }
    }

    if (btn.classList.contains("asp-name")) {
      const row = st.rows[idx];
      const name = await pickLora(row?.lora || "");
      if (name && row) {
        row.lora = name;
        renderRows(st);
        commit(st);
      }
    }
  });

  // 预设切换：立即应用
  el.querySelector(".asp-preset").addEventListener("change", async (e) => {
    const name = e.target.value;
    if (!name) { st.presetName = ""; commit(st); return; }
    const presets = await getPresets();
    const preset = presets[name];
    if (!preset || !Array.isArray(preset.rows)) return toast("error", `预设「${name}」读取失败`);
    st.rows = preset.rows.map((r) => ({ ...r }));
    st.presetName = name;
    renderRows(st);
    commit(st);
    toast("success", `已切换到预设「${name}」（${st.rows.length} 行）`);
  });
}

async function doSaveAs(st) {
  if (!st.rows.length) return toast("warn", "当前没有行可保存");
  const name = await askText({ title: "另存为新预设", placeholder: "输入预设名", okText: "保存" });
  if (name === null || !name.trim()) return;
  const resp = await requestJson("/asp/presets", { name: name.trim(), rows: st.rows });
  toast(resp.ok ? "success" : "error", resp.message || "");
  if (resp.ok) {
    st.presetName = name.trim();
    await getPresets(true);
    refreshPresetSelect(st);
    commit(st);
  }
}

// ---------------------------------------------------------------- 节点装配

function setupNode(node) {
  if (nodeState.has(node)) return;
  ensureStyle();
  const dataWidget = node.widgets?.find((w) => w.name === "lora_data");
  if (!dataWidget) {
    console.warn(TAG, "找不到 lora_data widget，前端 UI 未启用");
    return;
  }
  // 隐藏原始 STRING widget（保留序列化；value 由 DOM UI 维护）
  dataWidget.type = "hidden";
  dataWidget.computeSize = () => [0, -4];

  const el = buildUi(node, dataWidget);
  const st = { node, el, dataWidget, rows: [], presetName: "" };
  nodeState.set(node, st);

  const domWidget = node.addDOMWidget("asp_lora_ui", "aspLoraRows", el, {
    hideOnZoom: false,
    serialize: false,
    getMinHeight: () => Math.max(88, el.scrollHeight),
    getMaxHeight: () => 760,
    getHeight: () => Math.max(88, el.scrollHeight),
  });
  st.domWidget = domWidget;

  wireEvents(st);
  syncIn(node);

  // 工作流载入 / 复制粘贴：configure 应用 widgets_values 后按值重建 UI
  const origOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const r = origOnConfigure?.apply(this, args);
    syncIn(this);
    return r;
  };

  // 行数变化时让画布重新计算 DOM widget 高度
  const observer = new ResizeObserver(() => {
    node.setDirtyCanvas?.(true, true);
  });
  observer.observe(el);
  const origOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    try { observer.disconnect(); } catch (_) { /* noop */ }
    return origOnRemoved?.apply(this, args);
  };

  if ((node.size?.[0] || 0) < MIN_NODE_WIDTH) node.size[0] = MIN_NODE_WIDTH;
}

app.registerExtension({
  name: "asp.PowerLoraLoader",
  nodeCreated(node) {
    if (node?.type !== NODE_NAME && node?.comfyClass !== NODE_NAME) return;
    setupNode(node);
  },
  // 工作流加载完毕后兜底同步一次（覆盖 API JSON 导入等路径）
  loadedGraphNode(node) {
    if (node?.type !== NODE_NAME && node?.comfyClass !== NODE_NAME) return;
    setTimeout(() => syncIn(node), 0);
  },
  // ComfyUI 刷新节点定义（R 键）时让 LoRA 列表缓存过期
  async refreshComboInNode(defs) {
    if (defs && defs[NODE_NAME]) {
      loraCache.at = 0;
      await getLoras(true);
    }
  },
});

console.log("[asp-utils] 多 LoRA 加载器前端已加载");
