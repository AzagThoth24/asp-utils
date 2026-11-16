// 模式参照 KJNodes web/js/editors/editor_base.js（节点内嵌画布标杆）：
// 固定画布像素尺寸、addDOMWidget 传 getHeight 三回调、hideOnZoom:false、
// serialize:false、鼠标坐标用 getBoundingClientRect 换算成画布内坐标。
// v2：人物 LoRA/强度 从人物信息节点（输入连线）读取；输出重排后的兼容版。
//
// v3 修复（画布不显示 / 点击无效 / 节点忽大忽小）：
//   1) area() 改为纯"画布内坐标"（位图左上角为原点）。旧版 y0=node.size[1]-CH-30
//      混入了节点坐标（约 590+），而画布位图只有 CH(~250) 高：矩形全画在位图外
//      =画布空白；鼠标坐标又是画布内的，跟节点坐标一比永远不命中=点击无效，
//      新建矩形的 y 也恒被压成 0=拖框无效。
//   2) 删掉每 0.8s 强制 setSize 的兜底定时器：它和前端布局互相拉扯，节点会忽大
//      忽小。改为 KJNodes 的做法——onResize 钩子 + ResizeObserver 只负责重绘，
//      兜底放大只在节点创建时做一次（只放大、不缩小）。
//   3) 修复 widget 回调包装里引用了不存在的 `node` 变量（ReferenceError 被
//      try/catch 吞掉，导致改 width/height 后不重绘）。
//   4) 拖拽用 setPointerCapture：拖到画布外不丢手。
//
// v4（节点缩水 / 边框不贴内容）：
//   1) 控件高度改为按元素实际渲染高度（offsetHeight）回报，不再只按位图比例估算。
//   2) 最小高度兜底改用 litegraph 自己的 computeSize()（真实内容高度），取不到才用
//      估算值——节点边框正好贴住画布和提示文字，底部不再留大片空白。
//   3) onConfigure（载入工作流、恢复保存尺寸时）重新兜底一次最小高度，旧工作流里
//      存的小尺寸不会再把节点压瘪（这就是"还会变小"的来源：创建时兜底会被
//      configure 恢复的旧 size 覆盖）。
//   4) 画布加 1px 描边，永远能看清画布范围。
import { app } from "/scripts/app.js";

console.log("[RegionRectCanvas] extension script loaded");

const MAX_RECTS = 5;
const COLORS = ["#4da3ff", "#ff8a4d", "#7ddf64", "#c88cff", "#ffd84d"];
const CANVAS_W = 460;       // 画布位图宽（CSS 上按节点宽度 100% 缩放）
const MARGIN_BOTTOM = 30;   // 画布下方提示文字高度
const WIDGETS_H = 620;      // 画布上方控件区的估算高度（仅当取不到真实布局时兜底）
const HANDLE = 10;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const clamp01 = (v) => clamp(v, 0, 1);

function widgetVal(node, name, fallback) {
  const w = (node.widgets || []).find((w) => w.name === name);
  if (!w || w.value === undefined || w.value === null || w.value === "") return fallback;
  return w.value;
}

// width/height 若接了输入（如 PrimitiveInt），实时读源头节点的值
function getWH(node) {
  let W = +widgetVal(node, "width", 2048) || 2048;
  let H = +widgetVal(node, "height", 1136) || 1136;
  try {
    for (const [iname, k] of [["width", "W"], ["height", "H"]]) {
      const inp = (node.inputs || []).find((i) => i.name === iname);
      if (inp && inp.link && app.graph && app.graph.links) {
        const lk = app.graph.links[inp.link];
        const src = lk && app.graph._nodes_by_id ? app.graph._nodes_by_id[lk.origin_id] : null;
        if (src && src.widgets && src.widgets.length) {
          const v = +src.widgets[0].value;
          if (isFinite(v) && v > 0) { if (k === "W") W = v; else H = v; }
        }
      }
    }
  } catch (e) { /* 读不到就退回控件值 */ }
  return { W, H, CH: Math.max(90, Math.round((CANVAS_W * H) / W)) };
}

// rects 恒为 5 个槽位；w/h 同时为 0 的槽 = 空（零掩码）
function getRects(node) {
  let data;
  try { data = JSON.parse(widgetVal(node, "rects_data", "{}")); } catch (e) { data = {}; }
  const raw = Array.isArray(data && data.rects) ? data.rects : [];
  const rects = [];
  for (let i = 0; i < MAX_RECTS; i++) {
    const r = raw[i] || {};
    const w = clamp01(+r.w || 0), h = clamp01(+r.h || 0);
    rects.push({
      x: clamp01(+r.x || 0),
      y: clamp01(+r.y || 0),
      w: w < 0.02 ? 0 : w,
      h: h < 0.02 ? 0 : h,
    });
  }
  return rects;
}

function setRects(node, rects) {
  const w = (node.widgets || []).find((w) => w.name === "rects_data");
  if (w) w.value = JSON.stringify({ rects });
}

function shortLora(name) {
  return (name || "") === "None" ? "" :
    (name || "").split(/[\\/]/).pop().replace(/\.safetensors$/i, "");
}

// 人物 lora/强度 住在人物信息节点里：顺着输入连线读上游节点的 widget
function personInfoWidget(node, i, wname) {
  try {
    const inp = (node.inputs || []).find((x) => x.name === `人物信息${i + 1}`);
    if (inp && inp.link && app.graph && app.graph.links) {
      const lk = app.graph.links[inp.link];
      const src = lk && app.graph._nodes_by_id ? app.graph._nodes_by_id[lk.origin_id] : null;
      if (src) return widgetVal(src, wname, null);
    }
  } catch (e) { /* 读不到就退回默认 */ }
  return null;
}
function loraName(node, i) {
  return String(personInfoWidget(node, i, "lora") || "None");
}
function loraStrength(node, i) {
  return +personInfoWidget(node, i, "强度") || 1;
}

// DOM 控件实际渲染高度（画布 + 提示文字）；元素还没挂到页面上时退回估算值
function widgetH(node) {
  const el = node._rc && node._rc.element;
  const h = el ? el.offsetHeight : 0;
  return h > 0 ? h : getWH(node).CH + MARGIN_BOTTOM;
}

// 高度兜底：节点高度不得低于真实内容高度。computeSize() 是 litegraph 自己的
// 内容高度计算（含上面所有控件和本画布），用它节点边框就正好贴住内容；
// 只放大、不缩小，绝不跟前端布局对着干。
function ensureMin(node) {
  let need = 0;
  try {
    if (node.computeSize) need = node.computeSize(node.size[0])[1] || 0;
  } catch (e) { /* 忽略 */ }
  if (!need) need = WIDGETS_H + widgetH(node);
  if (node.size[1] < need) node.setSize([node.size[0], need]);
}

// 画布内坐标系：位图即坐标系，矩形、命中检测、鼠标全部用它（v3 修复点）
function area(node) {
  return { x0: 0, y0: 0, w: CANVAS_W, h: getWH(node).CH };
}

function hitTest(node, mx, my) {
  const a = area(node);
  const rects = getRects(node);
  for (let i = MAX_RECTS - 1; i >= 0; i--) {
    const r = rects[i];
    if (!r.w || !r.h) continue;
    const x0 = a.x0 + r.x * a.w, y0 = a.y0 + r.y * a.h;
    const x1 = x0 + r.w * a.w, y1 = y0 + r.h * a.h;
    if (mx >= x1 - HANDLE && my >= y1 - HANDLE && mx <= x1 && my <= y1) {
      return { i, part: "resize" };
    }
    if (mx >= x0 && mx <= x1 && my >= y0 && my <= y1) {
      return { i, part: "move" };
    }
  }
  return null;
}

function handleDown(node, mx, my) {
  const a = area(node);
  const nx = clamp01((mx - a.x0) / a.w);
  const ny = clamp01((my - a.y0) / a.h);
  const hit = hitTest(node, mx, my);
  const rects = getRects(node);
  if (hit) {
    const r = rects[hit.i];
    node._rc.drag = { i: hit.i, part: hit.part, px: nx, py: ny,
                      ox: r.x, oy: r.y, ow: r.w, oh: r.h };
    return;
  }
  // 拖空白：找一个空槽新建
  let slot = -1;
  for (let i = 0; i < MAX_RECTS; i++) {
    if (!rects[i].w || !rects[i].h) { slot = i; break; }
  }
  if (slot < 0) return;
  node._rc.drag = { i: slot, part: "create", px: nx, py: ny };
  rects[slot] = { x: nx, y: ny, w: 0, h: 0 };
  setRects(node, rects);
  paint(node);
}

function handleMove(node, mx, my) {
  const d = node._rc && node._rc.drag;
  if (!d) return;
  const a = area(node);
  const nx = clamp01((mx - a.x0) / a.w);
  const ny = clamp01((my - a.y0) / a.h);
  const rects = getRects(node);
  const r = rects[d.i];
  if (d.part === "create") {
    r.x = Math.min(d.px, nx);
    r.y = Math.min(d.py, ny);
    r.w = Math.abs(nx - d.px);
    r.h = Math.abs(ny - d.py);
  } else if (d.part === "move") {
    r.x = clamp01(d.ox + (nx - d.px));
    r.y = clamp01(d.oy + (ny - d.py));
  } else if (d.part === "resize") {
    r.w = clamp(d.ow + (nx - d.px), 0.02, 1 - r.x);
    r.h = clamp(d.oh + (ny - d.py), 0.02, 1 - r.y);
  }
  setRects(node, rects);
  paint(node);
}

function handleUp(node) {
  if (node._rc) node._rc.drag = null;
}

function handleDelete(node, mx, my) {
  const hit = hitTest(node, mx, my);
  if (hit) {
    const rects = getRects(node);
    rects[hit.i] = { x: 0, y: 0, w: 0, h: 0 }; // 清空该人物槽（槽位固定，不重排编号）
    setRects(node, rects);
    paint(node);
  }
}

function paint(node) {
  try {
    const rc = node._rc;
    if (!rc || !rc.canvas) return;
    const a = area(node);
    const rects = getRects(node);
    const ctx = rc.ctx;

    if (rc.canvas.width !== a.w) rc.canvas.width = a.w;
    if (rc.canvas.height !== a.h) rc.canvas.height = a.h;
    ctx.clearRect(0, 0, rc.canvas.width, rc.canvas.height);
    ctx.fillStyle = "#2b2b31";
    ctx.fillRect(0, 0, rc.canvas.width, rc.canvas.height);

    for (let i = 0; i < MAX_RECTS; i++) {
      const r = rects[i];
      if (!r.w || !r.h) continue;
      const x0 = a.x0 + r.x * a.w, y0 = a.y0 + r.y * a.h;
      const x1 = x0 + r.w * a.w, y1 = y0 + r.h * a.h;
      ctx.fillStyle = COLORS[i] + "55";
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeStyle = COLORS[i];
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.fillStyle = COLORS[i];
      ctx.font = "12px sans-serif";
      const front = i === 0 ? " (前景)" : "";
      ctx.fillText(`人物${i + 1}${front}`, x0 + 4, y0 + 14);
      const lname = shortLora(loraName(node, i));
      if (lname) {
        ctx.fillStyle = "#e8e8e8";
        ctx.fillText(lname, x0 + 4, y0 + 28);
      }
      // resize handle
      ctx.fillStyle = COLORS[i];
      ctx.fillRect(x1 - HANDLE, y1 - HANDLE, HANDLE, HANDLE);
    }
  } catch (e) { console.log("[RegionRectCanvas] paint error:", e); }
}

function setup(node) {
  node._rc = { drag: null, canvas: null, ctx: null };

  const element = document.createElement("div");
  const canvas = document.createElement("canvas");
  const { CH } = getWH(node);
  canvas.width = CANVAS_W;
  canvas.height = CH;
  canvas.style.cssText = "width:100%;display:block;cursor:crosshair;touch-action:none;border:1px solid #60606c;box-sizing:border-box;";
  element.appendChild(canvas);
  const hint = document.createElement("div");
  hint.style.cssText = "font-size:10px;color:#999;padding:2px 4px;line-height:1.5;";
  hint.textContent = "拖空白=新建(至多5个) 拖内部=移动 拖右下角=改大小 右键矩形=清空该人物槽 矩形1=人物1=前景(区域链首)";
  element.appendChild(hint);
  node._rc.canvas = canvas;
  node._rc.ctx = canvas.getContext("2d");
  node._rc.element = element;

  node.addDOMWidget("region_canvas", "RegionCanvasWidget", element, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => widgetH(node),
    getMaxHeight: () => widgetH(node),
    getHeight: () => widgetH(node),
  });

  const toLocal = (e) => {
    const b = canvas.getBoundingClientRect();
    return {
      mx: (e.clientX - b.left) * (canvas.width / Math.max(1, b.width)),
      my: (e.clientY - b.top) * (canvas.height / Math.max(1, b.height)),
    };
  };
  canvas.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    const p = toLocal(e);
    if (e.button === 2) { handleDelete(node, p.mx, p.my); return; }
    try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* 老前端没有就算了 */ }
    handleDown(node, p.mx, p.my);
  });
  canvas.addEventListener("pointermove", (e) => {
    const p = toLocal(e);
    if (!node._rc.drag) {
      const hit = hitTest(node, p.mx, p.my);
      canvas.style.cursor = hit ? (hit.part === "resize" ? "nwse-resize" : "move") : "crosshair";
      return;
    }
    handleMove(node, p.mx, p.my);
  });
  canvas.addEventListener("pointerup", () => handleUp(node));
  canvas.addEventListener("pointercancel", () => handleUp(node));
  canvas.addEventListener("pointerleave", () => handleUp(node));
  canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); e.stopPropagation(); });

  node._rc.repaint = () => paint(node);
  node._rcResizeObs = new ResizeObserver(() => {
    paint(node);
    try { ensureMin(node); } catch (e) { /* 忽略 */ }
  });
  node._rcResizeObs.observe(element);

  // 节点被拖角缩放时跟随重绘（只重绘、不反向 setSize，避免和前端布局拉扯）
  const prevOnResize = node.onResize;
  node.onResize = function (...args) {
    const rr = prevOnResize ? prevOnResize.apply(this, args) : undefined;
    paint(node);
    return rr;
  };

  // 宽度下限只在创建时做一次；高度交给 ensureMin（按真实内容贴齐）
  if (node.size[0] < 380) node.setSize([380, node.size[1]]);
  ensureMin(node);
  paint(node);

  node.onRemoved = () => {
    if (node._rcResizeObs) node._rcResizeObs.disconnect();
  };
}

app.registerExtension({
  name: "Krea2.RegionRectCanvas",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "RegionRectCanvas") return;
    console.log("[RegionRectCanvas] node definition matched");
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const self = this;
      console.log("[RegionRectCanvas] node created");
      setup(self);
      // 等前端把 widget 全部建好后：首绘 + 包装 width/height 回调（改尺寸后重绘 +
      // 高度兜底，只放大不缩小——不与前端布局拉扯）
      setTimeout(() => {
        for (const w of self.widgets || []) {
          if (!["width", "height"].includes(w.name)) continue;
          const cb = w.callback;
          w.callback = function (...args) {
            const rr = cb ? cb.apply(this, args) : undefined;
            try { ensureMin(self); paint(self); } catch (e) { /* 忽略 */ }
            return rr;
          };
        }
        try { self._rc.repaint && self._rc.repaint(); } catch (e) { /* 忽略 */ }
      }, 50);
      return r;
    };
    // 载入工作流时会用保存的旧尺寸覆盖创建时的尺寸，这里重新兜底一次，
    // 否则旧工作流会把节点压瘪（"还会变小"的另一个来源）
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const r = onConfigure ? onConfigure.apply(this, args) : undefined;
      if (this._rc) {
        try { ensureMin(this); paint(this); } catch (e) { /* 忽略 */ }
      }
      return r;
    };
  },
});

console.log("[RegionRectCanvas] all set");
