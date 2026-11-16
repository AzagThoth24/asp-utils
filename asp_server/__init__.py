"""asp-utils 的 server 端 HTTP 路由（前缀 /asp）。

在 ComfyUI 进程内自动注册（import 本包时）；无 PromptServer 的环境下
（单元测试 / 纯导入）静默跳过。

  GET  /asp/loras[?refresh=1]   LoRA 列表 + 触发词 + base_model
  GET  /asp/presets             预设列表
  POST /asp/presets             {name, rows}          新建/覆盖预设
  POST /asp/presets/delete      {name}                删除预设
  POST /asp/lora_tags           {lora, trigger_words} 手动覆盖触发词
"""

import logging

from . import lora_info, presets

_LOG = logging.getLogger("asp")

# 让 lora_info 感知手动覆盖文件的位置
lora_info.set_tags_override_path(presets.USER_DIR / presets.TAGS_FILE)

__all__ = ["register_routes", "lora_info", "presets"]


def register_routes(prompt_server):
    from aiohttp import web

    routes = prompt_server.routes

    async def get_loras(request):
        force = request.rel_url.query.get("refresh") in ("1", "true", "True")
        return web.json_response({"loras": lora_info.scan_loras(force=force)})

    async def get_presets(request):
        return web.json_response({"presets": presets.get_presets()})

    async def _json_body(request):
        try:
            data = await request.json()
        except Exception:
            return None, web.json_response({"ok": False, "message": "请求体不是合法 JSON"},
                                           status=400)
        if not isinstance(data, dict):
            return None, web.json_response({"ok": False, "message": "请求体必须是对象"},
                                           status=400)
        return data, None

    async def post_preset(request):
        data, err = await _json_body(request)
        if err:
            return err
        ok, message = presets.save_preset(data.get("name"), data.get("rows"))
        return web.json_response({"ok": ok, "message": message,
                                  "presets": presets.get_presets()})

    async def post_preset_delete(request):
        data, err = await _json_body(request)
        if err:
            return err
        ok, message = presets.delete_preset(data.get("name"))
        return web.json_response({"ok": ok, "message": message,
                                  "presets": presets.get_presets()})

    async def post_lora_tags(request):
        data, err = await _json_body(request)
        if err:
            return err
        words = data.get("trigger_words", data.get("words"))
        ok, message = presets.set_tags(data.get("lora"), words)
        if ok:
            # 覆盖变化后强制下次重扫（缓存键不含 tags 文件 mtime）
            lora_info._CACHE.clear()
        return web.json_response({"ok": ok, "message": message})

    routes.get("/asp/loras")(get_loras)
    routes.get("/asp/presets")(get_presets)
    routes.post("/asp/presets")(post_preset)
    routes.post("/asp/presets/delete")(post_preset_delete)
    routes.post("/asp/lora_tags")(post_lora_tags)
    _LOG.info("[asp] /asp/* 路由已注册")


# ComfyUI 进程内自动挂载；失败不影响节点加载
try:
    from server import PromptServer  # noqa: E402
except Exception:  # 无 server 环境（测试/纯导入）
    PromptServer = None

if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    try:
        register_routes(PromptServer.instance)
    except Exception as err:  # 路由注册失败只警告
        _LOG.warning("[asp] 路由注册失败: %s", err)
