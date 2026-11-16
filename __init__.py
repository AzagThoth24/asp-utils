# asp-utils 自定义节点包：
#   - asp_lora_loader.py  多 LoRA 加载器（触发词显示/输出、LoRA 组预设、
#                          Nodes 2.0 双渲染兼容的 DOM widget UI）
#   - region_canvas.py    画布 + 人物信息 + 提示词切段/编码/区域链（自 ComfyUI-Krea2-RegionCanvas 收编）
#   - krea2_regional.py   vendored：Krea2LoRA / Krea2RegionalPrompt / Krea2ApplyRegional
#   - krea2_detailer.py   vendored+扩展：Krea2RegionalDetailer + 四个部位细化节点
#   - asp_server/         /asp/* HTTP 路由（LoRA 列表/触发词、预设 CRUD）
#
# 区域节点沿用原包的注册 key（RegionRectCanvas 等），旧工作流零影响。
# 注意：不要与 ComfyUI-Krea2-RegionCanvas 部署在同一个 ComfyUI 实例（节点 key 重复）。

__version__ = "0.1.0"

import logging

_LOG = logging.getLogger("asp")

from .krea2_regional import (NODE_CLASS_MAPPINGS as _REGIONAL_MAPPINGS,  # noqa: E402
                             NODE_DISPLAY_NAME_MAPPINGS as _REGIONAL_DISPLAY)
from .krea2_detailer import (DETAILER_NODE_CLASS_MAPPINGS,  # noqa: E402
                             DETAILER_NODE_DISPLAY_NAME_MAPPINGS)
from .region_canvas import (NODE_CLASS_MAPPINGS as _CANVAS_MAPPINGS,  # noqa: E402
                            NODE_DISPLAY_NAME_MAPPINGS as _CANVAS_DISPLAY)
from .asp_lora_loader import (NODE_CLASS_MAPPINGS as _ASP_MAPPINGS,  # noqa: E402
                              NODE_DISPLAY_NAME_MAPPINGS as _ASP_DISPLAY)

NODE_CLASS_MAPPINGS = {**_REGIONAL_MAPPINGS,
                       **DETAILER_NODE_CLASS_MAPPINGS,
                       **_CANVAS_MAPPINGS,
                       **_ASP_MAPPINGS}

NODE_DISPLAY_NAME_MAPPINGS = {**_REGIONAL_DISPLAY,
                              **DETAILER_NODE_DISPLAY_NAME_MAPPINGS,
                              **_CANVAS_DISPLAY,
                              **_ASP_DISPLAY}

# server 端路由（/asp/*）；无 server 环境下仅告警不影响节点加载
try:
    from .asp_server import register_routes  # noqa: E402,F401
except Exception as err:  # pragma: no cover
    _LOG.warning("[asp] server 路由模块加载失败: %s", err)

# 前端（web/js/*.js）必须声明这个才会被加载
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
