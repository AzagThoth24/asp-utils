# asp-utils

个人 ComfyUI 自定义节点包。

## 功能

### 1. 多 LoRA 加载器（AspPowerLoraLoader）

一个节点加载任意多个 LoRA（思路参考 rgthree Power Lora Loader：逐行串行叠加，
复用官方 `LoraLoader`，双 CLIP 等特殊情况无需关心）：

- 每行独立：开关、LoRA 选择（可搜索弹窗，显示触发词 / base model）、
  Model 与 CLIP 双强度（-10 ~ 10）
- **自动显示触发词**：每行下方自动展示该 LoRA 的触发词，点击即复制；
  缺失时可点 ✎ 手动填写（保存到服务器，全员/全浏览器生效）
- **第三个输出 `trigger_words`**：已启用 LoRA 的触发词逗号拼接，可直接
  接进提示词链（配合文本拼接节点使用）
- **LoRA 组预设**：工具栏下拉切换预设、保存 / 另存为 / 删除；
  预设存在服务器端 `user/lora_presets.json`
- 行数据全部序列化进一个 `lora_data` STRING widget（JSON），不依赖动态输入
  hack —— UI 格式与 API 格式工作流都天然兼容，纯 API 调用也可以直接手写 JSON：

  ```json
  {"rows": [{"on": true, "lora": "my_lora.safetensors",
             "strength_model": 0.8, "strength_clip": 0.8}],
   "preset": ""}
  ```

**Nodes 2.0 兼容**：前端采用 DOM widget 方案（官方迁移指南推荐路径，
ADR ECS-WIDGETS-0023 双渲染桥接），不 patch 原型、不自绘 canvas，
旧画布与 Nodes 2.0 beta 渲染下均可正常交互。远端验证环境：
ComfyUI v0.33.1 / 前端 1.48.7。

### 2. Krea2 区域节点（自 ComfyUI-Krea2-RegionCanvas 收编）

注册 key 与原包完全一致（旧工作流零影响）：

| 分类 | 节点 |
|---|---|
| `mask/krea2_region` | RegionRectCanvas（多人区域画布）、Krea2CharacterInfo、Krea2RegionalDetailer、Krea2 细化-人体/脸部/眼睛/手部 |
| `conditioning/krea2_regional` | Krea2LoRA、Krea2RegionalPrompt、Krea2ApplyRegional |

迁移时顺带修复了原包 `krea2_detailer.py` 的一个 NameError
（`_detail_one_part` 循环里的 `pidx` 未定义，细化采样时会崩溃）。

> ⚠️ 本包含原包全部节点，**不要**与 `ComfyUI-Krea2-RegionCanvas`
> 同时部署在同一个 ComfyUI 实例（节点 key 重复注册）。

## 安装 / 部署

无额外 Python 依赖。有则拉取、无则克隆（服务器同步脚本 / Dockerfile `RUN` 均可直接用）：

```bash
cd /comfyui/custom_nodes && \
  if [ -d asp-utils/.git ]; then
    if git -C asp-utils diff --quiet && git -C asp-utils diff --cached --quiet; then
      git -C asp-utils pull --ff-only \
        || echo "拉取失败（多为网络问题）。AutoDL 可先执行: source /etc/network_turbo 再重试"
    else
      echo "asp-utils 本地有改动，处理方式: git -C asp-utils stash && git -C asp-utils pull --ff-only"
    fi
  else
    git clone https://github.com/AzagThoth24/asp-utils.git
  fi
```

大陆服务器直连 GitHub 报 `GnuTLS recv error` 时，先 `source /etc/network_turbo`（AutoDL）再拉取，完成后 `unset http_proxy https_proxy`。

装完重启 ComfyUI 即可。

## 触发词从哪来（优先级从高到低）

1. `user/lora_tags.json` —— UI 里 ✎ 手动填写的覆盖（自训 LoRA 推荐用这个）
2. LoRA 同目录的 `xxx.civitai.info` / `xxx.json` 边车文件里的 `trainedWords`
3. safetensors 内嵌 metadata 的 `modelspec.trigger_phrase`
4. `ss_tag_frequency` 高频标签（兜底，仅当前面都缺失时）

## HTTP 接口（本包自带，前缀 `/asp`）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/asp/loras?refresh=1` | LoRA 列表 + 触发词 + base_model |
| GET | `/asp/presets` | 预设列表 |
| POST | `/asp/presets` | `{name, rows}` 新建/覆盖预设 |
| POST | `/asp/presets/delete` | `{name}` 删除预设 |
| POST | `/asp/lora_tags` | `{lora, trigger_words}` 覆盖触发词 |

## 用户数据

`user/` 目录（首次运行自动创建，git 已忽略）：

- `lora_presets.json` —— LoRA 组预设
- `lora_tags.json` —— 触发词手动覆盖

迁移/备份：拷走这两个文件即可。

## 开发与测试

测试不依赖 ComfyUI 运行环境（AST 提取生产代码中的函数/类）：

```
python -m unittest discover -s tests -v
```

（`test_regressions.py` 需要 torch；`test_asp_lora.py` 纯标准库。）

## 远端 E2E 验收清单

- [ ] 节点可添加；默认宽度合适，工具栏/空行提示正常
- [ ] 添加 LoRA → 弹窗可搜索、显示触发词；选择后行内自动显示触发词
- [ ] 开关 / M、C 强度修改生效（强度 0 或关闭的行不加载）
- [ ] 工作流保存 → 重载，行数据完整（UI 格式与 API 格式各验证一次）
- [ ] 预设：另存为 → 下拉出现；改行 → 保存覆盖；切换预设行正确替换；删除生效
- [ ] 触发词 ✎ 编辑 → 刷新后仍在（`user/lora_tags.json` 有记录）
- [ ] `trigger_words` 输出接到文本节点，内容为启用 LoRA 的触发词
- [ ] 带 2 个 LoRA 实际出图，效果与分别串两个 LoraLoader 一致
- [ ] 前端 Logo 菜单切换 Nodes 2.0 开关，节点 UI 与交互正常
