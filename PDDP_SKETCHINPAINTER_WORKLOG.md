# PDDP × SketchInpainter 适配工作记录

最后更新：2026-08-10（训练链路与局部 Sketch 协议修复完成）

## 固定目标与路径

- Fork：`git@github.com:148here/PDDP.git`
- 初始 upstream/fork commit：`39a6dc1613d4e2af94db1e1553b2a5ea10304115`
- 服务器项目：`/cpfs01/projects-SSD/cfff-27504eab520e_SSD/zwz_42312/yza/PDDP_sketch_inpainting`
- Conda prefix：`/home/zwz_42312/conda_envs/pddp_sketch_inpainting`
- 产物根目录：`/home/zwz_42312/PDDPoutputs`
- 上传目录：`/home/zwz_42312/temp/`
- Canonical conditions：`/home/zwz_42312/SketchInpainter_outputs/difficulty_grouped_evaluation_v1/canonical_conditions.jsonl`

## 已锁定实验协议

- 原生 PDDP 分辨率：RGB/mask `256×256`、VQ token `32×32`、sketch `224×224`。
- 训练数据：ArtBench train 与 Mural1 train，原始权重 `1.0/0.1`；不加入 COCO replay。
- 约 1% 训练源图以 `sample_id` 稳定哈希划为 validation，不使用论文测试集调参。
- mask 使用 Mural1 mask pool，1/2 张概率各 0.5，允许 90 度旋转，白色表示 hole。
- MuGE：`alpha=1.0`、推理 seed 42、黑线白底。
- sketch 形变参数与当前 SketchInpainter Stage2 配置一致；随机条件由 `(global_seed, epoch, sample_id, index)` 稳定决定。
- PDDP sketch：只保留 hole 内线条，按 hole bbox 1.2 倍裁剪，等比缩放并白底 padding 到 224。
- PDDP 训练从官方 checkpoint 初始化，VQ-VAE 冻结，完整微调 PDDP 可训练模块；单 GPU 0。
- PDDP 无文本接口，canonical prompt 明确忽略；推理结果在原始分辨率按 exact hole 合成，hole 外严格为 GT。

## 第一阶段状态

| 状态 | 工作 |
|---|---|
| DONE | 克隆用户 fork 到固定服务器路径 |
| DONE | 新增精简服务器 Conda 环境清单，独立于现有 `sketch` 环境 |
| DONE | 修复 VQ device 判断、worker 配置、validation drop-last 与 epoch 传播 |
| DONE | 新增 collision-safe JSONL manifest 与逐样本 `.npy` VQ token 协议 |
| DONE | 新增 ArtBench/Mural1 manifest 构建、MuGE edge 和 VQ token 预处理入口 |
| DONE | 新增 SketchInpainter Stage2 mask/sketch 训练适配器 |
| DONE | 新增单 GPU 微调配置和启动脚本，支持 checkpoint resume 参数透传 |
| DONE | 新增 canonical 702 条批量推理脚本，支持 dry-run、哈希审计与缺失续跑 |
| DONE | 创建独立 Conda 环境并完成 CPU-only import 验证 |
| DONE | 完成真实 SketchInpainter 数据路径、DataLoader、cache schema 和确定性单元测试 |
| DONE | 完成训练集 manifest scan 与 canonical 702 条推理 dry-run |

## 第一阶段验收结果

- 第一阶段实现与验收 commit：`4af173c5d439e23370e48ed1510c3d3ef59090c6`（本段审计记录会产生一个后续文档 commit）。
- Python `3.8.20`、PyTorch `1.13.1`、torchvision `0.14.1`、OpenCV `4.6.0`。
- CPU import：PDDP、SketchInpainter `make_sketch_from_edge` 和适配数据集全部通过；验证时 `CUDA_VISIBLE_DEVICES` 为空，`torch.cuda.is_available() == false`。
- 单元测试：`8 passed`，包括实际 SketchInpainter sketch 构造和默认 DataLoader collate。
- manifest dry-run：ArtBench train `51,300`、Mural1 train `1,664`，合计 `52,964`；稳定哈希 train/validation 为 `52,407/557`；未写正式 manifest。
- canonical 推理 dry-run：`702` 条、`351` 个唯一源图、ArtBench/COCO/Mural1=`300/300/102`、两轮各 `351`、缺失或无效输入 `0`；未加载 checkpoint。
- 本阶段未生成 MuGE edge、未生成 VQ token、未加载上传权重、未执行 forward、训练或正式推理。
- Conda 环境占用约 `9.4G`；安装日志：`/home/zwz_42312/PDDPoutputs/setup/conda_create.log`。

关键文件 SHA-256：

- `environment.server.yml`：`6f5dbec8401a7f881f585334c03106139a719fbec842030c6730f42da4083a70`
- `configs/sketchinpainter_finetune.yaml`：`fa4ea31d802d2907399256adab1d6491e315fb97bf5c139720bf473057a2a7ad`
- `scripts/sketchinpainter/batch_inference.py`：`a01e50e24b10eeecf69492a92fc9b933b25a32bc5edc08cba5e79a1a9578a93f`
- `scripts/sketchinpainter/build_manifest.py`：`dd7cd94d2650aef5518714fb68be6f7b3f2416603b8dbb4019d453a24f883d5e`

## 已处理问题

- 原环境文件同时包含 CUDA 11.6 与 cudatoolkit 10.2；已改为最小 CUDA 11.6 runtime 环境。
- Conda 首次求解把 defaults Pillow 与 conda-forge libtiff 混装，造成 `libtiff.so.5` 缺失；环境清单已将 Pillow/libtiff 同时 pin 到 defaults ABI，CPU import 复验通过。
- 首次递归 manifest scan 将 Mural1 test 的 51 张图纳入；现已强制优先扫描数据根的 `train/`，复验为 Mural1 `1,664`。
- mask 数量可为 1 或 2 时，变长 rotation tensor 会破坏默认 collate；已改为审计字符串并通过 DataLoader 测试。

## 当前权重状态

下列文件仅为此前只读观察结果；上传未完成前不得标记为可用，也不得加载：

- `/home/zwz_42312/temp/downloaded_checkpoints/000297e_1343979iter.pth`：疑似官方 PDDP checkpoint，待大小稳定、结构审计和 SHA-256。
- `/home/zwz_42312/temp/downloaded_checkpoints/last.ckpt`：疑似 OpenImages VQ-VAE，待大小稳定、结构审计和 SHA-256。

## 下一阶段 TODO（不得在第一阶段自动执行）

1. 连续两次检查权重大小和 mtime，确认上传结束；计算 SHA-256 并只读检查 checkpoint 顶层键。
2. 运行 manifest 正式生成，审计 ArtBench/Mural1 数量、重复 ID 与 1% validation。
3. 在 detached screen 中依次生成 MuGE edge cache 和 VQ tokens；使用 `--resume` 恢复，审计覆盖率和哈希。
4. 构造 dataloader 并做一个 CPU batch 检查，再做一个受限 GPU forward/backward smoke test。
5. 在 detached screen 中启动单 GPU 微调，保存配置、commit、环境清单和日志。
6. 选定 checkpoint 后对 702 个 canonical conditions 批量推理，要求失败 0、hole 外 exact GT、输入哈希一致。
7. 接入既有 Mixed Easy/Medium/Hard 离线指标，写入对比模型大表和外部模型 TODO。

## 准备好的命令（尚未执行）

```bash
conda run -p /home/zwz_42312/conda_envs/pddp_sketch_inpainting \
  bash scripts/server/prepare_sketchinpainter_data.sh

conda run -p /home/zwz_42312/conda_envs/pddp_sketch_inpainting \
  bash scripts/server/train_sketchinpainter_finetune.sh

CHECKPOINT=/path/to/final.pth conda run -p /home/zwz_42312/conda_envs/pddp_sketch_inpainting \
  bash scripts/server/infer_canonical_702.sh
```

以上三条均属于后续阶段。本阶段不得执行。

## 2026-08-08 预训练 inference-only smoke

- 状态：`DONE`；未运行预处理、训练、backward 或正式 702 条推理。
- 代码 commit：`677d50cedbeb8716af2c3ca5dbf334e653b1112c`。
- PDDP 权重：`/home/zwz_42312/temp/downloaded_checkpoints/000297e_1343979iter.pth`，大小 `4,557,700,568`，SHA-256 `f16d0c6519601840b8f17645a5e3cea048ff135b0bb50c700fc0a97b695dcfe1`。
- VQ-VAE 权重：`/home/zwz_42312/temp/downloaded_checkpoints/last.ckpt`，大小 `376,581,823`，SHA-256 `5cd6c74810ab97e00e942c25403f73afc081e8b19987b31ec0d9ff5b68e7ab14`。
- 确定性选择：ArtBench、COCO、Mural1 各一张 Easy 和 Hard，均固定 `round_001`；清单见输出目录的 `selection.json`。
- 服务器输出：`/home/zwz_42312/PDDPoutputs/smoke/pretrained_canonical_v1/`。
- 本地人工检查副本：`D:/Coding/lab/TSA-inpainting/temp/PDDP_pretrained_smoke_v1/`。
- 选择、权重审计、统计和总图：服务器输出目录中的 `selection.json`、`checkpoint_audit.json`、`smoke_summary.json`、`contact_sheet.png`。
- 验收：6/6 成功；使用 EMA transformer；基础模型与 EMA 的核心 missing/unexpected key 均为空；6/6 composite 在 hole 外逐像素严格等于 GT；复制后的输出哈希与 metadata 全部一致；自动严重退化标记为 0。
- contact sheet SHA-256：`7d3ad17a691918737f38838a78033610788c45eae6838530fb90a1994a94fe6b`。
- 初步人工结论：权重和推理链路可用，输出有结构、有限且非近常量；但预训练质量不足以作为最终结果。COCO-Hard 有严重人物/肢体语义错误，ArtBench-Hard 与 Mural 样本存在接缝或局部色彩/内容偏差。正式预处理和微调前须等待人工许可。
- 恢复记录：首次 screen 在加载模型前因最小环境缺少上游 CLIP tokenizer 依赖而停止；补装并固化 `ftfy==6.1.1`、`regex==2022.8.17` 后，在同名 detached screen 中成功重跑。

## 2026-08-08 完整 sketch 微调 pilot

- 状态：`IN PROGRESS`；detached screen 为 `pddp_finetune_fullsketch_pilot_v1`。
- screen 日志：`/home/zwz_42312/PDDPoutputs/train_control/pilot_screen.log`。
- 协议：保留完整 canonical/训练 sketch，仅转换到 `224×224`；不再截取 hole 内线条或构造 bbox；原 free-form mask 直接作为 `obj_mask`。
- 数据：ArtBench/Mural1 共 `52,964` 张，稳定划分 train/validation 为 `52,407/557`。
- 流水线：manifest → MuGE batch 32 → VQ batch 32 → 单 GPU AMP 微调，所有阶段支持缓存续跑。
- pilot：真实 batch size `8`、gradient accumulation `1`、最多 `500` optimizer steps，每 `250` step 原子覆盖同一个 `checkpoint/last.pth`，不保留按 epoch 命名的历史权重。
- 初始实现 commit：`66826cb8628adf064f2d3d59e5a66c3580e36674`；预处理 batch 提升 commit：`8f4fd13`。
- 训练输出：`/home/zwz_42312/PDDPoutputs/train/sketchinpainter_finetune/`。
- 当前仅有预处理吞吐，尚无真实 forward/backward 速度；正式总步数将在 pilot 日志得到稳定 step time 和峰值显存后确定。

### Pilot 结果与一轮正式微调

- Pilot 已正常完成 `500/500` step；稳定 forward/backward 约 `0.4–0.5 s/step`，包含数据抖动和 checkpoint 写入后的平均约 `0.6 s/step`。
- Pilot 暴露官方 scratch scheduler 会在前 1000 step 将学习率从 `4e-6` 拉升到 `2e-4`；第 500 step 已约为 `1e-4`，不适合继续作为保守微调。
- 正式一轮从原始 PDDP checkpoint 重新初始化，使用固定 `4e-6`，不沿用 pilot optimizer/model 更新；目标为 `6364` step（一个有效 epoch）。
- 每 1000 step 原子覆盖同一个 `checkpoint/last.pth`，结束时强制覆盖最终权重；仍只保留一个训练 checkpoint。
- 正式 screen：`pddp_finetune_fullsketch_one_epoch_v1`；日志：`/home/zwz_42312/PDDPoutputs/train_control/formal_screen.log`。

## 2026-08-10 训练链路与局部 Sketch 协议修复

状态：`DONE（代码与 CPU 验证）`。本批未加载 checkpoint、未申请 GPU、未生成新缓存、未训练、未推理。

### 已确认失效的旧实验

- 旧完整 sketch 四轮训练实际结束于 `12,728` step；checkpoint 保留在 `/home/zwz_42312/PDDPoutputs/train/sketchinpainter_finetune/checkpoint/last.pth`，日志保留在 `/home/zwz_42312/PDDPoutputs/train_control/batch16_4epoch_screen.log`。
- 该 checkpoint 同时受旧输入协议、AMP 梯度裁剪顺序、裁剪区间、persistent worker epoch 不传播以及预训练 EMA/训练状态初始化问题影响，只供审计；不得 resume，也不得作为正式对比结果。

### 新固定协议

- `sketch_scope=bbox_crop`：free-form mask nearest resize 到 `256×256` 并原样作为 `obj_mask`；使用全部前景 union bbox，扩展 `1.2×`，裁剪框内全部 sketch 线条，等比例缩放并白底 padding 到 `224×224`。
- 框内位于 mask 外的线条保留，框外线条删除；空 mask 报错，边界确定性截断。训练与 canonical/selected-nine 推理共用同一 helper。
- 保留显式 `full` 与 `hole_crop` 作为历史兼容选项；推理 metadata 写入 scope、bbox scale、实际 bbox、源尺寸和输出尺寸。
- 完整 COCO train 纳入 manifest。只读扫描确认 ArtBench/COCO/Mural1 为 `51,300 / 118,287 / 1,664`，总计 `171,251`；稳定哈希划分 train/validation 为 `169,508 / 1,743`。
- 训练采样权重为 ArtBench `1.0`、COCO `0.43369`、Mural1 `0.1`。修复版配置为单卡 batch `16`、`max_epochs=4`、`max_iterations=-1`，仍仅原子覆盖 `last.pth`。

### 训练正确性修复

- AMP 顺序固定为 scaled backward → optimizer boundary → unscale → clip → scaler step/update；AMP/FP32 的 clip、optimizer、scheduler 和 EMA 都只在累积边界执行。
- `ClipGradNorm` 使用严格区间 `start_iteration <= step < end_iteration`，负 end 无上限。
- 官方预训练初始化仅加载 base model，并用官方 EMA 覆盖 live transformer 与 EMA shadow；不继承 iteration、optimizer、scheduler 或 clip 状态。只有正式 resume 才恢复完整训练状态，核心 key mismatch 直接失败。
- dataset epoch 改为共享内存 tensor，`persistent_workers=True` 时同 epoch 可复现、跨 epoch 条件变化。
- 训练前缓存检查改为逐 manifest 行验证 edge 可读、VQ token 为合法 1024-token schema，不再使用 `52,964` 硬编码文件数。

### 验证与交付

- 实现提交：`16c5ce7`；测试环境判断修复：`f168dc9`。均已推送到 `git@github.com:148here/PDDP.git` 的 `main`。
- 服务器在更新前把 tracked/untracked dirty worktree保存为命名 stash `pre_protocol_fix_20260810_`；CPU 测试生成的 tracked/untracked pycache 另存为 `post_cpu_tests_pycache_20260810`。随后 fast-forward 到最终提交；旧 checkpoint、日志和输出未删除。
- 服务器 CPU pytest：设置真实 `SKETCHINPAINTER_ROOT` 后 `37 passed`；覆盖 bbox crop、训练/推理逐像素一致、persistent worker、AMP/FP32 accumulation、clip 区间、EMA 初始化、resume 和三数据集 manifest。
- 配置 DataLoader dry-run 通过：基于旧 ArtBench/Mural1 manifest，修复版 batch 16 得到 effective train/validation `50,919 / 557`、iterations `3,182 / 34`。
- shell `bash -n` 通过。旧缓存逐行审计 `52,964/52,964`，错误 `0`。
- 完整 COCO manifest 仅运行 `--dry-run`，目标路径 `/home/zwz_42312/PDDPoutputs/preprocessed/training_manifest_with_coco.jsonl` 未写入（`written=false`）。

### 下一阶段（不得由本批自动执行）

1. 正式写入包含 COCO 的 `training_manifest.jsonl`，生成缺失的 `118,287` 组 MuGE edge 与 VQ token，并逐行复验。
2. 在无其他 GPU 作业时执行单 batch forward/backward smoke，检查显存、loss 和 EMA 初始化审计。
3. 从官方 checkpoint 全新初始化，在 detached screen 中训练 4 epochs；不得恢复旧 `12,728-step` checkpoint。
4. 使用最终 checkpoint 按 `bbox_crop/1.2×` 跑 selected-nine 与 canonical 702，之后接入 Mixed Easy/Medium/Hard 指标。
