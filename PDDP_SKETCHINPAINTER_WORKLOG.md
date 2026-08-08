# PDDP × SketchInpainter 适配工作记录

最后更新：2026-08-08（第一阶段验收完成）

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
