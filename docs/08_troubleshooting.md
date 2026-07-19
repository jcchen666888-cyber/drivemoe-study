# 易错点与诊断树

## 下载

**13.5 GB 文件存在但加载失败**：先比字节数，再算 SHA256。不同 HF/Xet 端点之间不要盲目续传；必须验证 `Content-Range` 起点。

**标签下载 429**：不要 snapshot 整个 2000+ 文件目录。最小复现只下载 3 路线 × 2 标签的白名单。

**PaliGemma 401**：这是 gated repo，不是网络故障。接受条款并登录 HF；只下载 tokenizer，避免误取全模型。

## 数据

**预处理退出 0 但样本为 0**：相对路径在 `generate_data` 工作目录下解析错。打印绝对路径并断言 `val episodes == 3`/完整协议的期望数量。

**图片路径存在但 DataLoader 报 `invalid literal for int()`**：发布版 dataset 用 `split("/")` 提取帧号，Windows 的反斜杠会使整条路径被当成帧号。本仓库转换器保存正斜杠绝对路径；换机器仍应重新生成 pickle。

**结果尺度怪异**：检查 `b2d_statistics.json` 和反归一化；模型输出是 `[-1,1]` 附近的归一化 action，不是直接米。

## 环境

**两个 `conda run` 并行时随机失败**：Conda 可能复用临时文件。并行任务直接调用环境中的 `python.exe`。

**Windows 多卡报 NCCL**：open-loop 最小复现设置 `CUDA_VISIBLE_DEVICES=0` 并使用自定义单卡 runner。

**加载时内存暴涨**：不要先建 fp32 13B 模型再转 bf16。用 bf16 默认 dtype 创建，再 `torch.load(mmap=True)` 与 `assign=True`。

## 模型

**strict load 失败**：核对固定 commit、Stage 2、`horizon_steps=10`、`cond_steps=5`、7 skill experts、top-3。

**论文说 6 experts，代码却 7**：这是论文/发布差异，不要手改模型；checkpoint 决定真实结构。

**router 看似错选**：单帧 top-1 与人工直觉不一致不等于模型无效。先检查标签映射；`NULL` 在代码中映射到 back。

## 评价

3 帧 RMSE、router 命中率都没有统计意义；它们只做工程闭环。论文指标必须用完整验证集或 CARLA closed-loop。
