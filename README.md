# cot_verifier_v1

内部 CoT 打标工具。给原版 Qwen3.8-27B 输入原始生成任务、4 张参考图和监督 CoT 原文，再用 Blender 5.0.1 执行模型生成的 Python。逐条记录这次生成能否产出可渲染的 3D 资产。

## 输入和服务器路径

| 内容 | 默认路径 |
|---|---|
| 训练数据 | /data/wangshuo/code/wuxiao-clean-cot-comparison-training/dataset/original_train.jsonl |
| 验证数据 | /data/wangshuo/code/wuxiao-clean-cot-comparison-training/dataset/original_val.jsonl |
| 图片根目录 | /home/rootroot/img2_3D |
| 原版模型权重 | /home/rootroot/models/Qwen/Qwen3.8-27B |
| vLLM | /data/wangshuo/nvme0/wangshuo/sft-training/checkpoints/vllm-venv/bin/vllm |
| Blender | /home/rootroot/opt/blender-5.0.1-linux-x64/blender |

现有快照有 2,834 条训练样本和 56 条验证样本。每行使用现有的 JSONL 结构。样本标识取自 id，问题取自 system 与 user 消息，CoT 取自 assistant.reasoning_content。图像按 user 消息的原顺序读取，CoT 原文追加到 user 内容。模型请求使用 temperature=0 和 enable_thinking=false。

## 判定逻辑

1. **校验输入。** 检查样本 ID、消息角色、CoT 和参考图路径。启动时记录 train、val 文件的 SHA-256，恢复时校验同一份输入快照。
2. **等待空闲 GPU。** 默认每 30 秒检查一次。连续两次采样确认至少 4 张 GPU 空闲时，启动最多 8 个本地 vLLM 副本，每张卡一个，端口仅绑定本机。每个副本默认最多 12 个并发请求。空闲判断综合计算进程、显存和 GPU 利用率。
3. **生成 Python。** 模型收到原始任务和监督 CoT，输出用于执行。若返回值是一个完整的 Markdown Python 代码块，剥掉围栏后原样检查。默认最多生成 8,192 token，遇到长度截断时在模型上下文允许的范围内提高预算，最多 16,384 token。瞬时推理错误有限次重试。
4. **解析、执行和渲染。** 先用 Python AST 校验语法，再在隔离的 Blender 进程里执行。Blender 后台以 CPU 渲染两个 512×512 视角。检查退出状态、脚本是否执行完毕、是否产出有效网格，以及两张 PNG 是否都有前景。
5. **写入标签。** 标签使用 id 和 split 对应原始样本，同时记录具体原因、推理尝试、CoT 与 prompt 的哈希和证据目录。源 JSONL 以只读方式打开。

| 标签 | 条件 |
|---|---|
| pass | 本次 Python 通过语法检查，Blender 正常执行并生成有效网格，两个视角都有前景。 |
| fail | 模型没有返回代码、Python 语法错误、生成代码在 Blender 中抛错、网格无效，或视角渲染为空。 |
| inconclusive | 输入缺失、推理或渲染环境异常、Blender 超时，或输出重试后仍被截断。 |

标签描述的是在固定模型与评测环境下，原始任务和 CoT 能否生成可渲染代码。判定无须匹配 GT 代码或 GT 外观。一次生成失败可能来自模型生成环节，标签适合做运行性筛查。

## 2026-11-22 全量实测

2026-11-22 18:52:05（北京时间）完成原始 CoT 数据快照的 2,890 条评测。使用原版 Qwen3.8-27B 和 Blender 5.0.1。请求包含原始生成任务、4 张参考图和监督 CoT 原文，thinking 关闭。

| 数据切分 | 总数 | pass | fail | inconclusive | 全样本可渲染率 |
|---|---:|---:|---:|---:|---:|
| train | 2,834 | 2,799 | 35 | 0 | 98.76% |
| val | 56 | 54 | 1 | 1 | 96.43% |
| 合计 | 2,890 | 2,853 | 36 | 1 | 98.72% |

36 条失败包括 Blender 执行错误 34 条、Python 语法错误 2 条。另有 1 条 Blender 超时，标为 inconclusive。仅在有明确判定的 2,889 条中计算，通过率为 98.75%。

[统计 JSON](results/original-20261122/summary.json) · [逐条标签 JSONL](results/original-20261122/labels.jsonl)

## 无 CoT 对照

两轮按相同的 2,870 条样本 ID 配对。均使用原版 Qwen3.8-27B、原始任务、4 张参考图及相同的生成参数和 Blender 判定。无 CoT 条件去掉附加的监督 CoT 原文。

| 输入 | 样本数 | pass | fail | inconclusive | 可渲染率 |
|---|---:|---:|---:|---:|---:|
| 原始任务 + 监督 CoT | 2,870 | 2,833 | 36 | 1 | 98.71% |
| 原始任务 | 2,870 | 1,672 | 1,172 | 26 | 58.26% |

加 CoT 的可渲染率高 **40.45 个百分点**。无 CoT 的失败包含 666 条 Blender 执行错误、472 条 Python 语法错误、31 条无效网格和 3 条空渲染。

[统计与配对矩阵](results/without-cot-20261123/summary.json) · [无 CoT 逐条标签](results/without-cot-20261123/labels.jsonl)

## 在服务器运行

使用当前服务器的 Linux Python 3.10+、Pillow、NVIDIA 驱动和已安装的 vLLM/Blender。隔离执行用到 unshare、mount、chroot、setpriv 和 prlimit；普通服务器账号需要支持非特权用户命名空间。

从项目根目录先跑预检，再启动后台进程：

```bash
RESULT="$HOME/sft_Data/cot_verifier_v1/run-original"
python3 cot_gpu_queue.py --sandbox "$PWD" --out "$RESULT" --preflight
nohup python3 -u cot_gpu_queue.py --sandbox "$PWD" --out "$RESULT" \
  > "$RESULT/controller.log" 2>&1 </dev/null &
```

默认的 GPU 范围是 4 到 8。可用 --min-gpus、--max-gpus 调整。恢复中断任务时使用同一个 --out，程序会跳过已有完整标签的记录，输出目录有进程锁。

```bash
cat "$RESULT/status.json"
tail -f "$RESULT/tags.jsonl"
```

| 文件 | 内容 |
|---|---|
| inputs.json | 输入快照、模型和运行参数 |
| status.json | 等卡、模型启动、逐条评测与结束状态 |
| tags.jsonl | 每个样本的最终标签和尝试信息 |
| summary.json | 结束后的总数、标签与异常统计、train/val 分布，以及已判定样本的通过率 |
| cases/ | 每次尝试的原始 prompt、监督 CoT、生成脚本、Blender 日志和渲染图 |
| logs/ | vLLM 副本日志 |

## 文件

- cot_gpu_queue.py 负责等卡、模型副本、批量评测和断点恢复。
- cot_text_smoke.py 负责组装原始任务和 CoT、请求模型、执行单条评测。
- isolated_blender.sh 负责启动隔离的 Blender 进程。
- batch_blender_worker.py 负责几何检查、取景和渲染。
