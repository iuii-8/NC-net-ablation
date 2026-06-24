# AutoTemplate + ncNet 实验结果


## 1. 实验目标

本实验在官方 ncNet 基础上加入 AutoTemplate 模块：

- 原始 ncNet 在输入中有两类样本：
  - `w/o chart template`：输入模板中包含 `[T]` 等占位符。
  - `with chart template`：输入中包含人工提供的图表模板。
- 我们的改进是：对原始 `w/o chart template` 样本，先用 AutoTemplate 自动预测图表模板，再将预测模板填入 ncNet 输入中。
- 目标是验证：自动预测模板能否接近人工模板效果，并改善无模板输入场景下的 ncNet 表现。

## 2. 已完成的主要工作

### 2.1 AutoTemplate 预测数据生成

已生成并保存 AutoTemplate 预测结果：

```text
ncnet/processed_data/train_with_predicted_template.json
ncnet/processed_data/dev_with_predicted_template.json
ncnet/processed_data/test_with_predicted_template.json
```

### 2.2 官方 ncNet 格式数据集构建

已将 AutoTemplate 预测结果转换成官方 ncNet 可读取的 CSV 格式：

```text
NC/dataset/dataset_autotemplate/train.csv
NC/dataset/dataset_autotemplate/dev.csv
NC/dataset/dataset_autotemplate/test.csv
```

生成逻辑：

- 保留官方 `dataset_final/*.csv` 的列结构；
- 对原始 `w/o template` 行，将 `<C> ... </C>` 中的模板替换为 AutoTemplate 预测模板；
- 对原始 `with template` 行，保持原人工模板不变；
- 重新生成 `token_types`。

对应脚本：

```text
ncnet/build_official_autotemplate_dataset.py
```

### 2.3 专用测试脚本

已新增专用测试脚本：

```text
NC/test_autotemplate.py
```

该脚本用于：

1. 读取官方原始测试集 `dataset_final/test.csv` 判断每条样本原本属于 `w/o template` 还是 `with template`；
2. 使用 `dataset_autotemplate/test.csv` 进行实际 ncNet 解码；
3. 分别统计：
   - AutoTemplate-origin 样本准确率；
   - 原人工模板样本准确率；
   - overall 准确率；
4. 输出错误样本到：

```text
NC/dataset/dataset_autotemplate/autotemplate_errors.csv
```

## 3. 云端实验环境

实验已在云算力平台完成，主要环境如下：

```text
GPU: Tesla V100 32GB
OS: Ubuntu 22.04
Conda env: ncnet
Python: 3.8
PyTorch: 1.7.1
torchtext: 0.8.1
```

说明：

- 本地 Windows 环境由于缺少旧版 `torchtext.data.Field`，曾使用兼容补丁，但该补丁会导致词表顺序和官方模型权重不完全匹配，因此本地结果不可作为正式实验结果。
- 正式结果以云端旧版 `torchtext==0.8.1` 环境下的运行结果为准。

## 4. 完整测试结果

云端完整运行命令：

```bash
python test_autotemplate.py
```

终端输出核心结果：

```text
AutoTemplate-origin samples: 2460
Original with-template samples: 2460
Overall evaluated samples: 4920
--------------------------------------------------------
ncNet + AutoTemplate on original w/o-template rows: 0.7829268292682927
ncNet with original template rows: 0.7853658536585366
ncNet AutoTemplate overall: 0.7841463414634147
```

换算为百分比：

| Setting | Test Subset | Accuracy |
|---|---:|---:|
| ncNet + AutoTemplate | Original w/o-template rows | **78.29%** |
| ncNet with original manual template | Original with-template rows | **78.54%** |
| ncNet + AutoTemplate overall | Full test set | **78.41%** |

## 5. 与原论文结果对比

原论文报告：

| Setting in Original Paper | Accuracy |
|---|---:|
| ncNet without chart template | **77.8%** |
| ncNet with chart template | **79.6%** |

当前实验结果：

| Method / Setting | Template Source | Accuracy |
|---|---|---:|
| ncNet, original paper | No chart template | 77.8% |
| ncNet, original paper | Manual chart template | 79.6% |
| ncNet reproduced in our run | Original manual template | 78.54% |
| ncNet + AutoTemplate, ours | Predicted chart template | **78.29%** |

关键差值：

| Comparison | Gap |
|---|---:|
| Ours vs. original paper no-template setting | **+0.49 pp** |
| Ours vs. our reproduced manual-template setting | **-0.25 pp** |
| Our reproduced manual-template setting vs. original paper manual-template setting | -1.06 pp |

其中 `pp` 表示 percentage points，即百分点。

## 6. 初步结论

1. AutoTemplate 在原始无模板样本上的准确率为 **78.29%**。
2. 该结果略高于原论文报告的 `without chart template` 平均结果 **77.8%**，提升约 **0.49 个百分点**。
3. 该结果仅低于本次复现的人工模板设置 **78.54%** 约 **0.25 个百分点**。
4. 这说明 AutoTemplate 能够在不依赖人工标注模板的情况下，为 ncNet 提供有效结构引导，效果几乎达到人工模板水平。

可用于报告的中文表述：

> 原论文报告 ncNet 在无 chart template 和有 chart template 两种设置下的平均准确率分别为 77.8% 和 79.6%。在我们的实验中，使用 AutoTemplate 为原始无模板样本自动预测 chart template 后，ncNet 在 2460 条原无模板测试样本上的准确率达到 78.29%。该结果相比原论文无模板设置高 0.49 个百分点，并且仅比我们复现的人工模板设置 78.54% 低 0.25 个百分点。实验结果表明，AutoTemplate 能够在不依赖人工模板标注的情况下，为 ncNet 提供有效的结构化引导，性能几乎达到人工模板输入水平。

可用于报告的英文表述：

> The original ncNet paper reports average accuracies of 77.8% and 79.6% for the settings without and with chart templates, respectively. In our experiment, after replacing the original no-template inputs with chart templates predicted by AutoTemplate, ncNet achieves 78.29% accuracy on the 2,460 original no-template test cases. This is 0.49 percentage points higher than the no-template result reported in the original paper and only 0.25 percentage points lower than our reproduced manual-template setting, which achieves 78.54%. These results indicate that AutoTemplate can provide effective structural guidance without requiring manually annotated chart templates, achieving performance close to that of manual templates.


## 7. 复现实验命令

云端推荐环境：

```bash
conda activate ncnet
cd /root/NC
python test_autotemplate.py
```

保存日志的推荐命令：

```bash
python test_autotemplate.py | tee autotemplate_full_result.log
```

小样本快速测试命令：

```bash
python test_autotemplate.py -limit 100
```

## 8. 注意事项

1. 正式结果必须基于云端 `Python 3.8 + PyTorch 1.7.1 + torchtext 0.8.1` 环境。
2. 本地 Windows 调试环境的 `torchtext` 兼容补丁可能造成词表顺序不一致，不能作为最终实验结果。
3. `autotemplate_errors.csv` 请优先使用云端完整测试生成的版本。
