# Karpathy Neural Networks: Zero to Hero 中文学习版

Andrej Karpathy《Neural Networks: Zero to Hero》课程的中文学习版 Notebook。

本仓库从自动微分与反向传播出发，依次实现 Bigram、MLP、BatchNorm、WaveNet、GPT、BPE tokenizer 和 GPT-2。Notebook 中的教学说明和代码注释已经汉化，公式、变量名、张量形状、API 和核心可执行逻辑保持原样，便于与英文课程对照学习。

> 本项目是非官方中文翻译与学习整理，与 Andrej Karpathy、OpenAI 无隶属关系。课程、视频和原始代码的权利归原作者及相关权利人所有。

## 课程目录

| 阶段 | 主题 | 主 Notebook | 练习 |
|---|---|---|---|
| 001 | Micrograd：自动微分与反向传播 | [Micrograd](001_micrograd/micrograd.ipynb) | [练习](001_micrograd/micrograd_exercises.ipynb) |
| 002 | Makemore 1：Bigram 语言模型 | [Bigram](002_makemore_Bigrams/makemore_Bigrams.ipynb) | [Bigram 练习](002_makemore_Bigrams/bigram_exercises.ipynb) / [Trigram 练习](002_makemore_Bigrams/trigrams_exercises.ipynb) |
| 003 | Makemore 2：MLP 语言模型 | [MLP](003_makemore_MLP/makemore_MLP.ipynb) | [练习](003_makemore_MLP/MLP_exercises.ipynb) |
| 004 | Makemore 3：激活值、梯度与 BatchNorm | [BatchNorm](004_makemore_BatchNorm/makemore_BatchNorm.ipynb) | [练习](004_makemore_BatchNorm/BatchNorm_exercises.ipynb) |
| 005 | Makemore 4：手写反向传播 | [Backprop Ninja](005_makemore_BackpropNinja/makemore_Backprop.ipynb) | — |
| 006 | Makemore 5：WaveNet | [WaveNet](006_makemore_WaveNet/makemore_WaveNet.ipynb) | [练习](006_makemore_WaveNet/WaveNet_Exercises.ipynb) |
| 007 | GPT：从 Bigram 到 Transformer | [GPT](007_GPT/gpt.ipynb) | [1 / 2a](007_GPT/ex1-2a.ipynb) / [2b](007_GPT/ex2b.ipynb) / [3](007_GPT/ex3.ipynb) |
| 008 | minBPE：Unicode、BPE 与 tokenizer | [minBPE](008_minBPE/minbpe.ipynb) | [练习](008_minBPE/minbpe-exercises.ipynb) |
| 009 | GPT-2：从零实现与训练 | [GPT-2](009_GPT2/gpt-2-from-scratch.ipynb) | — |

## 环境安装

使用 `pip`：

```bash
python -m pip install -r requirements.txt
```

或使用 Conda：

```bash
conda env create -f environment.yml
conda activate neural-networks-zero-to-hero
```

minBPE 中的 GPT tokenizer 正则表达式使用 Unicode 属性，因此需要 `regex` 依赖，不能用 Python 标准库 `re` 直接替代。

## 来源与致谢

- 原课程视频：[Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- Andrej Karpathy 原课程代码：[karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- 本项目所基于的英文学习仓库：[chizkidd/Karpathy-Neural-Networks-Zero-to-Hero](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero)
- 相关项目：[micrograd](https://github.com/karpathy/micrograd)、[makemore](https://github.com/karpathy/makemore)、[minBPE](https://github.com/karpathy/minBPE)、[nanoGPT](https://github.com/karpathy/nanoGPT)

## 许可证

本仓库沿用原英文学习仓库的 [MIT License](LICENSE)。使用、引用或转载时，请保留原作者、原课程和本仓库的来源信息。
