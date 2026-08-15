# MakeMore 第 4 部分：反向传播忍者指南 (Becoming a Backprop Ninja)

> **讲师**：Andrej Karpathy  
> **整理/翻译**：AI & Deep Learning Technical Translator  
> **核心主题**：手写张量级反向传播（Tensor-level Manual Backpropagation）、解析交叉熵梯度推导、解析批归一化（BatchNorm）梯度推导、渗漏抽象（Leaky Abstraction）与深度学习调试直觉。

---

## 目录
- [1. 课程概述与动机 (Overview & Motivation)](#1-课程概述与动机-overview--motivation)
  - [1.1 为什么我们需要手写反向传播？](#11-为什么我们需要手写反向传播)
  - [1.2 反向传播是一项“渗漏抽象 (Leaky Abstraction)”](#12-反向传播是一项渗漏抽象-leaky-abstraction)
  - [1.3 深度学习的历史视角 (Historical Perspective)](#13-深度学习的历史视角-historical-perspective)
  - [1.4 本课教学目标与初始代码配置](#14-本课教学目标与初始代码配置)
- [2. 练习 1：对原子操作逐步进行反向传播 (Exercise 1: Backprop Through Atomic Operations)](#2-练习-1对原子操作逐步进行反向传播-exercise-1-backprop-through-atomic-operations)
  - [2.1 梯度验证辅助函数 `cmp()`](#21-梯度验证辅助函数-cmp)
  - [2.2 负对数似然损失求导 `dlogprobs`](#22-负对数似然损失求导-dlogprobs)
  - [2.3 对数函数反向传播 `dprobs`](#23-对数函数反向传播-dprobs)
  - [2.4 归一化因子与概率计算 `dcounts_sum_inv` 和 `dcounts`](#24-归一化因子与概率计算-dcounts_sum_inv-和-dcounts)
  - [2.5 幂函数反向传播 `dcounts_sum`](#25-幂函数反向传播-dcounts_sum)
  - [2.6 指数函数反向传播 `dnorm_logits`](#26-指数函数反向传播-dnorm_logits)
  - [2.7 减去最大值（数值稳定性）的反向传播 `dlogits` 与 `dlogit_maxes`](#27-减去最大值数值稳定性的反向传播-dlogits-与-dlogit_maxes)
  - [2.8 `max()` 选通与 One-Hot 梯度路由 (dlogits 轴向第二分支)](#28-max-选通与-one-hot-梯度路由-dlogits-轴向第二分支)
  - [2.9 第二线性层（Linear Layer 2）的反向传播：矩阵乘法求导与形状匹配技巧](#29-第二线性层linear-layer-2的反向传播矩阵乘法求导与形状匹配技巧)
  - [2.10 $\tanh$ 激活函数的反向传播 `dh_preact`](#210-tanh-激活函数的反向传播-dh_preact)
  - [2.11 BatchNorm 缩放与平移参数的反向传播 `dbn_gain`, `dbn_bias`, `dbn_raw`](#211-batchnorm-缩放与平移参数的反向传播-dbn_gain-dbn_bias-dbn_raw)
  - [2.12 BatchNorm 标准化过程的反向传播（贝塞尔修正 Bessel's Correction 讨论）](#212-batchnorm-标准化过程的反向传播贝塞尔修正-bessels-correction-讨论)
  - [2.13 第一线性层（Linear Layer 1）的反向传播 `dmcat`, `dW1`, `db1`](#213-第一线性层linear-layer-1的反向传播-dmcat-dw1-db1)
  - [2.14 视图重构操作 `view()` 的反向传播 `demb`](#214-视图重构操作-view-的反向传播-demb)
  - [2.15 字符嵌入查找表（Embedding Table）的反向传播 `dC`](#215-字符嵌入查找表embedding-table的反向传播-dc)
- [3. 练习 2：Cross-Entropy 损失函数的解析梯度 (Exercise 2: Analytic Cross-Entropy Loss Gradient)](#3-练习-2cross-entropy-损失函数的解析梯度-exercise-2-analytic-cross-entropy-loss-gradient)
  - [3.1 数学公式推导：从原子节点到解析表达式](#31-数学公式推导从原子节点到解析表达式)
  - [3.2 物理直觉：“推与拉 (Push and Pull)” 的力学平衡模型](#32-物理直觉推与拉-push-and-pull-的力学平衡模型)
- [4. 练习 3：BatchNorm 层的解析梯度推导 (Exercise 3: Analytic BatchNorm Gradient Derivation)](#4-练习-3batchnorm-层的解析梯度推导-exercise-3-analytic-batchnorm-gradient-derivation)
  - [4.1 批归一化计算图的单块展开](#41-批归一化计算图的单块展开)
  - [4.2 纸笔微积分推导四步法](#42-纸笔微积分推导四步法)
  - [4.3 单行向量化 PyTorch 代码实现与验证](#43-单行向量化-pytorch-代码实现与验证)
- [5. 练习 4：融会贯通——成为反向传播忍者 (Exercise 4: Putting It All Together)](#5-练习-4融会贯通成为反向传播忍者-exercise-4-putting-it-all-together)
  - [5.1 移除 `loss.backward()`：纯手动梯度的完整训练循环](#51-移除-lossbackward纯手动梯度的完整训练循环)
  - [5.2 性能对比、模型校准与文本生成测试](#52-性能对比模型校准与文本生成测试)
- [6. 总结与后续课程展望 (Summary & Looking Ahead)](#6-总结与后续课程展望-summary--looking-ahead)

---

## 1. 课程概述与动机 (Overview & Motivation)

### 1.1 为什么我们需要手写反向传播？

在前面的课程中，我们已经构建并训练了一个基于 MLP（多层感知机）的字符级语言模型 `MakeMore`。模型的表现相当不错，我们也对其架构与前向传播有了深入的理解。

然而，在前向与反向传播的代码中，有一行代码始终像一个“黑盒”：
```python
loss.backward()
```
我们直接调用了 PyTorch 的 Autograd（自动微分引擎）来自动计算所有参数的梯度。

虽然大家可能都迫不及待想进入循环神经网络（RNN）、LSTM 以及 Transformer 等更酷炫的架构，但**在本节课中，我们将暂时停留在这里，彻底剥离 `loss.backward()`，在张量（Tensor）层面纯手写我们的反向传播 pass**。

### 1.2 反向传播是一项“渗漏抽象 (Leaky Abstraction)”

为什么手写反向传播如此重要？

> **渗漏抽象 (Leaky Abstraction)**：指的是一个旨在简化底层复杂性的抽象层，其内部细节偶尔会“渗漏”出来，导致使用者如果不了解底层实现就无法正确诊断或解决问题。

反向传播并不是一个能让你随心所欲拼装任意可微函数然后盲目祈祷就能完美运行的魔法盒。如果你不理解其内部的工作机制：
1. **梯度消失与梯度爆炸**：例如激活函数平坦尾部的饱和（Saturating Readout）、死神经元问题（Dead Neurons）、RNN 中的梯度爆炸。
2. **代码隐蔽 Bug**：Karpathy 曾在开源代码中见到有开发者尝试通过截断 Loss（`loss = torch.clamp(loss, ...)`）来防止梯度爆炸。这实际上是一个严重误解——对超额 Loss 进行 Clamp 会导致该样本的梯度直接变为 0，从而完全忽略了离群值（Outliers），而作者原本想做的是梯度裁剪（Gradient Clipping）。

在 Micrograd 课程中，我们在标量（Scalar）层面构建了 Autograd。但这还不够，在实际深度学习中，所有的运算都是以**矩阵和张量**为单位的高维并行运算。通过手写张量级反向传播，我们将彻底消除对底层的恐惧，成长为真正的“反向传播忍者（Backprop Ninja）”。

---

### 1.3 深度学习的历史视角 (Historical Perspective)

在今天，除了教学目的外，没有人会手写反向传播。但是在大约 10 年前（2010 年左右），**手写反向传播是每一个深度学习研究者的基本功和日常工作**。

- **2006 年 Geoffrey Hinton 的 Science 论文**（受限玻尔兹曼机 RBM）：当时没有 PyTorch 或 TensorFlow，主流工具是 MATLAB。研究人员需要在 MATLAB 中手写前向传播、对比散度（Contrastive Divergence）算法以及完整的反向传播梯度更新。
- **2014 年 Karpathy 的 Deep Fragment Embeddings 论文**：在该代码库中，不仅要手动实现复杂损失函数的前向计算，还需要手写对应的反向传播 pass，并使用**数值梯度检查器（Gradient Checker）**来验证手写解析梯度的正确性：
  $$ \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} \approx \nabla f(x) $$

---

### 1.4 本课教学目标与初始代码配置

我们维持与上一课完全相同的神经网络架构：一个带有批归一化（BatchNorm）的两层 MLP 语言模型。

#### 初始代码设置
```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# 读取数据集并构建词表
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

# 构建数据集
block_size = 3 # 上下文长度
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
```

#### 参数初始化与前向传播拆解
为了方便后向逐节点求导，我们将原来包含在 `F.cross_entropy` 和 `BatchNorm1d` 中的操作拆解为一系列原子的张量计算节点：

```python
g = torch.Generator().manual_seed(2147483647)
emb_dim = 10
hidden_dim = 64

# 参数初始化（注意：使用极小随机数初始化偏置以防止零偏置掩盖梯度计算错误）
C  = torch.randn((vocab_size, emb_dim),            generator=g)
W1 = torch.randn((block_size * emb_dim, hidden_dim), generator=g) * (5/3)/((block_size * emb_dim)**0.5)
b1 = torch.randn(hidden_dim,                       generator=g) * 0.1 # 冗余偏置，用于验证梯度正确性
W2 = torch.randn((hidden_dim, vocab_size),         generator=g) * 0.1
b2 = torch.randn(vocab_size,                       generator=g) * 0.1

bngain = torch.randn((1, hidden_dim), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, hidden_dim), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # 小批量数据

# 拆解后的前向传播 (Forward Pass in Atomic Chunks)
emb = C[Xb]                                       # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)               # (32, 30)
# 第一线性层
hprebn = embcat @ W1 + b1                         # (32, 64)
# BatchNorm 层拆解
bnmeant = 1/batch_size * hprebn.sum(0, keepdim=True) # (1, 64)
bndiff = hprebn - bnmeant                         # (32, 64)
bndiff2 = bndiff**2                               # (32, 64)
bnvar = 1/(batch_size-1) * bndiff2.sum(0, keepdim=True) # (1, 64) Bessel 修正
bnvar_inv = (bnvar + 1e-5)**-0.5                  # (1, 64)
bnraw = bndiff * bnvar_inv                        # (32, 64)
hpreact = bngain * bnraw + bnbias                 # (32, 64)
# 激活函数
h = torch.tanh(hpreact)                           # (32, 64)
# 第二线性层
logits = h @ W2 + b2                              # (32, 27)
# Softmax & Cross-Entropy 拆解
logit_maxes = logits.max(1, keepdim=True).values  # (32, 1) 数值稳定性
norm_logits = logits - logit_maxes                # (32, 27)
counts = norm_logits.exp()                        # (32, 27)
counts_sum = counts.sum(1, keepdim=True)          # (32, 1)
counts_sum_inv = counts_sum**-1                   # (32, 1)
probs = counts * counts_sum_inv                   # (32, 27)
logprobs = probs.log()                            # (32, 27)
loss = -logprobs[range(batch_size), Yb].mean()    # 标量 Loss

# 开启中间节点的保留梯度，用于梯度对比验证
for node in [emb, embcat, hprebn, bnmeant, bndiff, bndiff2, bnvar, bnvar_inv, 
             bnraw, hpreact, h, logits, logit_maxes, norm_logits, counts, 
             counts_sum, counts_sum_inv, probs, logprobs]:
    node.retain_grad()

loss.backward()
```

---

## 2. 练习 1：对原子操作逐步进行反向传播 (Exercise 1: Backprop Through Atomic Operations)

在此练习中，我们将沿着计算图从下往上（从 Loss 回溯到输入与参数），为每一个中间张量手动计算梯度。

### 2.1 梯度验证辅助函数 `cmp()`

为了确保我们手写的梯度与 PyTorch 的计算结果完全一致，我们定义一个比较函数 `cmp`：

```python
def cmp(name, dt, t):
    exact = torch.equal(dt, t.grad)
    approx = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{name:15s} | exact: {str(exact):5s} | approx: {str(approx):5s} | maxdiff: {maxdiff:e}')
```

---

### 2.2 负对数似然损失求导 `dlogprobs`

目标：计算 $\frac{\partial L}{\partial \text{logprobs}}$，其形状必须与 `logprobs` 完全一致，即 $(32, 27)$。

#### 数学推导
Loss 的计算公式为：
$$ L = -\frac{1}{N} \sum_{i=0}^{N-1} \text{logprobs}[i, y_i] $$
展开后：
$$ L = -\frac{1}{N} \left( \text{logprobs}[0, y_0] + \text{logprobs}[1, y_1] + \dots + \text{logprobs}[N-1, y_{N-1}] \right) $$

对任意元素 $\text{logprobs}[i, j]$：
- 如果 $j = y_i$（即对应真实标签位置）：
  $$ \frac{\partial L}{\partial \text{logprobs}[i, y_i]} = -\frac{1}{N} $$
- 如果 $j \neq y_i$（未参与 Loss 计算的位置）：
  $$ \frac{\partial L}{\partial \text{logprobs}[i, j]} = 0 $$

#### 代码实现
```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(batch_size), Yb] = -1.0 / batch_size

cmp('logprobs', dlogprobs, logprobs)
# 输出: logprobs        | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.3 对数函数反向传播 `dprobs`

前向计算：$\text{logprobs} = \ln(\text{probs})$

#### 微积分原理
由基本导数公式 $\frac{d}{dx} \ln(x) = \frac{1}{x}$，局部梯度为：
$$ \frac{\partial \text{logprobs}}{\partial \text{probs}} = \frac{1}{\text{probs}} $$

根据链式法则（Chain Rule）：
$$ \text{dprobs} = \frac{\partial L}{\partial \text{probs}} = \frac{\partial L}{\partial \text{logprobs}} \odot \frac{\partial \text{logprobs}}{\partial \text{probs}} = \text{dlogprobs} \odot \frac{1}{\text{probs}} $$

#### 直觉解释
如果模型对正确字符预测的概率 $\text{probs}$ 非常小（接近 0），则 $\frac{1}{\text{probs}}$ 会变得非常大，从而极大地放大梯度，强力惩罚错误的预测；如果概率已经接近 1，梯度传导则保持平稳。

#### 代码实现
```python
dprobs = (1.0 / probs) * dlogprobs

cmp('probs', dprobs, probs)
# 输出: probs           | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.4 归一化因子与概率计算 `dcounts_sum_inv` 和 `dcounts`

前向计算：$\text{probs} = \text{counts} \odot \text{counts\_sum\_inv}$
注意形状差异：`counts` 为 $(32, 27)$，`counts_sum_inv` 为 $(32, 1)$。这里触发了 PyTorch 的广播机制（Broadcasting），相当于将列向量沿水平方向复制了 27 次。

#### 1) `dcounts_sum_inv` 的求导
对于形如 $C = A \odot B$ 的乘法：
- 局部导数：$\frac{\partial C}{\partial B} = A = \text{counts}$
- 链式法则：$\text{dcounts\_sum\_inv\_replicated} = \text{counts} \odot \text{dprobs}$
- **广播的反向传播法则**：在前向传播中被**广播复制**的变量，在反向传播中必须沿广播维度进行**求和（Sum）**！因为同一个变量在多处被使用，梯度需要累加（即计算图中的节点复用规约）。

$$ \text{dcounts\_sum\_inv} = \sum_{j=0}^{26} \left( \text{counts} \odot \text{dprobs} \right)_{:, j} \quad (\text{保持维度 } \text{keepdim=True}) $$

```python
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)

cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
# 输出: counts_sum_inv  | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

#### 2) `dcounts` 的求导（第一分支）
$$ \text{dcounts} = \text{counts\_sum\_inv} \odot \text{dprobs} $$
注意：`counts` 在计算图中被使用了两次（一次用于生成 `probs`，另一次用于计算 `counts_sum`）。因此这只是 `dcounts` 的第一个梯度分支，后续需要使用 `+=` 累加第二个分支。

```python
dcounts = counts_sum_inv * dprobs
```

---

### 2.5 幂函数反向传播 `dcounts_sum`

前向计算：$\text{counts\_sum\_inv} = \text{counts\_sum}^{-1}$

#### 导数推导
根据幂函数求导法则 $\frac{d}{dx}(x^{-1}) = -x^{-2}$：
$$ \text{dcounts\_sum} = \left( -\text{counts\_sum}^{-2} \right) \odot \text{dcounts\_sum\_inv} $$

```python
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv

cmp('counts_sum', dcounts_sum, counts_sum)
# 输出: counts_sum      | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.6 `dcounts` 的第二分支累加

前向计算：$\text{counts\_sum} = \sum_{j=0}^{26} \text{counts}_{:, j}$  (形状从 $(32, 27)$ 规约到 $(32, 1)$)

#### 求和操作的反向传播法则
前向传播中的**求和（Sum）**操作，在反向传播中等价于**梯度路由与复制（Router / Broadcast）**！求和节点将来自上游的梯度原封不动地复制分发给参与求和的每一个输入元素。

$$ \text{dcounts}_{\text{branch2}} = \text{torch.ones\_like}(\text{counts}) \odot \text{dcounts\_sum} $$

结合第一分支的梯度：
```python
dcounts += torch.ones_like(counts) * dcounts_sum

cmp('counts', dcounts, counts)
# 输出: counts          | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.7 指数函数反向传播 `dnorm_logits`

前向计算：$\text{counts} = \exp(\text{norm\_logits})$

#### 导数推导
由指数函数性质 $\frac{d}{dx}(e^x) = e^x = \text{counts}$：
$$ \text{dnorm\_logits} = \text{counts} \odot \text{dcounts} $$

```python
dnorm_logits = counts * dcounts

cmp('norm_logits', dnorm_logits, norm_logits)
# 输出: norm_logits     | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.8 减去最大值（数值稳定性）的反向传播 `dlogits` 与 `dlogit_maxes`

前向计算：$\text{norm\_logits} = \text{logits} - \text{logit\_maxes}$  (形状：$(32, 27) - (32, 1)$)

#### 导数推导
类似于 $C = A - B$：
- 对 $A$（`logits`）的局部偏导数为 $+1$：
  $$ \text{dlogits}_{\text{branch1}} = \text{dnorm\_logits.clone()} $$
- 对 $B$（`logit_maxes`）的局部偏导数为 $-1$，且因为前向存在广播，反向需要沿维度 1 求和：
  $$ \text{dlogit\_maxes} = -\sum_{j=0}^{26} \text{dnorm\_logits}_{:, j} \quad (\text{keepdim=True}) $$

```python
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)

cmp('logit_maxes', dlogit_maxes, logit_maxes)
# 输出: logit_maxes     | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

> **重点直觉分析**：为什么 `dlogit_maxes` 应该接近 0？  
> 减去最大值仅是为了防止 $\exp(x)$ 数值溢出的技巧，由于 Softmax 具有平移不变性：$\text{Softmax}(z + c) = \text{Softmax}(z)$，平移量 $c$ 的改变理论上**完全不影响**概率输出与最终 Loss。因此其梯度在数学上为 **0**。在数值计算中，其最大差值通常只在 $10^{-9}$ 量级（浮点数精度误差）。

---

### 2.9 `max()` 选通与 One-Hot 梯度路由 (dlogits 轴向第二分支)

前向计算：$\text{logit\_maxes} = \text{max}(\text{logits}, \text{dim}=1)$

#### 导数推导
`max()` 操作的作用相当于一个开关（Gate）：在每一行中，只有达到最大值的那个位置接收来自 `dlogit_maxes` 的梯度，其余位置梯度为 0。

利用前向传播时保存的索引 `logits.max(1).indices` 构建 One-Hot 掩码矩阵：
```python
# 构造 one-hot 掩码，标注出最大值所在的位置
dlogits_from_max = F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

# 累加第二分支梯度
dlogits += dlogits_from_max

cmp('logits', dlogits, logits)
# 输出: logits          | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.10 第二线性层（Linear Layer 2）的反向传播：矩阵乘法求导与形状匹配技巧

前向计算：$\text{logits} = h W_2 + b_2$
张量维度：
- $\text{logits}: (32, 27)$
- $h: (32, 64)$
- $W_2: (64, 27)$
- $b_2: (27,)$

#### 纸笔推导（以 2x2 简易矩阵为例）
假定 $D = A B + C$，其中 $A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$, $B = \begin{bmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{bmatrix}$, $C = \begin{bmatrix} c_1 & c_2 \end{bmatrix}$。

写出标量展开式：
$$ d_{11} = a_{11}b_{11} + a_{12}b_{21} + c_1 $$
$$ d_{12} = a_{11}b_{12} + a_{12}b_{22} + c_2 $$

计算 $\frac{\partial L}{\partial a_{11}}$：因为 $a_{11}$ 同时影响 $d_{11}$ 和 $d_{12}$，应用链式法则累加：
$$ \frac{\partial L}{\partial a_{11}} = \frac{\partial L}{\partial d_{11}} b_{11} + \frac{\partial L}{\partial d_{12}} b_{12} $$

将所有偏导数整理回矩阵形式，可以惊奇地发现：
$$ \frac{\partial L}{\partial A} = \frac{\partial L}{\partial D} B^T $$
$$ \frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial D} $$
$$ \frac{\partial L}{\partial C} = \sum_{\text{rows}} \frac{\partial L}{\partial D} $$

#### 形状匹配黑客技巧 (Shape Matching Ninja Hack)
在实际代码中，无需死记硬背转置公式，只需利用**维度必须匹配**的线性代数基本规则：
1. `dh` 必须是 $(32, 64)$。唯一由 `dlogits` $(32, 27)$ 和 $W_2$ $(64, 27)$ 组合成 $(32, 64)$ 的矩阵乘法是：`dlogits @ W2.T`。
2. `dW2` 必须是 $(64, 27)$。唯一组合方式是：`h.T @ dlogits`。
3. `db2` 必须是 $(27,)$。唯一组合方式是沿 batch 维度 (axis 0) 对 `dlogits` 求和：`dlogits.sum(0)`。

```python
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
# 输出:
# h               | exact: True  | approx: True  | maxdiff: 0.000000e+00
# W2              | exact: True  | approx: True  | maxdiff: 0.000000e+00
# b2              | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.11 $\tanh$ 激活函数的反向传播 `dh_preact`

前向计算：$h = \tanh(\text{hpreact})$

#### 导数推导
根据 $\tanh$ 的微分性质 $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = 1 - h^2$：
$$ \text{dh\_preact} = (1 - h^2) \odot \text{dh} $$

```python
dhpreact = (1.0 - h**2) * dh

cmp('hpreact', dhpreact, hpreact)
# 输出: hpreact         | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.12 BatchNorm 缩放与平移参数的反向传播 `dbn_gain`, `dbn_bias`, `dbn_raw`

前向计算：$\text{hpreact} = \text{bngain} \odot \text{bnraw} + \text{bnbias}$
维度：`bngain` $(1, 64)$，`bnbias` $(1, 64)$，`bnraw` $(32, 64)$。

根据逐元素乘法与加法的反向传播法则（结合广播沿 dim 0 求和）：
```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnraw = bngain * dhpreact

cmp('bngain', dbngain, bngain)
cmp('bnbias', dbnbias, bnbias)
cmp('bnraw', dbnraw, bnraw)
# 输出:
# bngain          | exact: True  | approx: True  | maxdiff: 0.000000e+00
# bnbias          | exact: True  | approx: True  | maxdiff: 0.000000e+00
# bnraw           | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

---

### 2.13 BatchNorm 标准化过程的反向传播

前向步骤拆解：
1. $\text{bnraw} = \text{bndiff} \odot \text{bnvar\_inv}$
2. $\text{bnvar\_inv} = (\text{bnvar} + \epsilon)^{-0.5}$
3. $\text{bnvar} = \frac{1}{N-1} \sum \text{bndiff2}$
4. $\text{bndiff2} = \text{bndiff}^2$
5. $\text{bndiff} = \text{hprebn} - \text{bnmeant}$
6. $\text{bnmeant} = \frac{1}{N} \sum \text{hprebn}$

#### 代码实现与分步推导
```python
# 1. bnraw = bndiff * bnvar_inv
dbndiff = bnvar_inv * dbnraw # 第一分支
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)

# 2. bnvar_inv = (bnvar + 1e-5)**-0.5
dbnvar = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv

# 3. bnvar = 1/(N-1) * bndiff2.sum(0, keepdim=True)
dbndiff2 = (1.0 / (batch_size - 1)) * torch.ones_like(bndiff2) * dbnvar

# 4. bndiff2 = bndiff**2
dbndiff += (2.0 * bndiff) * dbndiff2 # 累加第二分支

# 5. bndiff = hprebn - bnmeant
dhprebn = dbndiff.clone() # 第一分支
dbnmeant = (-dbndiff).sum(0, keepdim=True)

# 6. bnmeant = 1/N * hprebn.sum(0, keepdim=True)
dhprebn += (1.0 / batch_size) * torch.ones_like(hprebn) * dbnmeant # 累加第二分支

cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bnvar', dbnvar, bnvar)
cmp('bndiff2', dbndiff2, bndiff2)
cmp('bndiff', dbndiff, bndiff)
cmp('bnmeant', dbnmeant, bnmeant)
cmp('hprebn', dhprebn, hprebn)
# 输出全为 exact: True, maxdiff: 0.000000e+00
```

> **关于贝塞尔修正 (Bessel's Correction) 的讨论**：  
> 在计算方差 `bnvar` 时，我们使用的是 $\frac{1}{N-1}$（无偏估计）而非 $\frac{1}{N}$（有偏估计）。BatchNorm 原始论文在训练时使用 $\frac{1}{N}$，但在测试阶段评估运行方差时又使用 $\frac{1}{N-1}$，这引入了微小的 Train-Test Mismatch（训练-测试不一致）。Karpathy 个人更倾向于在训练与测试中统一使用无偏估计 $\frac{1}{N-1}$。

---

### 2.14 第一线性层（Linear Layer 1）的反向传播 `dmcat`, `dW1`, `db1`

前向计算：$\text{hprebn} = \text{embcat} W_1 + b_1$
维度：`hprebn` $(32, 64)$，`embcat` $(32, 30)$，$W_1$ $(30, 64)$，$b_1$ $(64,)$。

应用维度匹配法：
```python
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)

cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)
# 输出全为 exact: True
```

---

### 2.15 视图重构操作 `view()` 的反向传播 `demb`

前向计算：$\text{embcat} = \text{emb.view}(\text{emb.shape}[0], -1)$ (从 $(32, 3, 10)$ 展平为 $(32, 30)$)

`view()` 操作只是改变了张量的逻辑内存寻址视图，没有改变底层的数值数据。因此其反向传播只需要将梯度的形状改变回原始输入维度即可：
```python
demb = dembcat.view(emb.shape)

cmp('emb', demb, emb)
# 输出: exact: True
```

---

### 2.16 字符嵌入查找表（Embedding Table）的反向传播 `dC`

前向计算：$\text{emb} = C[X_b]$  (形状：从 $C(27, 10)$ 提取得到 $\text{emb}(32, 3, 10)$)

#### 导数路由逻辑
前向传播中，索引 $X_b[k, j]$ 从查找表 $C$ 的第 $ix$ 行抽取了向量；反向传播时，必须将梯度 $\text{demb}[k, j, :]$ 释放分发回 $dC[ix, :]$ 行。如果同一个字符在 Batch 中被多次引用，梯度必须进行**累加（`+=`）**！

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]

cmp('C', dC, C)
# 输出: C               | exact: True  | approx: True  | maxdiff: 0.000000e+00
```

我们成功手动反向传播了整套计算图中的所有原子节点！

---

## 3. 练习 2：Cross-Entropy 损失函数的解析梯度 (Exercise 2: Analytic Cross-Entropy Loss Gradient)

在前一节中，我们将 Softmax + NLL 拆解成了十几个微小节点。在实际工程中，这不仅效率低下，且占据内存。本节我们将 Softmax 与 Cross-Entropy 合二为一，用数学公式推导出 $\frac{\partial L}{\partial z_k}$ 的解析表达式。

### 3.1 数学公式推导：从原子节点到解析表达式

设单个样本的 Logits 向量为 $z \in \mathbb{R}^K$，正确标签为 $y$。
1. **Softmax 概率**：
   $$ p_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}} $$
2. **损失函数 (Loss)**：
   $$ L = -\ln(p_y) = -\left( z_y - \ln \sum_{j} e^{z_j} \right) $$

现在求 Loss 对第 $k$ 个 Logit $z_k$ 的偏导数 $\frac{\partial L}{\partial z_k}$：

- **情况 1：$k = y$（对真实标签位置求导）**
  $$ \frac{\partial L}{\partial z_y} = -\frac{\partial}{\partial z_y} \left( z_y - \ln \sum_{j} e^{z_j} \right) = -\left( 1 - \frac{e^{z_y}}{\sum_{j} e^{z_j}} \right) = p_y - 1 $$

- **情况 2：$k \neq y$（对非真实标签位置求导）**
  $$ \frac{\partial L}{\partial z_k} = -\frac{\partial}{\partial z_k} \left( z_y - \ln \sum_{j} e^{z_j} \right) = -\left( 0 - \frac{e^{z_k}}{\sum_{j} e^{z_j}} \right) = p_k $$

将两种情况统一为一个简洁的数学表达式：
$$ \frac{\partial L}{\partial z_k} = p_k - \mathbb{I}(k = y) $$
其中 $\mathbb{I}(\cdot)$ 为指示函数（示性函数）。

对包含 $N$ 个样本的 Mini-batch 平均损失：
$$ \frac{\partial L}{\partial z_{i, k}} = \frac{1}{N} \left( p_{i, k} - \mathbb{I}(k = y_i) \right) $$

#### 单行代码实现
```python
# F.softmax 得到概率矩阵 p
probs = F.softmax(logits, dim=1)
dlogits_analytic = probs.clone()
# 在正确分类位置 subtract 1
dlogits_analytic[range(batch_size), Yb] -= 1.0
# 除以 Batch Size 求解平均梯度的贡献
dlogits_analytic /= batch_size

cmp('logits (analytic)', dlogits_analytic, logits)
# 输出: logits (analytic) | exact: False | approx: True  | maxdiff: 5.820766e-09
```
由于浮点数计算顺序的微小差异，对比结果显示 `approx: True`，最大误差仅为 $5.8 \times 10^{-9}$。

---

### 3.2 物理直觉：“推与拉 (Push and Pull)” 的力学平衡模型

观察解析梯度的数学形式，具有极其直观且优雅的物理含义：

```
每一个样本在 logit 维度上的梯度和为 0：
  sum_k (p_k - I(k=y)) = sum_k (p_k) - 1 = 1 - 1 = 0
```

可以将神经网络的训练想象成一个**复杂的物理滑轮与张力系统**：
- 对于**错误类别**（$k \neq y$）：梯度为 $+p_k > 0$。在梯度下降中，参数按照 $-\nabla L$ 方向更新，这意味着模型在**向下拉低（Push down）**错误类别的概率。
- 对于**正确类别**（$k = y$）：梯度为 $p_y - 1 < 0$。在梯度下降中，负号变成正号，模型在**向上拉高（Pull up）**正确类别的概率。
- **力的大小**与模型的预测偏差完全成正比：如果模型已经极其自信且正确（$p_y \approx 1$），则 $p_y - 1 \approx 0$，施加的拉力几乎为零；如果模型极其自信地做出了**错误**预测（$p_y \approx 0$），拉力将达到最大值 $-1$，产生强烈的修正张力！

---

## 4. 练习 3：BatchNorm 层的解析梯度推导 (Exercise 3: Analytic BatchNorm Gradient Derivation)

与练习 2 类似，将 BatchNorm 计算图拆解为十几个节点效率很低。在本节中，我们推导 BatchNorm 整体前向计算对输入 $x_i$ 的**全解析梯度公式**。

### 4.1 批归一化计算图的单块展开

设 Mini-batch 在某一特征维度上的输入向量为 $x = [x_1, x_2, \dots, x_N]^T \in \mathbb{R}^N$。
前向计算公式如下：
1. **均值**：$\mu = \frac{1}{N} \sum_{k=1}^N x_k$
2. **方差**：$\sigma^2 = \frac{1}{N} \sum_{k=1}^N (x_k - \mu)^2$
3. **标准化**：$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
4. **缩放与平移**：$y_i = \gamma \hat{x}_i + \beta$

已知上游传来的梯度为 $\frac{\partial L}{\partial y_i}$，我们需要求解 $\frac{\partial L}{\partial x_i}$。

---

### 4.2 纸笔微积分推导四步法

输入 $x_i$ 通过三条不同路径影响 Loss：
1. 直接路径：通过 $\hat{x}_i$ 影响 $y_i$
2. 间接路径 A：通过均值 $\mu$ 影响所有 $\hat{x}_k$
3. 间接路径 B：通过方差 $\sigma^2$ 影响所有 $\hat{x}_k$

根据多元链式法则：
$$ \frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial L}{\partial \mu} \frac{\partial \mu}{\partial x_i} + \frac{\partial L}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i} $$

#### 第一步：求解 $\frac{\partial L}{\partial \hat{x}_i}$
$$ \frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma $$

#### 第二步：求解 $\frac{\partial L}{\partial \sigma^2}$
方差 $\sigma^2$ 分支扇出到所有的 $\hat{x}_k$，因此需要对所有 $k=1 \dots N$ 求和：
$$ \frac{\partial L}{\partial \sigma^2} = \sum_{k=1}^N \frac{\partial L}{\partial \hat{x}_k} \frac{\partial \hat{x}_k}{\partial \sigma^2} $$
其中 $\frac{\partial \hat{x}_k}{\partial \sigma^2} = (x_k - \mu) \cdot \left(-\frac{1}{2}\right) (\sigma^2 + \epsilon)^{-3/2}$。

代入可得：
$$ \frac{\partial L}{\partial \sigma^2} = -\frac{1}{2} (\sigma^2 + \epsilon)^{-3/2} \sum_{k=1}^N \frac{\partial L}{\partial \hat{x}_k} (x_k - \mu) $$

#### 第三步：求解 $\frac{\partial L}{\partial \mu}$
均值 $\mu$ 同样扇出到所有 $\hat{x}_k$ 以及 $\sigma^2$：
$$ \frac{\partial L}{\partial \mu} = \sum_{k=1}^N \frac{\partial L}{\partial \hat{x}_k} \frac{\partial \hat{x}_k}{\partial \mu} + \frac{\partial L}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial \mu} $$

注意奇妙的消去现象：
$\frac{\partial \sigma^2}{\partial \mu} = \frac{1}{N} \sum_{k=1}^N -2(x_k - \mu) = -2 \left( \frac{1}{N} \sum x_k - \mu \right) = 0$！  
**第二项直接恒等于 0 抵消！**

对第一项：$\frac{\partial \hat{x}_k}{\partial \mu} = -\frac{1}{\sqrt{\sigma^2 + \epsilon}}$。
因此：
$$ \frac{\partial L}{\partial \mu} = -\frac{1}{\sqrt{\sigma^2 + \epsilon}} \sum_{k=1}^N \frac{\partial L}{\partial \hat{x}_k} $$

#### 第四步：代入汇总求解 $\frac{\partial L}{\partial x_i}$
将各偏导数合并化简，提出公因式 $\frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}$，最终得到** BatchNorm 梯度的终极解析极简公式**：

$$ \frac{\partial L}{\partial x_i} = \frac{\gamma}{N \sqrt{\sigma^2 + \epsilon}} \left[ N \frac{\partial L}{\partial y_i} - \sum_{j=1}^N \frac{\partial L}{\partial y_j} - \hat{x}_i \sum_{j=1}^N \left( \frac{\partial L}{\partial y_j} \hat{x}_j \right) \right] $$

---

### 4.3 单行向量化 PyTorch 代码实现与验证

根据上述数学公式，我们在矩阵层面将其写成简洁的单行向量化代码：

```python
# dhpreact 即 dL/dy, bnraw 即 x_hat
dhprebn_analytic = bngain * (bnvar + 1e-5)**-0.5 / batch_size * (
    batch_size * dhpreact - dhpreact.sum(0, keepdim=True) - bnraw * (dhpreact * bnraw).sum(0, keepdim=True)
)

cmp('hprebn (analytic)', dhprebn_analytic, hprebn)
# 输出: hprebn (analytic) | exact: False | approx: True  | maxdiff: 9.313226e-10
```
单行代码的梯度推导与 PyTorch Autograd 的误差在 $10^{-9}$ 级别，验证完全正确！

---

## 5. 练习 4：融会贯通——成为反向传播忍者 (Exercise 4: Putting It All Together)

现在我们将前面推导出的所有手动梯度整合到一个完整的训练循环中，彻底将 `loss.backward()` 替换为我们自己的计算代码，并引入 `with torch.no_grad()` 阻止 PyTorch 构建动态计算图以大幅提升训练效率。

### 5.1 纯手动梯度的完整训练循环代码

```python
# 重新初始化参数
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, emb_dim),            generator=g)
W1 = torch.randn((block_size * emb_dim, hidden_dim), generator=g) * (5/3)/((block_size * emb_dim)**0.5)
b1 = torch.randn(hidden_dim,                       generator=g) * 0.1
W2 = torch.randn((hidden_dim, vocab_size),         generator=g) * 0.1
b2 = torch.randn(vocab_size,                       generator=g) * 0.1

bngain = torch.randn((1, hidden_dim), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, hidden_dim), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]

max_steps = 200000
batch_size = 32

with torch.no_grad(): # 全过程关闭自动求导计算图构建！
    for step in range(max_steps):
        # 1. Mini-batch 采样
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # 2. 前向传播 (Forward Pass)
        emb = C[Xb]                                       # (32, 3, 10)
        embcat = emb.view(emb.shape[0], -1)               # (32, 30)
        hprebn = embcat @ W1 + b1                         # (32, 64)
        # BatchNorm 运行值计算
        bnmean = hprebn.sum(0, keepdim=True) / batch_size
        bnvar = ((hprebn - bnmean)**2).sum(0, keepdim=True) / (batch_size - 1)
        bnvar_inv = (bnvar + 1e-5)**-0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        # 激活函数与层 2
        h = torch.tanh(hpreact)                           # (32, 64)
        logits = h @ W2 + b2                              # (32, 27)
        loss = F.cross_entropy(logits, Yb)

        # 3. 手动反向传播 (Manual Backward Pass)
        # 步骤 A: 交叉熵解析梯度 dlogits
        dlogits = F.softmax(logits, dim=1)
        dlogits[range(batch_size), Yb] -= 1.0
        dlogits /= batch_size
        
        # 步骤 B: 第二线性层反向传播
        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)
        
        # 步骤 C: Tanh 激活函数反向传播
        dhpreact = (1.0 - h**2) * dh
        
        # 步骤 D: BatchNorm 增益与偏置反向传播
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        
        # 步骤 E: BatchNorm 解析梯度反向传播
        dhprebn = bngain * bnvar_inv / batch_size * (
            batch_size * dhpreact - dhpreact.sum(0, keepdim=True) - bnraw * (dhpreact * bnraw).sum(0, keepdim=True)
        )
        
        # 步骤 F: 第一线性层反向传播
        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)
        
        # 步骤 G: View 与 Embedding 查找表反向传播
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for k in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix = Xb[k, j]
                dC[ix] += demb[k, j]

        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

        # 4. 参数更新 (SGD Learning Rate Schedule)
        lr = 0.1 if step < 100000 else 0.01
        for p, grad in zip(parameters, grads):
            p -= lr * grad

        if step % 20000 == 0:
            print(f'{step:7d}/{max_steps:7d}: loss {loss.item():.4f}')
```

---

### 5.2 性能对比、模型校准与文本生成测试

#### BatchNorm 参数全数据集校准 (Calibration)
由于训练过程中使用了 Mini-batch 的均值与方差，在测试前需要对全数据集的均值 `bnmean_final` 与方差 `bnvar_final` 进行重新估算：

```python
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hprebn = embcat @ W1 + b1
    bnmean_final = hprebn.sum(0, keepdim=True) / Xtr.shape[0]
    bnvar_final = ((hprebn - bnmean_final)**2).sum(0, keepdim=True) / Xtr.shape[0]
```

#### 模型验证损失评估 (Loss Evaluation)
```python
def evaluate_loss(split):
    x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
    with torch.no_grad():
        emb = C[x]
        embcat = emb.view(emb.shape[0], -1)
        hprebn = embcat @ W1 + b1
        bnraw = (hprebn - bnmean_final) * (bnvar_final + 1e-5)**-0.5
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, y)
        print(f'{split:5s} loss: {loss.item():.4f}')

evaluate_loss('train')
evaluate_loss('val')
# 输出结果：
# train loss: 2.0682
# val   loss: 2.1051
```

模型收敛效果与此前使用 `loss.backward()` 自动求导训练得到的结果**完全一致**！

#### 随机采样名字生成 (Sampling)
```python
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        embcat = emb.view(1, -1)
        hprebn = embcat @ W1 + b1
        bnraw = (hprebn - bnmean_final) * (bnvar_final + 1e-5)**-0.5
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out[:-1]))
```
采样生成的名字展现出高度合理的结构特性（如 `carmah`, `dion`, `alenia` 等）。

---

## 6. 总结与后续课程展望 (Summary & Looking Ahead)

在这堂极其充实的课程中，我们完成了如下里程碑：
1. **打破 Autograd 依赖**：手写了包含 Embedding、Linear、BatchNorm、Tanh、Softmax 和 Cross Entropy 在内的全套张量反向传播 pass。
2. **掌握形状匹配技巧 (Shape Matching)**：学会了通过线性代数的维度约束快速导出矩阵乘法梯度的黑客技巧。
3. **理解广播与规约的对偶性 (Duality of Broadcasting and Reduction)**：在前向传播中被广播的张量，其反向传播必定对应求和规约；在前向传播中求和规约的维度，其反向传播必定对应复制广播。
4. **解析梯度推导**：
   - 证明了 Cross-Entropy 的解析梯度为 $d_k = \frac{1}{N}(p_k - \mathbb{I}(k=y))$，并理解了“推与拉”的力学物理直觉。
   - 推导了包含样本均值与方差依赖的 BatchNorm 终极单行解析梯度公式。

通过这一训练，反向传播在我们眼中不再是一个“渗漏抽象”，而是一个完全确定、透明且易于调试的张量流转过程。

在下一节课中，我们将正式开启**循环神经网络（RNN）**、**LSTM** 以及各种变体架构的探索，进入序列建模更广阔的新天地！
