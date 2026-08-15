# MakeMore 第二部分：多层感知机（MLP）语言模型

在上一堂课中，我们实现了基于二元语法（Bigram）的语言模型。我们分别探讨了**统计计数法**与**单层线性神经网络法**。在 Bigram 模型中，我们仅仅根据**前一个字符**来预测下一个字符的概率分布。

虽然 Bigram 模型易于理解和实现，但它的效果非常有限：因为模型只考虑了单字上下文（Context Length = 1），生成的名字往往缺乏整体结构，听起来不够自然。

如果我们想提升模型的效果，最直接的想法就是**增加上下文长度**（例如考虑前 2 个、前 3 个或更多字符）。然而，传统的统计计数方法会迅速陷入**“维度灾难 / 指数爆炸”**（Curse of Dimensionality）。

---

## 1. 理论背景与 Bengio et al. 2003 论文介绍

### 1.1 计数法在长上下文下的指数爆炸问题

假设我们字符库的大小 $|\mathcal{V}| = 27$（包括 26 个英文字母和 1 个特殊分隔符 `.`）：
- 当上下文长度为 $1$ 时：可能出现的上下文状态有 $27^1 = 27$ 种；
- 当上下文长度为 $2$ 时：可能出现的上下文状态激增至 $27^2 = 729$ 种；
- 当上下文长度为 $3$ 时：组合数达到 $27^3 = 19,683$ 种；
- 当上下文长度为 $N$ 时：状态空间按 $27^N$ 呈**指数级增长**。

如果使用基于计数的二维矩阵来记录概率，随着上下文变长，矩阵的行数会极其庞大。在有限的训练数据下，绝大多数组合的频次都是 0（数据极度稀疏），这导致传统的计数模型无法泛化，甚至完全无法运行。

### 1.2 Bengio et al. 2003 的解决方案：词/字符嵌入与 MLP

为了解决长上下文下的稀疏性与维度灾难，Yoshua Bengio 等人在 2003 年发表了极其经典的论文 **《A Neural Probabilistic Language Model》**（神经概率语言模型）。

> **论文核心思想**：
> 1. **连续特征向量嵌入（Distributed Representations / Embeddings）**：将词表中的每一个词/字符映射（嵌入）到一个低维的连续向量空间（例如 30 维空间）。
> 2. **参数共享与神经网络预测**：利用一个多层感知机（MLP）来根据前 $N$ 个词/字符的嵌入向量拼接结果，预测下一个词/字符的概率分布。
> 3. **反向传播联合优化**：通过反向传播（Backpropagation）同时优化 MLP 的权重、偏置以及嵌入矩阵 $C$ 本身。

#### 泛化机制（Generalization Intuition）
假设在训练集中，模型从未见过短语 `"a dog was running in a ___"`，但在测试时需要对其进行预测。
- 传统的计数模型在遇到未见过的短语组合时会彻底失效（概率为 0）。
- 而在 MLP 嵌入模型中：如果模型在训练过程中学会了将 `"a"` 和 `"the"` 映射到相近的向量位置，将 `"dog"` 和 `"cat"` 映射到相近的向量位置，那么即使未见过该特定短语，模型也能从相近短语（如 `"the cat was running in a ___"`）中迁移知识，从而做出合理的泛化预测。

---

## 2. 神经网络架构详解

结合 Bengio 2003 论文，我们将设计一个字符级（Character-level）的 MLP 语言模型。

假设我们的上下文长度（`block_size`）为 $N=3$，嵌入维度为 $d$（例如 $d=2$ 或 $d=10$），隐藏层神经元数量为 $h$（例如 $h=100, 200, 300$），字符库大小为 $|\mathcal{V}| = 27$。

模型的整体架构和前向传播流程如下：

```
[ 输入字符索引 (x1, x2, x3) ]  (每个索引 ∈ [0, 26])
          │
          ▼  查找共享嵌入矩阵 C (Shape: 27 × d)
[ C[x1], C[x2], C[x3] ]        (每个向量 Shape: d)
          │
          ▼  拼接 (Concatenation)
[ X_concat ]                   (Shape: 3d)
          │
          ▼  隐层全连接 W1 (Shape: 3d × h) + b1 (Shape: h)
[ Pre-activation ]             (Shape: h)
          │
          ▼  双曲正切激活函数 tanh
[ Hidden State (h_act) ]       (Shape: h)
          │
          ▼  输出层全连接 W2 (Shape: h × 27) + b2 (Shape: 27)
[ Logits (未归一化对数几率) ]    (Shape: 27)
          │
          ▼  Softmax 层
[ 概率分布 P(y | x1, x2, x3) ] (Shape: 27)
```

数学公式表达：
1. **嵌入向量查找与拼接**：
   $$X_{concat} = [ C[x_1] \,;\,\, C[x_2] \,;\,\, C[x_3] ] \in \mathbb{R}^{3d}$$
2. **隐藏层计算**：
   $$h_{act} = \tanh(X_{concat} W_1 + b_1) \in \mathbb{R}^{h}$$
   其中 $W_1 \in \mathbb{R}^{3d \times h}$，$b_1 \in \mathbb{R}^{h}$。
3. **输出对数几率（Logits）**：
   $$logits = h_{act} W_2 + b_2 \in \mathbb{R}^{27}$$
   其中 $W_2 \in \mathbb{R}^{h \times 27}$，$b_2 \in \mathbb{R}^{27}$。
4. **Softmax 归一化概率**：
   $$\hat{y}_k = P(Y = k \mid X) = \frac{e^{logits_k}}{\sum_{j=0}^{26} e^{logits_j}}$$

所有的可训练参数集合为 $\theta = \{ C, W_1, b_1, W_2, b_2 \}$。

---

## 3. 构建数据集与滑动窗口 (Rolling Window)

为了将名字文本转化为神经网络可以训练的离散样本组 $(X, Y)$，我们需要使用固定长度的滑动窗口。

设置上下文窗口大小 `block_size = 3`：
- 输入 $X$：包含 $3$ 个连续字符的整数索引矩阵，Shape 为 $(N_{samples}, 3)$；
- 目标 $Y$：对应的紧接着的第 $4$ 个字符的整数索引，Shape 为 $(N_{samples})$。

以名字 `"emma"` 为例，生成样本的过程如下（其中 `.` 表示索引为 `0` 的起始/结束掩码）：

| 上下文字符 $X$ (文字) | 输入索引 $X$ (整数) | 预测目标字符 $Y$ (文字) | 目标索引 $Y$ (整数) |
| :--- | :--- | :--- | :--- |
| `...` | `[0, 0, 0]` | `e` | `5` |
| `..e` | `[0, 0, 5]` | `m` | `13` |
| `.em` | `[0, 5, 13]` | `m` | `13` |
| `emm` | `[5, 13, 13]` | `a` | `1` |
| `mma` | `[13, 13, 1]` | `.` | `0` |

处理整个名字数据集（包含 32,000 个单词）后，共可得到约 228,146 个训练样本。

---

## 4. 嵌入查找矩阵 $C$ 与 PyTorch 高级索引机制

### 4.1 嵌入查找的本质：Indexing vs. One-Hot Matrix Multiplication

在数学和模型理解上，嵌入查找（Lookup）与独热编码（One-Hot）乘法是**完全等价**的：

1. **One-Hot 乘法视角**：
   把索引 `5` 编码为 27 维的 One-hot 向量 $x_{one\_hot} \in \mathbb{R}^{1 \times 27}$（除第 5 位为 1 外全为 0）。将该向量乘以矩阵 $C \in \mathbb{R}^{27 \times d}$：
   $$x_{one\_hot} \cdot C = C[5, :] \in \mathbb{R}^{1 \times d}$$
   这也表明：**嵌入层可以看作是没有激活函数的线性神经元层，其权重矩阵就是 $C$**。

2. **直接索引视角（Indexing）**：
   直接使用 PyTorch 数组切片 `C[5]` 取出第 5 行。
   
> **性能注意**：在实际代码中，独热编码相乘会浪费大量无效的零乘法运算和内存。因此我们直接使用 PyTorch 的索引机制取行，速度要快上几个数量级。

### 4.2 PyTorch 张量高级索引（Multidimensional Indexing）

PyTorch 允许我们直接传入任意维度的整数张量来索引矩阵：
如果 $X \in \mathbb{R}^{32 \times 3}$ 是一个二维的整数索引张量，那么 `C[X]` 的结果形状自动变为：
$$\text{Shape of } C[X] = (32, 3, d)$$
即每个位置上的整数索引都被替换为对应的 $d$ 维嵌入向量。

### 4.3 向量拼接的效率对比：`unbind` + `cat` vs. `view`

为了将 $(32, 3, d)$ 的张量转化为可以输入到第一隐藏层 $W_1$ 的 $(32, 3d)$ 矩阵，我们需要进行维度重组：

#### 方案 1：使用 `torch.unbind` 与 `torch.cat`
```python
# 将维度 1 上的 3 个切片拆分，并在维度 1 上拼接
X_concat = torch.cat(torch.unbind(emb, 1), 1)  # Shape: (32, 3*d)
```
**缺点**：`torch.cat` 会在内存中开辟全新的连续存储空间并进行数据拷贝，计算效率较低，且代码依赖硬编码的 `block_size`。

#### 方案 2：利用张量底层视图操作 `.view()`
```python
X_concat = emb.view(-1, 3 * d)  # Shape: (32, 3*d)
```
**优点与底层原理**：
在 PyTorch 中，`Tensor` 对象的底层数据以一维连续数组（`Storage`）的形式存储在计算机内存中。`Tensor` 的维度表现（`Shape`）、跨度步长（`Strides`）以及偏移量（`Storage Offset`）只是对同一块物理内存的逻辑视图映射。

调用 `.view()` 操作时，**绝对不会发生任何内存拷贝或分配**，其时间复杂度为 $O(1)$，极其高效！因此，`emb.view(-1, 3 * d)` 是拼接嵌入向量的理想方式。

---

## 5. 模型前向传播与交叉熵损失 (Cross-Entropy Loss)

### 5.1 手动实现 Softmax 与 负对数似然 (NLL) 损失

在显式计算过程中，前向传播代码如下：
```python
# 1. 隐藏层
h = torch.tanh(emb.view(-1, 3 * d) @ W1 + b1) # (N, h)

# 2. 输出层 (logits)
logits = h @ W2 + b2                          # (N, 27)

# 3. Softmax
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)  # (N, 27)

# 4. 负对数似然损失 (NLL Loss)
loss = -probs[torch.arange(N), Y].log().mean()
```

### 5.2 为什么必须使用 PyTorch 内置的 `F.cross_entropy`？

在实际工程中，我们**绝不应该**手动编写上面的 Softmax 和 NLL 损失代码，而应该使用 `torch.nn.functional.cross_entropy(logits, Y)`。原因包含以下三个核心优势：

#### 1. 算子融合与内存效率 (Fused Kernels & Memory Efficiency)
手动实现会创建大量的中间张量（如 `counts`, `probs`, `.log()` 结果等），这会频繁触发内存分配与显存读写。`F.cross_entropy` 会调用 CUDA/CPU 的融合算子（Fused Kernel），在单个循环内完成计算，不产生额外的中间张量。

#### 2. 反向传播解析导数化简 (Analytical Gradient Simplification)
手动实现反向传播时，Autograd 需要逐节点遍历 `exp`、除法、`log` 等微观节点；而 Softmax + 交叉熵结合后的解析梯度极其简洁：
$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$
内置函数直接利用这一数学化简，使反向传播的速度和内存消耗大幅改善。

#### 3. 数值稳定性与数值溢出防护 (Numerical Stability & Overflow Protection)
当 $logits$ 中包含较大的正数（如 $logits_i = 100$）时，$e^{100} \approx 2.68 \times 10^{43}$ 会超出浮点数（Float32）的表示范围，导致上溢出产生 `inf`，进而在计算概率和 Loss 时得到 `NaN`。

PyTorch 内置的 `cross_entropy` 利用了 Softmax 的**平移不变性**（Shift Invariance）：
$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - C_0}}{\sum_j e^{z_j - C_0}}, \quad \text{其中选择 } C_0 = \max_j (z_j)$$
通过在指数运算前减去对数几率的最大值 $\max(logits)$，保证了最大指数项为 $e^0 = 1$，彻底避免了上溢出 `NaN` 问题。

---

## 6. 模型优化与小批量梯度下降 (Mini-batch Gradient Descent)

### 6.1 单批次过拟合测试 (Sanity Check: Overfitting a Single Batch)

在全量数据训练前，通常先用一个非常小的批次（如 32 个样本）测试模型能否迅速过拟合：
如果模型在 32 个样本上的 Loss 能从 $17.0$ 迅速下降到接近 $0$（如 $0.2 \sim 0.3$），说明前向传播、反向传播与参数更新的代码实现完全正确。

> **为什么全匹配时 Loss 仍然无法达到绝对的 0？**
> 因为数据集中存在**固有模糊性/一值多译现象**：例如前缀 `...` 后面既可能是 `e`，也可能是 `o`、`a` 或 `s`。当相同的输入对应不同的正确标签时，信息论上的条件熵大于 0，因此交叉熵损失存在理论下限，无法降为 0。

### 6.2 小批量梯度下降 (Mini-batch Gradient Descent)

在全量数据集（228,000 个样本）上做一次完整的 Forward/Backward 非常耗时。在实际训练中，我们采用小批量梯度下降：
每次随机抽取 `batch_size = 32`（或 64）个样本：
```python
ix = torch.randint(0, X.shape[0], (32,))
Xb, Yb = X[ix], Y[ix]
```
虽然单个小批量的梯度估计带有一定的噪声（近似梯度），但计算速度提高了上千倍！“使用有噪梯度的多步快速更新”远比“使用精准梯度的极少步慢速更新”高效得多。

### 6.3 寻找最佳学习率：指数搜索网格 (Learning Rate Search Range)

如何科学确定学习率 $\eta$ 的范围？我们可以进行对数空间搜索：
1. 设定学习率指数 $\text{lre} \in [-3, 0]$（即学习率 $\eta = 10^{\text{lre}} \in [10^{-3}, 10^0]$）。
2. 在 1,000 次迭代中，使学习率随着步数在对数曲线上从 $0.001$ 动态递增至 $1.0$：
   ```python
   lre = torch.linspace(-3, 0, 1000)
   lrs = 10**lre
   ```
3. 记录每一步的 Loss 并绘制 $\text{lre}$ 对 Loss 的曲线图：
   - 当 $\eta < 10^{-3}$ 时：Loss 下降极慢，说明学习率过小；
   - 当 $\eta \approx 10^{-1} = 0.1$ 时：Loss 下降最为迅猛且稳定；
   - 当 $\eta > 1.0$ 时：Loss 开始发散抖动甚至爆炸。
   
由此我们可以确信，将初始学习率设为 $\eta = 0.1$ 是非常合理且优良的选择。

---

## 7. 数据集划分与超参数调优 (Dataset Splits & Hyperparameters)

为了防止模型“死记硬背”训练集，标准做法是将数据集严格划分为三个独立集合：

1. **训练集 (Train Set, 80%)**：用于更新模型参数（$W_1, b_1, W_2, b_2, C$）。
2. **验证集/开发集 (Dev / Validation Set, 10%)**：用于评估和选择模型的**超参数**（如隐藏层神经元数 $h$、嵌入维度 $d$、上下文块大小 `block_size`、正则化强度等）。
3. **测试集 (Test Set, 10%)**：仅在所有超参数选定、模型最终训练完成后评估一次，用来衡量模型的真实泛化能力。

> [!CAUTION]
> **绝对禁忌**：严禁在超参数调优过程中频繁在测试集上评估。如果根据测试集的结果来调整超参数，测试集就会隐式“污染”模型，导致评估结果过于乐观。

### 欠拟合与过拟合诊断
- 当 `Train Loss ≈ Dev Loss` 且数值较高时：说明模型处于**欠拟合（Underfitting）**状态，模型的容量（Capacity）不足。
- 当 `Train Loss << Dev Loss` 时：说明模型处于**过拟合（Overfitting）**状态，模型开始记忆训练噪声。

在我们初始的实验中（$d=2, h=100$），训练集 Loss 与验证集 Loss 均为 $2.37$ 左右，二者基本相等，说明模型严重欠拟合。

---

## 8. 模型容量扩增与学习率衰减 (Scaling Up & LR Decay)

为了克服欠拟合，我们尝试扩大模型的容量：

### 8.1 增加隐藏层神经元与嵌入维度
1. 将嵌入维度从 $d=2$ 提升到 $d=10$；
2. 将上下文长度维持 `block_size = 3`（输入维度 $3 \times 10 = 30$）；
3. 将隐藏层神经元数从 $h=100$ 提升到 $h=200$。

参数量从约 $3,400$ 增加到了约 $11,697$。

### 8.2 学习率衰减策略 (Learning Rate Decay)
在训练后期（例如第 100,000 步之后），当 Loss 出现平台期（Plateau）时，将学习率衰减 10 倍（从 $0.1$ 降低到 $0.01$），帮助模型在损失函数的鞍部/局部极小值区域做更精细的收敛。

通过上述改进，验证集 Loss 成功从 Bigram 模型的 $2.45$ 降低到了 **$2.17$**。

---

## 9. 嵌入特征空间可视化与模型采样 (Visualization & Sampling)

### 9.1 二维嵌入特征空间可视化 ($d=2$)

在 $d=2$ 的低维设置下，我们将嵌入矩阵 $C \in \mathbb{R}^{27 \times 2}$ 中的 27 个字符对应的 $(x, y)$ 坐标绘制到平面图上，观察训练后的结构：

- **元音字母聚类**：元音字母 $\{a, e, i, o, u\}$ 自动聚集在相近的平面区域。这说明神经网络学会了将它们视为功能类似、可互相替换的字符。
- **特殊字符孤立**：特殊分隔符 `.` 被推送到了平面的边缘极远位置。
- **罕见字母特例**：罕见字母如 `q` 被赋予了非常特殊的特征向量，独立在角落。

这完美印证了 Bengio 2003 论文的直觉：神经网络无需人工指定语法规则，就能通过梯度下降自动学习出富含语义结构的连续嵌入空间！

### 9.2 从训练好的 MLP 模型中采样生成文本

使用训练好的 MLP 模型生成新名字的算法流程：
1. 初始上下文设置为连续的 `.`（如 `[0, 0, 0]`）。
2. 在循环中：
   - 前向传播计算对数几率 $logits$；
   - 使用 `F.softmax(logits, dim=1)` 转化为概率分布 $P$；
   - 使用 `torch.multinomial(P, num_samples=1)` 采样出下一个字符索引；
   - 将采样的字符追加到当前生成序列，并滑动更新上下文窗口；
   - 当采样到终止符 `.` (0) 时，该名字生成结束。

采样示例输出：
```text
ham.
joes.
lila.
kalen.
emmilie.
```
对比 Bigram 模型的随机拼凑，MLP 模型生成的名字展现出了明显的英语语音发音规律与命名结构。

---

## 10. 完整代码实现参考

以下为包含完整训练、验证集分割及采样的代码实现：

```python
import torch
import torch.nn.functional as F
import random

# 1. 读取数据与构建字符映射表
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# 2. 构建数据集函数
block_size = 3 # 上下文长度

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # 滑动窗口更新
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# 划分数据集 (Train 80%, Dev 10%, Test 10%)
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte   = build_dataset(words[n2:])

# 3. 初始化模型参数
g = torch.Generator().manual_seed(2147483647)
n_embd = 10   # 嵌入维度 d
n_hidden = 200 # 隐藏层神经元数 h

C  = torch.randn((27, n_embd),            generator=g)
W1 = torch.randn((block_size * n_embd, n_hidden), generator=g) * 0.2
b1 = torch.randn(n_hidden,                generator=g) * 0.01
W2 = torch.randn((n_hidden, 27),          generator=g) * 0.01
b2 = torch.randn(27,                      generator=g) * 0

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# 4. 优化训练循环
max_steps = 200000
batch_size = 32

for i in range(max_steps):
    # 小批量抽样
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    
    # 前向传播 (Forward Pass)
    emb = C[Xb] # (batch_size, block_size, n_embd)
    h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1) # (batch_size, n_hidden)
    logits = h @ W2 + b2 # (batch_size, 27)
    loss = F.cross_entropy(logits, Yb)
    
    # 反向传播 (Backward Pass)
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # 学习率调度 (Learning Rate Schedule)
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

# 5. 在验证集上评估 Loss
emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size * n_embd) @ W1 + b1)
logits = h @ W2 + b2
dev_loss = F.cross_entropy(logits, Ydev)
print(f"Dev Loss: {dev_loss.item():.4f}")

# 6. 从模型中采样文本
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])] # (1, block_size, n_embd)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out[:-1]))
```

---

## 11. 总结与自我挑战练习

在本文中，我们成功实现了 Bengio et al. 2003 的多层感知机（MLP）字符语言模型，主要收获包括：
- 理解了高维上下文导致统计计数崩溃的原因（维度灾难）。
- 掌握了连续特征嵌入（Embedding）机制及其泛化优势。
- 深入理解了 PyTorch 高级索引与底层存储视图 `.view()` 的极速拼接原理。
- 剖析了 `F.cross_entropy` 的算子融合、导数化简与防溢出数值稳定性。
- 掌握了验证集/测试集分割原则、小批量梯度下降及学习率衰减策略。

### 动手挑战
尝试通过调节以下调节杠杆（Hyperparameters），将验证集 Loss 进一步降低（击败 $2.17$）：
1. **隐藏层与嵌入维度**：尝试增加 `n_hidden`（如 300, 500）与 `n_embd`（如 20, 30）。
2. **上下文块大小**：将 `block_size` 从 3 增加到 4、5 或 8。
3. **优化细节**：尝试改变 Batch Size、优化训练轮次或学习率衰减策略。
