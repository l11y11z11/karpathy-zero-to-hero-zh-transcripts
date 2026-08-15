# Andrej Karpathy 神经网络课程 04：Building MakeMore Part 3 —— 激活值、梯度流与批归一化 (BatchNorm)

在前面的课程中，我们基于 Bengio 等人（2003）的论文实现了一个多层感知机（MLP），用于字符级别的语言建模（Character-level Language Model）。我们输入前若干个字符的上下文，通过 MLP 预测序列中的下一个字符。

在转向更复杂、更庞大的神经网络架构（如循环神经网络 RNN、GRU、LSTM 以及 Transformer）之前，我们需要在多层感知机（MLP）的层面上多作留步。本节课的核心目标是：**建立对神经网络训练过程中“激活值（Activations）”和“反向传播梯度（Gradients）”行为特性的深刻直觉**。

理解激活值与梯度的分布和流动，对于理解神经网络的发展历史至关重要。RNN 等网络虽然理论上是通用近似器（Universal Approximator），能表达极复杂的函数，但传统的基于一阶梯度的优化方法极难对它们进行有效优化。其根本原因就在于深度网络中激活值与梯度的异常行为（如梯度消失和梯度爆炸）。

---

## 1. 初始状态分析与“冰球棍”损失曲线 (The Hockey-Stick Loss)

### 1.1 问题现象

观察未经特殊调优的 MLP 在训练初始阶段的损失曲线：
- **第 0 次迭代**：初始损失（Loss）高达 **27.0** 左右。
- **前几千次迭代**：损失在极短时间内迅速下降到约 2.1 左右。
- **损失曲线形状**：呈现出非常明显的“冰球棍/曲棍球棒”（Hockey-Stick）形状——开头有一个极陡峭的下降段，随后进入平缓改善的阶段。

```
Loss
 ^
 | \
27 |  \
 |   \
 |    \______________________
 3 |                        \________
   +-----------------------------------> Iterations
```

### 1.2 理论期望的初始损失

在神经网络训练中，根据损失函数和问题设置，我们在初始化时对“合理的初始损失值”应当有一个清晰的预期。

在字符预测任务中，词表大小（Vocabulary Size）为 $N = 27$（26 个小写英文字母 + 1 个特殊点号 `.` 标记）。
在初始化时刻，网络尚未学习到任何字符概率偏好，理论上应当给 27 个可能的下一个字符分配**均匀分布（Uniform Distribution）**，即每个字符的预测概率均等：

$$P(x_i) = \frac{1}{27}$$

由于我们采用交叉熵损失（Negative Log Likelihood Loss），理想的初始损失应为：

$$L_{\text{expected}} = -\ln\left(\frac{1}{27}\right) = \ln(27) \approx 3.29$$

实际记录到的初始损失 27.0 远高于理论期望值 3.29。这说明：**网络在初始化时输出的 Logits 具有极不合理的极端值，导致概率分布过度自信（Confidently Wrong），从而受到损失函数的巨额惩罚**。

### 1.3 极端的 Logits 与过度的信心

以 4 维分类为例直观感受 logits 对损失的影响：

```python
import torch

# 情况 A：Logits 接近于 0，输出均匀分布
logits_good = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs_good = torch.softmax(logits_good, dim=0)  # [0.25, 0.25, 0.25, 0.25]
loss_good = -torch.log(probs_good[2])          # -log(0.25) = 1.38 (预期损失)

# 情况 B：Logits 极不均匀（放大了极端值）
logits_bad = torch.tensor([10.0, -5.0, 0.0, -2.0])
probs_bad = torch.softmax(logits_bad, dim=0)    # [0.9999, 0.0000, 0.0000, 0.0000]
loss_bad = -torch.log(probs_bad[2])           # -log(0.0000...) = 10.36 (惩罚极高)
```

当 `logits` 出现很大的正数或负数时，`softmax` 会生成非常尖锐的概率分布。一旦该预测与真实标签不符，负对数似然损失就会急剧飙升。在初始化阶段，所有的自信预测本质上都是“盲目猜测”，因此绝大多数样本都会产生极高的初始损失。

训练最初的数千次迭代，优化器仅仅是在“压扁（squash）”这些过大的 `logits` 权重，将其拉回接近 0 的区域，这属于**无意义的浪费性计算**。

### 1.4 修复初始化 Logits

在 MLP 输出层中，$\text{logits} = h W_2 + b_2$。为使初始 logits 尽可能接近 0：

1. **偏置 `b2` 归零**：不需要在初始化时给 logits 叠加随机偏置，直接将其设为全 0。
2. **权重 `w2` 缩小**：将 $W_2$ 的随机初始化高斯张量乘以一个极小系数（如 $0.01$）。

```python
# 修改前
W2 = torch.randn((n_hidden, vocab_size)) * 1
b2 = torch.randn(vocab_size) * 1

# 修改后
W2 = torch.randn((n_hidden, vocab_size)) * 0.01
b2 = torch.zeros(vocab_size)
```

> [!NOTE]
> **为什么不将 $W_2$ 直接精确定为全 0？**  
> 虽然对最后一层输出层将权重设为全 0 在本例中也能得到 $3.29$ 的初始损失，但在一般神经网络中，将权重设为全 0 会导致对称性破坏失败（Symmetry Breaking Issue），使同层神经元接收完全相同的梯度并进行完全相同的更新。因此通常采用极小的非零随机值（如 $0.01$），保持一定的随机熵（Entropy）。

**修复效果**：
- 初始 Loss 降低至 **3.32**（非常接近理论预期 3.29）。
- 损失曲线中的“冰球棍”悬崖彻底消失。
- 在相同步数下，最终验证集 Loss 从 **2.16** 改善至 **2.13**。这是因为优化器把全部步数都花在了真正有效的特征学习上。

---

## 2. 隐藏层激活值与 Tanh 饱和及死神经元 (Tanh Saturation & Dead Neurons)

解决输出层的 Logits 之后，深入网络内部检查隐藏层激活值 $h = \tanh(\text{hpreact})$ 的分布。

### 2.1 Tanh 饱和现象分析

隐藏层的前向计算为：
$$\text{hpreact} = X_{\text{cat}} W_1 + b_1$$
$$h = \tanh(\text{hpreact})$$

统计隐藏层张量 $h$（维度为 $32 \times 200$，即 32 个 Batch 样本，200 个隐藏神经元）的直方图发现：绝大多数 $h$ 的取值都极端地集中在 $-1.0$ 和 $+1.0$ 处。

观察前激活值 $\text{hpreact}$ 的分布，发现其数值分布范围极宽（位于 $[-15, 15]$ 之间）。由于 $\tanh(x)$ 是一个平滑挤压函数（Squashing Function）：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

当输入 $|x| > 3$ 时，$\tanh(x)$ 就会进入两侧的平坦饱和区（Flat Tails），输出极度趋近于 $+1$ 或 $-1$。

```
Tanh(x)
  1 +          /----------------- (饱和区，梯度 -> 0)
    |         /
  0 +--------/----------> x
    |       /
 -1 +------/                     (饱和区，梯度 -> 0)
```

### 2.2 反向传播中的梯度摧毁机制

根据微积分链式法则，推导通过 $\tanh$ 单元的反向传播：

若 $t = \tanh(x)$，则局部导数（Local Gradient）为：

$$\frac{dt}{dx} = 1 - t^2$$

在反向传播过程中，传入该单元的梯度 $\text{out.grad}$ 与局部梯度相乘得到传出的梯度：

$$\text{in.grad} = \text{out.grad} \times (1 - t^2)$$

- **当 $t \approx 1$ 或 $t \approx -1$ 时**（处于饱和区）：
  $$\text{local\_grad} = 1 - (\pm 1)^2 = 0$$
  导致 $\text{in.grad} = 0$。**传入的梯度被截断归零（Gradient Destroyed/Killed）**！
- **当 $t \approx 0$ 时**（处于非饱和区）：
  $$\text{local\_grad} = 1 - 0^2 = 1$$
  梯度毫无损耗地直接通过（Pass-through）。

直观物理含义：若神经元的输出落入平坦区，改变其输入 $x$ 几乎不会对输出 $t$ 产生任何影响，因而对最终损失 Loss 也毫无贡献。反向传播梯度自然为 0。

### 2.3 深入探讨：死神经元 (Dead Neurons) 与不同激活函数

#### 1. Tanh 神经元的死亡
如果在初始化或训练过程中，某个 Tanh 神经元对于训练集中的**所有样本**，其前激活值 $\text{hpreact}$ 都落在极端的饱和区（例如输出恒为 $+1$ 或 $-1$），那么该神经元在所有样本上的梯度始终为 0。该神经元将永远无法更新权重，称之为**死神经元（Dead Neuron）**。

可以通过可视化分析每列神经元在各个样本上的激活情况：若某神经元整列均处于 $|h| > 0.99$ 的状态，即宣告死亡。

#### 2. ReLU 神经元的死亡现象 (Dead ReLU)
类似的问题在 **ReLU (Rectified Linear Unit)** 激活函数中更为严重：

$$\text{ReLU}(x) = \max(0, x)$$

- **负半轴 ($x < 0$)**：函数值为 0，局部导数**精确为 0**（而不是 Tanh 那样的渐近趋近于 0）。
- **永久性脑损伤（Permanent Brain Damage）**：在训练过程中，如果学习率（Learning Rate）设置过高，极大的梯度更新可能会将某个 ReLU 神经元的权重打出数据流形（Data Manifold）。此后对任意训练样本，该神经元的输入总是 negative，导致其永远无法激活。一旦进入该状态，其梯度永久为 0，再也无法恢复。

#### 3. 其它激活函数的对比
- **Sigmoid**：同样属于平滑挤压函数 $\sigma(x) = \frac{1}{1 + e^{-x}}$，导数为 $\sigma(x)(1 - \sigma(x))$，两端同样存在饱和死区问题。
- **Leaky ReLU**：$\text{LeakyReLU}(x) = \max(\alpha x, x)$ （其中 $\alpha \approx 0.01$）。负半轴保留微小斜率 $\alpha$，确保反向传播时始终有微弱梯度流过，从根本上解决了死神经元问题。

### 2.4 修复隐藏层前激活值分布

为避免隐藏层神经元饱和，需要将前激活值 $\text{hpreact} = X_{\text{cat}} W_1 + b_1$ 的标准差控制在合理范围（例如落在 $[-1.5, 1.5]$ 之间）。

通过缩小权重矩阵 $W_1$ 的缩放系数：

```python
# 修改前
W1 = torch.randn((n_inputs, n_hidden)) * 1
b1 = torch.randn(n_hidden) * 0.1

# 修改后 (适当缩小 W1 权重)
W1 = torch.randn((n_inputs, n_hidden)) * 0.2
b1 = torch.randn(n_hidden) * 0.01
```

**效果验证**：
- 隐藏层激活值 $h$ 的直方图呈现优美的正态分布形状，无极端的 $\pm 1.0$ 堆积。
- 饱和比例低于 $0.99$ 的极值点完全消失。
- 验证集 Loss 进一步下降至 **2.10**。

| 优化阶段 | 说明 | Validation Loss |
| :--- | :--- | :--- |
| **基线 (Baseline)** | 原始 MLP 默认初始化 | 2.17 |
| **阶段 1** | 修复 Softmax 初始 logits (W2 * 0.01, b2 = 0) | 2.13 |
| **阶段 2** | 修复 Tanh 隐藏层饱和 (W1 * 0.2, b1 * 0.01) | 2.10 |

---

## 3. Kaiming 初始化原理与增益 (Kaiming / He Initialization & Gain)

手动试错调整权重缩放系数（如 $0.2$、$0.01$）在浅层网络中可行，但在拥有数十层乃至数百层的深层网络中无法推广。我们需要一套系统化、基于数学原理的初始化策略。

### 3.1 矩阵乘法中的方差膨胀问题

假设有两个独立同分布的高斯随机变量向量：输入 $X \in \mathbb{R}^{m \times d_{\text{in}}}$ 与权重 $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$，其中 $X_{ij} \sim \mathcal{N}(0, 1)$，$W_{jk} \sim \mathcal{N}(0, 1)$。

计算矩阵乘法输出 $Y = X W$ 中单个元素 $y$ 的方差。假设输入神经元个数为 $\text{fan\_in} = d_{\text{in}}$：

$$y = \sum_{i=1}^{\text{fan\_in}} x_i w_i$$

由于 $x_i$ 与 $w_i$ 独立且均值为 0：

$$\text{Var}(y) = \sum_{i=1}^{\text{fan\_in}} \text{Var}(x_i w_i) = \sum_{i=1}^{\text{fan\_in}} \text{Var}(x_i) \cdot \text{Var}(w_i) = \text{fan\_in} \cdot 1 \cdot 1 = \text{fan\_in}$$

这意味着：**每经过一层未缩放的线性层，输出的方差就会扩大 $\text{fan\_in}$ 倍，标准差扩大 $\sqrt{\text{fan\_in}}$ 倍**！

```python
x = torch.randn(1000, 10) # 10维输入，均值 0，标准差 1
w = torch.randn(10, 200)   # 200个神经元，fan_in = 10
y = x @ w

print(x.std()) # tensor(1.002)
print(y.std()) # tensor(3.161)  <-- 恰好放大了 sqrt(10) ≈ 3.16 倍！
```

若要在矩阵乘法后**保持输出的标准差仍然为 1**，权重 $W$ 的初始标准差必须调整为：

$$\text{std}(W) = \frac{1}{\sqrt{\text{fan\_in}}}$$

```python
w = torch.randn(10, 200) / (10**0.5) # 除以 sqrt(fan_in)
y = x @ w
print(y.std()) # tensor(1.001)  <-- 完美保持标准差为 1！
```

### 3.2 Kaiming 初始化 (He et al., 2015) 与 Gain 增益

在实际网络中，线性层后面往往紧跟非线性激活函数。激活函数通常是**收缩性变换（Contractive Transformation）**，会压缩输出的分布和方差。

例如：
- **ReLU**：将所有负数清零，直接丢弃了一半的分布，使方差减半（标准差缩小至 $\frac{1}{\sqrt{2}}$）。因此需要乘以 $\text{gain} = \sqrt{2}$ 来补偿。
- **Tanh**：在平坦区挤压两端，需要乘以 $\text{gain} = \frac{5}{3} \approx 1.67$ 来补偿方差的收缩。

Kaiming He 等人在论文《Delving Deep into Rectifiers》中证明，考虑激活函数增益后，最合理的权重初始化标准差公式为：

$$\text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_in}}}$$

$$W \sim \mathcal{N}\left(0, \left(\frac{\text{gain}}{\sqrt{\text{fan\_in}}}\right)^2\right)$$

PyTorch 官方文档（`torch.nn.init.calculate_gain`）推荐的增益值（Gain）：

| 激活函数 (Non-Linearity) | 推荐增益系数 (Gain) | 数学直觉 / 解释 |
| :--- | :--- | :--- |
| **Linear / Identity** | $1$ | 无缩放变换 |
| **Sigmoid** | $1$ | 早期标准设定 |
| **Tanh** | $\frac{5}{3} \approx 1.67$ | 抵消 Tanh 双曲正切曲线两侧的收缩效应 |
| **ReLU** | $\sqrt{2} \approx 1.414$ | 补偿被负半轴直接裁切归零的一半方差 |
| **Leaky ReLU** | $\sqrt{\frac{2}{1 + \alpha^2}}$ | 考虑负半轴斜率 $\alpha$ 的修正因子 |

在我们的 MLP 隐藏层中（输入维度 $\text{fan\_in} = n_{\text{embed}} \times \text{block\_size} = 10 \times 3 = 30$），按 Kaiming 初始化公式设定标准差：

$$\text{std} = \frac{5/3}{\sqrt{30}} \approx \frac{1.667}{5.477} \approx 0.304$$

```python
# Kaiming 初始化应用于 Tanh 隐藏层
W1 = torch.randn((n_inputs, n_hidden)) * (5/3 / (n_inputs**0.5))
```

使用标准的 Kaiming 初始化，无需任何人工凭空凑出来的魔法数字，就能使网络激活值在前向传播中维持极其健康的正态分布，最终验证集 Loss 稳定收敛于 **2.10**。

---

## 4. 批归一化 (Batch Normalization, BatchNorm)

虽然 Kaiming 初始化给出了优美的数学推导，但在更深、更复杂的现代网络中（包含多种不同层、残差连接等），精确推导每一层的增益变得极其繁琐。

2015 年，Google 团队提出了 **Batch Normalization (Ioffe & Szegedy, 2015)**，这是一项改变现代深度学习格局的重大创新。

### 4.1 核心思想与数学公式

既然我们希望隐藏层的前激活值 $\text{hpreact}$ 在进入激活函数之前保持良好的标准高斯分布 $\mathcal{N}(0, 1)$，**为什么不直接在计算过程中将其强制标准化（Standardization）？**

标准化本身是一个完全可导（Differentiable）的算子，因此可以无缝嵌入到神经网络的前向与反向传播中。

假设一个小批次（Mini-batch）的前激活张量为 $X \in \mathbb{R}^{m \times d}$（$m$ 为 Batch 大小，$d$ 为神经元特征维度）。对于每个特征维度 $j$：

1. **计算批次均值 (Batch Mean)**：
   $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$$

2. **计算批次方差 (Batch Variance)**：
   $$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

3. **零均值单位方差标准化 (Normalize)**：
   $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
   *(其中 $\epsilon$ 为防止除以 0 的极小常数，如 $10^{-5}$)*

4. **可学习的缩放与平移 (Scale & Shift / Affine Transformation)**：
   $$y_i = \gamma \hat{x}_i + \beta$$
   *（其中 $\gamma$ 为增益参数 gain，初始为 1；$\beta$ 为偏置参数 bias，初始为 0）*

```
Input (hpreact)
   |
   v
[ 计算 Mini-Batch 均值 μ_B 和方差 σ_B^2 ]
   |
   v
[ 标准化: x_hat = (x - μ_B) / sqrt(σ_B^2 + ε) ]
   |
   v
[ 可学习变换: y = γ * x_hat + β ] (γ 初始 1, β 初始 0)
   |
   v
Output -> 进入 Tanh/ReLU
```

> [!IMPORTANT]
> **为什么引入可学习参数 $\gamma$ 与 $\beta$？**  
> 如果强制将激活值严格限制在标准高斯分布，会剥夺神经网络表达更复杂分布的能力。引入可学习的 $\gamma$ 和 $\beta$ 后，网络在初始化时刻处于高斯分布状态；若后续训练发现某些神经元需要更宽、更窄或有偏移的分布，可以通过反向传播自动学习调整 $\gamma$ 与 $\beta$。

### 4.2 BatchNorm 的关键特性与工程细节

#### 1. 偏置的冗余性 (Spurious / Redundant Bias)
在紧邻 BatchNorm 层之前的线性层（Linear Layer $y = X W + b$）中，偏置 $b$ 是完全多余的。

**推导**：
令 $x = X W + b$，计算均值时：

$$\mu_B = \frac{1}{m}\sum (X_i W + b) = \left(\frac{1}{m}\sum X_i W\right) + b$$

在标准化步骤中：

$$x - \mu_B = (X W + b) - \left(\frac{1}{m}\sum X_i W + b\right) = X W - \frac{1}{m}\sum X_i W$$

偏置 $b$ 在减去均值 $\mu_B$ 的瞬间被**完全抵消**！因此，在带有 BatchNorm 的 Linear/Conv 层中，应当显式设置 `bias=False`：

```python
# 推荐写法：取消前面线性层的 bias
hpreact = X @ W1  # 不需要 + b1
hpreact_bn = gamma * (hpreact - mean) / std + beta
```

#### 2. 样本间的数学耦合与正则化副作用 (Coupling & Regularization Effect)
在引入 BatchNorm 之前，样本在前向传播中是彼此独立计算的。

引入 BatchNorm 后，**单个样本的激活值与 logits 不再仅仅取决于其自身输入，还取决于同批次（Batch）中恰好被采样到的其他样本**。

这种因为随机 Batch 采样带来的扰动（Jitter / Noise）为网络注入了随机熵，效果类似于数据增强（Data Augmentation）或 Dropout，构成了一种强力的**正则化副作用（Regularizer）**，显著降低了过拟合风险。

虽然这带来了诸多优越特性，但样本间的数学耦合也经常在复杂的分布式训练中引发隐蔽的 Bug。因此，现代研究（如 GroupNorm, LayerNorm）一直在尝试寻找替代方案。

#### 3. 训练阶段与推理阶段的区别 (Training vs. Inference Mode)

在模型部署和测试（Inference）阶段，我们通常希望输入单个样本并立即得到确定的预测结果。此时 Batch Size 可能为 1，无法计算批次均值和方差 $\mu_B, \sigma_B^2$。

**解决方案：滑动运行均值与方差 (Running Mean & Running Variance)**：

在训练过程中，采用**指数移动平均（Exponential Moving Average, EMA）**在侧边隐式追踪全数据集的均值与方差（不参与梯度求导）：

$$\mu_{\text{running}} \leftarrow (1 - \eta) \cdot \mu_{\text{running}} + \eta \cdot \mu_B$$

$$\sigma_{\text{running}}^2 \leftarrow (1 - \eta) \cdot \sigma_{\text{running}}^2 + \eta \cdot \sigma_B^2$$

*(在 PyTorch 中，动量系数 $\eta = \text{momentum}$，默认值为 $0.1$)*

在推理阶段，直接固定使用训练过程中累计得到的 $\mu_{\text{running}}$ 与 $\sigma_{\text{running}}^2$：

$$\hat{x}_{\text{test}} = \frac{x_{\text{test}} - \mu_{\text{running}}}{\sqrt{\sigma_{\text{running}}^2 + \epsilon}}$$

$$y_{\text{test}} = \gamma \hat{x}_{\text{test}} + \beta$$

---

## 5. PyTorch 风格的模块重构 (PyTorch-ifying the Code)

为了直观体会 PyTorch 的底层机制，我们将网络结构封装为遵循 PyTorch API 风格的面向对象模块（Modules）。

```python
import torch

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Kaiming Normal 初始化
        self.weight = torch.randn((fan_in, fan_out)) / (fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        # 可学习参数 (Parameters)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        
        # 运行状态缓冲区 (Buffers, 不参与反向传播梯度计算)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self, x):
        if self.training:
            # 训练模式：计算 Mini-Batch 统计量
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True, unbiased=False)
        else:
            # 推理模式：使用积累的运行统计量
            xmean = self.running_mean
            xvar = self.running_var
            
        # 标准化
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        # 训练模式下更新 Running Buffers (必须在 torch.no_grad 下进行)
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
                
        return self.out
        
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
```

构建一个 6 层深的 MLP 网络框架：

```python
n_embd = 10
n_hidden = 100
block_size = 3
vocab_size = 27

C = torch.randn((vocab_size, n_embd))

layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size, bias=False),          BatchNorm1d(vocab_size),
]

# 对最后一层进行特殊增益调整，使初始 logits 不过于自信
with torch.no_grad():
    layers[-1].gamma *= 0.1
    # 隐藏层的 Linear 乘以 Kaiming Tanh Gain (5/3)
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True
```

---

## 6. 神经网络可视化与诊断分析 (Network Diagnostics)

在现代神经网络训练中，我们需要一套系统化的诊断工具（Diagnostics）来监控网络内部的健康状况。

### 6.1 前向激活值直方图 (Forward Activations)

监控各层 Tanh 激活值的平均值、标准差以及饱和率（Saturation Rate，即 $|h| > 0.97$ 的元素占比）。

期望状态：各层的标准差稳定在 $0.65$ 左右，饱和率保持在 $5\%$ 左右的较低水平，曲线在深层网络中不会发生衰减或爆炸。

```
Layer 1 (Tanh): std 0.64, saturation 4.8%  [分布均衡]
Layer 2 (Tanh): std 0.65, saturation 4.7%  [分布均衡]
Layer 3 (Tanh): std 0.65, saturation 4.9%  [分布均衡]
Layer 4 (Tanh): std 0.65, saturation 4.6%  [分布均衡]
```

### 6.2 反向梯度直方图 (Backward Gradients)

监控反向传播过程中流经各个隐藏层的梯度张量 `layer.out.grad`。

期望状态：从输出层到输入层，梯度的分布标准差应当在相同数量级保持稳定。若深度增加导致浅层梯度急剧缩小至 $10^{-10}$，说明发生了**梯度消失**；若急剧膨胀，则说明发生了**梯度爆炸**。

```
Layer 1 Grad: std 0.003
Layer 2 Grad: std 0.003
Layer 3 Grad: std 0.003
Layer 4 Grad: std 0.003  (梯度流非常健康稳定)
```

### 6.3 参数更新量与数据量之比 (Update-to-Data Ratio)

仅看梯度本身的绝对大小是不够的，真正决定权重发生多大相对变化的是**更新量（Update Step）与参数本身数值（Data Magnitude）的比值**：

$$\text{UD ratio} = \frac{\text{std}(\text{lr} \cdot \nabla_\theta)}{\text{std}(\theta)}$$

习惯上取对数尺度进行统计：

$$\log_{10}(\text{UD ratio}) = \log_{10}\left(\frac{\text{std}(\text{lr} \cdot \nabla_\theta)}{\text{std}(\theta)}\right)$$

经验法则（Rule of Thumb）：
- **最佳经验值**：$\log_{10}(\text{UD ratio}) \approx -3.0$ （即更新量约为参数数值的 $\frac{1}{1000}$，即 $10^{-3}$ 左右）。
- **如果比值高于 $10^{-1}$（对数化后 $\ge -1.0$）**：学习率过大或梯度过猛，参数变化过于剧烈，可能导致训练崩溃。
- **如果比值低于 $10^{-5}$（对数化后 $\le -5.0$）**：学习率过小，参数更新过于缓慢，网络几乎未在有效学习。

```python
# 监控所有权重矩阵的 UD ratio
ud_ratios = []
for p in weights:
    update = lr * p.grad
    ratio = update.std() / p.data.std()
    ud_ratios.append(torch.log10(ratio).item())

# 理想输出曲线应围绕 -3.0 上下平稳波动
```

---

## 7. 总结与理论全景图

本课程完整梳理了深度神经网络初始化与梯度控制的发展脉络：

```
[ 初始状态问题 ]
   ├── Logits 绝对值过大  ---> 出现"冰球棍"损失曲线 ---> 解决: 缩小最后一层 W2, b2=0
   └── Tanh 前激活值过大  ---> 饱和区导数归 0 (死神经元) ---> 解决: 缩小 W1, b1=0

[ 数学规范化策略 ]
   ├── Kaiming 初始化   ---> 方差维持 std = gain / sqrt(fan_in) (Tanh gain=5/3, ReLU gain=sqrt(2))
   └── BatchNorm (2015)  ├── 强制标准化: (x - μ_B) / sqrt(σ_B^2 + ε)
                        ├── 可学习参数: γ * x_hat + β
                        ├── 移除前面层的 Bias (被 μ_B 抵消)
                        ├── 引入批次噪声 (正则化副作用)
                        └── 维护 Running Mean / Var 供 Inference 推理使用

[ 网络健康度诊断 Diagnostics ]
   ├── 1. Forward Activations : 检查各层饱和度 (Tanh ~5%)
   ├── 2. Backward Gradients   : 检查梯度流是否均匀稳定 (无消失/爆炸)
   └── 3. Update-to-Data Ratio : 检查 log10(lr * grad.std() / data.std()) ≈ -3.0
```

得益于 BatchNorm 等归一化技术、残差连接（Residual Connections）以及现代高级优化器（如 Adam），当今训练深层神经网络对权重的初始值敏感度大为降低。然而，掌握激活值与梯度的底层微观特性，依然是深入研究复杂深度学习架构的核心基本功。
