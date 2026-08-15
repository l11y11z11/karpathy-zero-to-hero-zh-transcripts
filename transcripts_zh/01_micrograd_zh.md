# Andrej Karpathy《神经网络：从零手刷 Micrograd》硬核教程中文全解

> **讲师**：Andrej Karpathy (前 OpenAI 创始成员 / 前 Tesla AI 总监)  
> **整理与全解**：AI & Deep Learning 译者组  
> **项目源码**：[micrograd (GitHub)](https://github.com/karpathy/micrograd)

---

## 1. 课程引言与 Micrograd 概述

欢迎大家！我是 Andrej Karpathy，从事深度神经网络的训练工作已有十余年。在这堂课中，我希望带大家深入探究**神经网络底层究竟是如何运行的**。

具体的学习方式是：我们将从一个完全空白的 Jupyter Notebook 开始，一步步动手构建，直到最终定义并训练出一个完整的神经网络。你将直观且精准地看到底层发生的一切细节。

我们将要构建的这个核心项目叫做 **Micrograd**。Micrograd 是我在大约两年前发布到 GitHub 上的一个轻量级开源库。当时我只上传了源代码，大家需要自己去啃源码来理解它的工作原理。而在本次讲座中，我将带大家逐行实现它，并对每一个组件和细节进行详尽的讲解。

```
                       +-------------------+
                       |    Micrograd      |
                       +---------+---------+
                                 |
           +---------------------+---------------------+
           |                                           |
+----------v----------+                     +----------v----------+
|  engine.py (100 lines)|                     |   nn.py (50 lines)  |
|  - Value 类         |                     |  - Neuron 类        |
|  - 标量自动求导引擎   |                     |  - Layer 类         |
|  - 拓扑排序与反向传播 |                     |  - MLP 多层感知机   |
+---------------------+                     +---------------------+
```

### 1.1 什么是 Micrograd？它的重要性何在？

简单来说，**Micrograd 是一个标量级的自动求导引擎（Scalar-valued Autograd Engine）**。Autograd 是“自动梯度”（Automatic Gradient）的缩写，它本质上实现了**反向传播算法（Backpropagation）**。

反向传播是构建现代神经网络的核心算法。它能够高效地计算某个**损失函数（Loss Function）**关于神经网络所有**权重（Weights）**的梯度（Gradients）。有了梯度之后，我们就可以通过迭代的方式调整这些权重，从而最小化损失函数，持续提升神经网络预测的准确率。反向传播构成了现代所有深度学习框架（如 PyTorch、JAX 等）的最核心数学底层。

### 1.2 Micrograd 的使用示例

Micrograd 的功能可以通过一个简单的表达式构建过程来演示：

假设我们有两个输入标量 $a = -4.0$ 和 $b = 2.0$。在 Micrograd 中，我们将这两个数值包裹在 `Value` 对象中：

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
```

在这段代码中，我们使用 `a` 和 `b` 构建了一个复杂的计算图，最终得到了输出 `g`。Micrograd 会在后台完整记录这一数学表达式的图拓扑关系（Computational Graph）。例如，它知道 `c` 是加法节点产生的结果，并且 `c` 的前驱子节点是 `a` 和 `b`。

当我们做**正向传播（Forward Pass）**时，访问 `g.data` 就可以得到表达式的标量计算结果（本例中为 $24.7$）。而最神奇的部分在于，我们可以对输出节点 `g` 直接调用：

```python
g.backward()
```

这一行代码会自动初始化从节点 `g` 开始的反向传播。Micrograd 将从 `g` 开始沿着计算图逆向递归应用微积分中的**链式法则（Chain Rule）**，计算出 `g` 关于内部所有中间节点（如 `e`, `d`, `c`）以及输入节点 `a` 和 `b` 的导数（即梯度）。例如：

- `a.grad` 计算结果为 $138.0$，表示 $\frac{\partial g}{\partial a} = 138.0$
- `b.grad` 计算结果为 $645.0$，表示 $\frac{\partial g}{\partial b} = 645.0$

梯度的物理含义非常明确：它告诉我们输入微小变动对最终输出 `g` 的影响敏感度。$a.\text{grad} = 138.0$ 意味着如果我们给 $a$ 增加一个极小的正向扰动 $\epsilon$，最终输出 $g$ 将以 $138 \cdot \epsilon$ 的变化率剧烈增长。

### 1.3 为什么要基于“标量”构建？与张量（Tensor）的关系

大家可能会问：上面这个包含各种加减乘除和幂运算的表达式没有任何实际物理含义，为什么我们要写这样的代码？

答案是：**神经网络本质上就是这样的数学表达式，甚至比上述表达式还要更加简单规律**。神经网络将输入数据和网络权重作为输入，通过数学表达式计算出预测值与损失函数。反向传播算法并不关心这个表达式是不是神经网络，它能对**任何任意的数学表达式**求导。

另外需要特别强调的是：**Micrograd 是一个标量级（Scalar-valued）的引擎**。它直接运行在单个数字（标量）层面。而在生产环境的深度学习框架（如 PyTorch）中，我们使用的是 **$n$ 维张量（N-dimensional Tensors）**。

那么为什么要使用标量来讲解呢？
因为张量本质上只是将成千上万个标量打包封装成了数组（Arrays）。使用张量的**唯一目的就是为了并行计算效率**，在硬件层（CPU/GPU）并发执行数组运算以加速训练。**从数学本质和反向传播的原理来看，张量与标量没有任何区别！**

在教学中直接引入张量会引入大量维度对齐、张量广播和矩阵乘法形状等繁杂细节，遮蔽了反向传播最核心的数学美感。因此，Micrograd 剥离了所有效率优化，仅用最底层的标量原子向你彻底展示反向传播的灵魂。

### 1.4 代码量：轻巧而强大的 150 行代码

你可能会以为实现一个支持神经网络训练的自动求导引擎需要成千上万行代码，但事实上 Micrograd 极其简洁：

- `engine.py`：自动求导引擎核心，**仅约 100 行 Python 代码**。
- `nn.py`：建立在引擎之上的神经网络库（定义神经元 Neuron、单层 Layer 和多层感知机 MLP），**仅约 50 行 Python 代码**。

整整 150 行简单优雅的 Python 代码，就涵盖了训练现代神经网络所需要的全部核心机制！

---

## 2. 导数与梯度的直观数学理解

在正式动手写代码之前，我们需要确保大家在直觉层面彻底理解：**什么是导数？导数究竟向我们传递了什么信息？**

### 2.1 单变量函数的导数

让我们先引入最基础的 Python 环境，并定义一个单变量标量函数 $f(x)$：

```python
import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5
```

这是一个常见的二次抛物线函数 $f(x) = 3x^2 - 4x + 5$。如果传入输入 $x = 3.0$，函数输出为 $f(3.0) = 3(9) - 4(3) + 5 = 20.0$。

我们可以绘制出该函数在区间 $[-5, 5]$ 上的曲线图：

```python
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
```

```
       y ^
         |      *             *
      20 |-------o-----------
         |        \         /
       5 |---------\---o---/
         +----------+--+---+--------> x
                   -3  2/3 3
```

现在思考一个问题：**在特定的输入点 $x$，函数的“导数”代表什么？**

在高中或大学微积分课程中，我们通常会在纸上使用求导公式（如乘积法则、幂法则）：
$$\frac{d}{dx}(3x^2 - 4x + 5) = 6x - 4$$
然后代入 $x=3$ 算出 $6(3)-4 = 14$。

但在神经网络中，网络表达式极为庞大，拥有成千上万甚至数千亿个项，没有任何人会在纸上推导符号导数。因此，我们需要从**导数的极限定义（Limit Definition of Derivative）**出发去直观感受它：

$$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

这个公式的物理含义是：**如果在感兴趣的点 $x$ 处给它一个微小的正向增量 $h$，函数值 $f(x)$ 会发生怎样的变化？响应的敏感度（斜率）是多少？**

我们可以用数值近似（Numerical Approximation）的方法在代码中直接测试：

```python
h = 0.001
x = 3.0
d1 = f(x)         # 20.0
d2 = f(x + h)     # f(3.001) = 20.014003
slope = (d2 - d1) / h # (20.014003 - 20.0) / 0.001 = 14.003
```

- 当 $x = 3.0$ 时：给 $x$ 加上 $h=0.001$，$f(x+h)$ 变大到了 $20.014$。说明函数在正向响应，斜率（上升速率）为 $+14$。
- 当 $x = -3.0$ 时：$f(-3) = 44$。如果给 $x$ 加上 $h=0.001$，变为 $x = -2.999$，$f(-2.999) = 43.978$。函数值降低了！因此变化量 $(d2 - d1)$ 为负数，斜率为 $-22$。
- 当 $x = \frac{2}{3} \approx 0.6667$ 时：此时位于抛物线极小值底部，斜率为 $0$。给 $x$ 微小扰动，$f(x)$ 几乎保持不变。

---

### 2.2 多变量函数的偏导数直观

神经网络包含多个输入变量。现在让我们增加一点复杂度，考虑一个多变量标量函数 $d = a \cdot b + c$。

已知输入点：$a = 2.0$, $b = -3.0$, $c = 10.0$，输出为 $d = 2.0 \cdot (-3.0) + 10.0 = 4.0$。

我们希望评估输出 $d$ 关于各个输入变量 $a, b, c$ 的**偏导数（Partial Derivatives）** $\frac{\partial d}{\partial a}$, $\frac{\partial d}{\partial b}$, $\frac{\partial d}{\partial c}$。

我们可以编写一个小的测试脚本，通过对单个变量施加 $h$ 的扰动，观察 $d$ 的响应：

```python
h = 0.0001

# 基础点计算
a, b, c = 2.0, -3.0, 10.0
d1 = a * b + c

# 1. 测试 a 的偏导数 \partial d / \partial a
a += h
d2 = a * b + c
print("slope w.r.t a:", (d2 - d1) / h)  # 输出 -3.0

# 2. 测试 b 的偏导数 \partial d / \partial b
a, b, c = 2.0, -3.0, 10.0
b += h
d2 = a * b + c
print("slope w.r.t b:", (d2 - d1) / h)  # 输出 2.0

# 3. 测试 c 的偏导数 \partial d / \partial c
a, b, c = 2.0, -3.0, 10.0
c += h
d2 = a * b + c
print("slope w.r.t c:", (d2 - d1) / h)  # 输出 1.0
```

让我们从数学解析角度验证上述结果：
1. **对 $a$ 微分**：$\frac{\partial d}{\partial a} = \frac{\partial}{\partial a}(a \cdot b + c) = b$。因为 $b = -3.0$，所以偏导数为 $-3.0$。直观理解：因为 $b$ 是负数，给 $a$ 增加正增量，实际上是在减去更多的值，因此 $d$ 必然下降。
2. **对 $b$ 微分**：$\frac{\partial d}{\partial b} = \frac{\partial}{\partial b}(a \cdot b + c) = a$。因为 $a = 2.0$，偏导数为 $2.0$。
3. **对 $c$ 微分**：$\frac{\partial d}{\partial c} = \frac{\partial}{\partial c}(a \cdot b + c) = 1.0$。偏导数为 $1.0$。给 $c$ 增加多少，$d$ 就增加相同的量。

---

## 3. 构建表达式图与 Value 数据结构

有了对导数的直观感受后，我们需要构建一个数据结构来维护复杂的数学表达式图。这就是 `Value` 类。

### 3.1 Value 类的第一版封装

我们首先定义 `Value` 类，用它来包裹标量数字，并重载 `+` 和 `*` 运算符：

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # 初始梯度默认为 0 (假设该变量改变时不影响输出)
        self._prev = set(_children) # 记录产生当前节点的子节点集合
        self._op = _op             # 记录产生当前节点的操作符 (如 '+', '*')
        self.label = label         # 节点名称标签，方便调试可视化

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
```

在这里：
- `self.data` 存放实际的浮点数值。
- `self.grad` 存放输出 $L$ 关于该节点的梯度 $\frac{\partial L}{\partial \text{self}}$。初始设为 `0.0`。
- `self._prev` 保持指向子节点的指针集合，从而维持整个表达式的计算拓扑树。
- `self._op` 记录创建该节点的运算符。

测试其基础功能：
```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
```

此时，表达式 $L = (a \cdot b + c) \cdot f$ 在内存中构建了一个指向拓扑图：

```
a (2.0)  ---* (mul) ---> e (-6.0) 
b (-3.0) --/             \
                          + (add) ---> d (4.0) ---* (mul) ---> L (-8.0)
c (10.0) ----------------/                       /
f (-2.0) ---------------------------------------/
```

### 3.2 图可视化工具 (Graphviz Visualization)

为了能够直观地观察我们构建的复杂计算图，我们可以利用开源图可视化软件 Graphviz 编写一个绘图辅助函数 `draw_dot`：

```python
from graphviz import Digraph

def trace(root):
    # 递归收集计算图中所有的节点 (nodes) 和边 (edges)
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # 从左到右布局
    nodes, edges = trace(root)
    
    for n in nodes:
        uid = str(id(n))
        # 为每个 Value 节点绘制矩形框，显示 label, data 和 grad
        dot.node(name=uid, label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}", shape='record')
        if n._op:
            # 如果该节点由某个操作生成，创建一个圆形的虚拟操作符节点
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # 将子节点连接到操作符节点
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
```

调用 `draw_dot(L)`，我们就可以渲染出带有数据 `data` 和梯度 `grad` 状态的完整计算图！

---

## 4. 手动反向传播与链式法则详解

在自动编写递归代码之前，我们将对上述表达式 $L = (a \cdot b + c) \cdot f$ 进行**纯手动反向传播**推导。这是全课最精髓的部分：**只要你理解了这一个表达式的求导过程，你就理解了所有神经网络训练的核心原理！**

我们目前关注的目标是：计算输出 $L$ 关于图中每一个中间节点和叶子节点的偏导数 $\frac{\partial L}{\partial v}$，并填入各节点的 `.grad` 属性中。

---

### 4.1 根节点 $L$ 的基准情况 (Base Case)

首先从最右端的根节点 $L$ 开始。

**问题**：$\frac{\partial L}{\partial L}$ 是多少？
**答案**：显然是 $1.0$。当 $L$ 改变 $\epsilon$ 时，$L$ 自身的变化量也是 $\epsilon$。
因此：
```python
L.grad = 1.0
```

---

### 4.2 倒数第二层：乘法节点 $L = d \cdot f$

接下来，我们逆向倒退一步，求 $\frac{\partial L}{\partial d}$ 和 $\frac{\partial L}{\partial f}$。

根据已知关系 $L = d \cdot f$：
- 关于 $d$ 的偏导数：$$\frac{\partial L}{\partial d} = f$$
- 关于 $f$ 的偏导数：$$\frac{\partial L}{\partial f} = d$$

证明推导（基于导数定义）：
$$\frac{\partial L}{\partial d} = \lim_{h \to 0} \frac{(d+h)\cdot f - d \cdot f}{h} = \lim_{h \to 0} \frac{d\cdot f + h\cdot f - d\cdot f}{h} = \lim_{h \to 0} \frac{h\cdot f}{h} = f$$

代入具体的数值：
- $f.\text{data} = -2.0 \implies \frac{\partial L}{\partial d} = -2.0$
- $d.\text{data} = 4.0 \implies \frac{\partial L}{\partial f} = 4.0$

因此：
```python
d.grad = -2.0
f.grad = 4.0
```

---

### 4.3 核心枢纽：加法节点 $d = e + c$ 与微积分链式法则

现在我们需要推导偏导数 $\frac{\partial L}{\partial c}$ 和 $\frac{\partial L}{\partial e}$。

这是整个反向传播中最核心的一步：**节点 $c$ 和 $e$ 并不直接连接到最终输出 $L$，它们是通过中间节点 $d$ 间接影响 $L$ 的！**

已知 $d = e + c$，我们知道局部偏导数（Local Derivatives）：
$$\frac{\partial d}{\partial c} = 1.0, \quad \frac{\partial d}{\partial e} = 1.0$$

然而，我们真正需要的是全局偏导数（Global Derivative） $\frac{\partial L}{\partial c}$。如何将局部偏导数与上游已算得的偏导数 $\frac{\partial L}{\partial d}$ 结合起来？

答案就是**微积分链式法则（Chain Rule）**！

```
     链式法则 (Chain Rule)
  If z depends on y, and y depends on x:
      dz / dx = (dz / dy) * (dy / dx)
```

维基百科给出的经典直观解释：
> **如果一辆汽车的速度是自行车的 2 倍（$\frac{d\text{Car}}{d\text{Bike}} = 2$），而自行车的速度是行人步行速度的 4 倍（$\frac{d\text{Bike}}{d\text{Walk}} = 4$），那么汽车的速度就是行人步行速度的 $2 \times 4 = 8$ 倍（$\frac{d\text{Car}}{d\text{Walk}} = 2 \times 4 = 8$）。**

链式法则表明：**跨越复合函数的求导，只需将沿途的瞬时变化率（局部梯度）相乘！**

应用链式法则求解 $\frac{\partial L}{\partial c}$：
$$\frac{\partial L}{\partial c} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c} = (-2.0) \cdot 1.0 = -2.0$$

同理，求解 $\frac{\partial L}{\partial e}$：
$$\frac{\partial L}{\partial e} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial e} = (-2.0) \cdot 1.0 = -2.0$$

> [!IMPORTANT]
> **加法节点的反向传播重要规律**：
> 因为加法节点的局部偏导数永远是 $1.0$，所以在反向传播中，**加法节点本质上只是一个“梯度的分发器（Gradient Router）”**！它把上游传进来的梯度（此处为 $-2.0$）原封不动、等额地分发给它的所有子节点。

因此：
```python
c.grad = -2.0
e.grad = -2.0
```

---

### 4.4 乘法节点 $e = a \cdot b$ 的链式法则求导

现在继续逆向推进到叶子节点 $a$ 和 $b$。

已知 $e = a \cdot b$，且从上一步已经算出上游梯度 $\frac{\partial L}{\partial e} = -2.0$。

首先计算局部偏导数：
$$\frac{\partial e}{\partial a} = b = -3.0, \quad \frac{\partial e}{\partial b} = a = 2.0$$

再次应用链式法则求解最终梯度 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial b}$：

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial a} = (-2.0) \cdot (-3.0) = +6.0$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial e} \cdot \frac{\partial e}{\partial b} = (-2.0) \cdot 2.0 = -4.0$$

因此：
```python
a.grad = 6.0
b.grad = -4.0
```

总结当前所有节点的梯度信息：

| 节点 (Node) | 数据 (`data`) | 梯度 (`grad`) | 含义 ($\frac{\partial L}{\partial \text{Node}}$) |
| :--- | :--- | :--- | :--- |
| `L` | $-8.0$ | $+1.0$ | 基准梯度 |
| `f` | $-2.0$ | $+4.0$ | $\frac{\partial L}{\partial f} = d = 4.0$ |
| `d` | $+4.0$ | $-2.0$ | $\frac{\partial L}{\partial d} = f = -2.0$ |
| `c` | $+10.0$ | $-2.0$ | 梯路由：$1.0 \times (-2.0) = -2.0$ |
| `e` | $-6.0$ | $-2.0$ | 梯路由：$1.0 \times (-2.0) = -2.0$ |
| `a` | $+2.0$ | $+6.0$ | 链式乘法：$b \times (-2.0) = (-3.0) \times (-2.0) = +6.0$ |
| `b` | $-3.0$ | $-4.0$ | 链式乘法：$a \times (-2.0) = 2.0 \times (-2.0) = -4.0$ |

---

### 4.5 梯度下降（Gradient Step）直观验证

知道了上述所有梯度后，我们能拿它们做什么？

假设我们希望**增大输出 $L$ 的值**（使其从 $-8.0$ 往更大的方向增长），我们可以根据梯度的指引，微调可控的叶子节点 $a, b, c, f$：

根据梯度更新公式（沿着梯度方向移动一小步 $\eta = 0.01$）：
- $a.\text{data} += 0.01 \times a.\text{grad} = 2.0 + 0.01 \times (6.0) = 2.06$
- $b.\text{data} += 0.01 \times b.\text{grad} = -3.0 + 0.01 \times (-4.0) = -3.04$
- $c.\text{data} += 0.01 \times c.\text{grad} = 10.0 + 0.01 \times (-2.0) = 9.98$
- $f.\text{data} += 0.01 \times f.\text{grad} = -2.0 + 0.01 \times (4.0) = -1.96$

重新跑一遍正向传播：
```python
e = a * b          # 2.06 * (-3.04) = -6.2624
d = e + c          # -6.2624 + 9.98 = 3.7176
L = d * f          # 3.7176 * (-1.96) = -7.2865
```

我们可以看到，$L$ 的值成功从 $-8.0000$ 上升到了 $-7.2865$！

这证明了：**梯度精准地指示了所有参数对最终输出影响的方向与强度。只要我们顺着/逆着梯度的方向微调参数，就能精准地掌控输出值的上升或下降！** 

在实际神经网络训练中，由于我们的目标是**最小化损失函数（Loss）**，所以我们会向**梯度的反方向**微调网络参数（即 **梯度下降法 Gradient Descent**）。

---

## 5. 神经元数学模型与 Tanh 激活函数

掌握了基础计算图的反向传播后，我们将其应用到真正的**生物/数学神经元模型（Neuron Model）**中。

### 5.1 神经元的数学建模

在生物学中，神经元通过树突接收电信号，在细胞体中积聚信号，当电位超过一定阈值时通过轴突释放冲动。

数学上对单神经元的最简建模如下：

```
Inputs (x)    Weights (w)
  x1 ---------> ( * w1 ) ----\
                              +----> Sum ( + b ) ----> Activation f(x) ----> Output (o)
  x2 ---------> ( * w2 ) ----/
```

- 输入向量 $\mathbf{x} = [x_1, x_2, \dots, x_n]$
- 突触权重向量 $\mathbf{w} = [w_1, w_2, \dots, w_n]$ (控制每个输入的连接强度)
- 偏置 $b$ (Bias，控制神经元触发激活的难易程度/内在倾向性)
- 细胞体未激活累加值：$$n = \sum_i w_i x_i + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
- 激活函数（Activation Function / Squashing Function）：$o = f(n)$

---

### 5.2 $\tanh(x)$ 激活函数及其导数

在本例中，我们选择经典的双曲正切函数 **$\tanh(x)$** 作为激活函数。

$\tanh(x)$ 的数学定义为：
$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

画出 $\tanh(x)$ 的函数图像：

```
       y ^
       +1 +------------------...-- (Saturates at +1)
          |                .
        0 +-------------o------------- (Zero crossing at 0)
          |           .
       -1 +--...------------------ (Saturated at -1)
          +-----------+-----------+---> x
                     -2     0     2
```

从图像可以看出：$\tanh$ 函数将任意实数范围 $(-\infty, +\infty)$ 的输入非线性地挤压（Squash）到 $(-1, +1)$ 区间内。当输入非常大时平滑饱和收敛至 $+1$，当输入非常小时收敛至 $-1$。

**$\tanh(x)$ 的导数公式**：
根据微积分求导规则，$\tanh(x)$ 的导数形式非常简洁美丽：
$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

这一公式在反向传播实现中极为高效：**因为我们在正向传播时已经计算出了 $o = \tanh(x)$，那么在反向传播时，局部导数直接就是 $1 - o^2$！**

---

### 5.3 手动反向传播单个神经元

让我们在 Python 中构建一个双输入神经元 $x_1, x_2$ 的计算图：

```python
# 输入 x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# 权重 w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# 偏置 b
b = Value(6.8813735870195432, label='b') # 选取特殊值方便结果凑整

# 点积与累加过程
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2.label = 'x1*w1 + x2*w2'
n = x1w1_x2w2 + b; n.label = 'n' # 未激活细胞体信号
```

现在我们需要将 `n` 传过 $\tanh$ 激活函数。因为当前的 `Value` 类还不支持 $\tanh$，我们先在 `Value` 类中添加 `tanh()` 方法：

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    return out
```

执行正向传播：
```python
o = n.tanh(); o.label = 'o'
# 计算得出: n.data = 0.8813735, o.data = 0.7071
```

接下来，我们对这个神经元从输出 `o` 开始手动倒推所有梯度：

1. **根节点基准**：
   $$o.\text{grad} = 1.0$$
2. **通过 $\tanh$ 节点反向传播**：
   已知 $o = \tanh(n)$，局部导数 $\frac{\partial o}{\partial n} = 1 - o^2$。
   由于 $o.\text{data} = 0.7071$，则 $1 - (0.7071)^2 = 1 - 0.5 = 0.5$。
   应用链式法则：
   $$n.\text{grad} = o.\text{grad} \cdot \frac{\partial o}{\partial n} = 1.0 \times 0.5 = 0.5$$
3. **通过加法节点 `n = x1w1_x2w2 + b` 反向传播**：
   加法节点原封不动分发梯度：
   $$b.\text{grad} = 0.5, \quad x1w1\_x2w2.\text{grad} = 0.5$$
4. **通过加法节点 `x1w1_x2w2 = x1w1 + x2w2` 反向传播**：
   $$x1w1.\text{grad} = 0.5, \quad x2w2.\text{grad} = 0.5$$
5. **通过乘法节点 `x1w1 = x1 * w1` 反向传播**：
   - $x1.\text{grad} = w1.\text{data} \times x1w1.\text{grad} = -3.0 \times 0.5 = -1.5$
   - $w1.\text{grad} = x1.\text{data} \times x1w1.\text{grad} = 2.0 \times 0.5 = +1.0$
6. **通过乘法节点 `x2w2 = x2 * w2` 反向传播**：
   - $x2.\text{grad} = w2.\text{data} \times x2w2.\text{grad} = 1.0 \times 0.5 = +0.5$
   - $w2.\text{grad} = x2.\text{data} \times x2w2.\text{grad} = 0.0 \times 0.5 = 0.0$

看！因为输入 $x_2 = 0.0$，在乘法作用下，$x_2$ 屏蔽了权重 $w_2$ 的改变对输出的影响，因此 $w_2.\text{grad} = 0.0$！这与物理直觉完全契合！

---

## 6. 自动化反向传播实现：闭包与拓扑排序

手动一步步写代码赋予 `.grad` 赋值显然效率太低。现在我们要将这一全过程**自动化**。

### 6.1 在节点中保存局部反向传播闭包 (`_backward`)

我们要让每个 `Value` 节点在被创建时，自动知道如何将自己的 `.grad` 链式传播给前驱子节点。

我们在 `Value` 中引入 `self._backward` 闭包函数：

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None # 默认无操作 (如叶子节点)
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # 链式法则：局部梯度 (1.0) * 上游梯度 (out.grad)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            # 链式法则：局部梯度 (1 - t^2) * 上游梯度 (out.grad)
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out
```

---

### 6.2 拓扑排序 (Topological Sort) 的引入

只定义好 `_backward()` 函数还不够。在什么顺序下依次调用各个节点的 `_backward()`？

思考：**如果我们还没有计算出上游节点 `out` 的最终 `grad`，就先去调用前驱子节点的 `_backward()`，那么子节点算出的梯度必然是错误的！**

因此，在有向无环计算图（DAG）中，**必须保证所有下游节点先完成反向传播，才能轮到上游节点**。在图论中，这种节点遍历顺序称为**拓扑排序（Topological Sort）**。

```
拓扑排序示例:
  x1, w1 ---> x1w1 ---\
                       +---> n ---> o (根节点)
  x2, w2 ---> x2w2 ---/
  
正向拓扑链: [x1, w1, x1w1, x2, w2, x2w2, b, n, o]
逆向反向传播链: [o, n, b, x2w2, w2, x2, x1w1, w1, x1]
```

我们可以编写深度优先搜索 (DFS) 算法来实现拓扑排序，并封装进 `Value` 类的 `backward()` 方法中：

```python
    def backward(self):
        # 1. 构建拓扑排序序列
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2. 将根节点梯度初始化为 1.0
        self.grad = 1.0

        # 3. 逆序遍历拓扑链，依次触发每个节点的 _backward()
        for node in reversed(topo):
            node._backward()
```

此时，用户只需在根节点上简单调用：
```python
o.backward()
```
整张计算图中成百上千个节点的梯度就会在毫秒内自动递归计算完毕！

---

### 6.3 重要 Bug 修复：多元链式法则与梯度的累加 (+=)

在上文的代码中，大家可能注意到我们在 `_backward()` 闭包中使用的是 `+=` 而不是简单的 `=`：
```python
self.grad += 1.0 * out.grad
```

**为什么要使用 `+=` 进行梯度累加？** 这是一个极其关键且隐蔽的底层 Bug！

假设存在如下计算图，变量 $a$ 被使用了多次：
```python
a = Value(3.0)
b = a + a
b.backward()
```
从数学上看：$b = 2a \implies \frac{\partial b}{\partial a} = 2$。

如果我们在 `_backward()` 中写的是赋值语句 `self.grad = 1.0 * out.grad`：
1. 第一次加法分支计算 $a$ 的梯度：`a.grad = 1.0`
2. 第二次加法分支计算 $a$ 的梯度：`a.grad = 1.0` (覆盖掉了第一次的结果！)

最终得到的 `a.grad` 是 $1.0$，这显然是错的！

根据多元微积分的链式法则（Multivariate Chain Rule）：**当一个变量向后续多条路径分叉流动时，该变量的总梯度等于所有流动路径梯度的累加和！**

$$\frac{\partial L}{\partial a} = \sum_i \frac{\partial L}{\partial \text{path}_i} \cdot \frac{\partial \text{path}_i}{\partial a}$$

因此，**所有梯度的更新必须统一采用 `+=` 进行累加！**

---

## 7. 扩展 Value 类的运算符与基础函数

为了让我们的 `Value` 类能够构建更加复杂的现代神经网络（如包括除法、幂运算、减法、ReLU 等），我们需要补全 Python 的重载运算符和标量数学函数。

### 7.1 标量混合运算与 `__radd__`, `__rmul__`

目前执行 `a + 1` 或 `a * 2` 会报错，因为 `1` 或 `2` 不是 `Value` 对象。

我们可以通过自动类型转换来解决：
```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        ...
```
但是对于 `2 * a` 或 `1 + a`，Python 会首先在整型 `2` 上调用 `__mul__`，导致失败。此时 Python 会回退尝试调用 `Value` 类的**反向重载运算符 `__rmul__` 和 `__radd__`**：

```python
    def __rmul__(self, other): # 应对 2 * self
        return self * other

    def __radd__(self, other): # 应对 1 + self
        return self + other
```

---

### 7.2 幂运算 `__pow__` 与 Power Rule 求导

我们要支持 `a ** k`（其中 $k$ 为常数实数或整数）：

根据微积分中的**幂法则（Power Rule）**：
$$\frac{d}{dx}(x^k) = k \cdot x^{k-1}$$

结合链式法则实现：
```python
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "目前仅支持常数次幂"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            # 链式法则: k * x^(k-1) * out.grad
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out
```

---

### 7.3 减法与除法的巧妙转换

我们不需要为减法和除法单独推导复杂的导数逻辑，因为它们可以自然转换为加法、乘法和幂运算的组合：

1. **减法**：$a - b = a + (-1 \cdot b)$
2. **除法**：$\frac{a}{b} = a \cdot b^{-1}$

```python
    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __truediv__(self, other): # self / other
        return self * (other**-1)

    def __rtruediv__(self, other): # other / self
        return other * (self**-1)
```

---

### 7.4 指数函数 `exp()`

定义 $e^x$ 及其导数 $\frac{d}{dx} e^x = e^x$：

```python
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
```

---

### 7.5 拆解 $\tanh$ 验证计算图完备性

有了 `exp()`、`-` 和 `/` 之后，我们甚至不再需要把 `tanh` 作为一个原子节点。我们可以将 $\tanh$ 直接拆解为它的底层算式：

$$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$

```python
# 拆解方式实现 tanh
two_n = 2 * n
e = two_n.exp()
o = (e - 1) / (e + 1)
o.backward()
```

运行代码后，我们可以看到：将 $\tanh$ 拆解为 5 个原子节点（包含指数、减法、加法、除法）构建的极其冗长的计算图，计算得出的叶子节点梯度与此前直接使用复合 `tanh` 节点算出的梯度**完全一模一样**！

> [!NOTE]
> 这深刻地说明了：**在自动求导引擎中，你可以随意选择在什么粒度抽象算子！** 只要你能准确写出该模块的局部导数，你可以把它写成极小的原子算子（如 `+`, `*`），也可以打包成复杂的复合算子（如 `tanh`, `softmax`, `attention`）。

---

## 8. 对比 PyTorch 官方 API

在完成了 Micrograd 的核心引擎后，我们把它与现代工业级框架 **PyTorch** 进行一次直接对比。

让我们用 PyTorch 重写上述相同的双输入神经元代码：

```python
import torch

# PyTorch 中默认创建 Single Precision (float32)，我们需要显式声明为 double (float64) 以保持浮点精度完全一致
x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
w1 = torch.tensor([-3.0], dtype=torch.float64, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
b  = torch.tensor([6.8813735870195432], dtype=torch.float64, requires_grad=True)

# 正向传播
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print("Forward Pass Output (PyTorch):", o.item()) # 输出 0.7071067811865476

# 反向传播
o.backward()

print("--- Gradients (PyTorch vs Micrograd) ---")
print("x1.grad:", x1.grad.item()) # -1.5
print("w1.grad:", w1.grad.item()) # 1.0
print("x2.grad:", x2.grad.item()) # 0.5
print("w2.grad:", w2.grad.item()) # 0.0
```

测试结果表明：**Micrograd 计算出的数值与 PyTorch 官方 API 的输出结果精度达到了小数点后 16 位的完全重合！**

这印证了我们的结论：Micrograd 的 API 架构（`.data`, `.grad`, `.backward()`, `requires_grad` 等）完全是对 PyTorch 的高度精确致敬。两者唯一的区别在于 PyTorch 底层处理的是高效并发的张量。

---

## 9. 搭建神经网络库：Neuron, Layer, MLP

引擎部分彻底竣工。现在我们在 `engine.py` 的基础上，编写 `nn.py` 构建一个完整的神经网络库。

根据深度学习的层级结构：
$$\text{Value (标量节点)} \longrightarrow \text{Neuron (神经元)} \longrightarrow \text{Layer (神经元层)} \longrightarrow \text{MLP (多层感知机)}$$

```
[Layer 1 (Input)] ---> [Layer 2 (Hidden)] ---> [Layer 3 (Output)]
 (3 inputs)              (4 neurons)              (1 neuron)
```

---

### 9.1 神经元类 `Neuron`

单个神经元接收 $n_{\text{in}}$ 个输入，创建 $n_{\text{in}}$ 个随机初始化的权重和一个偏置：

```python
import random

class Module:
    """参考 PyTorch 中的 nn.Module 基类"""
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
        # 随机初始化权重在 [-1.0, 1.0] 区间内
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # 计算 w * x + b
        # zip(self.w, x) 产生 (w_i, x_i) 元素对
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
```

在 Python 中，重载 `__call__` 方法允许我们像调用函数一样使用对象：`n(x)`。

---

### 9.2 神经元层类 `Layer`

一个神经元层由 $n_{\text{out}}$ 个平行的神经元构成，它们共享相同的输入向量：

```python
class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

---

### 9.3 多层感知机类 `MLP`

多层感知机（Multilayer Perceptron）由多个 `Layer` 依次首尾相连构成：

```python
class MLP(Module):
    def __init__(self, nin, nouts):
        # nouts 是一个包含各层输出神经元数量的列表，如 [4, 4, 1]
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

验证 MLP 的构建：
```python
x = [2.0, 3.0, -1.0] # 输入维度为 3
n = MLP(3, [4, 4, 1]) # 构建架构为 3 -> 4 -> 4 -> 1 的三层网络
out = n(x)            # 前向传播，输出单个 Value 标量
```

检查参数总量：
```python
len(n.parameters()) # 输出来自所有权重的偏置标量参数总数 (本例中为 41 个)
```

---

## 10. 完整的神经网络训练实战与梯度下降

我们现在万事俱备，准备在一个二分类数据集上完成神经网络的完整训练流程。

### 10.1 构建训练数据集与损失函数

假设我们有一个只有 4 个样本的小型二分类数据集：

```python
# 4 个三维输入样本
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

# 对应的目标标签 y (Ground Truth)
ys = [1.0, -1.0, -1.0, 1.0]
```

目前未经过训练的网络预测结果可能非常糟糕（例如全部预测出接近 0 的数字）。

为了衡量神经网络当前的表现好坏，我们需要定义**损失函数（Loss Function）**。本例中使用经典的**均方误差损失（Mean Squared Error Loss, MSE）**：

$$L = \sum_{i=1}^{N} (y_{\text{pred}}^{(i)} - y_{\text{truth}}^{(i)})^2$$

```python
def compute_loss(model, xs, ys):
    ypred = [model(x) for x in xs]
    # MSE 损失计算
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    return loss, ypred
```

- 当预测值 $y_{\text{pred}}$ 完全等于真实标签 $y_{\text{truth}}$ 时，损失 $L = 0$。
- 偏离越远，损失 $L$ 越大。我们的目标就是**将损失 $L$ 优化下降至接近 0**。

---

### 10.2 标准训练循环 (Training Loop) 四步曲

现代深度学习模型的训练循环始终遵循固定的四步步奏：

1. **正向传播 (Forward Pass)**：输入数据，计算预测值 `ypred` 和当前损失值 `loss`。
2. **梯度清零 (Zero Grad)**：清空上一次迭代残留在所有参数上的 `.grad` 状态。
3. **反向传播 (Backward Pass)**：调用 `loss.backward()` 自动求导，计算所有参数关于当前损失的梯度。
4. **参数更新 (SGD Step)**：按照梯度下降更新公式微调所有参数。

梯度下降的参数更新公式：
$$p.\text{data} \mathrel{-}= \eta \cdot p.\text{grad}$$

其中 $\eta$ 为**学习率（Learning Rate）**。负号的原因是：**梯度指向损失增加最快的方向，而我们要最小化损失，因此必须沿着梯度的反方向迈步！**

```python
# 初始化网络
model = MLP(3, [4, 4, 1])
learning_rate = 0.05

print("--- 开始训练网络 ---")
for k in range(100): # 迭代 100 步
    # 1. 正向传播
    loss, ypred = compute_loss(model, xs, ys)

    # 2. 梯度清零 (极其重要!)
    model.zero_grad()

    # 3. 反向传播
    loss.backward()

    # 4. 参数更新 (SGD)
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if k % 10 == 0:
        print(f"Step {k:2d} | Loss: {loss.data:.6f}")
```

输出日志示例：
```
Step  0 | Loss: 7.124589
Step 10 | Loss: 2.341205
Step 20 | Loss: 0.895123
Step 30 | Loss: 0.231456
Step 40 | Loss: 0.078412
...
Step 90 | Loss: 0.000845
```

观察预测值与真实的对比：
```python
print("目标值 (ys):", ys)
print("预测值 (ypred):", [f"{y.data:.4f}" for y in ypred])
# 输出: ['0.9821', '-0.9754', '-0.9689', '0.9891']
```

损失从初始的 **$7.12$ 一路猛降到了 $0.0008$**！预测值完全精准地逼近了目标标签 $[1, -1, -1, 1]$！网络成功学会了这组数据的二分类规则！

---

### 10.3 深度剖析：为什么漏掉 `zero_grad()` 会导致灾难性 Bug？

在上述训练循环中，如果漏掉了 `model.zero_grad()`，会发生什么？

在讲座中，Karpathy 坦言自己曾多次在镜头前犯过这个经典错误！

**Bug 原因解释**：
因为我们在 `Value` 类的反向传播中采用了 `+=` 梯度累加机制：
```python
self.grad += ...
```
如果在开启新一轮 `loss.backward()` 之前，不手动将所有 `p.grad` 重置清零为 `0.0`，那么**前一次迭代算出来的梯度就会直接叠加在新的梯度上面**！

这将导致：
- 梯度值以指数级不断爆炸膨胀。
- 参数更新步子迈得过大，直接跳出了损失函数的正常区域。
- 训练过程迅速失控崩溃，损失直接爆炸飞到无穷大 (NaN/Inf)。

> [!WARNING]
> **记住**：在每次执行 `loss.backward()` 前，必须无条件执行 `model.zero_grad()`！（在 PyTorch 中对应 `optimizer.zero_grad()`）。

---

## 11. 总结与 Micrograd 源码全貌回顾

### 11.1 神经网络与 AI 的本质哲学

通过这堂课从零构建 Micrograd，我们应当洞察到神经网络极其朴素而美丽的本质：

1. **神经网络是什么？**  
   神经网络不过是一个庞大而规则的**数学表达式**。它接收输入数据和可调权重参数，通过正向传播算出预测值和损失。
2. **损失函数是什么？**  
   损失函数是一个把网络表现的好坏硬性量化为一个单标量数字的评估标准。
3. **反向传播是什么？**  
   反向传播是递归应用微积分**链式法则**逆向穿过计算图，精确得知每个参数微调对损失的影响（梯度）。
4. **梯度下降是什么？**  
   梯度下降是根据梯度指引，向着降低损失的方向微幅微调参数，并通过成千上万次迭代使网络性能达到顶峰。

目前最前沿的 AI 大语言模型（如 GPT-4），尽管包含了上千亿个参数，采用了更加复杂的神经网络结构（如 Transformer 自注意力机制）、交叉熵损失函数和大规模 GPU 张量并行计算，**但其底层的训练逻辑与 Micrograd 的这 150 行代码 100% 完全相同！**

---

### 11.2 Micrograd 源码目录一览

完整的 Micrograd 项目在 GitHub 上仅包含极其精简的文件组织：

- `micrograd/engine.py`: `Value` 类，包含完整自动求导引擎（拓扑排序与 10 余种重载运算符）。
- `micrograd/nn.py`: `Module`, `Neuron`, `Layer`, `MLP` 定义。
- `test/test_engine.py`: 单元测试文件，对比 Micrograd 与 PyTorch 在复杂表达式下的正向与反向梯度数值一致性。
- `demo.ipynb`: 一个更高级的二维双螺旋（Moon Data）分类 Demo，展示了 Mini-batch 批处理、学习率衰减（Learning Rate Decay）和 L2 正则化（Regularization）的实际用法。

如果你完整跟随本教程编写出了上述代码，那么恭喜你，你已经 100% 掌握了 Micrograd 的全部秘密，也彻底揭开了神经网络和深度学习最核心的神秘面纱！

---

> **结语**：希望大家喜欢这次手刷 Micrograd 的硬核之旅！如果大家觉得本教程有帮助，请不要忘记在 GitHub 上给 Micrograd Star，并关注 Karpathy 的后续神经网络课程。Happy Coding!
