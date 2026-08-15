#!/usr/bin/env python3
"""Generate clean, text-accurate SVG teaching diagrams for the Markdown notes."""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', sans-serif"
INK, MUTED = "#172033", "#526079"
BLUE, BLUE_BG = "#2563eb", "#dbeafe"
ORANGE, ORANGE_BG = "#ea580c", "#ffedd5"
GREEN, GREEN_BG = "#15803d", "#dcfce7"
PURPLE, PURPLE_BG = "#7c3aed", "#ede9fe"
RED, RED_BG = "#b91c1c", "#fee2e2"
PAPER, GRID = "#fffdf8", "#e5e7eb"


def esc(s):
    return escape(str(s))


def t(x, y, s, size=17, color=INK, weight=400, anchor="middle"):
    return f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>'


def lines(x, y, values, size=15, color=MUTED, gap=22):
    return "".join(t(x, y + i * gap, v, size, color, 500) for i, v in enumerate(values))


def arrow(x1, y1, x2, y2, color=BLUE, dashed=False, width=3):
    marker = {BLUE: "blue", ORANGE: "orange", GREEN: "green", PURPLE: "purple", RED: "red"}.get(color, "blue")
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="round" marker-end="url(#arrow-{marker})"{dash}/>'


def box(x, y, w, h, title, detail="", bg=BLUE_BG, stroke=BLUE):
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{bg}" stroke="{stroke}" stroke-width="2"/>', t(x + w / 2, y + 32, title, 18, INK, 700)]
    if detail:
        parts.append(t(x + w / 2, y + 66, detail, 15, MUTED, 500))
    return "".join(parts)


def base(title, w, h, body, desc):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">
  <title id="title">{esc(title)}</title><desc id="desc">{esc(desc)}</desc>
  <defs>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{BLUE}"/></marker>
    <marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{ORANGE}"/></marker>
    <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{GREEN}"/></marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{PURPLE}"/></marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{RED}"/></marker>
  </defs>
  <rect width="100%" height="100%" rx="20" fill="{PAPER}"/>
  <text x="40" y="48" font-family="{FONT}" font-size="26" font-weight="700" fill="{INK}">{esc(title)}</text>
  {body}
</svg>\n'''


def save(name, title, body, w=1200, h=540, desc=""):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(base(title, w, h, body, desc or title), encoding="utf-8")


def flow(name, title, stages, caption, accent=BLUE, w=1200, h=540, vertical=False):
    body = t(w / 2, 102, caption, 17, MUTED, 500)
    colors = [(BLUE_BG, BLUE), (ORANGE_BG, ORANGE), (PURPLE_BG, PURPLE), (GREEN_BG, GREEN), (RED_BG, RED)]
    if vertical:
        x, y, bw, bh, gap = (w - 430) / 2, 135, 430, 66, 50
        for i, (title2, detail) in enumerate(stages):
            bg, stroke = colors[i % len(colors)]
            body += box(x, y, bw, bh, title2, detail, bg, stroke)
            if i + 1 < len(stages):
                body += arrow(w / 2, y + bh, w / 2, y + bh + gap, accent)
            y += bh + gap
    else:
        bw, bh, gap = 180, 112, 28
        total = len(stages) * bw + (len(stages) - 1) * gap
        x, y = (w - total) / 2, 220
        for i, (title2, detail) in enumerate(stages):
            bg, stroke = colors[i % len(colors)]
            body += box(x, y, bw, bh, title2, detail, bg, stroke)
            if i + 1 < len(stages):
                body += arrow(x + bw, y + bh / 2, x + bw + gap, y + bh / 2, accent)
            x += bw + gap
    save(name, title, body, w, h, caption)


def curve(name, title, caption, points, labels, color=BLUE):
    body = t(600, 100, caption, 17, MUTED, 500)
    body += arrow(150, 450, 1080, 450, MUTED, width=2)
    body += arrow(150, 450, 150, 120, MUTED, width=2)
    body += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'
    body += t(1090, 460, labels[0], 17, MUTED, 700)
    body += t(130, 110, labels[1], 17, MUTED, 700, "end")
    save(name, title, body, 1200, 540, caption)


def bars(name, title, caption, data, colors=None):
    body = t(600, 100, caption, 17, MUTED, 500)
    base_y = 455
    colors = colors or [BLUE, ORANGE, PURPLE, GREEN, RED]
    for i, (label, value, display) in enumerate(data):
        x = 125 + i * 210
        height = max(55, value * 300)
        color = colors[i % len(colors)]
        body += f'<rect x="{x}" y="{base_y - height}" width="105" height="{height}" rx="13" fill="{color}" opacity="0.88"/>'
        body += t(x + 52, base_y + 30, label, 15, MUTED, 600)
        body += t(x + 52, base_y - height - 14, display, 16, color, 700)
    body += arrow(90, base_y, 1090, base_y, MUTED, width=2)
    save(name, title, body, 1200, 560, caption)


def main():
    body = box(450, 145, 300, 78, "Micrograd", "标量自动微分", BLUE_BG, BLUE)
    body += box(120, 315, 360, 112, "engine.py", "Value · 拓扑排序 · 反向传播", BLUE_BG, BLUE)
    body += box(720, 315, 360, 112, "nn.py", "Neuron · Layer · MLP", ORANGE_BG, ORANGE)
    body += arrow(600, 223, 300, 315) + arrow(600, 223, 900, 315)
    body += t(600, 500, "先造出自动求导发动机，再把它组装成神经网络", 18, INK, 600)
    save("micrograd_modules.svg", "Micrograd 的两层结构", body, 1200, 560, "Micrograd 由标量自动微分 engine.py 与神经网络组件 nn.py 组成")
    curve("micrograd_parabola.svg", "梯度与局部斜率", "曲线上的局部斜率决定参数更新方向", "170,400 250,330 330,270 410,225 490,190 570,165 650,150 730,145 810,148 890,158 970,175 1050,200", ("x", "y"), ORANGE)
    flow("micrograd_chain_rule.svg", "链式法则与反向传播", [("x", "输入变量"), ("y = f(x)", "中间变量"), ("z = g(y)", "最终输出")], "前向计算从左到右，梯度沿相反方向传播", ORANGE)
    flow("micrograd_neuron_forward.svg", "神经元的前向计算", [("x₁ × w₁", "加权输入"), ("Σ + b", "求和并加偏置"), ("f(·)", "激活函数"), ("o", "输出")], "一个神经元就是：加权求和 → 非线性变换")
    curve("micrograd_tanh.svg", "tanh 激活函数与梯度", "tanh 在中间区域梯度较大，在两端饱和区梯度接近零", "165,180 245,185 325,205 405,250 485,320 565,385 645,420 725,435 805,440 885,442 965,443 1045,443", ("x", "tanh(x)"), BLUE)
    flow("micrograd_topological_order.svg", "计算图的拓扑顺序", [("x₁, w₁, x₂, w₂", "输入与参数"), ("x₁w₁ / x₂w₂", "局部乘法"), ("n", "汇合"), ("o", "根节点")], "正向按依赖关系计算，反向按相反顺序累加梯度", BLUE, 1200, 540)
    flow("micrograd_loss_curve.svg", "损失函数下降", [("预测", "当前输出"), ("Loss", "衡量错误"), ("梯度", "指出方向"), ("更新参数", "重新预测")], "梯度下降的目标：让损失沿训练过程持续下降", GREEN)
    flow("bigram_pipeline.svg", "Bigram 语言模型流程", [("字符 x", "例如：a"), ("One-hot", "(1, 27)"), ("× W", "27 × 27"), ("Logits", "(1, 27)"), ("Softmax", "概率分布")], "Bigram 只根据当前字符预测下一个字符", GREEN, 1250, 540)
    flow("mlp_pipeline.svg", "MLP 语言模型数据流", [("输入索引", "x₁, x₂, x₃"), ("Embedding", "27 × d"), ("拼接", "Shape 3d"), ("隐藏层", "W₁ + b₁ + tanh"), ("输出层", "27 logits"), ("Softmax", "P(y | x₁,x₂,x₃)")], "多个历史字符共享嵌入矩阵，再联合预测下一个字符", GREEN, 1400, 540)
    curve("batchnorm_loss_curve.svg", "BatchNorm 前后的损失曲线", "合理初始化与 BatchNorm 让训练从一开始就进入健康区间", "180,170 250,205 320,245 390,290 460,330 530,360 600,385 670,402 740,415 810,425 900,432 1050,438", ("Iterations", "Loss"), RED)
    curve("batchnorm_tanh.svg", "BatchNorm 把激活值拉回健康区", "标准化让 tanh 输入集中在中间区域，避免梯度消失", "165,180 245,185 325,205 405,250 485,320 565,385 645,420 725,435 805,440 885,442 965,443 1045,443", ("x", "tanh(x)"), BLUE)
    flow("batchnorm_pipeline.svg", "BatchNorm 前向流程", [("输入 hpreact", "一批激活值"), ("统计 μ_B, σ²_B", "均值与方差"), ("标准化 x̂", "(x−μ)/√(σ²+ε)"), ("γx̂ + β", "可学习变换"), ("tanh / ReLU", "继续前向")], "BatchNorm 的四步：统计 → 标准化 → 仿射变换 → 继续计算", PURPLE, 1400, 540)
    bars("batchnorm_activation_health.svg", "激活值健康度", "四层 Tanh 的标准差接近，激活分布保持健康", [("Layer 1", .64, "std 0.64"), ("Layer 2", .65, "std 0.65"), ("Layer 3", .65, "std 0.65"), ("Layer 4", .65, "std 0.65")])
    bars("batchnorm_gradient_health.svg", "梯度流健康度", "各层梯度尺度接近：没有明显的梯度消失或爆炸", [("Layer 1", .72, "std 0.003"), ("Layer 2", .72, "std 0.003"), ("Layer 3", .72, "std 0.003"), ("Layer 4", .72, "std 0.003")], [GREEN])
    flow("batchnorm_diagnostics.svg", "网络健康度诊断", [("1 观察激活值", "饱和度约 5%"), ("2 观察梯度", "避免消失 / 爆炸"), ("3 观察更新", "log10 比例约 -3")], "先看数据分布，再看梯度流，最后看参数更新是否合适", GREEN)
    body = t(625, 100, "字符表示沿树状路径逐层合并，感受野随深度扩大", 17, MUTED, 500)
    for x, label in [(120, "C₁ C₂"), (340, "C₃ C₄"), (560, "C₅ C₆"), (780, "C₇ C₈")]:
        body += box(x, 145, 150, 65, label, "字符", BLUE_BG, BLUE)
    body += box(225, 280, 185, 70, "B₁", "2-gram", ORANGE_BG, ORANGE) + box(665, 280, 185, 70, "B₂", "2-gram", ORANGE_BG, ORANGE)
    body += box(445, 410, 220, 70, "F₁", "4-gram", PURPLE_BG, PURPLE) + box(515, 540, 220, 70, "Predict", "输出 logits", GREEN_BG, GREEN)
    body += arrow(195, 210, 290, 280) + arrow(415, 210, 345, 280) + arrow(635, 210, 730, 280) + arrow(855, 210, 790, 280)
    body += arrow(410, 315, 500, 410) + arrow(760, 350, 610, 410) + arrow(555, 480, 625, 540)
    save("wavenet_tree.svg", "WaveNet 层次化感受野", body, 1250, 660, "输入字符经过 2-gram、4-gram 等树状层次化融合后预测输出")
    body = t(600, 100, "下三角允许注意力，上三角遮住未来信息", 17, MUTED, 500)
    x0, y0, cell = 405, 155, 78
    for i, label in enumerate(["t−3", "t−2", "t−1", "t"]):
        body += t(x0 + i * cell + cell / 2, y0 - 18, label, 15, MUTED, 700)
        body += t(x0 - 24, y0 + i * cell + cell / 2 + 5, label, 15, MUTED, 700, "end")
        for j in range(4):
            allowed = j <= i
            fill, stroke, label2 = (GREEN_BG, GREEN, "看") if allowed else (RED_BG, RED, "遮")
            body += f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            body += t(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2 + 7, label2, 18, stroke, 700)
    body += t(600, 560, "当前位置只能看见自己和历史信息，绝不泄漏未来", 18, INK, 600)
    save("causal_mask.svg", "Transformer 因果掩码", body, 1200, 640, "因果掩码用下三角矩阵阻断未来 Token 的信息")
    flow("wavenet_context_reuse.svg", "WaveNet 滑动窗口与特征复用", [("位置 i", "(C₁,C₂) → H₁"), ("位置 i+1", "(C₂,C₃) → H₁′"), ("下一层", "(H₁,H₂) → Q₁")], "窗口向右移动一个位置，旧特征和新特征一起被重新组合", ORANGE)
    body = t(600, 100, "同一个 Token 并行投影成 Query、Key、Value 三种角色", 17, MUTED, 500)
    body += box(500, 150, 200, 82, "Token X", "输入表示", BLUE_BG, BLUE)
    body += box(110, 350, 280, 105, "Query (Q)", "我在寻找什么？", ORANGE_BG, ORANGE)
    body += box(460, 350, 280, 105, "Key (K)", "我能提供什么？", PURPLE_BG, PURPLE)
    body += box(810, 350, 280, 105, "Value (V)", "真正传递的内容", GREEN_BG, GREEN)
    body += arrow(570, 232, 250, 350, ORANGE) + arrow(600, 232, 600, 350, PURPLE) + arrow(630, 232, 950, 350, GREEN)
    body += t(600, 520, "先用 Q·K 计算相关性，再用权重加权聚合 V", 18, INK, 600)
    save("qkv_roles.svg", "自注意力中的 Q K V", body, 1200, 590, "一个 Token 并行生成 Query Key Value")
    body = t(600, 100, "残差路径绕过子层，直接把输入加回输出", 17, MUTED, 500)
    body += box(100, 260, 150, 82, "x", "输入", BLUE_BG, BLUE)
    body += box(430, 170, 250, 82, "LayerNorm", "归一化", PURPLE_BG, PURPLE)
    body += box(430, 350, 250, 82, "Sub-Layer", "Attention / FFN", ORANGE_BG, ORANGE)
    body += box(950, 260, 170, 82, "x_out", "输出", GREEN_BG, GREEN)
    body += arrow(250, 301, 430, 211) + arrow(555, 252, 555, 350) + arrow(680, 391, 950, 301)
    body += '<path d="M250 301 C390 60, 820 60, 950 301" fill="none" stroke="#15803d" stroke-width="4" stroke-dasharray="9 8" marker-end="url(#arrow-green)"/>'
    body += t(600, 530, "x_out = x + SubLayer(LayerNorm(x))", 19, INK, 700)
    save("residual_block.svg", "Transformer 残差块", body, 1200, 600, "输入经过归一化和子层，同时沿跳连路径与输出相加")
    bars("transformer_loss_progression.svg", "Transformer 验证损失进步", "从 Bigram 到多层 Transformer 的验证损失逐步下降", [("Bigram", .18, "4.87"), ("单头", .48, "2.40"), ("多头+FFN", .53, "2.24"), ("残差+LN", .57, "2.06"), ("6层10M", .78, "1.48")])
    flow("encoder_decoder.svg", "Encoder–Decoder 结构", [("Encoder", "读取完整输入"), ("K / V", "提供上下文"), ("Decoder", "因果生成输出")], "编码器提供上下文，解码器结合交叉注意力自回归生成", PURPLE)
    flow("pretrain_alignment.svg", "从预训练到对齐", [("预训练", "海量无标注文本"), ("SFT", "监督微调"), ("RM", "奖励模型"), ("PPO / RLHF", "强化学习对齐"), ("对话助手", "ChatGPT")], "预训练学习世界知识，对齐阶段学习如何帮助用户", ORANGE, 1400, 540)
    body = t(600, 100, "新增一个特殊 Token，需要同时扩展两张矩阵", 17, MUTED, 500)
    body += box(470, 150, 260, 82, "新增 Special Token", "词表 V → V + 1", BLUE_BG, BLUE)
    body += box(120, 350, 350, 110, "Embedding Wₑ", "新增 1 行：R^(V+1) × d", ORANGE_BG, ORANGE)
    body += box(730, 350, 350, 110, "LM Head W_h", "新增 1 列：R^d × (V+1)", GREEN_BG, GREEN)
    body += arrow(560, 232, 300, 350, ORANGE) + arrow(640, 232, 900, 350, GREEN)
    body += t(600, 535, "否则模型能读入新 Token，却无法为它输出概率", 18, INK, 600)
    save("special_token_surgery.svg", "特殊 Token 的模型手术", body, 1200, 610, "新增特殊 Token 时同步扩展 Embedding 与 LM Head")
    flow("tokenizer_comparison.svg", "Tokenizer 路线对比", [("原始文本", "str"), ("UTF-8 / Unicode", "底层编码"), ("BPE 合并", "学习高频片段"), ("Token IDs", "模型输入")], "TikToken / minBPE 与 SentencePiece 的共同目标是稳定编码文本", BLUE)
    flow("vocab_tradeoff.svg", "词表大小的双刃剑", [("V 过小", "序列长 · O(N²) 贵"), ("平衡点", "V ≈ 32k–100k"), ("V 过大", "矩阵大 · Softmax 慢")], "词表大小是序列长度与模型容量之间的权衡", GREEN)
    flow("untrained_token_flow.svg", "未训练 Token 的异常路径", [("Tokenizer 训练集", "包含用户名"), ("专属 Token", "一个独立 ID"), ("LLM 训练集", "剔除数据"), ("随机向量", "从未更新"), ("OOD 异常", "推理时崩溃")], "词表中存在，不代表 Token 在模型训练中真正被学过", RED, 1400, 540)
    body = box(110, 155, 220, 76, "a", "2.0", BLUE_BG, BLUE) + box(110, 300, 220, 76, "b", "−3.0", BLUE_BG, BLUE) + box(450, 225, 220, 76, "e = a × b", "−6.0", ORANGE_BG, ORANGE) + box(760, 225, 220, 76, "L", "最终输出 −8", GREEN_BG, GREEN)
    body += arrow(330, 190, 450, 255) + arrow(330, 335, 450, 275) + arrow(670, 263, 760, 263)
    body += t(600, 100, "每个节点保存自己的值，并在反向传播时累加梯度", 17, MUTED, 500)
    save("micrograd_computation_graph.svg", "Micrograd 计算图", body, 1200, 460, "输入节点经过运算得到最终损失")
    body = box(120, 180, 220, 86, "输入层", "3 inputs", BLUE_BG, BLUE) + box(490, 180, 220, 86, "隐藏层", "4 neurons", ORANGE_BG, ORANGE) + box(860, 180, 220, 86, "输出层", "1 neuron", GREEN_BG, GREEN)
    for y in [210, 235, 260]:
        body += arrow(340, y, 490, 210, BLUE, width=1)
    for y in [210, 235, 260, 285]:
        body += arrow(710, y, 860, 223, ORANGE, width=2)
    body += t(600, 390, "MLP 只是把神经元按层连接起来", 18, INK, 600)
    save("micrograd_mlp_architecture.svg", "多层感知机 MLP", body, 1200, 460, "输入层、隐藏层和输出层的全连接结构")
    body = t(600, 105, "链式法则：dz/dx = (dz/dy) × (dy/dx)", 22, ORANGE, 700) + box(160, 220, 180, 76, "x", "输入", BLUE_BG, BLUE) + box(510, 220, 180, 76, "y", "中间变量", ORANGE_BG, ORANGE) + box(860, 220, 180, 76, "z", "输出", GREEN_BG, GREEN)
    body += arrow(340, 258, 510, 258) + arrow(690, 258, 860, 258) + arrow(950, 350, 600, 350, ORANGE) + arrow(470, 350, 300, 350, ORANGE)
    body += t(600, 455, "前向从左到右；反向沿相反方向传递局部梯度", 18, INK, 600)
    save("micrograd_chain_rule.svg", "链式法则与反向传播", body, 1200, 520, "前向计算和反向传播的方向相反")


if __name__ == "__main__":
    main()
