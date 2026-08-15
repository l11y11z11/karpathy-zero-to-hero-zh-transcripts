#!/usr/bin/env python3
"""Replace selected ASCII/Mermaid teaching blocks with project-local SVG links."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    "transcripts_zh/01_micrograd_zh.md": {
        17: ("micrograd_modules.svg", "Micrograd 的两层结构"),
        127: ("micrograd_parabola.svg", "梯度与局部斜率"),
        255: ("micrograd_computation_graph.svg", "Micrograd 计算图"),
        363: ("micrograd_chain_rule.svg", "链式法则与反向传播"),
        464: ("micrograd_neuron_forward.svg", "神经元的前向计算"),
        488: ("micrograd_tanh.svg", "tanh 激活函数与梯度"),
        634: ("micrograd_topological_order.svg", "计算图的拓扑顺序"),
        864: ("micrograd_mlp_architecture.svg", "多层感知机 MLP"),
        1040: ("micrograd_loss_curve.svg", "损失函数下降"),
    },
    "transcripts_zh/02_makemore1_zh.md": {
        351: ("bigram_pipeline.svg", "Bigram 语言模型流程"),
    },
    "transcripts_zh/03_makemore2_zh.md": {
        47: ("mlp_pipeline.svg", "MLP 语言模型数据流"),
    },
    "transcripts_zh/04_makemore3_zh.md": {
        20: ("batchnorm_loss_curve.svg", "BatchNorm 前后的损失曲线"),
        114: ("batchnorm_tanh.svg", "BatchNorm 把激活值拉回健康区"),
        294: ("batchnorm_pipeline.svg", "BatchNorm 前向流程"),
        481: ("batchnorm_activation_health.svg", "激活值健康度"),
        494: ("batchnorm_gradient_health.svg", "梯度流健康度"),
        533: ("batchnorm_diagnostics.svg", "网络健康度诊断"),
    },
    "transcripts_zh/06_makemore5_zh.md": {
        149: ("wavenet_tree.svg", "WaveNet 层次化感受野"),
        346: ("causal_mask.svg", "Transformer 因果掩码"),
        359: ("wavenet_context_reuse.svg", "WaveNet 滑动窗口与特征复用"),
    },
    "transcripts_zh/07_nanoGPT_zh.md": {
        355: ("qkv_roles.svg", "自注意力中的 Q K V"),
        522: ("residual_block.svg", "Transformer 残差块"),
        654: ("transformer_loss_progression.svg", "Transformer 验证损失进步"),
        686: ("encoder_decoder.svg", "Encoder–Decoder 结构"),
        714: ("pretrain_alignment.svg", "从预训练到对齐"),
    },
    "transcripts_zh/08_minbpe_zh.md": {
        406: ("special_token_surgery.svg", "特殊 Token 的模型手术"),
        468: ("tokenizer_comparison.svg", "Tokenizer 路线对比"),
        538: ("vocab_tradeoff.svg", "词表大小的双刃剑"),
        645: ("untrained_token_flow.svg", "未训练 Token 的异常路径"),
    },
}


def replace_file(relative, replacements):
    path = ROOT / relative
    source = path.read_text(encoding="utf-8").splitlines(keepends=True)
    result = []
    line_no = 1
    replaced = 0
    while line_no <= len(source):
        if line_no in replacements and source[line_no - 1].startswith("```"):
            asset, alt = replacements[line_no]
            end = line_no
            while end < len(source) and not (end > line_no and source[end].startswith("```")):
                end += 1
            if end >= len(source):
                raise RuntimeError(f"unterminated code block at {relative}:{line_no}")
            result.append(f"![{alt}](../assets/diagrams/{asset})\n")
            line_no = end + 2
            replaced += 1
            continue
        result.append(source[line_no - 1])
        line_no += 1
    path.write_text("".join(result), encoding="utf-8")
    return replaced


def main():
    total = 0
    for relative, replacements in REPLACEMENTS.items():
        total += replace_file(relative, replacements)
    print(f"replaced {total} teaching diagram blocks")


if __name__ == "__main__":
    main()
