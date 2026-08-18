import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数 (hyperparameters)
batch_size = 32 # 我们将并行处理多少个独立序列？
block_size = 8 # 预测的最大上下文长度是多少？
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 文本中出现的所有唯一字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 创建字符到整数的映射字典
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # 编码器：接收字符串，输出整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # 解码器：接收整数列表，输出字符串

# 训练集与验证集切分
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 前 90% 用于训练，其余用于验证
train_data = data[:n]
val_data = data[n:]

# 数据加载
def get_batch(split):
    # 生成一个小批量的输入数据 x 和目标数据 y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 极简 Bigram 语言模型
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # 每个词元直接从查找表中读取下一个词元的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置嵌入表
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        # idx 和 targets 都是形状为 (B, T) 的整数张量
        tok_emb = self.token_embedding_table(idx) # (B, T, C=嵌入维度)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T, C)
        x = tok_emb + pos_emb # 加上位置编码 [广播机制: (B,T,C) + (1,T,C)]
        logits = self.lm_head(x) # (B, T, 词表大小)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前上下文中形状为 (B, T) 的索引数组
        for _ in range(max_new_tokens):
            # 获取预测结果
            logits, loss = self(idx)
            # 仅关注最后一个时间步
            logits = logits[:, -1, :] # 形状变为 (B, C)
            # 应用 softmax 获取概率分布
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 将采样的索引追加到当前序列中
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# 创建 PyTorch 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # 定期评估训练集和验证集上的损失
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 采样一个批次的数据
    xb, yb = get_batch('train')

    # 计算损失
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 从模型中生成文本
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
