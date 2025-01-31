import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from einops import rearrange

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class Config: vocab_size: int = 50257; max_seq_len: int = 1024; d_model: int = 768; n_layers: int = 12; n_heads: int = 12

def create_mask(t, device):
    mask = torch.tril(torch.ones((t, t), dtype=torch.float32))
    return mask.view(1, 1, t, t).to(device=device)

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads, self.d_model = config.n_heads, config.d_model
        self.d_head = self.d_model // self.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v)
        return self.out(rearrange(out, 'b h n d -> b n (h d)'))

class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ff = FeedForward(config)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        
    def forward(self, idx):
        b, t = idx.size()
        mask = create_mask(t, idx.device)
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))

def benchmark_torch(batch_size=128, seq_len=512, n_layers=12, n_heads=12, d_model=768, steps=30, warmup=1, device='cuda', dtype=torch.float32):
    assert torch.cuda.is_available(), "CUDA device required"
    device = 'cuda'

    torch.manual_seed(0)
    config = Config(n_layers=n_layers, n_heads=n_heads, d_model=d_model, max_seq_len=seq_len)
    model = LM(config).to(device=device, dtype=dtype)
    model = torch.compile(model)
    model.eval()
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        for _ in range(steps):
            model(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps

if __name__ == "__main__":
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"Using device: {torch.cuda.get_device_name(0)}\n")
    trials = 5
    times = []
    for i in range(trials):
        ms = benchmark_torch(
            batch_size=128,
            seq_len=512,
            steps=30,
            warmup=1
        )
        times.append(ms)
        print(f"Trial {i+1}: {ms:.2f} ms per step")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage: {avg_time:.2f} ms per step")
