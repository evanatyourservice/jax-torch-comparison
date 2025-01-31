from typing import Optional
import jax, jax.numpy as jnp
from flax import linen as nn
import time
from einops import rearrange
from dataclasses import dataclass

@dataclass
class Config: vocab_size: int = 50257; max_seq_len: int = 1024; d_model: int = 768; n_layers: int = 12; n_heads: int = 12

def create_mask(t):
    mask = jnp.tril(jnp.ones((t, t), dtype=jnp.float32))
    return mask.reshape(1, 1, t, t)

class Attention(nn.Module):
    config: Config
    
    def setup(self):
        self.n_heads, self.d_model = self.config.n_heads, self.config.d_model
        self.d_head = self.d_model // self.n_heads
        self.qkv = nn.Dense(3 * self.config.d_model, use_bias=False, kernel_init=nn.initializers.normal(0.02))
        self.out = nn.Dense(self.config.d_model, use_bias=False, kernel_init=nn.initializers.normal(0.02))
        
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        qkv = self.qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))
        attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(self.d_head)
        if mask is not None:
            attn = jnp.where(mask == 0, float('-inf'), attn)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.matmul(attn, v)
        return self.out(rearrange(out, 'b h n d -> b n (h d)'))

class FeedForward(nn.Module):
    config: Config
    
    def setup(self):
        self.fc1 = nn.Dense(4 * self.config.d_model, use_bias=False, kernel_init=nn.initializers.normal(0.02))
        self.fc2 = nn.Dense(self.config.d_model, use_bias=False, kernel_init=nn.initializers.normal(0.02))
        
    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))

class Block(nn.Module):
    config: Config
    
    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.attn = Attention(self.config)
        self.ff = FeedForward(self.config)
        
    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class LM(nn.Module):
    config: Config
    
    def setup(self):
        self.tok_emb = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=nn.initializers.normal(0.02),
        )
        self.pos_emb = self.param('pos_emb', nn.initializers.zeros, (1, self.config.max_seq_len, self.config.d_model))
        self.blocks = [Block(self.config) for _ in range(self.config.n_layers)]
        self.ln_f = nn.LayerNorm()
        self.head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
        )
        
    def __call__(self, idx):
        b, t = idx.shape
        mask = create_mask(t)
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))
@jax.jit
def model_step(params, x):
    return model.apply({'params': params}, x)

def benchmark_jax(batch_size=32, seq_len=512, n_layers=12, n_heads=12, d_model=768, steps=100, warmup=3, dtype=jnp.float32):
    key = jax.random.PRNGKey(0)
    config = Config(n_layers=n_layers, n_heads=n_heads, d_model=d_model, max_seq_len=seq_len)
    global model
    model = LM(config)
    
    x = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    params = model.init(key, x)
    
    print(f"JAX devices: {jax.devices()}")
    x = jax.device_put(x, jax.devices()[0])
    
    for _ in range(warmup):
        model_step(params, x).block_until_ready()
    
    start = time.time()
    for _ in range(steps):
        model_step(params, x).block_until_ready()
    end = time.time()
    
    total_time_ms = (end - start) * 1000
    return total_time_ms / steps

if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices available: {jax.devices()}\n")
    trials = 5
    times = []
    for i in range(trials):
        ms = benchmark_jax(
            batch_size=32,
            seq_len=512,
            steps=50,
            warmup=3
        )
        times.append(ms)
        print(f"Trial {i+1}: {ms:.2f} ms per step")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage: {avg_time:.2f} ms per step")
