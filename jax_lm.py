from typing import Optional
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange

jax.config.update('jax_default_matmul_precision', 'tensorfloat32')  # tensorfloat32


@dataclass
class Config: vocab_size: int = 50257; max_seq_len: int = 1024; d_model: int = 1024; n_layers: int = 24; n_heads: int = 16; scan: bool = False

def create_mask(t):
    mask = jnp.tril(jnp.ones((t, t), dtype=jnp.bool))
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
            attn = jnp.where(mask, attn, 0.7 * jnp.finfo(attn.dtype).min)
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

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6
    
    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.dim,))
        
    def __call__(self, x):
        x_in = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_in * x_in, axis=-1, keepdims=True) + self.eps)
        out = x / rms * self.weight
        return out.astype(x.dtype)

class Block(nn.Module):
    config: Config
    scanning: bool = False
    
    def setup(self):
        self.ln1 = RMSNorm(self.config.d_model)
        self.ln2 = RMSNorm(self.config.d_model)
        self.attn = Attention(self.config)
        self.ff = FeedForward(self.config)
        
    def __call__(self, x, mask=None):
        if self.scanning:
            x, mask = x
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        if self.scanning:
            return x, mask
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
        if self.config.scan:
            # (remat_scan only applies remat if len(lengths) > 1)
            self.blocks = nn.remat_scan(Block, lengths=(self.config.n_layers,))(self.config, True)
        else:
            self.blocks = [Block(self.config) for _ in range(self.config.n_layers)]
        self.ln_f = RMSNorm(self.config.d_model)
        self.head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
        )

        self.scanning = self.config.scan

    def __call__(self, idx):
        b, t = idx.shape
        mask = create_mask(t)
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        if self.scanning:
            x = self.blocks((x, mask))[0]
        else:
            for block in self.blocks:
                x = block(x, mask)
        return self.head(self.ln_f(x))

def benchmark_jax(batch_size=128, seq_len=512, n_layers=24, n_heads=16, d_model=1024, steps=10, warmup=3, dtype=jnp.bfloat16, scan=False):
    key = jax.random.PRNGKey(0)
    config = Config(n_layers=n_layers, n_heads=n_heads, d_model=d_model, max_seq_len=seq_len, scan=scan)
    model = LM(config)

    @jax.jit
    def model_step(variables, x):
        return model.apply(variables, x)
    
    x = jax.random.randint(key, (batch_size, seq_len), 0, config.vocab_size)
    variables = model.init(key, x)
    variables = jax.tree.map(lambda x: x.astype(dtype) if x.dtype == jnp.float32 else x, variables)
    
    n_params = sum(x.size for x in jax.tree.leaves(variables))
    print(f"Starting trial with {n_params/1e6:.1f}M parameter model...")

    for _ in range(warmup):
        model_step(variables, x).block_until_ready()
    
    start = time.time()
    for _ in range(steps):
        out = model_step(variables, x)
    out = jax.block_until_ready(out)
    end = time.time()
    
    total_time_ms = (end - start) * 1000
    return total_time_ms / steps

if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices available: {jax.devices()}\n")
    
    batch_sizes = [64, 128, 256]

    print("Testing without scan:\n")
    for bs in batch_sizes:
        try:
            ms = benchmark_jax(
                batch_size=bs,
                seq_len=1024,
                n_layers=24,
                n_heads=16,
                d_model=1024,
                steps=10,
                warmup=3,
                scan=False
            )
            print(f"Batch size {bs}: {ms:.2f} ms per step")
        except Exception as e:
            print(f"Error with batch size {bs}: {str(e)}")

    print("\nTesting with scan:\n")
    for bs in batch_sizes:
        try:
            ms = benchmark_jax(
                batch_size=bs,
                seq_len=1024,
                n_layers=24,
                n_heads=16,
                d_model=1024,
                steps=10,
                warmup=3,
                scan=True
            )
            print(f"Batch size {bs}: {ms:.2f} ms per step")
        except Exception as e:
            print(f"Error with batch size {bs}: {str(e)}")
