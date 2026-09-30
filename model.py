import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    n_layer: int = 5
    n_embd: int = 384
    n_head: int = 6
    head_dim: int = 64
    block_size: int = 256
    rope_base: float = 10000.0
    dropout: float = 0.0


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


class RotaryEmbedding:
    def __init__(self, head_dim, block_size, base=10000.0):
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
        positions = jnp.arange(block_size)
        freqs = jnp.outer(positions, inv_freq)
        self.cos = jnp.concatenate((jnp.cos(freqs), jnp.cos(freqs)), axis=-1)
        self.sin = jnp.concatenate((jnp.sin(freqs), jnp.sin(freqs)), axis=-1)

    def __call__(self, x):
        sequence_length = x.shape[2]
        cos = self.cos[:sequence_length][None, None, :, :]
        sin = self.sin[:sequence_length][None, None, :, :]
        x1, x2 = jnp.split(x, 2, axis=-1)
        return x * cos + jnp.concatenate((-x2, x1), axis=-1) * sin


class MLP:
    def __init__(self, cfg, key):
        key_fc1, key_fc2 = jax.random.split(key)
        self.fc1 = jax.random.normal(key_fc1, (cfg.n_embd, 4 * cfg.n_embd)) * 0.02
        self.fc2 = jax.random.normal(key_fc2, (4 * cfg.n_embd, cfg.n_embd)) * 0.02

    def __call__(self, x):
        return jax.nn.gelu(x @ self.fc1) @ self.fc2


class CausalSelfAttention:
    def __init__(self, cfg, key):
        if cfg.n_embd != cfg.n_head * cfg.head_dim:
            raise ValueError("n_embd must equal n_head * head_dim")
        key_qkv, key_out = jax.random.split(key)
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = jax.random.normal(
            key_qkv, (cfg.n_embd, 3 * cfg.n_head * cfg.head_dim)
        ) * 0.02
        self.out_proj = jax.random.normal(
            key_out, (cfg.n_head * cfg.head_dim, cfg.n_embd)
        ) * 0.02
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.block_size, cfg.rope_base)
        self.attn_scale = 1.0 / math.sqrt(cfg.head_dim)

    def __call__(self, x):
        batch, sequence_length, channels = x.shape
        q, k, v = jnp.split(x @ self.qkv, 3, axis=-1)
        shape = (batch, sequence_length, self.n_head, self.head_dim)
        q = self.rope(jnp.transpose(q.reshape(shape), (0, 2, 1, 3)))
        k = self.rope(jnp.transpose(k.reshape(shape), (0, 2, 1, 3)))
        v = jnp.transpose(v.reshape(shape), (0, 2, 1, 3))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.attn_scale
        causal = jnp.tril(jnp.ones((sequence_length, sequence_length), dtype=bool))
        scores = jnp.where(causal[None, None, :, :], scores, -jnp.inf)
        weights = jax.nn.softmax(scores, axis=-1)
        y = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(batch, sequence_length, channels)
        return y @ self.out_proj


class Block:
    def __init__(self, cfg, key):
        key_attn, key_mlp = jax.random.split(key)
        self.ln1_gamma = jnp.ones((cfg.n_embd,))
        self.ln1_beta = jnp.zeros((cfg.n_embd,))
        self.ln2_gamma = jnp.ones((cfg.n_embd,))
        self.ln2_beta = jnp.zeros((cfg.n_embd,))
        self.attn = CausalSelfAttention(cfg, key_attn)
        self.mlp = MLP(cfg, key_mlp)

    def __call__(self, x):
        x = x + self.attn(layer_norm(x, self.ln1_gamma, self.ln1_beta))
        x = x + self.mlp(layer_norm(x, self.ln2_gamma, self.ln2_beta))
        return x


class GPT:
    def __init__(self, cfg=None, key=None):
        self.cfg = ModelConfig() if cfg is None else cfg
        key = jax.random.key(1337) if key is None else key
        keys = jax.random.split(key, self.cfg.n_layer + 2)
        self.tok_emb = jax.random.normal(
            keys[0], (self.cfg.vocab_size, self.cfg.n_embd)
        ) * 0.02
        self.blocks = [Block(self.cfg, block_key) for block_key in keys[1:-1]]
        self.ln_f_gamma = jnp.ones((self.cfg.n_embd,))
        self.ln_f_beta = jnp.zeros((self.cfg.n_embd,))

    def forward(self, idx, targets=None):
        x = self.tok_emb[idx]
        for block in self.blocks:
            x = block(x)
        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        logits = x @ self.tok_emb.T
        loss = None
        if targets is not None:
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            token_loss = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
            valid = targets != -1
            loss = jnp.sum(jnp.where(valid, token_loss, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, key=None):
        key = jax.random.key(0) if key is None else key
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self.forward(idx_cond)
            key, sample_key = jax.random.split(key)
            next_token = jax.random.categorical(
                sample_key, logits[:, -1, :] / max(temperature, 1e-8), axis=-1
            )[:, None]
            idx = jnp.concatenate((idx, next_token), axis=1)
        return idx


def _gpt_flatten(model):
    children = [model.tok_emb]
    for block in model.blocks:
        children.extend([
            block.ln1_gamma,
            block.ln1_beta,
            block.ln2_gamma,
            block.ln2_beta,
            block.attn.qkv,
            block.attn.out_proj,
            block.mlp.fc1,
            block.mlp.fc2,
        ])
    children.extend([model.ln_f_gamma, model.ln_f_beta])
    return tuple(children), model.cfg


def _gpt_unflatten(cfg, children):
    model = GPT.__new__(GPT)
    model.cfg = cfg
    index = 0
    model.tok_emb = children[index]
    index += 1
    model.blocks = []
    for _ in range(cfg.n_layer):
        block = Block.__new__(Block)
        block.ln1_gamma = children[index]
        block.ln1_beta = children[index + 1]
        block.ln2_gamma = children[index + 2]
        block.ln2_beta = children[index + 3]
        index += 4
        attention = CausalSelfAttention.__new__(CausalSelfAttention)
        attention.n_head = cfg.n_head
        attention.head_dim = cfg.head_dim
        attention.qkv = children[index]
        attention.out_proj = children[index + 1]
        attention.rope = RotaryEmbedding(cfg.head_dim, cfg.block_size, cfg.rope_base)
        attention.attn_scale = 1.0 / math.sqrt(cfg.head_dim)
        index += 2
        block.attn = attention
        mlp = MLP.__new__(MLP)
        mlp.fc1 = children[index]
        mlp.fc2 = children[index + 1]
        index += 2
        block.mlp = mlp
        model.blocks.append(block)
    model.ln_f_gamma = children[index]
    model.ln_f_beta = children[index + 1]
    return model


jax.tree_util.register_pytree_node(GPT, _gpt_flatten, _gpt_unflatten)

