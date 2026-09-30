from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    n_layer: int = 5
    n_head: int = 6
    n_embd: int = 384
    head_dim: int = 64
    block_size: int = 256
    rope_base: float = 10000.0
    dropout: float = 0.0


@dataclass
class TrainConfig:
    seed: int = 1337
    micro_batch: int = 32
    grad_accum: int = 2
    max_steps: int = 5000
    eval_interval: int = 250
    log_interval: int = 20
    eval_batches: int = 100
    lr: float = 3e-4
    lr_min: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    data_dir: str = "torch-version/data"
    out_dir: str = "out"
    tokenizer_path: str = "torch-version/data/tokenizer.json"
    train_bin: str = "torch-version/data/train.bin"
    val_bin: str = "torch-version/data/val.bin"
    dtype: str = "bf16"
    devices: int = 1

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum * ModelConfig.block_size
