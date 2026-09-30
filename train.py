import argparse
import math
import os
import pickle
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from config import ModelConfig, TrainConfig
from data import BatchSampler, BinDataset
from model import GPT
from tokenizer import decode_tokens, load_tokenizer


def get_lr(it, cfg: TrainConfig):
    if it < cfg.warmup_steps:
        return cfg.lr * (it + 1) / cfg.warmup_steps
    if it > cfg.max_steps:
        return cfg.lr_min
    progress = (it - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * progress))


def tree_norm(tree):
    return jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree)))


def init_optimizer(model):
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, model),
        "v": jax.tree_util.tree_map(jnp.zeros_like, model),
        "step": 0,
    }


def adamw_update(model, grads, state, lr, cfg):
    step = state["step"] + 1
    norm = tree_norm(grads)
    scale = jnp.minimum(1.0, cfg.grad_clip / (norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    m = jax.tree_util.tree_map(
        lambda old, grad: cfg.beta1 * old + (1 - cfg.beta1) * grad,
        state["m"],
        grads,
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: cfg.beta2 * old + (1 - cfg.beta2) * grad * grad,
        state["v"],
        grads,
    )
    mh = jax.tree_util.tree_map(lambda x: x / (1 - cfg.beta1**step), m)
    vh = jax.tree_util.tree_map(lambda x: x / (1 - cfg.beta2**step), v)

    def update(param, mean, variance):
        value = mean / (jnp.sqrt(variance) + 1e-8)
        if param.ndim >= 2:
            value = value + cfg.weight_decay * param
        return param - lr * value

    return (
        jax.tree_util.tree_map(update, model, mh, vh),
        {"m": m, "v": v, "step": step},
        norm,
    )


def evaluate(model, val_ds, cfg, loss_fn, rng):
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = val_ds.sample_batch(rng, 64)
        losses.append(float(loss_fn(model, jnp.asarray(x), jnp.asarray(y))))
    avg = float(np.mean(losses))
    return avg, math.exp(avg)


def save_checkpoint(path, model, optimizer, step, val_loss):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {"model": model, "optim": optimizer, "step": step, "val_loss": val_loss}, f
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="tinystories-gpt-10m-jax")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    tcfg = TrainConfig()
    mcfg = ModelConfig()
    if args.steps:
        tcfg.max_steps = args.steps
    if args.data_dir:
        tcfg.data_dir = args.data_dir
        tcfg.tokenizer_path = os.path.join(args.data_dir, "tokenizer.json")
        tcfg.train_bin = os.path.join(args.data_dir, "train.bin")
        tcfg.val_bin = os.path.join(args.data_dir, "val.bin")

    os.makedirs(tcfg.out_dir, exist_ok=True)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("wandb is not installed. Run: pip install wandb") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=args.wandb_name,
            config={
                "model": asdict(mcfg),
                "training": asdict(tcfg),
                "effective_batch_tokens": tcfg.effective_batch,
            },
            resume="allow" if args.resume else None,
        )

    rng = np.random.default_rng(tcfg.seed)
    key = jax.random.key(tcfg.seed)
    train_ds = BinDataset(tcfg.train_bin, mcfg.block_size)
    val_ds = BinDataset(tcfg.val_bin, mcfg.block_size)
    sampler = BatchSampler(train_ds, tcfg.micro_batch, tcfg.seed)
    model = GPT(mcfg, key)
    optimizer = init_optimizer(model)
    step = 0

    if args.resume and os.path.exists(args.resume):
        with open(args.resume, "rb") as f:
            checkpoint = pickle.load(f)
        model = checkpoint["model"]
        optimizer = checkpoint["optim"]
        step = checkpoint["step"]

    def loss_fn(current_model, x, y):
        _, loss = current_model.forward(x, y)
        return loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    eval_fn = jax.jit(loss_fn)
    best_val = float("inf")
    t0 = time.time()

    print(f"train tokens: {train_ds.n_tokens:,}  val tokens: {val_ds.n_tokens:,}")
    print("model params: JAX object model; parameter counting is not exposed")
    if wandb_run:
        wandb_run.config.update(
            {
                "train_tokens": train_ds.n_tokens,
                "val_tokens": val_ds.n_tokens,
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )

    for it in range(step, tcfg.max_steps):
        lr = get_lr(it, tcfg)
        if it % tcfg.eval_interval == 0 or it == tcfg.max_steps - 1:
            val_loss, val_ppl = evaluate(model, val_ds, tcfg, eval_fn, rng)
            print(
                f"step {it}: val loss {val_loss:.4f}  val ppl {val_ppl:.2f}", flush=True
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                    },
                    step=it,
                )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    os.path.join(tcfg.out_dir, "best.pkl"),
                    model,
                    optimizer,
                    it,
                    val_loss,
                )

        grads = None
        loss_value = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = next(sampler)
            loss, micro_grads = grad_fn(model, jnp.asarray(x), jnp.asarray(y))
            grads = (
                micro_grads
                if grads is None
                else jax.tree_util.tree_map(lambda a, b: a + b, grads, micro_grads)
            )
            loss_value += float(loss) / tcfg.grad_accum
        grads = jax.tree_util.tree_map(lambda x: x / tcfg.grad_accum, grads)
        model, optimizer, grad_norm = adamw_update(model, grads, optimizer, lr, tcfg)

        if it % tcfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens = tcfg.micro_batch * mcfg.block_size * tcfg.grad_accum
            throughput = tokens / max(dt, 1e-9)
            grad_norm_value = float(grad_norm)
            print(
                f"step {it}/{tcfg.max_steps} loss {loss_value:.4f} lr {lr:.2e} "
                f"gnorm {grad_norm_value:.2f} {throughput:.0f} tok/s",
                flush=True,
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": loss_value,
                        "train/learning_rate": lr,
                        "train/gradient_norm": grad_norm_value,
                        "train/tokens_per_second": throughput,
                    },
                    step=it,
                )

    val_loss, val_ppl = evaluate(model, val_ds, tcfg, eval_fn, rng)
    save_checkpoint(
        os.path.join(tcfg.out_dir, "final.pkl"),
        model,
        optimizer,
        tcfg.max_steps,
        val_loss,
    )
    print(
        f"done. final val loss {val_loss:.4f} ppl {val_ppl:.2f}  (best {best_val:.4f})"
    )
    if wandb_run:
        wandb_run.log(
            {
                "val/final_loss": val_loss,
                "val/final_perplexity": val_ppl,
                "val/best_loss": best_val,
            },
            step=tcfg.max_steps,
        )

    tok = load_tokenizer(tcfg.tokenizer_path)
    for prompt in (
        "Once upon a time, a little",
        "Tom and Lily were",
        "The cat was very",
    ):
        ids = jnp.asarray([tok.encode(prompt).ids], dtype=jnp.int32)
        key, sample_key = jax.random.split(key)
        output = model.generate(ids, 100, temperature=0.8, key=sample_key)
        sample = decode_tokens(tok, output[0].tolist())
        print("-", sample)
        if wandb_run:
            wandb_run.log(
                {f"samples/{prompt}": wandb.Html(f"<pre>{sample}</pre>")},
                step=tcfg.max_steps,
            )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
