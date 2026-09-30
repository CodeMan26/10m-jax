import os

import numpy as np

from tokenizer import load_tokenizer


def tokenize_to_bin(txt_path: str, bin_path: str, tokenizer, batch: int = 10_000):
    wrote = 0
    with open(txt_path, "r") as f, open(bin_path, "wb") as out:
        while True:
            lines = [f.readline().strip() for _ in range(batch)]
            lines = [line for line in lines if line]
            if not lines:
                break
            encs = tokenizer.encode_batch(lines)
            arr = np.concatenate([np.asarray(e.ids, dtype=np.uint16) for e in encs])
            arr.tofile(out)
            wrote += arr.size
            print(f"  tokenized {wrote:,} tokens ({wrote / 1e6:.1f}M)", flush=True)
    print(f"  done: {bin_path} {wrote:,} tokens ({wrote * 2 / 1e6:.1f} MB)", flush=True)


class BinDataset:
    def __init__(self, bin_path: str, block_size: int):
        self._bin = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_tokens = int(self._bin.size)
        self.block_size = block_size
        assert self.n_tokens > block_size

    def __len__(self):
        return self.n_tokens // self.block_size - 1

    def __getitem__(self, i):
        i *= self.block_size
        x = self._bin[i : i + self.block_size].astype(np.int32)
        y = self._bin[i + 1 : i + 1 + self.block_size].astype(np.int32)
        return x, y

    def sample_batch(self, rng, batch_size):
        ixs = rng.integers(0, len(self), size=batch_size)
        xs, ys = zip(*(self[i] for i in ixs))
        return np.stack(xs), np.stack(ys)


class BatchSampler:
    def __init__(self, ds: BinDataset, micro_batch: int, seed: int = 1337):
        self.ds = ds
        self.micro_batch = micro_batch
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return self.ds.sample_batch(self.rng, self.micro_batch)


def count_data(data_dir: str):
    for name in ("train.bin", "val.bin"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            n = np.memmap(path, dtype=np.uint16, mode="r").size
            print(f"{name}: {n:,} tokens ({n // (1 << 20):,}M)", flush=True)
