from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def train_tokenizer(files, vocab_size: int, out_path: str):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
        show_progress=False,
    )
    tokenizer.train(files, trainer)
    tokenizer.save(out_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode_tokens(tokenizer, ids) -> str:
    return tokenizer.decode(ids)
