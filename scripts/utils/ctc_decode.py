"""
Centralized CTC Decoding — Greedy and Beam Search + LM
=======================================================
USE THIS for all evaluation scripts. Replaces the duplicated
ctc_greedy_decode() found across 7+ scripts.

Functions:
  1. ctc_greedy_decode(logits, idx_to_char) -> str
  2. build_ctc_decoder(vocab_path, lm_path, alpha, beta) -> BeamSearchDecoderCTC
  3. ctc_beam_decode(logits, decoder, beam_width) -> str
"""

import json
import torch
import numpy as np


def ctc_greedy_decode(logits, idx_to_char, blank_idx=0):
    """
    Greedy CTC decode from logits (T, vocab_size).
    Returns decoded text string.
    """
    indices = torch.argmax(logits, dim=-1)  # (T,)
    collapsed = torch.unique_consecutive(indices)
    chars = []
    for idx in collapsed:
        idx = idx.item()
        if idx == blank_idx:
            continue
        token = idx_to_char.get(idx, "")
        if token == "<space>":
            chars.append(" ")
        elif token in ("<blank>", "<unk>"):
            continue
        else:
            chars.append(token)
    return "".join(chars).strip()


def build_ctc_decoder(vocab_path, lm_path=None, alpha=0.5, beta=1.0):
    """
    Build a pyctcdecode BeamSearchDecoderCTC.

    Args:
        vocab_path: path to vocab.json (85 tokens)
        lm_path: path to KenLM .arpa or .bin file (None = no LM)
        alpha: LM weight (higher = trust LM more)
        beta: word insertion bonus (higher = prefer more words)

    Returns:
        BeamSearchDecoderCTC decoder object
    """
    from pyctcdecode import build_ctcdecoder

    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_idx = json.load(f)

    # Build labels list ordered by index
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    labels = []
    for i in range(len(idx_to_char)):
        token = idx_to_char[i]
        if token == "<blank>":
            labels.append("")  # pyctcdecode blank
        elif token == "<space>":
            labels.append(" ")  # word boundary for LM
        elif token == "<unk>":
            labels.append("⁇")  # pyctcdecode UNK
        else:
            labels.append(token)

    return build_ctcdecoder(
        labels=labels,
        kenlm_model_path=lm_path,
        alpha=alpha,
        beta=beta,
    )


def ctc_beam_decode(logits, decoder, beam_width=100):
    """
    Beam search CTC decode with optional LM.

    Args:
        logits: torch.Tensor of shape (T, vocab_size) — raw logits (NOT softmax)
        decoder: BeamSearchDecoderCTC from build_ctc_decoder()
        beam_width: beam size (default 100)

    Returns:
        str: decoded text
    """
    # pyctcdecode expects numpy array of log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
    text = decoder.decode(log_probs, beam_width=beam_width)
    return text.strip()
