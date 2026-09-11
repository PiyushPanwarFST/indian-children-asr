# 01: MSE with Raw Whisper Decoder (Semantic Branch)

**Status:** Not done yet - discuss with professor

**Idea:** Use the semantic MSE-trained encoder (from step1) with Whisper's original decoder (beam search) to generate transcriptions. This would be the semantic equivalent of the acoustic branch's first experiment (167.10% WER).

**Why it may not make sense:** The semantic encoder was trained to match IndicConformer's logit space (CTC-based), while Whisper's decoder expects attention-based encoder features. The mismatch might produce meaningless output.

**Decision:** Ask professor whether to include this for comparison.
