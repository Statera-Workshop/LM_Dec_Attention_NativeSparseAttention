# LM_Dec_Attention_NativeSparseAttention

NativeSparseAttention - Attention family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Smoke-Tested**
- Failure class: Hardware Infeasible: pure-PyTorch path on Windows (no Triton / flex attention); 23M tokens in 4 h

## Provenance

- Source repository: (new) lucidrains/native-sparse-attention-pytorch 0.2.3
- Upstream / license: lucidrains/native-sparse-attention-pytorch (MIT), pip 0.2.3

## Differences from the source ([Unified] protocol)

- sliding window 64, compress block 16, select 4 blocks of 16, kv heads 4; pure PyTorch path (flex attention and Triton kernels are not available on Windows)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 31.09 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.798 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.951 / test 1.928 bits per byte (time-capped)
- Peak GPU memory at length 2048 (batch 1): train 5617 MB, inference 322 MB
- Peak GPU memory at length 8192 (batch 1): train OOM, inference 1218 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.02 (keys removed 0.02); 4 pairs @ 2048: 0.02 (keys removed 0.02); 16 pairs @ 512: 0.02 (keys removed 0.02); 16 pairs @ 2048: 0.01 (keys removed 0.02)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy native-sparse-attention-pytorch==0.2.3
python run.py check nsa
python run.py lm02 nsa      # enwik8 is downloaded to data/ on first use
```
