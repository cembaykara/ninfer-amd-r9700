# v2v M-RoPE discrepancy and fix

**Date:** 2026-10-22
**Status:** planned. Everything else in the v2v path is verified against the reference; this is the last discrepancy.

## Context

Qwen-Image-2.1-Turbo image editing (v2v) runs: Qwen3-VL encoder (ViT + 36-layer
text stack) -> hidden states as DiT context -> 8-step denoise -> VAE. The
condition image is interleaved into the text stack as merged visual tokens.

Verified against the checkpoint on disk (models/hf/Qwen/Qwen-Image-2.1-Turbo)
and the diffusers/transformers reference, in order:

| Piece | Reference | Engine | Status |
|---|---|---|---|
| Image preprocessing | Qwen3VLProcessor: patch 16, merge 2, 65536-16777216 px, bicubic, (x/255-0.5)/0.5 | vision_preprocess.cpp | match (byte-exact replica tested) |
| ViT patch_embed | Conv3d(3, 1152, (2,16,16)) = [1152, 1536] | was [1152, 1152] + scrambled layout | fixed (073e826c) |
| Deepstack | ViT layers 8/16/24, separate mergers (postshuffle norm), injected at text layers 0/1/2 as plain add at visual positions | same | match |
| Prompt template | ti2i: system(14 tok) + user + <image1><\|vision_start\|>pads<\|vision_end\|>prompt (no newline) | was missing <image1>, had extra newline | fixed (073e826c) |
| Text-stack RoPE | 3-D M-RoPE, grid positions for visual tokens | 1-D sequential | **this report** |

## The discrepancy

The Qwen3-VL text model (text_encoder/config.json) is configured:

    rope_theta = 5000000
    rope_scaling = { mrope_interleaved: true, mrope_section: [24, 20, 20] }
    head_dim = 128

The reference (transformers modeling_qwen3_vl.py, Qwen3VLTextRotaryPosEmbed)
builds position_ids as [3, bs, seq] and applies per-frequency-pair phases:

- inv_freq[j] = theta^(-2j/128), j = 0..63
- axis(j): j < 60 -> j % 3 (0=T, 1=H, 2=W); j >= 60 -> T
  (i.e. T,H,W repeated 20 times, then T 4 times: 24/20/20 per mrope_section)
- cos/sin per pair: pos[axis(j)] * inv_freq[j]; final 128-dim vector is
  cat(freqs, freqs) (rotate-half layout)

Positions (get_rope_index + get_vision_position_ids):

- text token at sequence position p: (p, p, p)
- visual tokens: for a still image, grid (T=1, H=gh, W=gw) merged by
  spatial_merge_size=2 -> llm grid (1, gh/2, gw/2); merged token v =
  h_m * (gw/2) + w_m (row-major) gets (start, h_m + start, w_m + start),
  where start is the sequence position of the first <|image_pad|> token.
  The text position counter advances through the visual block, so tokens
  after the block continue at start + V.

The engine (encoder_forward_v2v, v2v_forward.hip) applies 1-D rope_neox with
sequential positions [0..seq) to every token. Text tokens are unaffected
((p,p,p) collapses to 1-D p exactly), but visual tokens get (p,p,p) instead
of (start, h_m+start, w_m+start): the 2-D spatial structure of the image is
lost in the encoder attention, and the hidden states the DiT consumes at the
visual positions do not carry coherent spatial content. For the 320x224 test
image (7x10 merged grid, 70 tokens) the first token matches the reference
and the rest drift, growing with grid depth.

## Why the existing MRoPE domain does not cover this

include/ninfer/ops/rope.h already registers a Text MRoPE mode, but for the
Qwen3.8 text model: head_dim=256, rotary_dim=64, axis = i%3 for all 64 pairs
(22/21/21 split). Qwen3-VL needs head_dim=rotary_dim=128, theta=5e6, and the
[24,20,20] interleaved split (24/20/20). Different geometry and different
axis schedule, so a new registered domain is required rather than reusing the
existing one.

## Fix

1. rope op: register a new domain - Text MRoPE D128/R128: positions I32 [T,3]
   (contiguous, token-major), head_dim=rotary_dim=128, theta=5e6; pair j uses
   axis (j<60 ? j%3 : 0) with frequency theta^(-2j/128); rotate-half layout
   as the existing modes.
2. Host (engine.cpp image_encode_v2v): build the position table [seq][3]:
   - i < visual_start or i >= visual_start+V: (i, i, i)
   - visual token v = i - visual_start: h_m = v / (gw/2), w_m = v % (gw/2);
     (visual_start, visual_start + h_m, visual_start + w_m)
   gw comes from the preprocess result (grid_w).
3. encoder_forward_v2v: replace the two rope_neox calls with the new MRoPE
   rope op over the [seq][3] table (ws.positions storage resized to [seq][3]).
4. Rebuild local/ninfer-r9700:image, re-run the add-glasses edit test.

## Verification

- Unit: engine M-RoPE cos/sin vs a Python reference (the transformers
  recomposition_frequencies formula) on a known [seq][3] table - bit-exact
  after BF16 rounding, per the rope op oracle convention.
- End-to-end: the standing test - shutterstock_648907024.jpg (woman, grey
   tee, white wall) with prompt "The same young woman, now wearing glasses,
   keeping her face, hair, pose, and heather grey t-shirt unchanged, plain
   white background, photorealistic portrait" at 512x512. Pass: identity
   preserved (face/hair/pose/tee) and glasses present.
