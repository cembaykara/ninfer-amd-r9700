# Qwen3.8-27B R9700 Vision parity tool

The diagnostic captures 471 represented boundaries per media item: preprocessing, patch projection, all 27 Vision blocks and the merger. It compares HIP with the artifact-native implementation reference, reruns operations from captured inputs to isolate local error, checks MRoPE metadata, and compares 333 Vision tensors and native tower outputs directly with the source BF16 model. Each local Linear oracle evaluates the complete dot in FP64 from represented BF16 inputs and exact BF16 weights or stored signed codes times scales; it does not reproduce private activation quantization or round decoded weights to BF16. The local attention oracle evaluates segmented softmax(Q Kᵀ / √D)V in FP64, retaining FP64 output. CPU schedule references use the same mathematical attention followed by their explicit BF16 boundary; library BF16 SDPA remains a supplementary implementation comparison through the source Hugging Face final check. Use the existing AMD artifact binder; NVFP4 artifacts are unsupported.

Run with an explicitly selected artifact and existing Python 3.11 reference environment:

```bash
cmake --build build-r9700 --target ninfer_qwen3_8_27b_vision_trace
source .ninfer-local.env
flock --exclusive "$NINFER_GPU_LOCK" "$NINFER_PYTHON" -m tools.parity.qwen3_8_27b.vision \
  --weights /absolute/path/to/selected-r9700.ninfer \
  --model-dir /path/to/models/qwen3.8-27b-bf16 \
  --messages examples/cli/messages/image_chart.json \
  --trace-exe build-r9700/tests/ninfer_qwen3_8_27b_vision_trace \
  --device cpu --output /absolute/path/to/new-vision-report.json
```

The selected-artifact command planner in `tools/bench/prepare_selected_vision_diagnostic.py` requires this tracer in the selected build's `tests/` directory. Its schema-2 plan records and validates the executable, artifact, source receipt, prepared input and diagnostic implementation. The completion validator requires all 471 captures in each of the three comparison profiles and reevaluates their numerical gates from the metrics; historical four-capture reports are not accepted or reinterpreted.

The native tracer requires sole R9700 ownership; `--device` selects only the Python reference device. The existing environment must provide torch, safetensors and Transformers; nothing is downloaded.

`--trace-dir` consumes an existing manifest; otherwise native traces are temporary. `--prepared-input` reuses a previously frozen single-image reference input, still checking its metadata against the native trace. The tracer checks exact traced/untraced output and aggregate and per-item agreement.

The version-3 report names the direct native/source comparison `production_vs_source_bf16`. Final embeddings must have relative L2 error at most 5% and cosine at least 0.999 against the BF16 source. This engineering admission target supplements the unchanged source-weight row/column/group guards and per-Op checks; it does not establish visual task accuracy by itself. The prior 25%/0.97 final gate and the artifact-reference/source substitution are superseded. A failed diagnostic remains a failed diagnostic even when final embeddings meet this target.

Reports preserve source provenance, finite shapes, whole-tensor, worst-token, worst-feature and stored-group metrics with explicit local and accumulated criteria. Reports never overwrite existing files. These diagnostics do not replace independent Op or model-quality qualification.

Metric fault tests check that dropped tokens, features and quantization groups cannot hide in aggregate averages: `python3.11 -m pytest tools/parity/qwen3_8_27b/test_vision.py tools/parity/qwen3_8_27b/test_vision_parity_metrics.py`.
