# Developer notes

Internal design notes, decision logs and work-in-progress guides. These are **not** part of the published documentation (Sphinx builds only `docs/source/`), and may describe unfinished work.

| File | Content |
|---|---|
| `gemma3_conversion.md` | Gemma3-270M → Mammoth: converter usage, architecture facts, decision log |
| `post_emb_norm_scope.md` | Per-language vs. per-xcoder post-embedding norm layout |
| `litert.md` | LiteRT conversion and Android / NPU deployment (in progress, not public yet) |
| `tflops_explained.md` | How the training log computes FLOPs / TFLOPS |
| `ROCTX_PROFILING.md` | ROCm profiling with roctx markers (outdated: still uses the removed `--enable_profiling` option) |
| `scalability/` | Multi-node scaling benchmark on LUMI (config, plot script, result) |
