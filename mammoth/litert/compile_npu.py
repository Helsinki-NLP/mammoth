#!/usr/bin/env python3
"""
AOT-compile a Mammoth .tflite for Qualcomm (and optionally MediaTek) NPUs.

Requires Linux + ai-edge-litert + ai-edge-litert-sdk-qualcomm-nightly.
Does NOT run on macOS or Windows.

Usage:
    # All registered backends (Qualcomm + any other installed SDK):
    python mammoth/litert/compile_npu.py \\
        --tflite mammoth_eng_spa.tflite \\
        --output ./npu_output

    # Specific SoC only:
    python mammoth/litert/compile_npu.py \\
        --tflite mammoth_eng_spa.tflite \\
        --output ./npu_output \\
        --targets SM8650 SM8750

    # By vendor shorthand:
    python mammoth/litert/compile_npu.py \\
        --tflite mammoth_multi.tflite \\
        --output ./npu_output \\
        --targets Qualcomm

Qualcomm SoC map:
    SM8450  Snapdragon 8 Gen 1   (Galaxy S22, Pixel 7)
    SM8550  Snapdragon 8 Gen 2   (Galaxy S23)
    SM8650  Snapdragon 8 Gen 3   (Galaxy S24)
    SM8750  Snapdragon 8 Elite   (Galaxy S25, Pixel 9)

Output layout (example for SM8650 + SM8750):
    npu_output/
      mammoth_eng_spa_fallback.tflite          <- CPU/GPU fallback (always)
      mammoth_eng_spa_Qualcomm_SM8650.tflite
      mammoth_eng_spa_Qualcomm_SM8750.tflite
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Platform guard ────────────────────────────────────────────────────────────

if sys.platform != "linux":
    print(
        f"[ERROR] AOT NPU compilation requires Linux "
        f"(current platform: {sys.platform}).\n"
        "Run this script on LUMI or another Linux host."
    )
    sys.exit(1)

# ── Dependency imports (fail fast with a clear message) ───────────────────────

try:
    from ai_edge_litert.aot import aot_compile as aot
except ImportError:
    print(
        "[ERROR] ai-edge-litert is not installed.\n"
        "  pip install ai-edge-litert"
    )
    sys.exit(1)

try:
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
    _QNN_AVAILABLE = True
except ImportError:
    _QNN_AVAILABLE = False

try:
    from ai_edge_litert.aot.vendors.mediatek import target as mtk_target
    _MTK_AVAILABLE = True
except ImportError:
    _MTK_AVAILABLE = False

if not _QNN_AVAILABLE and not _MTK_AVAILABLE:
    print(
        "[ERROR] No vendor SDK installed. For Qualcomm:\n"
        "  pip install ai-edge-litert-sdk-qualcomm-nightly"
    )
    sys.exit(1)

# ── QNN library path setup ────────────────────────────────────────────────────

def _setup_qnn_lib_path() -> None:
    """Prepend the bundled x86_64 QNN libs to LD_LIBRARY_PATH if needed.

    ai_edge_litert_sdk_qualcomm ships libQnnSystem.so etc. under
    data/lib/x86_64-linux-clang/ but does NOT add that path to
    LD_LIBRARY_PATH, so dlopen() inside the compiler plugin fails unless
    we do it ourselves before the first aot_compile() call.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("ai_edge_litert_sdk_qualcomm")
        if spec is None:
            return
        pkg_dir = os.path.dirname(spec.origin)
        qnn_lib_dir = os.path.join(pkg_dir, "data", "lib", "x86_64-linux-clang")
        if not os.path.isdir(qnn_lib_dir):
            return
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if qnn_lib_dir not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (
                qnn_lib_dir + (":" + existing if existing else "")
            )
            print(f"  QNN libs prepended to LD_LIBRARY_PATH: {qnn_lib_dir}")
    except Exception as e:
        print(f"  [WARN] Could not auto-configure QNN lib path: {e}")


# ── Target resolution ─────────────────────────────────────────────────────────

_QNN_ALIASES = {"Qualcomm", "QNN", "qnn"}
_MTK_ALIASES = {"MediaTek", "MTK", "mtk"}


def resolve_targets(target_strs: list[str]):
    """Return a list of vendor Target objects, or None (= all registered)."""
    if not target_strs:
        return None  # aot_compile compiles for all registered backends

    targets = []
    for t in target_strs:
        if t in _QNN_ALIASES:
            if not _QNN_AVAILABLE:
                print(f"[WARN] Skipping {t!r}: Qualcomm SDK not installed.")
                continue
            targets.append(qnn_target.Target(qnn_target.SocModel.ALL))
        elif t in _MTK_ALIASES:
            if not _MTK_AVAILABLE:
                print(f"[WARN] Skipping {t!r}: MediaTek SDK not installed.")
                continue
            targets.append(mtk_target.Target(mtk_target.SocModel.ALL))
        elif _QNN_AVAILABLE and t in qnn_target.SocModel.__members__:
            targets.append(qnn_target.Target(qnn_target.SocModel[t]))
        elif _MTK_AVAILABLE and t in mtk_target.SocModel.__members__:
            targets.append(mtk_target.Target(mtk_target.SocModel[t]))
        else:
            known = []
            if _QNN_AVAILABLE:
                known += list(qnn_target.SocModel.__members__)
            if _MTK_AVAILABLE:
                known += list(mtk_target.SocModel.__members__)
            raise ValueError(
                f"Unknown target: {t!r}.  "
                f"Known SoC models: {', '.join(known)}"
            )

    if not targets:
        raise ValueError("No valid targets after filtering — nothing to compile.")
    return targets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AOT-compile a Mammoth .tflite for Qualcomm NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tflite", required=True,
                        help="Input .tflite file (from convert.py or convert_multi.py)")
    parser.add_argument("--output", default="./npu_output",
                        help="Output directory for compiled .tflite files")
    parser.add_argument("--targets", nargs="*", default=[],
                        metavar="TARGET",
                        help=(
                            "SoC targets to compile for. "
                            "Empty = all registered backends. "
                            "Examples: Qualcomm  SM8650  SM8750  MediaTek"
                        ))
    parser.add_argument("--keep-going", action="store_true", default=True,
                        help="Skip failed backends instead of aborting (default: on)")
    parser.add_argument("--no-keep-going", dest="keep_going", action="store_false",
                        help="Abort on first backend failure")
    args = parser.parse_args()

    if not os.path.isfile(args.tflite):
        print(f"[ERROR] Input file not found: {args.tflite}")
        sys.exit(1)

    targets = resolve_targets(args.targets)

    if targets is None:
        print("Targets: all registered backends")
    else:
        print(f"Targets: {[str(t) for t in targets]}")

    # Ensure bundled QNN libs are on LD_LIBRARY_PATH before the plugin dlopen()s them.
    if _QNN_AVAILABLE:
        _setup_qnn_lib_path()

    # Snapshot /tmp before compilation so we can find new error files afterwards.
    import glob
    tmp_errors_before = set(glob.glob("/tmp/*.error"))

    print(f"Compiling {args.tflite} ...")
    compiled_models = aot.aot_compile(
        args.tflite,
        keep_going=args.keep_going,
        target=targets,
    )

    print("\n── Compilation report ───────────────────────────────────────────")
    print(compiled_models.compilation_report())

    # Print any error files created by the plugin during this run.
    tmp_errors_after = set(glob.glob("/tmp/*.error"))
    new_errors = tmp_errors_after - tmp_errors_before
    if new_errors:
        print("\n── Plugin error details ─────────────────────────────────────────")
        for path in sorted(new_errors):
            print(f"\n{path}:")
            try:
                with open(path) as f:
                    print(f.read())
            except Exception as read_err:
                print(f"  (could not read: {read_err})")

    os.makedirs(args.output, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.tflite))[0]
    print(f"\nExporting to {args.output}/{model_name}_*.tflite ...")
    compiled_models.export(output_dir=args.output, model_name=model_name)
    print("Done.")


if __name__ == "__main__":
    main()
