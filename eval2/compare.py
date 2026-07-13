#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, re, math
from pathlib import Path
from collections import defaultdict
import itertools

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


METRICS = {"BLEU", "chrF2", "TER"}

def read_makefile_joined(path: Path) -> str:
    lines = path.read_text().splitlines()
    out = []
    buf = ""

    for line in lines:
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            out.append(buf + line)
            buf = ""

    if buf:
        out.append(buf)

    return "\n".join(out)


def parse_make_models(path: Path):
    text = read_makefile_joined(path)
    vars = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_]+)\s*:?=\s*(.*)$", line)
        if m:
            vars[m.group(1)] = m.group(2).strip()

    def expand(s):
        old = None
        while old != s:
            old = s
            for k, v in vars.items():
                s = s.replace(f"$({k})", v)
        return s

    aliases_raw = vars.get("MODEL_ALIASES", "")
    aliases_raw = aliases_raw.replace("\\", " ")
    aliases = aliases_raw.split()

    models = {}
    for a in aliases:
        key = f"MODEL_{a}"
        if key in vars:
            models[a] = Path(expand(vars[key]))
    return models


def parse_lang_token(tok):
    m = re.fullmatch(r"([A-Z]{2})\.([A-Za-z_]+)", tok)
    if not m:
        return None
    country, lang = m.groups()
    lang = lang.split("_", 1)[0]
    return f"{lang}+{country}" if country != "XX" else lang


def parse_score_filename(fn, kind):
    m = re.fullmatch(
        rf"{re.escape(kind)}_([A-Z]{{2}}\.[A-Za-z_]+)-([A-Z]{{2}}\.[A-Za-z_]+)\.(wmt|flo|bqt|bqtpar)\.(0s)?sacre",
        fn,
    )
    if not m:
        return None
    src_tok, tgt_tok, dataset, zs = m.groups()
    src = parse_lang_token(src_tok)
    tgt = parse_lang_token(tgt_tok)
    if not src or not tgt:
        return None
    return dataset, src, tgt, bool(zs)


def parse_sacre(path: Path, metric: str):
    text = path.read_text(errors="replace")
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        arr = json.loads(text[start:end + 1])
    except Exception:
        return None
    for item in arr:
        if item.get("name") == metric and isinstance(item.get("score"), (int, float)):
            return float(item["score"])
    return None


def load_model_scores(model_dir: Path, kind: str, metric: str):
    scores = {}
    score_dir = model_dir / "inf_scores"
    if not score_dir.is_dir():
        return scores

    bad_names = 0
    bad_scores = 0
    
    for p in score_dir.iterdir():
        if not (p.name.endswith(".sacre") or p.name.endswith(".0ssacre")):
            continue

        meta = parse_score_filename(p.name, kind)
        if meta is None:
            bad_names += 1
            print(f"SKIP name: {alias if 'alias' in globals() else ''} {p.name}", file=sys.stderr)
            continue

        val = parse_sacre(p, metric)
        if val is None:
            bad_scores += 1
            print(f"SKIP score: {p.name}", file=sys.stderr)
            continue

        dataset, src, tgt, zeroshot = meta
        scores[(dataset, src, tgt, zeroshot)] = val

    print(f"{model_dir}: parsed={len(scores)} bad_names={bad_names} bad_scores={bad_scores}", file=sys.stderr)
    return scores


def collect(models, kind, metric):
    by_model = {
        alias: load_model_scores(path, kind, metric)
        for alias, path in models.items()
    }
    keys = sorted(set().union(*(s.keys() for s in by_model.values())))
    return by_model, keys


def winner_data(by_model, keys, min_models=2):
    out = {}
    for key in keys:
        vals = {
            model: scores[key]
            for model, scores in by_model.items()
            if key in scores
        }
        if len(vals) < min_models:
            continue
        ordered = sorted(vals.items(), key=lambda x: x[1], reverse=True)
        winner, best = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else best
        margin = best - second
        out[key] = winner, best, margin, vals
    return out


def medal_table(by_model, keys, tie_eps=0.05):
    stats = {
        m: {"wins": 0, "ties": 0, "sum": 0.0, "n": 0}
        for m in by_model
    }

    for key in keys:
        vals = {
            m: scores[key]
            for m, scores in by_model.items()
            if key in scores
        }
        if len(vals) < 2:
            continue
        best = max(vals.values())
        winners = [m for m, v in vals.items() if abs(v - best) <= tie_eps]

        for m, v in vals.items():
            stats[m]["sum"] += v
            stats[m]["n"] += 1

        if len(winners) == 1:
            stats[winners[0]]["wins"] += 1
        else:
            for m in winners:
                stats[m]["ties"] += 1

    rows = []
    for m, s in stats.items():
        avg = s["sum"] / s["n"] if s["n"] else float("nan")
        rows.append((m, s["wins"], s["ties"], s["n"], avg))
    rows.sort(key=lambda x: (-x[1], -x[2], -x[4] if not math.isnan(x[4]) else 0))
    return rows


def dominance_table(by_model, keys, tie_eps=0.05):
    models = list(by_model)
    dom = {(a, b): 0 for a in models for b in models if a != b}

    for key in keys:
        vals = {
            m: scores[key]
            for m, scores in by_model.items()
            if key in scores
        }
        for a, b in itertools.permutations(vals, 2):
            if vals[a] > vals[b] + tie_eps:
                dom[(a, b)] += 1
    return models, dom


def write_medals(path: Path, rows):
    with path.open("w") as f:
        f.write("model\twins\tties\tcovered\taverage\n")
        for m, wins, ties, n, avg in rows:
            f.write(f"{m}\t{wins}\t{ties}\t{n}\t{avg:.4f}\n")


def write_dominance(path: Path, models, dom):
    with path.open("w") as f:
        f.write("model\t" + "\t".join(models) + "\n")
        for a in models:
            row = []
            for b in models:
                row.append("-" if a == b else str(dom[(a, b)]))
            f.write(a + "\t" + "\t".join(row) + "\n")


def write_dot(path: Path, models, dom, min_wins=1):
    with path.open("w") as f:
        f.write("digraph dominance {\n")
        f.write("  rankdir=LR;\n")
        f.write("  node [shape=box, style=rounded];\n")
        for m in models:
            f.write(f'  "{m}";\n')
        for a, b in itertools.permutations(models, 2):
            w = dom[(a, b)]
            back = dom.get((b, a), 0)
            if w >= min_wins and w > back:
                penwidth = 1 + min(6, w / 20)
                f.write(f'  "{a}" -> "{b}" [label="{w}", penwidth={penwidth:.2f}];\n')
        f.write("}\n")


def plot_winner_heatmap(out, dataset, outfile: Path):
    cells = {
        (src, tgt): (winner, margin)
        for (ds, src, tgt, zs), (winner, best, margin, vals) in out.items()
        if ds == dataset
    }
    if not cells:
        return

    srcs = sorted({s for s, t in cells})
    tgts = sorted({t for s, t in cells})
    models = sorted({winner for winner, margin in cells.values()})
    model_to_idx = {m: i for i, m in enumerate(models)}

    max_margin = max((m for _, m in cells.values()), default=1.0) or 1.0

    fig_w = max(8, 0.35 * len(tgts) + 3)
    fig_h = max(6, 0.30 * len(srcs) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("tab10")

    for y, src in enumerate(srcs):
        for x, tgt in enumerate(tgts):
            cell = cells.get((src, tgt))
            if not cell:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="white", edgecolor="lightgray"))
                continue
            winner, margin = cell
            base = cmap(model_to_idx[winner] % 10)
            alpha = 0.25 + 0.75 * (margin / max_margin)
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=base, alpha=alpha, edgecolor="white"))
            if len(tgts) <= 40 and len(srcs) <= 40:
                ax.text(x + 0.5, y + 0.5, winner[:3], ha="center", va="center", fontsize=6)

    ax.set_xlim(0, len(tgts))
    ax.set_ylim(0, len(srcs))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(tgts))])
    ax.set_xticklabels(tgts, rotation=90, fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(len(srcs))])
    ax.set_yticklabels(srcs, fontsize=7)
    ax.set_title(f"Winner heatmap — {dataset}")
    ax.set_xlabel("Target language")
    ax.set_ylabel("Source language")

    patches = [
        mpatches.Patch(color=cmap(model_to_idx[m] % 10), label=m)
        for m in models
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def plot_delta_heatmap(by_model, keys, model_a, model_b, dataset, outfile: Path):
    vals = {}
    for key in keys:
        ds, src, tgt, zs = key
        if ds != dataset:
            continue
        if key in by_model[model_a] and key in by_model[model_b]:
            vals[(src, tgt)] = by_model[model_a][key] - by_model[model_b][key]

    if not vals:
        return

    srcs = sorted({s for s, t in vals})
    tgts = sorted({t for s, t in vals})
    max_abs = max(abs(v) for v in vals.values()) or 1.0

    fig_w = max(8, 0.35 * len(tgts) + 3)
    fig_h = max(6, 0.30 * len(srcs) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("coolwarm")

    for y, src in enumerate(srcs):
        for x, tgt in enumerate(tgts):
            v = vals.get((src, tgt))
            if v is None:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor="white", edgecolor="lightgray"))
                continue
            color = cmap((v / max_abs + 1) / 2)
            ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white"))
            if len(tgts) <= 35 and len(srcs) <= 35:
                ax.text(x + 0.5, y + 0.5, f"{v:+.1f}", ha="center", va="center", fontsize=6)

    ax.set_xlim(0, len(tgts))
    ax.set_ylim(0, len(srcs))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(tgts))])
    ax.set_xticklabels(tgts, rotation=90, fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(len(srcs))])
    ax.set_yticklabels(srcs, fontsize=7)
    ax.set_title(f"BLEU delta — {model_a} minus {model_b} — {dataset}")
    ax.set_xlabel("Target language")
    ax.set_ylabel("Source language")
    plt.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def short_name(name: str) -> str:
    return name.replace("docmt4denhalf", "4dh").replace("docmt10denhalf", "10dh")


def write_pretty_medals(path: Path, rows, metric: str):
    with path.open("w") as f:
        f.write(f"Medal table ({metric})\n\n")
        f.write(f"{'Rank':>4}  {'Model':<24} {'Wins':>6} {'Ties':>6} {'Avg':>8} {'Covered':>8}\n")
        f.write(f"{'-'*4}  {'-'*24} {'-'*6} {'-'*6} {'-'*8} {'-'*8}\n")
        for i, (m, wins, ties, n, avg) in enumerate(rows, start=1):
            f.write(f"{i:>4}  {m:<24} {wins:>6} {ties:>6} {avg:>8.2f} {n:>8}\n")


def write_pretty_dominance(path: Path, models, dom):
    labels = [short_name(m) for m in models]
    width = max(8, max(len(x) for x in labels) + 2)

    with path.open("w") as f:
        f.write("Pairwise wins: row beats column\n\n")
        f.write(" " * width + "".join(f"{x:>{width}}" for x in labels) + "\n")
        f.write("-" * (width * (len(labels) + 1)) + "\n")

        for a, la in zip(models, labels):
            f.write(f"{la:<{width}}")
            for b in models:
                if a == b:
                    cell = "---"
                else:
                    cell = str(dom[(a, b)])
                f.write(f"{cell:>{width}}")
            f.write("\n")


def write_pretty_dominance_percent(path: Path, models, dom):
    labels = [short_name(m) for m in models]
    width = max(8, max(len(x) for x in labels) + 2)

    with path.open("w") as f:
        f.write("Pairwise win percentages: row beats column\n\n")
        f.write(" " * width + "".join(f"{x:>{width}}" for x in labels) + "\n")
        f.write("-" * (width * (len(labels) + 1)) + "\n")

        for a, la in zip(models, labels):
            f.write(f"{la:<{width}}")
            for b in models:
                if a == b:
                    cell = "---"
                else:
                    ab = dom[(a, b)]
                    ba = dom[(b, a)]
                    total = ab + ba
                    cell = "n/a" if total == 0 else f"{100 * ab / total:.0f}%"
                f.write(f"{cell:>{width}}")
            f.write("\n")


def write_condorcet(path: Path, models, dom):
    rows = []
    for a in models:
        wins = 0
        total = 0
        for b in models:
            if a == b:
                continue
            ab = dom[(a, b)]
            ba = dom[(b, a)]
            if ab + ba == 0:
                continue
            total += 1
            if ab > ba:
                wins += 1
        rows.append((a, wins, total))

    rows.sort(key=lambda x: (-x[1], x[0]))

    with path.open("w") as f:
        f.write("Condorcet-style pairwise victories\n\n")
        f.write(f"{'Model':<24} {'Pairwise victories':>20}\n")
        f.write(f"{'-'*24} {'-'*20}\n")
        for m, wins, total in rows:
            f.write(f"{m:<24} {wins:>8}/{total:<11}\n")


def write_pretty_reports(outdir: Path, medals, models, dom, metric: str):
    write_pretty_medals(outdir / "medal_table.txt", medals, metric)
    write_pretty_dominance(outdir / "dominance_pretty.txt", models, dom)
    write_pretty_dominance_percent(outdir / "dominance_percent.txt", models, dom)
    write_condorcet(outdir / "condorcet.txt", models, dom)
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mk-models", default="mk-model.mk")
    ap.add_argument("--models", nargs="+", required=True, help="2–8 model aliases")
    ap.add_argument("--kind", required=True, choices=["docmt", "sentmt", "mt"])
    ap.add_argument("--metric", default="BLEU", choices=sorted(METRICS))
    ap.add_argument("--dataset", default="bqtpar", choices=["wmt", "flo", "bqt", "bqtpar"])
    ap.add_argument("--outdir", default="bleu_compare")
    ap.add_argument("--delta", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    ap.add_argument("--tie-eps", type=float, default=0.05)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    
    if not (2 <= len(args.models) <= 8):
        raise SystemExit("Use 2–8 models.")

    all_models = parse_make_models(Path(args.mk_models))
    if args.list_models:
        for a, p in sorted(all_models.items()):
            print(f"{a}\t{p}")
        return

    selected = {}
    for m in args.models:
        if m not in all_models:
            raise SystemExit(f"Unknown model alias: {m}")
        selected[m] = all_models[m]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    by_model, keys = collect(selected, args.kind, args.metric)
    for alias, scores in by_model.items():
        print(f"{alias}: {len(scores)} parsed scores from {selected[alias] / 'inf_scores'}")
    
    winners = winner_data(by_model, keys)
    plot_winner_heatmap(winners, args.dataset, outdir / "winner_heatmap.png")

    medals = medal_table(by_model, keys, tie_eps=args.tie_eps)
    write_medals(outdir / "medal_table.tsv", medals)

    models, dom = dominance_table(by_model, keys, tie_eps=args.tie_eps)
    write_dominance(outdir / "dominance.tsv", models, dom)
    write_dot(outdir / "dominance_graph.dot", models, dom)
    write_pretty_reports(outdir, medals, models, dom, metric=args.metric)
    
    if args.delta:
        a, b = args.delta
        if a not in selected or b not in selected:
            raise SystemExit("--delta models must be among --models")
        plot_delta_heatmap(
            by_model, keys, a, b, args.dataset,
            outdir / f"delta_{a}_minus_{b}.png"
        )

    print(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
    
