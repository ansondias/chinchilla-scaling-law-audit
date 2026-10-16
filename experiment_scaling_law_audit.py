"""
experiment_scaling_law_audit.py

Chinchilla Scaling-Law Audit
============================
Collects N (non-embedding/total parameter count, as reported) and D (training
tokens) for 27 real, publicly documented LLMs spanning:
  - pre-Chinchilla era   (released/designed before Mar 2022, typically under-trained: D/N << 20)
  - Chinchilla-era       (D/N ~ 20, i.e. "compute-optimal" per Hoffmann et al. 2022)
  - post-Chinchilla / LLaMA-style (deliberately over-trained for cheaper inference: D/N >> 20)

For each model we compute:
  - ratio      = D / N               (tokens per parameter)
  - log10(N), log10(D)               (for the log-log plot)
  - deviation  = log10(ratio / 20)   (how far above/below the Chinchilla line, in decades)

Outputs:
  - chinchilla_audit_results.csv   (full table + computed columns)
  - chinchilla_audit_plot.png      (log-log N vs D scatter with D = 20N reference line)

Run:
    python3 experiment_scaling_law_audit.py
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. The dataset
# ---------------------------------------------------------------------------
# N_params      : total parameter count (source: paper / model card, as stated)
# D_tokens       : training tokens actually seen during pretraining (source: paper / blog)
# era            : pre_chinchilla | chinchilla_era | post_chinchilla
#   pre_chinchilla   -> announced/trained before the Hoffmann et al. paper (Mar 2022)
#                       or explicitly following the earlier "Kaplan et al. 2020" recipe
#   chinchilla_era   -> explicitly designed to match the ~20 tokens/parameter rule
#   post_chinchilla  -> LLaMA-style: deliberately trained past compute-optimal on many
#                       more tokens than 20N, to shrink inference cost for a fixed capability
#
# All figures are the authors' own reported numbers (or a straightforward estimate
# clearly marked "(est.)" when the source didn't give an exact token count).

MODELS = [
    # model_name, org, year, N_params, D_tokens, era, source_url, note
    ("GPT-1", "OpenAI", 2018, 117e6, 1.0e9, "pre_chinchilla",
     "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
     "D est. from BooksCorpus ~800M words (~1B tokens); not reported precisely in paper"),
    ("GPT-2 (1.5B)", "OpenAI", 2019, 1.5e9, 8e9, "pre_chinchilla",
     "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
     "D est. from WebText 40GB text (~8B tokens); exact count not published"),
    ("GPT-3 175B", "OpenAI", 2020, 175e9, 300e9, "pre_chinchilla",
     "https://arxiv.org/abs/2005.14165",
     "Brown et al. 2020, Table 2.2"),
    ("GPT-J-6B", "EleutherAI", 2021, 6e9, 402e9, "pre_chinchilla",
     "https://huggingface.co/EleutherAI/gpt-j-6b",
     "402B tokens over 383,500 steps on The Pile"),
    ("Gopher 280B", "DeepMind", 2021, 280e9, 300e9, "pre_chinchilla",
     "https://arxiv.org/abs/2112.11446",
     "Rae et al. 2021, the motivating case for the Chinchilla study"),
    ("MT-NLG 530B", "Microsoft/NVIDIA", 2022, 530e9, 270e9, "pre_chinchilla",
     "https://arxiv.org/abs/2201.11990",
     "Smith et al. 2022"),
    ("GPT-NeoX-20B", "EleutherAI", 2022, 20e9, 400e9, "pre_chinchilla",
     "https://arxiv.org/abs/2204.06745",
     "Black et al. 2022, ~400B tokens from The Pile"),
    ("Chinchilla 70B", "DeepMind", 2022, 70e9, 1400e9, "chinchilla_era",
     "https://arxiv.org/abs/2203.15556",
     "Hoffmann et al. 2022 - the compute-optimal reference model itself"),
    ("PaLM 540B", "Google", 2022, 540e9, 780e9, "pre_chinchilla",
     "https://arxiv.org/abs/2204.02311",
     "Chowdhery et al. 2022; trained concurrently with/just after Chinchilla, ratio ~1.4"),
    ("OPT-175B", "Meta", 2022, 175e9, 180e9, "pre_chinchilla",
     "https://arxiv.org/abs/2205.01068",
     "Zhang et al. 2022"),
    ("BLOOM 176B", "BigScience", 2022, 176e9, 366e9, "pre_chinchilla",
     "https://arxiv.org/abs/2211.05100",
     "BigScience Workshop 2022; training started before Chinchilla's release"),
    ("Cerebras-GPT 13B", "Cerebras", 2023, 13e9, 257e9, "chinchilla_era",
     "https://arxiv.org/abs/2304.03208",
     "Dey et al. 2023, explicitly built to the ~20 tokens/param Chinchilla recipe"),
    ("LLaMA-7B", "Meta", 2023, 6.7e9, 1000e9, "post_chinchilla",
     "https://arxiv.org/abs/2302.13971",
     "Touvron et al. 2023a"),
    ("LLaMA-65B", "Meta", 2023, 65.2e9, 1400e9, "post_chinchilla",
     "https://arxiv.org/abs/2302.13971",
     "Touvron et al. 2023a"),
    ("MPT-7B", "MosaicML", 2023, 6.7e9, 1000e9, "post_chinchilla",
     "https://www.mosaicml.com/blog/mpt-7b",
     "MosaicML blog, 2023"),
    ("Falcon-40B", "TII", 2023, 40e9, 1000e9, "post_chinchilla",
     "https://arxiv.org/abs/2311.16867",
     "Almazrouei et al. 2023, RefinedWeb corpus"),
    ("Falcon-180B", "TII", 2023, 180e9, 3500e9, "post_chinchilla",
     "https://arxiv.org/abs/2311.16867",
     "Almazrouei et al. 2023"),
    ("Pythia-12B", "EleutherAI", 2023, 12e9, 300e9, "chinchilla_era",
     "https://arxiv.org/abs/2304.01373",
     "Biderman et al. 2023, ~1 epoch on The Pile (300B tokens)"),
    ("LLaMA2-7B", "Meta", 2023, 6.7e9, 2000e9, "post_chinchilla",
     "https://arxiv.org/abs/2307.09288",
     "Touvron et al. 2023b"),
    ("LLaMA2-70B", "Meta", 2023, 68.9e9, 2000e9, "post_chinchilla",
     "https://arxiv.org/abs/2307.09288",
     "Touvron et al. 2023b"),
    ("StableLM-Alpha-7B", "Stability AI", 2023, 7e9, 1500e9, "post_chinchilla",
     "https://stability.ai/news-updates/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models",
     "Stability AI blog, 2023; ~3x The Pile"),
    ("Qwen-7B", "Alibaba", 2023, 7e9, 2400e9, "post_chinchilla",
     "https://arxiv.org/abs/2309.16609",
     "Bai et al. 2023, Qwen Technical Report"),
    ("Baichuan2-7B", "Baichuan Inc.", 2023, 7e9, 2600e9, "post_chinchilla",
     "https://arxiv.org/abs/2309.10305",
     "Yang et al. 2023"),
    ("T5-11B", "Google", 2019, 11e9, 1000e9, "pre_chinchilla",
     "https://arxiv.org/abs/1910.10683",
     "Raffel et al. 2020; ~1M steps x 2^11 seq x 512 tok (repeats over 750GB C4)"),
    ("LLaMA3-8B", "Meta", 2024, 8e9, 15000e9, "post_chinchilla",
     "https://ai.meta.com/blog/meta-llama-3/",
     "Meta AI blog, Apr 2024"),
    ("LLaMA3-70B", "Meta", 2024, 70e9, 15000e9, "post_chinchilla",
     "https://ai.meta.com/blog/meta-llama-3/",
     "Meta AI blog, Apr 2024"),
    ("LLaMA3.1-405B", "Meta", 2024, 405e9, 15000e9, "post_chinchilla",
     "https://arxiv.org/abs/2407.21783",
     "Dubey et al. 2024, The Llama 3 Herd of Models"),
]

COLUMNS = ["model", "org", "year", "N_params", "D_tokens", "era", "source_url", "note"]


def build_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(MODELS, columns=COLUMNS)

    # --- sanity checks: unit mistakes are the #1 way this kind of table breaks ---
    # N should plausibly be between 1e6 (1M) and 1e13 (10T) params
    # D should plausibly be between 1e8 (100M) and 1e14 (100T) tokens
    assert (df["N_params"].between(1e6, 1e13)).all(), "N_params out of plausible range (unit error?)"
    assert (df["D_tokens"].between(1e8, 1e14)).all(), "D_tokens out of plausible range (unit error?)"
    # D/N should realistically fall roughly between 0.1 and 10,000 tokens/param.
    # Pre-Chinchilla models (e.g. MT-NLG, ratio ~0.5) can legitimately dip below 1 -
    # that under-training is exactly the phenomenon this audit is meant to surface -
    # but a ratio far outside this band almost always means an M/B/T unit slip.
    ratio_check = df["D_tokens"] / df["N_params"]
    assert (ratio_check.between(0.1, 1e4)).all(), "D/N ratio implausible (check units: M vs B vs T)"

    df["ratio_D_over_N"] = df["D_tokens"] / df["N_params"]
    df["log10_N"] = np.log10(df["N_params"])
    df["log10_D"] = np.log10(df["D_tokens"])
    # how many decades above (+) or below (-) the Chinchilla-optimal D = 20N line
    df["log10_deviation_from_20N"] = np.log10(df["ratio_D_over_N"] / 20.0)
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out = out.sort_values("N_params").reset_index(drop=True)
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {path} ({len(out)} rows)")


ERA_STYLE = {
    "pre_chinchilla":  dict(color="#d95f02", label="Pre-Chinchilla (under-trained, D/N < 20)"),
    "chinchilla_era":  dict(color="#1b9e77", label="Chinchilla-era (compute-optimal, D/N ≈ 20)"),
    "post_chinchilla": dict(color="#7570b3", label="Post-Chinchilla / LLaMA-style (over-trained, D/N > 20)"),
}


def make_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    # Reference line: D = 20N (Chinchilla compute-optimal ratio)
    n_line = np.logspace(math.log10(df["N_params"].min()) - 0.3,
                          math.log10(df["N_params"].max()) + 0.3, 100)
    d_line = 20 * n_line
    ax.plot(n_line, d_line, linestyle="--", color="gray", linewidth=1.5,
            label="D = 20N (Chinchilla compute-optimal)", zorder=1)

    for era, style in ERA_STYLE.items():
        sub = df[df["era"] == era]
        ax.scatter(sub["N_params"], sub["D_tokens"], s=70, color=style["color"],
                   edgecolor="black", linewidth=0.5, label=style["label"], zorder=3)

    # label each point with a short model name
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["N_params"], row["D_tokens"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N — Parameters (log scale)")
    ax.set_ylabel("D — Training tokens (log scale)")
    ax.set_title("Chinchilla Scaling-Law Audit: D vs. N for 27 Real LLMs")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Wrote {path}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Summary by era ===")
    summary = df.groupby("era")["ratio_D_over_N"].agg(["count", "median", "min", "max"])
    print(summary.round(1))

    print("\n=== Furthest above the D=20N line (most over-trained) ===")
    print(df.sort_values("ratio_D_over_N", ascending=False)
            [["model", "N_params", "D_tokens", "ratio_D_over_N"]].head(5).to_string(index=False))

    print("\n=== Furthest below the D=20N line (most under-trained) ===")
    print(df.sort_values("ratio_D_over_N", ascending=True)
            [["model", "N_params", "D_tokens", "ratio_D_over_N"]].head(5).to_string(index=False))


def main() -> None:
    df = build_dataframe()
    save_csv(df, HERE / "chinchilla_audit_results.csv")
    make_plot(df, HERE / "chinchilla_audit_plot.png")
    print_summary(df)


if __name__ == "__main__":
    main()
