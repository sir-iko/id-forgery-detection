import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np, pandas as pd

from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef



SCORE_FILES = {"ResNet50":    "checkpoints/scores_test_resnet50.csv",

               "DenseNet121": "checkpoints/scores_test_densenet121.csv",

               "ViT-B/16":    "checkpoints/scores_test_vit.csv"}

PARTITION_SEED, PARTITION_SIZE = 42, 150

CALIB_SIZES, N_REPEATS, TARGET_FPR = [10, 25, 50], 20, 0.10



def stratified_draw(df, n, seed, key="attack_type"):

    rng = np.random.default_rng(seed)

    groups = {k: sub.index.to_numpy() for k, sub in df.groupby(key)}

    total = len(df)

    raw = {k: n * len(i) / total for k, i in groups.items()}

    floor = {k: int(np.floor(v)) for k, v in raw.items()}

    rem = n - sum(floor.values())

    for k in sorted(raw, key=lambda k: raw[k] - floor[k], reverse=True)[:rem]:

        floor[k] += 1

    picked = []

    for k, idx in groups.items():

        picked.extend(rng.choice(idx, size=min(floor[k], len(idx)), replace=False).tolist())

    return df.loc[picked]



def fit_threshold(draw):

    bona = draw[draw["attack_type"] == "none"]["p_forged"].to_numpy()

    return None if len(bona) == 0 else float(np.quantile(bona, 1 - TARGET_FPR, method="higher"))



def subset_by_type(df, atk):

    return df if atk == "global" else df[(df["attack_type"] == atk) | (df["attack_type"] == "none")]



results = {}

for name, path in SCORE_FILES.items():

    df = pd.read_csv(path)

    part = stratified_draw(df, PARTITION_SIZE, PARTITION_SEED)

    rep = df.drop(index=part.index)

    results[name] = {a: {} for a in ("face", "text")}

    for size in CALIB_SIZES:

        runs = {a: [] for a in ("face", "text")}

        for r in range(N_REPEATS):

            t = fit_threshold(stratified_draw(part, size, seed=1000 * size + r))

            if t is None: continue

            for a in ("face", "text"):

                rp = subset_by_type(rep, a)

                y = rp["y_true"].to_numpy()

                pred = (rp["p_forged"].to_numpy() >= t).astype(int)

                runs[a].append({"balanced_accuracy": balanced_accuracy_score(y, pred),

                                "f1": f1_score(y, pred, pos_label=1, zero_division=0),

                                "mcc": matthews_corrcoef(y, pred)})

        for a in ("face", "text"):

            results[name][a][size] = {m: (float(np.mean([x[m] for x in runs[a]])),

                                          float(np.std([x[m] for x in runs[a]])))

                                      for m in ("balanced_accuracy", "f1", "mcc")}



print("VERIFY against Table 6 (size 50):")

for n in SCORE_FILES:

    for a in ("face", "text"):

        m, s = results[n][a][50]["balanced_accuracy"]

        mc, _ = results[n][a][50]["mcc"]

        print("  %-12s %-5s BA=%.3f +/- %.3f   MCC=%+.3f" % (n, a, m, s, mc))



x = np.arange(len(CALIB_SIZES))

colors = {"ResNet50": "#1f77b4", "DenseNet121": "#d62728", "ViT-B/16": "#2ca02c"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=False, sharey=False)

for ax, metric, label in zip(axes, ["balanced_accuracy", "f1"], ["Balanced accuracy", "F1"]):

    for n in SCORE_FILES:

        for a, ls, mk in (("face", "-", "o"), ("text", "--", "s")):

            y = [results[n][a][s][metric][0] for s in CALIB_SIZES]

            ax.plot(x, y, ls=ls, marker=mk, color=colors[n], label="%s (%s)" % (n, a))

    if metric == "balanced_accuracy":

        ax.axhline(0.5, ls=":", lw=1.2, color="grey")

    ax.set_xticks(x); ax.set_xticklabels(CALIB_SIZES)

    ax.set_xlim(-0.15, len(CALIB_SIZES) - 0.85)

    ax.set_ylim(0, 1)

    ax.set_xlabel("Calibration-set size (bonafide images)")

    ax.set_ylabel(label); ax.set_title(label)



handles, labels = axes[0].get_legend_handles_labels()

fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,

           frameon=False, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.08, 1, 1])

fig.savefig("figure3_calibration_recovery.png", dpi=300, bbox_inches="tight")

print("\nwrote figure3_calibration_recovery.png")
