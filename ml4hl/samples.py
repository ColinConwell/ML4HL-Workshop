"""Real-data samples for each widget's data mode.

Used by the widget gallery (``scripts/build_gallery.py``) and in the tutorial
for interactive explorations. Uses the audience-aligned domain datasets from
``ml4hl.data`` (Wordbank CDI child vocabulary, IHDP perinatal stress & child
reading/attention, The Music Lab cross-cultural vocalizations, Cowen 13-dim
music emotions, and THINGS human similarity judgments), while preserving legacy
helpers (``penguins``, ``diabetes_bmi``, ``digits_images``) for compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _z(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


# ─── Audience-aligned domain sample helpers ─────────────────────────────

def wordbank_vocab(n: int = 180, seed: int = 0, artifacts=None) -> dict:
    """Age in months (8-30m) vs vocabulary size (standardized 680-word CDI scale)."""
    from .data import load_wordbank

    wb = load_wordbank("English (American)", artifacts=artifacts)
    rng = np.random.default_rng(seed)
    # Stratified sample across age months 8..30 so every developmental stage is represented
    parts = []
    per_age = max(3, round(n / 23))
    for _, grp in wb.groupby("age"):
        k = min(len(grp), per_age)
        parts.append(grp.iloc[rng.choice(len(grp), k, replace=False)])
    sub = pd.concat(parts, ignore_index=True)
    if len(sub) > n:
        sub = sub.iloc[rng.choice(len(sub), n, replace=False)].reset_index(drop=True)
    sub = sub.sort_values("age").reset_index(drop=True)
    # Age-month empirical mean trajectory
    curve = wb.groupby("age")["vocab_share"].mean() * 680.0
    x_age = sub["age"].to_numpy(dtype=float) + rng.uniform(-0.35, 0.35, len(sub))
    y_words = (sub["vocab_share"].to_numpy(dtype=float) * 680.0).round(1)
    return dict(
        x=np.round(x_age, 2),
        y=y_words,
        curve_x=curve.index.to_numpy(dtype=float),
        curve_y=np.round(curve.to_numpy(dtype=float), 1),
        xlabel="child age (months)",
        ylabel="productive vocabulary (words)",
    )


def perinatal_reading(n: int | None = 150, seed: int = 0, artifacts=None) -> dict:
    """Perinatal stress & adversity index vs age-8 reading achievement (IHDP cohort)."""
    from .data import load_perinatal_dev

    df = load_perinatal_dev(artifacts=artifacts)
    if n and n < len(df):
        df = df.sample(n, random_state=seed).reset_index(drop=True)
    return dict(
        x=df["perinatal_stress_index"].to_numpy(dtype=float),
        y=df["reading_96m"].to_numpy(dtype=float),
        xlabel="perinatal stress & risk index (0-10)",
        ylabel="child reading score at age 8 (WJ)",
    )


def perinatal_attention_groups(n_per_group: int = 32, seed: int = 0, artifacts=None) -> dict:
    """Child externalizing/attention scores across low vs high perinatal stress cohorts."""
    from .data import load_perinatal_dev

    df = load_perinatal_dev(artifacts=artifacts)
    q_lo = df["perinatal_stress_index"].quantile(0.30)
    q_hi = df["perinatal_stress_index"].quantile(0.70)
    low = df[df["perinatal_stress_index"] <= q_lo]["externalizing_96m"].sample(n_per_group, random_state=seed).to_numpy()
    high = df[df["perinatal_stress_index"] >= q_hi]["externalizing_96m"].sample(n_per_group, random_state=seed).to_numpy()
    return dict(
        a=low,
        b=high,
        names=["low prenatal risk", "high prenatal risk"],
        unit="CBCL externalizing/attention",
    )


def musiclab_songs(artifacts=None) -> pd.DataFrame:
    """Natural History of Song Discography (Lullaby, Healing, Dance, Love across 86 societies)."""
    from .data import load_musiclab

    return load_musiclab("nhs", artifacts=artifacts)


# ─── Legacy helpers kept for backwards compatibility ────────────────────

def penguins() -> pd.DataFrame:
    """Palmer penguins, complete cases only."""
    import seaborn as sns

    return sns.load_dataset("penguins").dropna().reset_index(drop=True)


def diabetes_bmi(n: int | None = 160, seed: int = 0) -> dict:
    """Body-mass index against one-year disease progression (sklearn diabetes)."""
    from sklearn.datasets import load_diabetes

    d = load_diabetes(as_frame=True, scaled=False).frame
    if n:
        d = d.sample(n, random_state=seed)
    return dict(x=d["bmi"].to_numpy(), y=d["target"].to_numpy(),
                xlabel="body-mass index", ylabel="disease progression (1 yr)")


def digits_images(n: int = 300, seed: int = 0) -> dict:
    from sklearn.datasets import load_digits

    d = load_digits()
    idx = np.random.default_rng(seed).choice(len(d.images), n, replace=False)
    return dict(images=(d.data[idx] / 16.0).round(3), side=8, labels=d.target[idx])


def gallery_samples(artifacts=None) -> dict:
    """Return ``{widget: {"label": str, "data": ..., "params": {...}}}``."""
    from .data import load_music_emotions, load_musiclab, load_perinatal_dev
    from .zoo import decision_grids

    rng = np.random.default_rng(0)
    out: dict = {}

    # 1. Regression & Splines: Perinatal Stress / Reading + Wordbank Child Vocabulary
    peri = perinatal_reading(120, seed=2, artifacts=artifacts)
    wb = wordbank_vocab(160, seed=1, artifacts=artifacts)
    wb_small = wordbank_vocab(48, seed=3, artifacts=artifacts)

    out["ols_fit"] = dict(label="perinatal stress vs age-8 reading", data=perinatal_reading(85, seed=2, artifacts=artifacts))
    out["ols_robust"] = dict(label="perinatal stress vs age-8 reading", data=perinatal_reading(75, seed=4, artifacts=artifacts))
    out["gd_regression"] = dict(label="perinatal stress vs age-8 reading", data=peri)
    out["spline_scaling"] = dict(label="Wordbank CDI: child age vs vocabulary", data=wb)
    out["spline_relu_basis"] = dict(label="Wordbank CDI: child age vs vocabulary", data=wb)
    out["overfitting"] = dict(label="Wordbank CDI: child age vs vocabulary", data=wb_small)

    # 2. Categorization & Clustering: The Music Lab (Lullaby, Healing, Dance) & Infant-Directed Vocalizations
    nhs = load_musiclab("nhs", artifacts=artifacts)
    nhs3 = nhs[nhs["type"].isin(["Lullaby", "Dance", "Healing"])].reset_index(drop=True)
    xy_nhs = nhs3[["form_fast", "form_melo"]].to_numpy(dtype=float)
    sp_nhs = nhs3["type"].astype("category")
    out["category_manifolds"] = dict(
        label="Music Lab NHS: Lullaby, Dance & Healing",
        data=dict(
            X=xy_nhs,
            y=sp_nhs.cat.codes.to_numpy(),
            names=list(sp_nhs.cat.categories),
            xlabel="perceived tempo / beat",
            ylabel="melodic pleasantness",
        ),
    )

    nhs2 = nhs[nhs["type"].isin(["Lullaby", "Dance"])].reset_index(drop=True)
    xy_bin = _z(nhs2[["form_fast", "form_melo"]])
    y_bin = (nhs2["type"] == "Dance").astype(int).to_numpy()
    out["logistic_gd"] = dict(
        label="Music Lab NHS: Lullaby vs Dance",
        data=dict(X=xy_bin, y=y_bin, names=["Lullaby", "Dance"], xlabel="tempo / beat (z)", ylabel="melodic contour (z)"),
    )
    out["logistic_link"] = dict(
        label="Music Lab NHS: Lullaby vs Dance",
        data=dict(x=xy_bin[:, 0], y=y_bin, names=["Lullaby", "Dance"], xlabel="perceived tempo / beat (z)"),
    )
    out["perceptron"] = dict(
        label="Music Lab NHS: Lullaby vs Dance",
        data=dict(X=xy_bin / 2.0, y=y_bin, names=["Lullaby", "Dance"]),
    )
    out["mlp_xor"] = dict(
        label="Music Lab NHS: Lullaby vs Dance",
        data=dict(X=xy_bin / 2.0, y=y_bin, names=["Lullaby", "Dance"]),
    )

    from sklearn.datasets import make_moons

    Xm, ym = make_moons(220, noise=0.28, random_state=0)
    zoo = [
        decision_grids(Xm, ym, name="moons (simulated)", class_names=["A", "B"], xlabel="x1", ylabel="x2"),
        decision_grids(
            xy_nhs, sp_nhs.to_numpy(), name="Music Lab songs",
            xlabel="perceived tempo / beat", ylabel="melodic pleasantness",
        ),
    ]
    out["classifier_zoo"] = dict(label="moons + Music Lab songs (sklearn-fit)", data=zoo, sim=zoo[:1])

    # Clustering: Cowen 13-dim music emotions & Music Lab
    emo = load_music_emotions(artifacts=artifacts)["df"]
    emo_sub = emo.sample(170, random_state=2)
    xy_emo = _z(emo_sub[["calm/relaxing/serene", "energizing/pump-up"]].to_numpy() + rng.normal(0, 0.03, (len(emo_sub), 2)))
    out["percolation"] = dict(label="music emotions: calm vs energizing (z)", data=dict(X=xy_emo))
    out["kmeans"] = dict(label="music emotions: calm vs energizing (z)", data=dict(X=xy_emo[:130]))
    out["gmm_em"] = dict(label="music emotions: calm vs energizing (z)", data=dict(X=xy_emo[:130]))

    dsub = nhs3.sample(min(36, len(nhs3)), random_state=4).reset_index(drop=True)
    feats_nhs = _z(dsub[["form_fast", "form_beat", "form_melo", "form_plea", "form_exci", "func_baby", "func_danc"]])
    out["dendrogram"] = dict(
        label="Music Lab NHS: 7 song dimensions",
        data=dict(X=feats_nhs, xy=feats_nhs[:, [0, 2]], labels=[f"{t[:4]}·{c[:6]}" for t, c in zip(dsub["type"], dsub["culture"])]),
    )

    # 3. Dimensionality reduction
    dig = digits_images(240)
    out["autoencoder"] = dict(label="sklearn digits (8x8)", data=dict(images=dig["images"], side=8))

    peri_df = load_perinatal_dev(artifacts=artifacts).sample(150, random_state=5)
    emo_dims = [c for c in emo.columns if c not in ("top_emotion", "therapy_quadrant")]
    emo_high = emo[emo_dims].sample(160, random_state=5).to_numpy(dtype=float)
    emo_labs = emo.loc[emo[emo_dims].sample(160, random_state=5).index, "therapy_quadrant"].astype("category").cat.codes.to_numpy()
    out["projection"] = dict(
        label="perinatal language/reading (PCA) + 28-d music emotions (random)",
        data=dict(
            X2=_z(peri_df[["vocab_96m", "reading_96m"]]),
            Xhigh=emo_high,
            labels=emo_labs,
        ),
    )
    out["pca_reconstruct"] = dict(
        label="music emotions: 13 affective dimensions",
        data=dict(
            X=_z(emo[load_music_emotions(artifacts=artifacts)["core13"]].sample(120, random_state=7).to_numpy()),
            feature_names=[c.split("/")[0] for c in load_music_emotions(artifacts=artifacts)["core13"]],
        ),
    )

    # 4. Resampling & RSA
    out["resampling"] = dict(
        label="IHDP: attention scores by prenatal risk",
        data=perinatal_attention_groups(32, seed=6, artifacts=artifacts),
    )

    rdm = _things_rdm_sample(artifacts)
    if rdm is not None:
        out["rdm_builder"] = dict(label="THINGS: SPoSE vs image-text model", data=rdm)
        out["rsa_pipeline"] = dict(label="THINGS: SPoSE vs image-text model", data=rdm)
    return out


def _things_rdm_sample(artifacts=None, per_group: int = 6):
    try:
        from .data import load_things

        th = load_things(artifacts)
    except Exception:
        return None
    cats = ["animal", "food", "tool", "vehicle"]
    rows = []
    for c in cats:
        sub = th["meta"][th["meta"]["category"] == c].head(per_group)
        rows.extend(sub.index.tolist())
    if len(rows) < 8:
        return None
    meta = th["meta"].loc[rows]
    return dict(names=meta["concept"].tolist(), A=th["spose"][rows], B=th["model"][rows],
                groups=meta["category"].tolist(), labelA="human similarity (SPoSE)", labelB=th["model_name"])
