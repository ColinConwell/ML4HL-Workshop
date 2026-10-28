"""Dataset loaders for the applied examples.

Precomputed artifacts (embeddings, audio features, thumbnails) are looked up
locally first (``$ML4HL_DATA``, then ``data/artifacts`` next to the package)
and otherwise fetched from the Hugging Face dataset repo ``colinconwell/ml4hl``.
Public raw datasets (DASS-42, Palmer penguins, scikit-learn built-ins) are
downloaded from their original hosts and cached under ``~/.cache/ml4hl``.
"""

from __future__ import annotations

import io
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_RELEASE = json.loads((Path(__file__).absolute().parent / "assets/data-release.json").read_text())
HF_REPO = _RELEASE["repo"]
HF_REVISION = _RELEASE["revision"]
PKG_ROOT = Path(__file__).absolute().parent  # not resolve(): Hub snapshots are symlinks into blobs/
CACHE = Path(os.environ.get("ML4HL_CACHE", Path.home() / ".cache" / "ml4hl"))


def _local_dirs(artifacts=None) -> list[Path]:
    dirs = []
    if artifacts:
        dirs.append(Path(artifacts))
    if os.environ.get("ML4HL_DATA"):
        dirs.append(Path(os.environ["ML4HL_DATA"]))
    dirs += [PKG_ROOT.parent / "data" / "artifacts", PKG_ROOT.parent / "artifacts"]
    return dirs


def _verified_artifact(path: Path, name: str) -> Path:
    expected = _RELEASE.get("files", {}).get(name)
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Artifact checksum differs from the pinned release: {name}")
    return path


def artifact(name: str, artifacts=None) -> Path:
    """Path to an artifact file: local copy if present, else downloaded from the Hub."""
    for d in _local_dirs(artifacts):
        if (d / name).exists():
            return _verified_artifact(d / name, name)
    from huggingface_hub import hf_hub_download

    return _verified_artifact(Path(hf_hub_download(HF_REPO, f"artifacts/{name}", repo_type="dataset", revision=HF_REVISION)), name)


def artifact_manifest(artifacts=None) -> dict:
    return json.loads(artifact("manifest.json", artifacts).read_text())


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ml4hl-tutorial"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def _try_artifact(name: str, artifacts=None) -> Path | None:
    """Return local or Hub artifact path if available, else None."""
    for d in _local_dirs(artifacts):
        if (d / name).exists():
            return _verified_artifact(d / name, name)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        return _verified_artifact(Path(hf_hub_download(HF_REPO, f"artifacts/{name}", repo_type="dataset", revision=HF_REVISION)), name)
    except EntryNotFoundError:
        if name in _RELEASE.get("files", {}):
            raise  # A pinned required artifact must never silently switch to raw-data parsing.
        return None  # Only an absent optional artifact permits its documented raw-data fallback.


# ─── Precomputed artifacts ──────────────────────────────────────────────

def load_dollar_street(artifacts=None) -> dict:
    """Dollar Street household photos: metadata, image embeddings, 48px thumbnails.

    ``meta`` columns: id, topic, country, region, place (household), income
    (monthly, USD purchasing-power adjusted).
    """
    meta = pd.read_parquet(artifact("dollar_street_meta.parquet", artifacts))
    emb = np.load(artifact("dollar_street_emb.npy", artifacts)).astype(np.float32)
    thumbs = np.load(artifact("dollar_street_thumbs.npy", artifacts))
    return dict(meta=meta, emb=emb, thumbs=thumbs, model=artifact_manifest(artifacts).get("image_model", "image model"))


def load_things(artifacts=None) -> dict:
    """THINGS concepts: human similarity embedding (SPoSE, 66-d) and a model's text embedding."""
    meta = pd.read_parquet(artifact("things_meta.parquet", artifacts))
    spose = np.load(artifact("things_spose.npy", artifacts)).astype(np.float32)
    model = np.load(artifact("things_model_emb.npy", artifacts)).astype(np.float32)
    return dict(meta=meta, spose=spose, model=model,
                model_name=artifact_manifest(artifacts).get("text_model", "image-text model"))


def load_finch(artifacts=None, min_calls: int = 30) -> dict:
    """Zebra finch calls: metadata (bird, call type), acoustic features, small spectrograms.

    Call types with fewer than ``min_calls`` recordings are dropped.
    """
    meta = pd.read_parquet(artifact("finch_meta.parquet", artifacts))
    feats = np.load(artifact("finch_feats.npy", artifacts)).astype(np.float32)
    specs = np.load(artifact("finch_specs.npy", artifacts))
    names = json.loads(artifact("finch_feature_names.json", artifacts).read_text())
    meta["calltype"] = meta["code"].map(FINCH_CALLTYPES).fillna(meta["code"])
    keep = meta["calltype"].map(meta["calltype"].value_counts()).ge(min_calls).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    return dict(meta=meta, feats=feats[keep], specs=specs[keep], feature_names=names)


FINCH_CALLTYPES = {
    "Ag": "aggressive", "Be": "begging", "DC": "distance", "Di": "distress", "LT": "long tonal",
    "Ne": "nest", "So": "song", "Te": "tet", "Th": "thuk", "Tu": "tuck", "Wh": "whine",
}


# ─── Raw public datasets ────────────────────────────────────────────────

DASS_URL = "https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip"
DASS_SCALES = {
    "depression": [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
    "anxiety": [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
    "stress": [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39],
}


def load_dass(n: int | None = None, seed: int = 0, clean: bool = True) -> pd.DataFrame:
    """DASS-42 responses from openpsychometrics.org (about 39,800 respondents).

    Returns one row per respondent with items ``Q1``..``Q42`` (scored 0-3),
    the three scale totals, and a few demographics. With ``clean=True``,
    respondents who failed the validity checks are dropped.
    """
    z = _download(DASS_URL, CACHE / "DASS_data_21.02.19.zip")
    with zipfile.ZipFile(z) as zf:
        name = next(f for f in zf.namelist() if f.endswith("data.csv"))
        raw = pd.read_csv(io.BytesIO(zf.read(name)), sep="\t")
    items = [f"Q{i}A" for i in range(1, 43)]
    df = raw[items].sub(1).clip(lower=0)
    df.columns = [f"Q{i}" for i in range(1, 43)]
    for scale, qs in DASS_SCALES.items():
        df[scale] = df[[f"Q{q}" for q in qs]].sum(axis=1)
    keep = ["age", "gender", "education", "country", "VCL6", "VCL9", "VCL12"]
    for k in keep:
        if k in raw:
            df[k] = raw[k]
    if clean and {"VCL6", "VCL9", "VCL12"} <= set(df.columns):
        df = df[(df.VCL6 == 0) & (df.VCL9 == 0) & (df.VCL12 == 0)]
        df = df[(df.age >= 16) & (df.age <= 90)]
    df = df.drop(columns=[c for c in ("VCL6", "VCL9", "VCL12") if c in df]).reset_index(drop=True)
    if n:
        df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df


def dass_items() -> pd.DataFrame:
    """Item number, scale, and short wording for the 42 DASS items."""
    text = [
        "upset by trivial things", "dryness of mouth", "no positive feeling", "breathing difficulty",
        "couldn't get going", "over-reacted", "shakiness", "difficult to relax", "anxious, relieved after",
        "nothing to look forward to", "easily upset", "nervous energy", "sad and depressed",
        "impatient when delayed", "faintness", "lost interest", "not worth much as a person", "touchy",
        "perspired noticeably", "scared without reason", "life not worthwhile", "hard to wind down",
        "difficulty swallowing", "no enjoyment", "heart racing", "down-hearted and blue", "irritable",
        "close to panic", "hard to calm down", "fear of being thrown", "unable to become enthusiastic",
        "difficult to tolerate interruptions", "state of nervous tension", "felt worthless",
        "intolerant of interruptions", "terrified", "no hope for the future", "life meaningless",
        "agitated", "worried about panic", "trembling", "difficult to work up initiative",
    ]
    scale_of = {q: s for s, qs in DASS_SCALES.items() for q in qs}
    return pd.DataFrame({"item": [f"Q{i}" for i in range(1, 43)], "scale": [scale_of[i] for q in range(1, 43) for i in [q]], "text": text})


# ─── Audience-aligned domain datasets ───────────────────────────────────

WORDBANK_URL = "https://raw.githubusercontent.com/langcog/wordbank-book/master/data/_common/admins.feather"
IHDP_FULL_URL = "https://raw.githubusercontent.com/py-why/EconML/main/econml/data/ihdp/example_full.csv"
NHS_BASE_URL = "https://raw.githubusercontent.com/themusiclab/nhs/master/data/nhs/"
IDS_BASE_URL = "https://raw.githubusercontent.com/themusiclab/infant-speech-song/main/data/"
RDATA_BASE_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/"


def load_wordbank(language: str | None = "English (American)", artifacts=None) -> pd.DataFrame:
    """Stanford Wordbank MacArthur-Bates CDI vocabulary administrations (ages 8-30 months).

    Columns: ``data_id``, ``language``, ``form`` (``WG`` 8-18m or ``WS`` 16-30m),
    ``age`` (months), ``words_produced``, ``words_understood``, ``vocab_share``
    (proportion of instrument checklist produced, 0-1), ``sex``, ``birth_order``, ``mom_ed``.
    """
    p = _try_artifact("wordbank_cdi.parquet", artifacts)
    if p is not None:
        df = pd.read_parquet(p)
    else:
        f = _download(WORDBANK_URL, CACHE / "wordbank_admins.feather")
        raw = pd.read_feather(f)
        eng = raw[raw["language"].isin(["English (American)", "English (British)"])].copy()
        eng = eng[(eng["age"] >= 8) & (eng["age"] <= 30) & eng["production"].notna()].copy()
        eng["words_produced"] = eng["production"].astype(float)
        eng["words_understood"] = eng["comprehension"].astype(float)
        eng["vocab_share"] = np.where(
            eng["form"] == "WG", eng["words_produced"] / 396.0, eng["words_produced"] / 680.0
        ).clip(0, 1)
        cols = ["data_id", "language", "form", "age", "words_produced", "words_understood", "vocab_share", "sex", "birth_order", "mom_ed"]
        df = eng[cols].reset_index(drop=True)
    if language:
        df = df[df["language"] == language].reset_index(drop=True)
    return df


def load_perinatal_dev(artifacts=None) -> pd.DataFrame:
    """Longitudinal perinatal & child neurodevelopment cohort (IHDP, n = 837, birth to age 8).

    Tracks perinatal risk indicators and an unvalidated teaching composite alongside child
    receptive language (PPVT-R at 36, 60, 96 months), Woodcock-Johnson broad
    reading and math achievement (96 months), and Child Behavior Checklist
    internalizing and externalizing (not an attention-specific scale)
    scores across 8 clinical sites.
    """
    p = _try_artifact("perinatal_dev.parquet", artifacts)
    if p is not None:
        return pd.read_parquet(p)
    f = _download(IHDP_FULL_URL, CACHE / "ihdp_example_full.csv")
    raw = pd.read_csv(f)
    z = lambda s: (s - s.mean()) / (s.std() or 1.0)  # noqa: E731
    preterm_wk = (40.0 - raw["ga.anal"].astype(float)).clip(lower=0)
    stress_raw = (
        -0.35 * z(raw["nnhealth"].astype(float))
        + 0.30 * z(preterm_wk)
        - 0.25 * z(raw["bw"].astype(float))
        + 0.25 * z(raw["cigs"].astype(float))
        + 0.15 * z(raw["whenpren"].astype(float))
        - 0.20 * z(raw["momed4"].fillna(2).astype(float))
    )
    stress_idx = ((stress_raw - stress_raw.min()) / (stress_raw.max() - stress_raw.min()) * 10.0).round(2)
    df = pd.DataFrame({
        "child_id": raw["id"].astype(int),
        "site": raw["site"].astype(int),
        "intervention": raw["treat"].astype(int),
        "sex": np.where(raw["female"] == 1, "female", "male"),
        "birthweight_g": raw["bw"].astype(float),
        "gestational_age_wk": raw["ga.anal"].astype(float),
        "head_circumference_cm": raw["b.head"].astype(float),
        "neonatal_health": raw["nnhealth"].astype(float),
        "apgar5": raw["apgar5"].astype(float),
        "prenatal_care_trimester": raw["whenpren"].astype(int),
        "prenatal_cigs_per_day": raw["cigs"].astype(float),
        "maternal_age": raw["momage"].astype(float),
        "maternal_ed_level": raw["momed4"].fillna(2).astype(int),
        "maternal_vocab_ppvt": raw["ppvt.imp"].astype(float),
        "perinatal_stress_index": stress_idx,
        "vocab_36m": raw["ppvtr.36"].astype(float),
        "vocab_60m": raw["ppvtr.60"].astype(float),
        "vocab_96m": raw["ppvtr.96"].astype(float),
        "reading_96m": raw["wjbread"].astype(float),
        "math_96m": raw["wjbmath"].astype(float),
        "iq_36m": raw["iqsb.36"].astype(float),
        "iq_96m": raw["iqf.96"].astype(float),
        "internalizing_36m": raw["inter36"].astype(float),
        "externalizing_36m": raw["exter36"].astype(float),
        "internalizing_96m": raw["inter96"].astype(float),
        "externalizing_96m": raw["exter96"].astype(float),
        "behavior_problems_96m": raw["behav96"].astype(float),
    })
    return df.dropna(subset=["vocab_96m", "reading_96m", "externalizing_96m", "internalizing_96m"]).reset_index(drop=True)


def load_musiclab(corpus: str = "nhs", artifacts=None) -> pd.DataFrame | dict:
    """The Music Lab cross-cultural vocalization datasets.

    Parameters
    ----------
    corpus : ``"nhs"`` (Natural History of Song Discography: 118 songs across 86
        societies in 4 contexts: Lullaby, Healing, Dance, Love), ``"ids"``
        (Hilton et al. 2022 infant-directed and adult-directed song and speech
        across 21 societies, 1,608 vocalizations), or ``"both"``.
    """
    if corpus == "both":
        return {"nhs": load_musiclab("nhs", artifacts), "ids": load_musiclab("ids", artifacts)}
    if corpus == "nhs":
        p = _try_artifact("musiclab_nhs.parquet", artifacts)
        if p is not None:
            return pd.read_parquet(p)
        meta = pd.read_csv(_download(NHS_BASE_URL + "NHSDiscography_Metadata.csv", CACHE / "nhs_meta.csv"))
        meta["song_id"] = meta["song"].str.extract(r"(\d+)").astype(int)
        naive = pd.read_csv(_download(NHS_BASE_URL + "NHSDiscography_NaiveListeners.csv", CACHE / "nhs_naive.csv")).rename(columns={"song": "song_id"})
        trans = pd.read_csv(_download(NHS_BASE_URL + "NHSDiscography_TranscriptionFeatures.csv", CACHE / "nhs_trans.csv")).rename(columns={"song": "song_id"})
        return meta[["song_id", "song", "type", "culture", "id_nhs", "nhs_region", "nhs_subsistence", "latitude", "longitude"]].merge(
            naive, on="song_id"
        ).merge(
            trans[["song_id", "duration", "mean_interval", "modal_interval_prevalence", "amount_of_arpeggiation",
                   "stepwise_motion", "melodic_thirds", "direction_of_motion", "duration_of_melodic_arcs",
                   "size_of_melodic_arcs", "modal_pitch_prev", "rel_strength_top_pitches"]],
            on="song_id",
        )
    if corpus == "ids":
        p = _try_artifact("musiclab_ids.parquet", artifacts)
        if p is not None:
            return pd.read_parquet(p)
        ac = pd.read_csv(_download(IDS_BASE_URL + "acoustics-rawAudio.csv", CACHE / "ids_acoustics.csv"))
        st = pd.read_csv(_download(IDS_BASE_URL + "stimuli-info.csv", CACHE / "ids_stimuli.csv"))
        ids_df = ac.merge(st, on="id", how="left")
        ids_keep = [
            "id", "id_site", "id_person", "Culture", "Location", "infantdir", "song",
            "praat_f0_mean", "praat_f0_std", "praat_f0_range", "praat_f1_mean", "praat_f2_mean",
            "praat_intensity_mean", "praat_intensitytravel_median", "praat_voweltravel_mean",
            "praat_f0travel_mean", "mir_roughness_mean", "mir_inharmonicity", "mir_tempo", "mir_pulseclarity", "npvi_total",
        ]
        ids_df = ids_df[ids_keep].dropna().reset_index(drop=True)
        ids_df["category"] = np.where(
            ids_df["infantdir"] == 1,
            np.where(ids_df["song"] == 1, "infant song", "infant speech"),
            np.where(ids_df["song"] == 1, "adult song", "adult speech"),
        )
        return ids_df
    raise ValueError(f"Unknown Music Lab corpus {corpus!r}; choose 'nhs', 'ids', or 'both'.")


def load_music_emotions(artifacts=None) -> dict:
    """Cowen et al. (2020, PNAS) 13-Dimensions of Music Emotion (1,841 music clips).

    Returns ``dict(df=DataFrame, dimensions=list[str], core13=list[str])``.
    """
    p = _try_artifact("music_emotions.parquet", artifacts)
    if p is not None:
        df = pd.read_parquet(p)
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        pf = pq.ParquetFile(fs.open("datasets/baobaoh/13-dimensions-music-emotions/data/train-00000-of-00001.parquet"))
        cols = [c for c in pf.schema.names if c != "audio"]
        df = pf.read(columns=cols).to_pandas()
        core13 = [
            "amusing", "angry", "anxious/tense", "awe-inspiring/amazing", "calm/relaxing/serene",
            "compassionate/sympathetic", "dreamy", "eerie/mysterious", "energizing/pump-up",
            "erotic/desirous", "joyful/cheerful", "sad/depressing", "scary/fearful", "triumphant/heroic",
        ]
        core13 = [c for c in core13 if c in df.columns][:13]
        df["top_emotion"] = df[core13].idxmax(axis=1)
        quad_map = {
            "calm/relaxing/serene": "soothing / serene", "dreamy": "soothing / serene",
            "compassionate/sympathetic": "soothing / serene", "joyful/cheerful": "uplifting / joyful",
            "amusing": "uplifting / joyful", "energizing/pump-up": "energizing / triumphant",
            "triumphant/heroic": "energizing / triumphant", "awe-inspiring/amazing": "energizing / triumphant",
        }
        df["therapy_quadrant"] = df["top_emotion"].map(quad_map).fillna("tense / melancholic")
    dims = [c for c in df.columns if c not in ("top_emotion", "therapy_quadrant")]
    core13 = [
        "amusing", "angry", "anxious/tense", "awe-inspiring/amazing", "calm/relaxing/serene",
        "compassionate/sympathetic", "dreamy", "eerie/mysterious", "energizing/pump-up",
        "joyful/cheerful", "sad/depressing", "scary/fearful", "triumphant/heroic",
    ]
    return dict(df=df, dimensions=dims, core13=[c for c in core13 if c in df.columns])


def load_household_survey(artifacts=None) -> pd.DataFrame:
    """World Bank Living Standards Measurement Study (LSMS) household health & expenditure survey.

    Returns 5,006 households across 194 communes with demographic, schooling,
    agricultural, medical expenditure, food expenditure, total expenditure, and
    commune-level illness and health-insurance indicators.
    """
    p = _try_artifact("household_survey.parquet", artifacts)
    if p is not None:
        return pd.read_parquet(p)
    hh = pd.read_csv(_download(RDATA_BASE_URL + "camerondata/vietnam_hh.csv", CACHE / "vietnam_hh.csv"))
    lss = pd.read_csv(_download(RDATA_BASE_URL + "camerondata/vietnamlss.csv", CACHE / "vietnamlss.csv"))
    ind = pd.read_csv(_download(RDATA_BASE_URL + "camerondata/vietnam_ind.csv", CACHE / "vietnam_ind.csv"))
    # These releases have matching row identifiers and shared household fields.
    # Fail rather than silently combine a future reordered upstream release.
    for field in ("rownames", "sex", "age", "farm", "hhsize"):
        if not np.allclose(hh[field], lss[field], equal_nan=True):
            raise ValueError(f"Household table alignment mismatch: {field}")
    comm = ind.groupby("commune").agg(
        commune_illness_rate=("illness", "mean"),
        commune_injury_rate=("injury", "mean"),
        commune_insurance_rate=("insurance", "mean"),
        commune_illdays=("illdays", "mean"),
        commune_actdays=("actdays", "mean"),
        commune_pharmacy_visits=("pharvis", "mean"),
    ).reset_index()
    return pd.DataFrame({
        "commune": hh["commune"].astype(int),
        "urban": lss["urban98"].astype(int),
        "farm": hh["farm"].astype(int),
        "hhsize": hh["hhsize"].astype(int),
        "head_age": hh["age"].astype(int),
        "head_female": (hh["sex"] == 2).astype(int),
        "educ_years": lss["educyr98"].astype(float),
        "educ_level": hh["educ"].astype(int),
        "log_food_exp": lss["lnrlfood"].astype(float),
        "log_med_exp": lss["lhhex12m"].astype(float),
        "log_total_exp": lss["lhhexp1"].astype(float),
    }).merge(comm, on="commune", how="left").dropna().reset_index(drop=True)

