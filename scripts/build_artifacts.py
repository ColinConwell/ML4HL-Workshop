"""Precompute the heavy artifacts and curated domain tables used by the applied examples.

    python scripts/build_artifacts.py all               # everything
    python scripts/build_artifacts.py audience          # fast tabular & domain artifacts
    python scripts/build_artifacts.py dollar-street     # image embeddings + thumbnails
    python scripts/build_artifacts.py things            # SPoSE + text embeddings
    python scripts/build_artifacts.py finch             # zebra finch acoustic features
    python scripts/build_artifacts.py wordbank          # Stanford Wordbank MB-CDI vocabulary curves
    python scripts/build_artifacts.py perinatal         # IHDP perinatal stress, language, reading & attention
    python scripts/build_artifacts.py musiclab          # The Music Lab NHS & infant-directed vocalizations
    python scripts/build_artifacts.py music-emotions    # Cowen et al. (2020) 13-dim music emotions
    python scripts/build_artifacts.py household-survey  # World Bank LSMS household health & expenditure

Outputs go to data/artifacts/ (git-ignored). The tutorial uses the immutable
public release pinned in ml4hl/assets/data-release.json; rebuilding may produce
different artifacts if upstream sources or model weights have changed. The
tutorial itself loads the prepared arrays and tables without a GPU.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "artifacts"

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DS_REPO = "iamshnoo/dollarstreet"
DS_TOPICS = ["home", "stove_hob", "toothbrush", "bed", "front_door", "kitchen"]
THINGS_RAW = "https://raw.githubusercontent.com/aaronmueller/lm-property-inheritance/main/data/things/"
FINCH_REPO = "cgeorgiaw/animal-sounds"


def log(msg: str) -> None:
    print(f"[artifacts {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str):
    from transformers import AutoModel, AutoProcessor

    dev = device()
    log(f"loading {name} on {dev}")
    model = AutoModel.from_pretrained(name).to(dev).eval()
    proc = AutoProcessor.from_pretrained(name)
    return model, proc, dev


def _feats(out):
    """Pull the pooled embedding tensor out of whatever transformers returned."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def embed_images(images, model, proc, dev, batch: int = 32) -> np.ndarray:
    import torch

    out = []
    for i in range(0, len(images), batch):
        inputs = proc(images=images[i:i + batch], return_tensors="pt").to(dev)
        with torch.no_grad():
            f = _feats(model.get_image_features(**inputs))
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def embed_texts(texts, model, proc, dev, batch: int = 64) -> np.ndarray:
    import torch

    out = []
    for i in range(0, len(texts), batch):
        inputs = proc(text=texts[i:i + batch], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            f = _feats(model.get_text_features(**inputs))
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def update_manifest(**kw) -> None:
    path = OUT / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    man.update(kw)
    man["built"] = time.strftime("%Y-%m-%d")
    path.write_text(json.dumps(man, indent=2))


# ─── Dollar Street ──────────────────────────────────────────────────────

def _parse_income(v) -> float:
    if v is None:
        return float("nan")
    m = re.search(r"[\d.,]+", str(v))
    return float(m.group(0).replace(",", "")) if m else float("nan")


def build_dollar_street(model_name: str, shards_per_topic: int = 0) -> None:
    """Shards are sorted by income, so read the (smaller) last row group of
    every shard of each topic over HTTP range requests instead of taking whole
    shards. That spans the full income range at a fraction of the download."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    from PIL import Image

    fs = HfFileSystem()
    files = [s.rfilename for s in HfApi().dataset_info(DS_REPO).siblings]
    rows, imgs, thumbs = [], [], []
    for topic in DS_TOPICS:
        shards = sorted(f for f in files if re.match(rf"data/{re.escape(topic)}-\d+-of-\d+-.*\.parquet$", f))
        if shards_per_topic:
            shards = [shards[i] for i in np.linspace(0, len(shards) - 1, min(shards_per_topic, len(shards))).round().astype(int)]
        for sh in shards:
            log(f"dollar street: reading last row group of {sh}")
            pf = pq.ParquetFile(fs.open(f"datasets/{DS_REPO}/{sh}"))
            tab = pf.read_row_group(pf.metadata.num_row_groups - 1).to_pandas()
            for _, r in tab.iterrows():
                img = r["image"]
                raw = img.get("bytes") if isinstance(img, dict) else img
                try:
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    continue
                im.thumbnail((384, 384))
                imgs.append(im)
                t = im.copy()
                w, h = t.size
                s = min(w, h)
                t = t.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((48, 48))
                thumbs.append(np.asarray(t, dtype=np.uint8))
                rows.append(dict(id=str(r["id"]), topic=topic, country=r.get("country_name"), region=r.get("region_id"),
                                 place=str(r.get("place")), income=_parse_income(r.get("income"))))
            log(f"dollar street: {len(rows)} images so far")
    meta = pd.DataFrame(rows)
    model, proc, dev = load_model(model_name)
    log(f"dollar street: embedding {len(imgs)} images")
    emb = embed_images(imgs, model, proc, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT / "dollar_street_meta.parquet", index=False)
    np.save(OUT / "dollar_street_emb.npy", emb.astype(np.float16))
    np.save(OUT / "dollar_street_thumbs.npy", np.stack(thumbs))
    update_manifest(image_model=model_name, dollar_street_n=len(meta), dollar_street_topics=DS_TOPICS)
    log(f"dollar street: wrote {len(meta)} rows, emb {emb.shape}")


# ─── THINGS ─────────────────────────────────────────────────────────────

def _fetch(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ml4hl"}), timeout=120) as r:
        return r.read()


def build_things(model_name: str) -> None:
    log("things: downloading SPoSE embedding + concept metadata")
    spose = np.loadtxt(io.BytesIO(_fetch(THINGS_RAW + "spose_embedding_66d_sorted.txt")))
    uids = _fetch(THINGS_RAW + "unique_id.txt").decode().split()
    cats = pd.read_csv(io.BytesIO(_fetch(THINGS_RAW + "category53_longFormat.tsv")), sep="\t")
    cats.columns = [c.strip().lower() for c in cats.columns]
    ucol = next(c for c in cats.columns if "uniqueid" in c.replace("_", "") or c in ("uid", "unique_id", "word"))
    ccol = next(c for c in cats.columns if "categ" in c)
    cat_of = cats.groupby(ucol)[ccol].first().to_dict()
    concept = [u.replace("_", " ").rstrip("0123456789").strip() for u in uids]
    meta = pd.DataFrame(dict(uid=uids, concept=concept, category=[cat_of.get(u, "other") for u in uids]))
    log(f"things: {len(meta)} concepts, SPoSE {spose.shape}, {meta.category.nunique()} categories")
    model, proc, dev = load_model(model_name)
    emb = embed_texts([f"a photo of a {c}." for c in concept], model, proc, dev)
    OUT.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT / "things_meta.parquet", index=False)
    np.save(OUT / "things_spose.npy", spose.astype(np.float32))
    np.save(OUT / "things_model_emb.npy", emb.astype(np.float16))
    update_manifest(text_model=model_name, things_n=len(meta))
    log(f"things: wrote meta + spose + text embeddings {emb.shape}")


# ─── Zebra finch ────────────────────────────────────────────────────────

FINCH_RE = re.compile(r"^(?P<bird>[A-Za-z]+\d+)_(?P<date>\d+)-(?P<type>[A-Za-z]+)-(?P<idx>\d+)")


def build_finch(per_type: int = 160, sr: int = 22050) -> None:
    import librosa
    from huggingface_hub import HfApi, hf_hub_download

    files = [s.rfilename for s in HfApi().dataset_info(FINCH_REPO).siblings if s.rfilename.startswith("zebra_finch/")]
    recs = []
    for f in files:
        m = FINCH_RE.match(Path(f).name)
        if m:
            t = m.group("type")
            recs.append(dict(file=f, bird=m.group("bird"), date=m.group("date"), code=t[:2] if t[:2] != "Ag" else "Ag"))
    df = pd.DataFrame(recs)
    df = pd.concat([g.sample(min(per_type, len(g)), random_state=0) for _, g in df.groupby("code")], ignore_index=True)
    log(f"finch: {len(df)} calls across {df.code.nunique()} types, {df.bird.nunique()} birds")

    n_mels, feats, specs, durs, keep = 48, [], [], [], []
    for i, r in df.iterrows():
        try:
            path = hf_hub_download(FINCH_REPO, r.file, repo_type="dataset")
            y, _ = librosa.load(path, sr=sr, mono=True)
        except Exception as exc:
            log(f"finch: skip {r.file} ({exc})")
            continue
        if len(y) < sr * 0.02:
            continue
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=300, fmax=10000, hop_length=128), ref=np.max)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=128)[0]
        flat = librosa.feature.spectral_flatness(y=y, hop_length=128)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=128)[0]
        dur = len(y) / sr
        f = np.concatenate([S.mean(1), S.std(1), [cent.mean(), cent.std(), flat.mean(), zcr.mean(), dur]])
        feats.append(f)
        img = librosa.util.fix_length(S, size=96, axis=1)[:, :96]
        img = (np.clip((img + 80) / 80, 0, 1) * 255).astype(np.uint8)
        specs.append(img[::-1][::2, ::4])
        durs.append(dur)
        keep.append(i)
        if len(keep) % 200 == 0:
            log(f"finch: {len(keep)} calls processed")
    meta = df.loc[keep].reset_index(drop=True)
    meta["duration"] = durs
    from ml4hl.data import FINCH_CALLTYPES

    meta["calltype"] = meta.code.map(FINCH_CALLTYPES).fillna(meta.code)
    names = [f"mel{i}_mean" for i in range(n_mels)] + [f"mel{i}_sd" for i in range(n_mels)] + \
            ["centroid_mean", "centroid_sd", "flatness", "zero_crossings", "duration"]
    OUT.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT / "finch_meta.parquet", index=False)
    np.save(OUT / "finch_feats.npy", np.stack(feats).astype(np.float32))
    np.save(OUT / "finch_specs.npy", np.stack(specs))
    (OUT / "finch_feature_names.json").write_text(json.dumps(names))
    update_manifest(finch_n=len(meta), finch_types=sorted(meta.calltype.unique().tolist()))
    log(f"finch: wrote {len(meta)} calls, feats {np.stack(feats).shape}")


# ─── Audience-aligned domain tables ─────────────────────────────────────

WORDBANK_URL = "https://raw.githubusercontent.com/langcog/wordbank-book/master/data/_common/admins.feather"
IHDP_FULL_URL = "https://raw.githubusercontent.com/py-why/EconML/main/econml/data/ihdp/example_full.csv"
NHS_BASE_URL = "https://raw.githubusercontent.com/themusiclab/nhs/master/data/nhs/"
IDS_BASE_URL = "https://raw.githubusercontent.com/themusiclab/infant-speech-song/main/data/"
MUSIC_EMO_REPO = "baobaoh/13-dimensions-music-emotions"
RDATA_BASE_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/"


def build_wordbank() -> None:
    log("wordbank: downloading MB-CDI administrations from langcog/wordbank-book")
    raw = pd.read_feather(io.BytesIO(_fetch(WORDBANK_URL)))
    eng = raw[raw["language"].isin(["English (American)", "English (British)"])].copy()
    eng = eng[(eng["age"] >= 8) & (eng["age"] <= 30) & eng["production"].notna()].copy()
    # Normalize words produced onto the 680-word WS scale (WG max is 396 words; WS is 680)
    eng["words_produced"] = eng["production"].astype(float)
    eng["words_understood"] = eng["comprehension"].astype(float)
    eng["vocab_share"] = np.where(eng["form"] == "WG", eng["words_produced"] / 396.0, eng["words_produced"] / 680.0).clip(0, 1)
    keep_cols = ["data_id", "language", "form", "age", "words_produced", "words_understood", "vocab_share", "sex", "birth_order", "mom_ed"]
    out = eng[keep_cols].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "wordbank_cdi.parquet", index=False)
    update_manifest(wordbank_n=len(out))
    log(f"wordbank: wrote {len(out)} rows to wordbank_cdi.parquet")


def build_perinatal() -> None:
    log("perinatal: downloading longitudinal IHDP perinatal & child development cohort")
    raw = pd.read_csv(io.BytesIO(_fetch(IHDP_FULL_URL)))
    # Construct a standardized perinatal stress & adversity index from prenatal/birth risk factors:
    # lower neonatal health index, prematurity (38 - ga.anal), low birthweight, prenatal smoking/alcohol, late prenatal care, low maternal education
    z = lambda s: (s - s.mean()) / (s.std() or 1.0)
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
    df = df.dropna(subset=["vocab_96m", "reading_96m", "externalizing_96m", "internalizing_96m"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "perinatal_dev.parquet", index=False)
    update_manifest(perinatal_n=len(df))
    log(f"perinatal: wrote {len(df)} rows to perinatal_dev.parquet")


def build_musiclab() -> None:
    log("musiclab: downloading Natural History of Song (NHS) & Infant-Directed Vocalization corpora")
    meta = pd.read_csv(io.BytesIO(_fetch(NHS_BASE_URL + "NHSDiscography_Metadata.csv")))
    meta["song_id"] = meta["song"].str.extract(r"(\d+)").astype(int)
    naive = pd.read_csv(io.BytesIO(_fetch(NHS_BASE_URL + "NHSDiscography_NaiveListeners.csv"))).rename(columns={"song": "song_id"})
    trans = pd.read_csv(io.BytesIO(_fetch(NHS_BASE_URL + "NHSDiscography_TranscriptionFeatures.csv"))).rename(columns={"song": "song_id"})
    nhs = meta[["song_id", "song", "type", "culture", "id_nhs", "nhs_region", "nhs_subsistence", "latitude", "longitude"]].merge(
        naive, on="song_id"
    ).merge(
        trans[["song_id", "duration", "mean_interval", "modal_interval_prevalence", "amount_of_arpeggiation",
               "stepwise_motion", "melodic_thirds", "direction_of_motion", "duration_of_melodic_arcs",
               "size_of_melodic_arcs", "modal_pitch_prev", "rel_strength_top_pitches"]],
        on="song_id",
    )

    ac = pd.read_csv(io.BytesIO(_fetch(IDS_BASE_URL + "acoustics-rawAudio.csv")))
    st = pd.read_csv(io.BytesIO(_fetch(IDS_BASE_URL + "stimuli-info.csv")))
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

    OUT.mkdir(parents=True, exist_ok=True)
    nhs.to_parquet(OUT / "musiclab_nhs.parquet", index=False)
    ids_df.to_parquet(OUT / "musiclab_ids.parquet", index=False)
    update_manifest(musiclab_nhs_n=len(nhs), musiclab_ids_n=len(ids_df))
    log(f"musiclab: wrote NHS ({len(nhs)} songs) and IDS ({len(ids_df)} vocalizations)")


def build_music_emotions() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    log("music-emotions: reading 28 emotion dimensions from baobaoh/13-dimensions-music-emotions")
    fs = HfFileSystem()
    pf = pq.ParquetFile(fs.open(f"datasets/{MUSIC_EMO_REPO}/data/train-00000-of-00001.parquet"))
    cols = [c for c in pf.schema.names if c != "audio"]
    df = pf.read(columns=cols).to_pandas()
    # Also compute each clip's primary emotion among the 13 core dimensions (Cowen et al. 2020)
    core13 = [
        "amusing", "angry", "anxious/tense", "awe-inspiring/amazing", "calm/relaxing/serene",
        "compassionate/sympathetic", "dreamy", "eerie/mysterious", "energizing/pump-up",
        "erotic/desirous", "joyful/cheerful", "sad/depressing", "scary/fearful", "triumphant/heroic",
    ]
    core13 = [c for c in core13 if c in df.columns][:13]
    df["top_emotion"] = df[core13].idxmax(axis=1)
    # Group into 4 broad clinical/music-therapy affective quadrants for classification/clustering demos
    quad_map = {
        "calm/relaxing/serene": "soothing / serene",
        "dreamy": "soothing / serene",
        "compassionate/sympathetic": "soothing / serene",
        "joyful/cheerful": "uplifting / joyful",
        "amusing": "uplifting / joyful",
        "energizing/pump-up": "energizing / triumphant",
        "triumphant/heroic": "energizing / triumphant",
        "awe-inspiring/amazing": "energizing / triumphant",
        "anxious/tense": "tense / melancholic",
        "scary/fearful": "tense / melancholic",
        "eerie/mysterious": "tense / melancholic",
        "sad/depressing": "tense / melancholic",
        "angry": "tense / melancholic",
    }
    df["therapy_quadrant"] = df["top_emotion"].map(quad_map).fillna("soothing / serene")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "music_emotions.parquet", index=False)
    update_manifest(music_emotions_n=len(df), music_emotions_dims=len(cols))
    log(f"music-emotions: wrote {len(df)} rows x {len(df.columns)} cols to music_emotions.parquet")


def build_household_survey() -> None:
    log("household-survey: downloading World Bank LSMS household health & expenditure tables")
    hh = pd.read_csv(io.BytesIO(_fetch(RDATA_BASE_URL + "camerondata/vietnam_hh.csv")))
    lss = pd.read_csv(io.BytesIO(_fetch(RDATA_BASE_URL + "camerondata/vietnamlss.csv")))
    ind = pd.read_csv(io.BytesIO(_fetch(RDATA_BASE_URL + "camerondata/vietnam_ind.csv")))

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

    df = pd.DataFrame({
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

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "household_survey.parquet", index=False)
    update_manifest(household_survey_n=len(df))
    log(f"household-survey: wrote {len(df)} households across {df.commune.nunique()} communes")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "what",
        choices=[
            "all", "audience", "dollar-street", "things", "finch",
            "wordbank", "perinatal", "musiclab", "music-emotions", "household-survey",
        ],
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="image-text model for embeddings")
    ap.add_argument("--shards-per-topic", type=int, default=0, help="0 = every shard (one row group each)")
    ap.add_argument("--finch-per-type", type=int, default=160)
    a = ap.parse_args()
    if a.what in ("all", "audience", "wordbank"):
        build_wordbank()
    if a.what in ("all", "audience", "perinatal"):
        build_perinatal()
    if a.what in ("all", "audience", "musiclab"):
        build_musiclab()
    if a.what in ("all", "audience", "music-emotions"):
        build_music_emotions()
    if a.what in ("all", "audience", "household-survey"):
        build_household_survey()
    if a.what in ("all", "things"):
        build_things(a.model)
    if a.what in ("all", "finch"):
        build_finch(a.finch_per_type)
    if a.what in ("all", "dollar-street"):
        build_dollar_street(a.model, a.shards_per_topic)


if __name__ == "__main__":
    main()
