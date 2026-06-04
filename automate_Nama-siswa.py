import os
import glob
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ── Konfigurasi logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Konstanta default ─────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH = (
    "/kaggle/input/datasets/yudhaislamisulistya/"
    "plants-type-datasets/split_ttv_dataset_type_of_plants"
)
DEFAULT_OUTPUT_DIR  = "data/processed"
DEFAULT_TARGET_SIZE = (128, 128)
VALID_EXTS          = (".jpg", ".jpeg", ".png", ".bmp")
VALID_SPLITS        = ["train", "test", "valid"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_metadata(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di: '{dataset_path}'\n"
            "Pastikan path sudah benar sebelum menjalankan skrip ini."
        )

    records = []
    for split in sorted(os.listdir(dataset_path)):
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for img_path in glob.glob(os.path.join(cls_path, "*")):
                if img_path.lower().endswith(VALID_EXTS):
                    records.append(
                        {"split": split, "class": cls, "path": img_path}
                    )

    df = pd.DataFrame(records)
    logger.info(
        "Metadata dimuat: %d gambar, %d kelas, split: %s",
        len(df),
        df["class"].nunique(),
        df["split"].unique().tolist(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Label Encoding
# ─────────────────────────────────────────────────────────────────────────────

def build_label_encoder(df_meta: pd.DataFrame) -> dict:    
    classes = sorted(df_meta["class"].unique())
    encoder = {cls: idx for idx, cls in enumerate(classes)}
    logger.info("Label encoder dibuat untuk %d kelas.", len(encoder))
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing per Gambar
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_single_image(
    image_path: str,
    target_size: tuple = (128, 128),
) -> "np.ndarray | None":    
    if not image_path.lower().endswith(VALID_EXTS):
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
        return arr
    except Exception as exc:
        logger.warning("Gagal memproses '%s': %s", image_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Preprocessing Seluruh Dataset
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    df_meta: pd.DataFrame,
    label_encoder: dict,
    target_size: tuple = (128, 128),
) -> dict:    
    processed = defaultdict(lambda: {"X": [], "y": []})
    skipped = 0
    total = len(df_meta)

    for i, (_, row) in enumerate(df_meta.iterrows(), start=1):
        if i % 500 == 0 or i == total:
            logger.info("  Memproses gambar %d / %d ...", i, total)
        arr = preprocess_single_image(row["path"], target_size)
        if arr is None:
            skipped += 1
            continue
        processed[row["split"]]["X"].append(arr)
        processed[row["split"]]["y"].append(label_encoder[row["class"]])

    logger.info("Total gambar dilewati (corrupt/format): %d", skipped)

    # Konversi list → numpy array
    result = {}
    for split, data in processed.items():
        X = np.array(data["X"], dtype=np.float32)
        y = np.array(data["y"], dtype=np.int64)
        result[split] = {"X": X, "y": y}
        logger.info("[%s]  X: %s, y: %s", split, X.shape, y.shape)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Simpan Hasil
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_data(
    processed: dict,
    label_encoder: dict,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for split, data in processed.items():
        x_path = os.path.join(output_dir, f"X_{split}.npy")
        y_path = os.path.join(output_dir, f"y_{split}.npy")
        np.save(x_path, data["X"])
        np.save(y_path, data["y"])
        logger.info("Tersimpan: %s, %s", x_path, y_path)

    # Simpan label encoder sebagai CSV
    enc_path = os.path.join(output_dir, "label_encoder.csv")
    pd.DataFrame(
        list(label_encoder.items()), columns=["class", "label_index"]
    ).to_csv(enc_path, index=False)
    logger.info("Tersimpan: %s", enc_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fungsi Utama (pipeline lengkap)
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str   = DEFAULT_OUTPUT_DIR,
    target_size: tuple = DEFAULT_TARGET_SIZE,
) -> dict:
    logger.info("=" * 60)
    logger.info("Memulai pipeline preprocessing")
    logger.info("  Dataset  : %s", dataset_path)
    logger.info("  Output   : %s", output_dir)
    logger.info("  Ukuran   : %s", target_size)
    logger.info("=" * 60)

    # Langkah 1 – Load metadata
    df_meta = load_dataset_metadata(dataset_path)

    # Langkah 2 – Label encoder
    label_encoder = build_label_encoder(df_meta)

    # Langkah 3 – Preprocessing
    processed = preprocess_dataset(df_meta, label_encoder, target_size)

    # Langkah 4 – Simpan
    save_processed_data(processed, label_encoder, output_dir)

    logger.info("=" * 60)
    logger.info("✅  Preprocessing selesai. Data tersimpan di '%s'", output_dir)
    logger.info("=" * 60)

    return processed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Otomasi preprocessing dataset Plants Type."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path ke folder dataset (default: path Kaggle bawaan).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Direktori output untuk menyimpan array numpy (default: data/processed).",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=list(DEFAULT_TARGET_SIZE),
        metavar=("WIDTH", "HEIGHT"),
        help="Ukuran resize gambar dalam piksel (default: 128 128).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
    )
