"""Pipeline reproduzível para classificação de recall em mamografias.

O rótulo é definido no nível do exame (AccessionNumber). As imagens de todas
as incidências são classificadas individualmente e as probabilidades são
agregadas por média para produzir uma predição por exame.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import struct
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import timm
import torch
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


BINARY_ID2LABEL = {0: "no_recall", 1: "recall"}
BINARY_LABEL2ID = {value: key for key, value in BINARY_ID2LABEL.items()}
REQUIRED_COLUMNS = {"AccessionNumber", "PatientID", "target"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Revisões imutáveis consultadas no Hugging Face Hub em 2026-09-07.
# Todos os modelos usam implementações nativas do Transformers; código remoto
# não é executado.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "resnet50_local": {
        "source": "resnet50_mammo",
        "revision": None,
        "license": "não documentada no checkpoint local",
        "pretraining": "não documentado no checkpoint local",
        "architecture": "ResNet-50",
        "expected_resolution": 224,
        "safe_weights": True,
        "limitations": "A procedência dos pesos locais não está documentada.",
    },
    "efficientnet_b0": {
        "source": "google/efficientnet-b0",
        "revision": "3d8d75d0812b65064f0151b456f869c6344bf20c",
        "license": "apache-2.0",
        "pretraining": "ImageNet-1k",
        "architecture": "EfficientNet-B0",
        "expected_resolution": 224,
        "safe_weights": False,
        "limitations": (
            "Pré-treinamento em imagens naturais; o repositório fixado oferece "
            "pesos PyTorch, mas não safetensors."
        ),
    },
    "convnext_tiny": {
        "source": "facebook/convnext-tiny-224",
        "revision": "6166b7613034066690a621d8bf25ffdf181a34f0",
        "license": "apache-2.0",
        "pretraining": "ImageNet-1k",
        "architecture": "ConvNeXT-Tiny",
        "expected_resolution": 224,
        "safe_weights": False,
        "limitations": (
            "Pré-treinamento em imagens naturais e redução para 224x224; "
            "achados mamográficos pequenos podem perder resolução."
        ),
    },
    "vgg16": {
        "source": "timm/vgg16.tv_in1k",
        "revision": "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f",
        "license": "bsd-3-clause",
        "pretraining": "ImageNet-1k (pesos originais do torchvision)",
        "architecture": "VGG16",
        "expected_resolution": 224,
        "safe_weights": True,
        "limitations": (
            "Pré-treinamento em imagens naturais, redução para 224x224 e cerca "
            "de 138 milhões de parâmetros; o classificador denso aumenta muito "
            "o custo de memória e o risco de overfitting."
        ),
    },
    "densenet121": {
        "source": "timm/densenet121.ra_in1k",
        "revision": "92007b6200e0b4a4fe68cb4e3947022a928aaaae",
        "license": "apache-2.0",
        "pretraining": "ImageNet-1k (receita RandAugment do timm)",
        "architecture": "DenseNet-121",
        "expected_resolution": 224,
        "safe_weights": True,
        "limitations": (
            "Pré-treinamento em imagens naturais e redução para 224x224; "
            "achados mamográficos pequenos podem perder resolução."
        ),
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    """Captura RNGs em tipos aceitos por ``torch.load(weights_only=True)``."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def canonical_id(value: Any) -> str:
    """Normaliza identificadores lidos como texto, inteiro ou float."""
    if pd.isna(value):
        raise ValueError("Identificador ausente.")
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Identificador inválido: {value!r}")
    return text.lstrip("0") or "0"


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        dtype={"AccessionNumber": "string", "PatientID": "string"},
    )
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} não contém as colunas obrigatórias: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()
    for column in ("AccessionNumber", "PatientID"):
        df[column] = df[column].map(canonical_id)

    df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)
    invalid_targets = sorted(set(df["target"]) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Rótulos fora do domínio binário: {invalid_targets}")

    conflicting = df.groupby("AccessionNumber")["target"].nunique()
    if (conflicting > 1).any():
        examples = conflicting[conflicting > 1].index.tolist()[:5]
        raise ValueError(f"Exames com rótulos conflitantes: {examples}")

    return df


def discover_manifest(
    labels: pd.DataFrame,
    image_root: str | Path,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Relaciona imagens a exames pelo nome da pasta (AccessionNumber)."""
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_root}")

    exam_rows = (
        labels.sort_values("AccessionNumber")
        .drop_duplicates("AccessionNumber")
        .set_index("AccessionNumber")
    )
    records: list[dict[str, Any]] = []
    unknown_directories: list[str] = []

    for exam_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        try:
            accession = canonical_id(exam_dir.name)
        except ValueError:
            unknown_directories.append(exam_dir.name)
            continue
        if accession not in exam_rows.index:
            unknown_directories.append(exam_dir.name)
            continue

        row = exam_rows.loc[accession]
        for image_path in sorted(exam_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(
                    {
                        "AccessionNumber": accession,
                        "PatientID": canonical_id(row["PatientID"]),
                        "target": int(row["target"]),
                        "image_path": str(image_path.resolve()),
                    }
                )

    if strict and unknown_directories:
        preview = ", ".join(unknown_directories[:5])
        raise ValueError(
            f"{len(unknown_directories)} pastas não correspondem ao CSV "
            f"(exemplos: {preview})."
        )
    if not records:
        raise RuntimeError(f"Nenhuma imagem válida encontrada em {image_root}.")

    return pd.DataFrame.from_records(records)


def limit_exams(
    manifest: pd.DataFrame,
    max_exams: int | None,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    if max_exams is None:
        return manifest.reset_index(drop=True)
    if max_exams <= 0:
        raise ValueError("--max-exams deve ser positivo.")

    exams = manifest[["AccessionNumber", "target"]].drop_duplicates()
    if len(exams) <= max_exams:
        return manifest.reset_index(drop=True)

    classes = sorted(exams["target"].unique())
    if max_exams >= len(classes):
        # Reserva uma amostra por classe antes de completar aleatoriamente.
        # Isso impede que o arredondamento elimine a classe minoritária.
        sampled = pd.concat(
            [
                group.sample(n=1, random_state=seed)
                for _, group in exams.groupby("target")
            ],
            ignore_index=True,
        )
        remaining = exams[
            ~exams["AccessionNumber"].isin(sampled["AccessionNumber"])
        ]
        additional = min(max_exams - len(sampled), len(remaining))
        if additional:
            sampled = pd.concat(
                [
                    sampled,
                    remaining.sample(n=additional, random_state=seed),
                ],
                ignore_index=True,
            )
    else:
        sampled = exams.sample(n=max_exams, random_state=seed)
    return manifest[
        manifest["AccessionNumber"].isin(sampled["AccessionNumber"])
    ].reset_index(drop=True)


def check_patient_disjoint(
    train_manifest: pd.DataFrame, validation_manifest: pd.DataFrame
) -> None:
    overlap = set(train_manifest["PatientID"]) & set(validation_manifest["PatientID"])
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            f"Vazamento entre treino e validação: {len(overlap)} pacientes "
            f"aparecem nos dois conjuntos (exemplos: {examples})."
        )


def read_safetensors_shapes(model_path: str | Path) -> dict[str, list[int]]:
    """Lê somente o cabeçalho do arquivo, sem depender de torch/safetensors."""
    model_path = Path(model_path)
    with model_path.open("rb") as file:
        header_length = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_length))
    return {
        key: value["shape"]
        for key, value in header.items()
        if key != "__metadata__" and isinstance(value, dict) and "shape" in value
    }


def audit_project(
    csv_path: str | Path,
    train_images: str | Path,
    validation_images: str | Path,
    model_dir: str | Path,
    *,
    image_sample: int = 100,
) -> dict[str, Any]:
    labels = load_labels(csv_path)
    train_manifest = discover_manifest(labels, train_images)
    validation_manifest = discover_manifest(labels, validation_images)
    check_patient_disjoint(train_manifest, validation_manifest)

    def summarize(manifest: pd.DataFrame) -> dict[str, Any]:
        exams = manifest.drop_duplicates("AccessionNumber")
        counts = manifest.groupby("AccessionNumber").size()
        return {
            "exams": int(exams["AccessionNumber"].nunique()),
            "patients": int(exams["PatientID"].nunique()),
            "images": int(len(manifest)),
            "positive_exams": int(exams["target"].sum()),
            "positive_rate": float(exams["target"].mean()),
            "images_per_exam_min": int(counts.min()),
            "images_per_exam_max": int(counts.max()),
            "images_per_exam_mean": float(counts.mean()),
        }

    sampled_paths = pd.concat(
        [
            train_manifest["image_path"].head(image_sample),
            validation_manifest["image_path"].head(image_sample),
        ],
        ignore_index=True,
    )
    unreadable: list[str] = []
    black_images: list[str] = []
    dimensions: dict[str, int] = {}
    for path_text in sampled_paths:
        try:
            with Image.open(path_text) as image:
                image.verify()
            with Image.open(path_text) as image:
                gray = image.convert("L")
                dimensions[f"{gray.width}x{gray.height}"] = (
                    dimensions.get(f"{gray.width}x{gray.height}", 0) + 1
                )
                if gray.getextrema() == (0, 0):
                    black_images.append(path_text)
        except Exception:
            unreadable.append(path_text)

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Modelo incompleto em {model_dir}: config.json e "
            "model.safetensors são obrigatórios."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    shapes = read_safetensors_shapes(weights_path)
    classifier_shapes = {
        name: shape for name, shape in shapes.items() if "classifier" in name
    }

    id2label = config.get("id2label", {})
    if len(id2label) != 2:
        raise ValueError(
            f"Configuração do modelo declara {len(id2label)} classes; eram esperadas 2."
        )
    if not any(shape and shape[0] == 2 for shape in classifier_shapes.values()):
        raise ValueError("O checkpoint não contém um classificador binário.")
    if unreadable:
        raise ValueError(f"{len(unreadable)} imagens da amostra não puderam ser lidas.")
    if black_images:
        raise ValueError(
            f"{len(black_images)} imagens totalmente pretas encontradas na amostra."
        )

    laterality_table = pd.crosstab(labels.get("Laterality"), labels["target"])
    laterality_determines_target = False
    if "Laterality" in labels:
        laterality_determines_target = bool(
            labels.groupby("Laterality")["target"].nunique().max() == 1
        )

    return {
        "csv": {
            "rows": int(len(labels)),
            "exams": int(labels["AccessionNumber"].nunique()),
            "patients": int(labels["PatientID"].nunique()),
            "target_counts": {
                str(key): int(value)
                for key, value in labels["target"].value_counts().sort_index().items()
            },
            "laterality_determines_target": laterality_determines_target,
            "laterality_by_target": {
                str(index): {
                    str(column): int(value)
                    for column, value in row.items()
                    if int(value) != 0
                }
                for index, row in laterality_table.iterrows()
            },
        },
        "train": summarize(train_manifest),
        "validation": summarize(validation_manifest),
        "patient_overlap": 0,
        "image_sample": {
            "checked": int(len(sampled_paths)),
            "dimensions": dimensions,
            "unreadable": 0,
            "black": 0,
        },
        "model": {
            "directory": str(model_dir.resolve()),
            "labels": id2label,
            "classifier_shapes": classifier_shapes,
        },
    }


class MammographyImageDataset:
    """Dataset definido sem importar torch no modo de auditoria."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        processor: Any,
        *,
        augment: bool,
    ) -> None:

        self.manifest = manifest.reset_index(drop=True)
        self.processor = processor
        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomRotation(degrees=5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:

        row = self.manifest.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        pixel_values = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(int(row["target"]), dtype=torch.long),
            "accession": str(row["AccessionNumber"]),
            "patient": str(row["PatientID"]),
        }


def load_processor_and_model(
    model_source: str | Path,
    *,
    for_training: bool,
    device: Any,
    revision: str | None = None,
    use_safetensors: bool | None = None,
) -> tuple[Any, Any]:

    model_source = str(model_source)
    processor = AutoImageProcessor.from_pretrained(
        model_source,
        revision=revision,
        use_fast=False,
        trust_remote_code=False,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_source,
        revision=revision,
        num_labels=2,
        id2label=BINARY_ID2LABEL,
        label2id=BINARY_LABEL2ID,
        ignore_mismatched_sizes=for_training,
        trust_remote_code=False,
        use_safetensors=use_safetensors,
    )
    if model.config.num_labels != 2:
        raise RuntimeError(
            f"O modelo carregado possui {model.config.num_labels} classes, não 2."
        )
    return processor, model.to(device)


def configure_processor_image_size(processor: Any, image_size: int) -> None:
    """Configura e reconstrói transforms quando o processador é do timm."""
    if image_size <= 0:
        raise ValueError("--image-size deve ser positivo.")
    if hasattr(processor, "data_config"):

        data_config = dict(processor.data_config)
        input_size = data_config.get("input_size", (3, image_size, image_size))
        channels = int(input_size[0])
        data_config["input_size"] = (channels, image_size, image_size)
        processor.data_config = data_config
        processor.val_transforms = timm.data.create_transform(
            **data_config, is_training=False
        )
        processor.train_transforms = timm.data.create_transform(
            **data_config, is_training=True
        )
        processor._not_supports_tensor_input = any(
            transform.__class__.__name__ == "ToTensor"
            for transform in processor.val_transforms.transforms
        )
    else:
        processor.size = {"height": image_size, "width": image_size}


def processor_metadata(processor: Any) -> dict[str, Any]:
    data_config = getattr(processor, "data_config", {})
    return {
        "type": type(processor).__name__,
        "mean": getattr(processor, "image_mean", None)
        or data_config.get("mean"),
        "std": getattr(processor, "image_std", None)
        or data_config.get("std"),
        "size": getattr(processor, "size", None)
        or data_config.get("input_size"),
        "interpolation": data_config.get("interpolation"),
        "crop_pct": data_config.get("crop_pct"),
    }


def save_processor_artifact(processor: Any, destination: Path) -> None:
    """Persiste ao menos a configuração efetiva dos processadores Timm."""
    if hasattr(processor, "data_config"):
        destination.mkdir(parents=True, exist_ok=True)
        write_json(
            destination / "preprocessor_config.json",
            {
                "image_processor_type": type(processor).__name__,
                "architecture": getattr(processor, "architecture", None),
                "data_config": processor.data_config,
            },
        )
    else:
        processor.save_pretrained(destination)


def make_loader(
    manifest: pd.DataFrame,
    processor: Any,
    *,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    workers: int,
    pin_memory: bool = False,
) -> Any:

    dataset = MammographyImageDataset(manifest, processor, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def predict_manifest(
    model: Any,
    processor: Any,
    manifest: pd.DataFrame,
    *,
    device: Any,
    batch_size: int,
    workers: int,
) -> pd.DataFrame:

    loader = make_loader(
        manifest,
        processor,
        batch_size=batch_size,
        augment=False,
        shuffle=False,
        workers=workers,
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inferência", unit="batch"):
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            labels = batch["label"].cpu().numpy()
            for accession, patient, target, probability in zip(
                batch["accession"],
                batch["patient"],
                labels,
                probabilities,
            ):
                rows.append(
                    {
                        "AccessionNumber": canonical_id(accession),
                        "PatientID": canonical_id(patient),
                        "target": int(target),
                        "probability": float(probability),
                    }
                )

    image_predictions = pd.DataFrame(rows)
    return (
        image_predictions.groupby("AccessionNumber", as_index=False)
        .agg(
            PatientID=("PatientID", "first"),
            target=("target", "first"),
            probability=("probability", "mean"),
            n_images=("probability", "size"),
        )
        .sort_values("AccessionNumber")
        .reset_index(drop=True)
    )


def binary_metrics(predictions: pd.DataFrame, threshold: float) -> dict[str, float]:

    y_true = predictions["target"].to_numpy()
    probabilities = predictions["probability"].to_numpy()
    if len(np.unique(y_true)) < 2:
        raise ValueError("A avaliação requer exemplos das duas classes.")
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(tp / (tp + fp)) if tp + fp else math.nan,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "f1": (
            float(2 * tp / (2 * tp + fp + fn))
            if 2 * tp + fp + fn
            else math.nan
        ),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_operating_threshold(
    predictions: pd.DataFrame,
    *,
    policy: str,
    fixed_threshold: float,
) -> dict[str, Any]:
    """Escolhe o threshold somente na validação, sem consultar o holdout."""

    if policy == "fixed":
        metrics = binary_metrics(predictions, fixed_threshold)
        return {
            "policy": policy,
            "threshold": float(fixed_threshold),
            "selection_partition": "validation",
            "criterion_value": None,
            "validation_metrics": metrics,
        }
    if policy != "youden":
        raise ValueError(f"Política de threshold desconhecida: {policy}")
    y_true = predictions["target"].to_numpy()
    probabilities = predictions["probability"].to_numpy()
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, probabilities
    )
    finite = np.isfinite(thresholds)
    scores = true_positive_rate - false_positive_rate
    candidates = np.flatnonzero(finite)
    if candidates.size == 0:
        raise ValueError("Não foi possível selecionar um threshold finito.")
    best_score = float(np.max(scores[candidates]))
    tied = candidates[np.isclose(scores[candidates], best_score)]
    # Em empate, prioriza maior sensibilidade; depois, o menor threshold.
    best_index = int(
        sorted(
            tied,
            key=lambda index: (
                -float(true_positive_rate[index]),
                float(thresholds[index]),
            ),
        )[0]
    )
    threshold = float(np.clip(thresholds[best_index], 0.0, 1.0))
    return {
        "policy": policy,
        "threshold": threshold,
        "selection_partition": "validation",
        "criterion": "maximizar sensibilidade + especificidade - 1",
        "criterion_value": best_score,
        "validation_metrics": binary_metrics(predictions, threshold),
    }


def patient_bootstrap_intervals(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    iterations: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, dict[str, float]]:
    """IC percentil com reamostragem de pacientes e métricas por exame."""
    if iterations <= 0:
        return {}
    patients = predictions["PatientID"].astype(str).unique()
    if len(patients) < 2:
        raise ValueError("Bootstrap requer pelo menos dois pacientes.")
    grouped = {
        patient: frame.copy()
        for patient, frame in predictions.groupby(
            predictions["PatientID"].astype(str), sort=False
        )
    }
    rng = np.random.default_rng(seed)
    metric_names = (
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "f1",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    attempts = 0
    max_attempts = max(iterations * 10, iterations)
    while len(samples["roc_auc"]) < iterations and attempts < max_attempts:
        attempts += 1
        selected = rng.choice(patients, size=len(patients), replace=True)
        frames = []
        for draw_index, patient in enumerate(selected):
            frame = grouped[str(patient)].copy()
            frame["PatientID"] = f"{patient}__bootstrap_{draw_index}"
            frames.append(frame)
        sample = pd.concat(frames, ignore_index=True)
        if sample["target"].nunique() < 2:
            continue
        metrics = binary_metrics(sample, threshold)
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    if len(samples["roc_auc"]) < iterations:
        raise RuntimeError(
            f"Bootstrap gerou somente {len(samples['roc_auc'])}/{iterations} "
            "amostras válidas."
        )
    alpha = (1.0 - confidence) / 2.0
    intervals = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        intervals[name] = {
            "lower": float(np.quantile(array, alpha)),
            "upper": float(np.quantile(array, 1.0 - alpha)),
            "confidence": float(confidence),
            "iterations": int(len(array)),
            "unit": "patient",
        }
    return intervals


def train_model(args: argparse.Namespace) -> None:

    seed_everything(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    labels = load_labels(args.csv)
    train_manifest = limit_exams(
        discover_manifest(labels, args.train_images),
        args.max_train_exams,
        seed=args.seed,
    )
    validation_manifest = limit_exams(
        discover_manifest(labels, args.validation_images),
        args.max_validation_exams,
        seed=args.seed,
    )
    check_patient_disjoint(train_manifest, validation_manifest)
    processor, model = load_processor_and_model(
        args.model,
        for_training=True,
        device=device,
    )
    loader = make_loader(
        train_manifest,
        processor,
        batch_size=args.batch_size,
        augment=True,
        shuffle=True,
        workers=args.workers,
    )

    exam_labels = train_manifest.drop_duplicates("AccessionNumber")["target"]
    positive_rate = float(exam_labels.mean())
    if not 0 < positive_rate < 1:
        raise ValueError("O conjunto de treino precisa conter as duas classes.")
    class_weights = torch.tensor(
        [1.0, (1.0 - positive_rate) / positive_rate],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Época {epoch + 1}/{args.epochs}", unit="batch")
        for step, batch in enumerate(loop, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            loop.set_postfix(loss=running_loss / step)

        predictions = predict_manifest(
            model,
            processor,
            validation_manifest,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        metrics = binary_metrics(predictions, args.threshold)
        print(json.dumps({"epoch": epoch + 1, **metrics}, indent=2))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.id2label = BINARY_ID2LABEL
    model.config.label2id = BINARY_LABEL2ID
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Modelo salvo em {output_dir.resolve()}")


def evaluate_model(args: argparse.Namespace) -> None:

    seed_everything(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    labels = load_labels(args.csv)
    manifest = limit_exams(
        discover_manifest(labels, args.images),
        args.max_exams,
        seed=args.seed,
    )
    processor, model = load_processor_and_model(
        args.model,
        for_training=False,
        device=device,
    )
    predictions = predict_manifest(
        model,
        processor,
        manifest,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    metrics = binary_metrics(predictions, args.threshold)
    print(json.dumps(metrics, indent=2))
    if args.predictions:
        output_path = Path(args.predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Predições salvas em {output_path.resolve()}")


def manifest_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    exams = manifest.drop_duplicates("AccessionNumber")
    class_counts = exams["target"].value_counts().sort_index()
    return {
        "exams": int(len(exams)),
        "patients": int(exams["PatientID"].nunique()),
        "images": int(len(manifest)),
        "class_counts": {
            str(label): int(count) for label, count in class_counts.items()
        },
        "positive_rate": float(exams["target"].mean()),
    }


def stratified_patient_split(
    manifest: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa exames por paciente, preservando aproximadamente a prevalência."""

    if not 0.1 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction deve estar entre 0.1 e 0.5.")
    exams = manifest[
        ["AccessionNumber", "PatientID", "target"]
    ].drop_duplicates("AccessionNumber")
    n_splits = max(2, round(1.0 / validation_fraction))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    train_indices, validation_indices = next(
        splitter.split(
            exams,
            y=exams["target"],
            groups=exams["PatientID"],
        )
    )
    train_accessions = set(exams.iloc[train_indices]["AccessionNumber"])
    validation_accessions = set(
        exams.iloc[validation_indices]["AccessionNumber"]
    )
    train = manifest[
        manifest["AccessionNumber"].isin(train_accessions)
    ].reset_index(drop=True)
    validation = manifest[
        manifest["AccessionNumber"].isin(validation_accessions)
    ].reset_index(drop=True)
    check_patient_disjoint(train, validation)
    for name, partition in (("treino", train), ("validação", validation)):
        if partition["target"].nunique() != 2:
            raise ValueError(f"A partição de {name} não contém as duas classes.")
    return train, validation


def runtime_metadata() -> dict[str, Any]:

    packages = {}
    for name in (
        "numpy",
        "pandas",
        "Pillow",
        "scikit-learn",
        "transformers",
        "torchvision",
        "matplotlib",
        "seaborn",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
    }


def json_safe(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(value) for value in payload]
    if isinstance(payload, (np.integer, np.floating)):
        payload = payload.item()
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def log_line(path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def aggregate_image_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    image_predictions = pd.DataFrame(rows)
    return (
        image_predictions.groupby("AccessionNumber", as_index=False)
        .agg(
            PatientID=("PatientID", "first"),
            target=("target", "first"),
            probability=("probability", "mean"),
            n_images=("probability", "size"),
        )
        .sort_values("AccessionNumber")
        .reset_index(drop=True)
    )


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: Any,
    threshold: float,
    optimizer: Any | None,
    scaler: Any | None,
    accumulation_steps: int,
    gradient_clip: float | None,
    mixed_precision: bool,
    description: str,
) -> tuple[dict[str, float], pd.DataFrame]:

    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    sample_count = 0
    loop = tqdm(loader, desc=description, unit="batch")
    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for step, batch in enumerate(loop, start=1):
            images = batch["pixel_values"].to(
                device, non_blocking=device.type == "cuda"
            )
            targets = batch["label"].to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(pixel_values=images).logits
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise RuntimeError(
                        f"Logits incompatíveis: esperado N x 2, recebido {tuple(logits.shape)}."
                    )
                raw_loss = criterion(logits, targets)
            if not torch.isfinite(raw_loss):
                raise FloatingPointError(
                    f"Loss NaN/Inf detectada em {description}, batch {step}."
                )
            if training:
                scaled_loss = raw_loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = step % accumulation_steps == 0 or step == len(loader)
                if should_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    finite_gradients = all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    )
                    if not finite_gradients:
                        raise FloatingPointError(
                            f"Gradiente NaN/Inf detectado em {description}, batch {step}."
                        )
                    if gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip
                        )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = int(targets.shape[0])
            loss_sum += float(raw_loss.detach().cpu()) * batch_size
            sample_count += batch_size
            probabilities = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            target_values = targets.detach().cpu().numpy()
            for accession, patient, target, probability in zip(
                batch["accession"],
                batch["patient"],
                target_values,
                probabilities,
            ):
                rows.append(
                    {
                        "AccessionNumber": canonical_id(accession),
                        "PatientID": canonical_id(patient),
                        "target": int(target),
                        "probability": float(probability),
                    }
                )
            loop.set_postfix(loss=loss_sum / max(sample_count, 1))

    predictions = aggregate_image_rows(rows)
    metrics = binary_metrics(predictions, threshold)
    metrics["loss"] = float(loss_sum / sample_count)
    return metrics, predictions


def smoke_test_training_step(
    model: Any,
    loader: Any,
    criterion: Any,
    *,
    device: Any,
) -> dict[str, Any]:
    """Executa forward, backward e optimizer.step e restaura os pesos originais."""

    batch = next(iter(loader))
    snapshot = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    images = batch["pixel_values"].to(device)
    targets = batch["label"].to(device)
    logits = model(pixel_values=images).logits
    loss = criterion(logits, targets)
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(loss):
        raise RuntimeError("Falha no smoke test de forward/loss.")
    loss.backward()
    if not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("Gradientes inválidos no smoke test.")
    optimizer.step()
    model.load_state_dict(snapshot)
    optimizer.zero_grad(set_to_none=True)
    return {
        "input_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "target_dtype": str(targets.dtype),
        "logit_shape": list(logits.shape),
        "logit_dtype": str(logits.dtype),
        "loss": float(loss.detach().cpu()),
        "optimizer_step": "executado e pesos restaurados",
    }


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:

    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def plot_training_artifacts(
    history: pd.DataFrame,
    validation_metrics: dict[str, float],
    output_dir: Path,
) -> None:

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss"], marker="o", label="Treino")
    axis.plot(
        history["epoch"], history["validation_loss"], marker="o", label="Validação"
    )
    axis.set(title="Loss por época", xlabel="Época", ylabel="Cross-entropy ponderada")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    for split, line_style in (("train", "--"), ("validation", "-")):
        for metric in ("accuracy", "sensitivity", "specificity", "f1"):
            axis.plot(
                history["epoch"],
                history[f"{split}_{metric}"],
                linestyle=line_style,
                marker="o",
                label=f"{split} {metric}",
            )
    axis.set(
        title="Métricas por época no nível do exame",
        xlabel="Época",
        ylabel="Métrica",
        ylim=(-0.03, 1.03),
    )
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "metrics_curves.png", dpi=160)
    plt.close(figure)

    matrix = np.array(
        [
            [validation_metrics["tn"], validation_metrics["fp"]],
            [validation_metrics["fn"], validation_metrics["tp"]],
        ]
    )
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["no_recall", "recall"],
        yticklabels=["no_recall", "recall"],
        ax=axis,
    )
    axis.set(
        title="Matriz de confusão — melhor checkpoint (validação)",
        xlabel="Predito",
        ylabel="Real",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def run_single_benchmark(
    *,
    model_name: str,
    model_spec: dict[str, Any],
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    run_dir: Path,
    args: argparse.Namespace,
    device: Any,
    versions: dict[str, Any],
) -> dict[str, Any]:

    model_dir = run_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=False)
    log_path = model_dir / "run.log"
    log_line(log_path, f"Carregando {model_name} de {model_spec['source']}.")
    processor, model = load_processor_and_model(
        model_spec["source"],
        revision=model_spec["revision"],
        for_training=True,
        device=device,
        use_safetensors=True if model_spec["safe_weights"] else None,
    )
    if args.image_size is not None:
        configure_processor_image_size(processor, int(args.image_size))
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metadata = {
        **model_spec,
        "local_name": model_name,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "classifier_change": "cabeça substituída por Linear(..., 2)",
        "fine_tuning": "todos os parâmetros treináveis",
        "normalization": processor_metadata(processor),
    }
    write_json(model_dir / "model_metadata.json", metadata)
    save_processor_artifact(processor, model_dir / "processor")

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_manifest,
        processor,
        batch_size=args.batch_size,
        augment=True,
        shuffle=True,
        workers=args.workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        validation_manifest,
        processor,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_manifest,
        processor,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        workers=args.workers,
        pin_memory=pin_memory,
    )
    exam_labels = train_manifest.drop_duplicates("AccessionNumber")["target"]
    positives = int(exam_labels.sum())
    negatives = int(len(exam_labels) - positives)
    if not positives or not negatives:
        raise ValueError("O treino precisa conter exames das duas classes.")
    class_weights = torch.tensor(
        [1.0, negatives / positives], dtype=torch.float32, device=device
    )
    # Dois logits exigem CrossEntropyLoss. O peso positivo é calculado somente
    # no treino para compensar o desbalanceamento sem consultar validação/teste.
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    smoke = smoke_test_training_step(
        model, train_loader, criterion, device=device
    )
    write_json(model_dir / "smoke_test.json", smoke)
    log_line(log_path, f"Smoke test aprovado: {smoke}.")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    config = {
        "model": model_name,
        "source": model_spec["source"],
        "revision": model_spec["revision"],
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": int(args.image_size),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR por época",
        "loss": "CrossEntropyLoss com peso positivo n_neg/n_pos",
        "positive_class_weight": float(negatives / positives),
        "threshold": args.threshold,
        "threshold_policy": args.threshold_policy,
        "bootstrap_iterations": args.bootstrap_iterations,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "gradient_accumulation_steps": args.accumulation_steps,
        "gradient_clip": args.gradient_clip,
        "mixed_precision_requested": bool(args.mixed_precision),
        "mixed_precision_enabled": amp_enabled,
        "finalize_only": bool(args.finalize_only),
        "partitions": {
            "train": manifest_summary(train_manifest),
            "validation": manifest_summary(validation_manifest),
            "test": manifest_summary(test_manifest),
        },
        "versions": versions,
    }
    start_epoch = 1
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []
    prior_training_seconds = 0.0
    resume_checkpoint_path = (
        Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    )
    if resume_checkpoint_path is not None:
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint de retomada não encontrado: {resume_checkpoint_path}"
            )
        checkpoint = torch.load(
            resume_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        checkpoint_config = checkpoint.get("config", {})
        compatibility_fields = (
            "model",
            "source",
            "revision",
            "seed",
            "batch_size",
            "image_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "threshold",
            "threshold_policy",
            "patience",
            "min_delta",
            "gradient_accumulation_steps",
            "gradient_clip",
            "mixed_precision_enabled",
            "partitions",
        )
        mismatches = [
            field
            for field in compatibility_fields
            if checkpoint_config.get(field) != config.get(field)
        ]
        if mismatches:
            raise ValueError(
                "O checkpoint não é compatível com esta execução nos campos: "
                + ", ".join(mismatches)
            )
        resumed_epoch = int(checkpoint["epoch"])
        if resumed_epoch >= args.epochs:
            raise ValueError(
                f"O checkpoint já está na época {resumed_epoch}; "
                f"--epochs={args.epochs} não deixa épocas para retomar."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = resumed_epoch + 1
        best_epoch = resumed_epoch
        best_loss = float(checkpoint["best_metric"])
        prior_history_path = resume_checkpoint_path.with_name("history.json")
        if not prior_history_path.is_file():
            raise FileNotFoundError(
                f"Histórico associado ao checkpoint não encontrado: {prior_history_path}"
            )
        prior_history = json.loads(prior_history_path.read_text(encoding="utf-8"))
        history_rows = (
            prior_history
            if args.finalize_only
            else [
                row
                for row in prior_history
                if int(row["epoch"]) <= resumed_epoch
            ]
        )
        history_epochs = {int(row["epoch"]) for row in history_rows}
        if not history_rows or (
            resumed_epoch not in history_epochs
            if args.finalize_only
            else int(history_rows[-1]["epoch"]) != resumed_epoch
        ):
            raise ValueError(
                "O histórico não contém a época correspondente ao checkpoint."
            )
        prior_training_seconds = float(
            sum(float(row["duration_seconds"]) for row in history_rows)
        )
        discarded_epochs = (
            []
            if args.finalize_only
            else [
                int(row["epoch"])
                for row in prior_history
                if int(row["epoch"]) > resumed_epoch
            ]
        )
        rng_restored = "rng_state" in checkpoint
        config["resume"] = {
            "checkpoint": str(resume_checkpoint_path),
            "checkpoint_epoch": resumed_epoch,
            "start_epoch": start_epoch,
            "discarded_history_epochs": discarded_epochs,
            "rng_state_restored": rng_restored,
            "note": (
                "Estado aleatório restaurado do checkpoint."
                if rng_restored
                else "Checkpoint legado sem RNG; a trajetória após a retomada pode divergir."
            ),
        }
        checkpoint["config"] = config
        checkpoint["resumed_from"] = str(resume_checkpoint_path)
        atomic_torch_save(checkpoint, model_dir / "best_checkpoint.pth")
        history_frame = pd.DataFrame(history_rows)
        history_frame.to_csv(model_dir / "history.csv", index=False)
        write_json(model_dir / "history.json", history_rows)
        log_line(
            log_path,
            f"Retomando da época {resumed_epoch}; próxima época {start_epoch}. "
            f"Épocas históricas descartadas por não corresponderem ao checkpoint: "
            f"{discarded_epochs or 'nenhuma'}. RNG restaurado: {rng_restored}.",
        )
        if rng_restored:
            restore_rng_state(checkpoint["rng_state"])
    write_json(model_dir / "config.json", config)

    training_started = time.perf_counter()
    epoch_range = (
        range(start_epoch, args.epochs + 1) if not args.finalize_only else ()
    )
    if args.finalize_only:
        log_line(
            log_path,
            "Modo de finalização: treinamento ignorado; executando apenas "
            "validação, teste holdout e bootstrap do checkpoint.",
        )
    for epoch in epoch_range:
        epoch_started = time.perf_counter()
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            threshold=args.threshold,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            gradient_clip=args.gradient_clip,
            mixed_precision=amp_enabled,
            description=f"{model_name} treino {epoch}/{args.epochs}",
        )
        validation_metrics, _ = run_epoch(
            model,
            validation_loader,
            criterion,
            device=device,
            threshold=args.threshold,
            optimizer=None,
            scaler=None,
            accumulation_steps=1,
            gradient_clip=None,
            mixed_precision=amp_enabled,
            description=f"{model_name} validação {epoch}/{args.epochs}",
        )
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": float(time.perf_counter() - epoch_started),
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"validation_{key}": value for key, value in validation_metrics.items()}
        )
        history_rows.append(row)
        history_frame = pd.DataFrame(history_rows)
        history_frame.to_csv(model_dir / "history.csv", index=False)
        write_json(model_dir / "history.json", history_rows)
        log_line(
            log_path,
            f"Época {epoch}: train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_AUROC={validation_metrics['roc_auc']:.4f}, "
            f"val_AUPRC={validation_metrics['average_precision']:.4f}.",
        )
        if validation_metrics["loss"] < best_loss - args.min_delta:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "best_metric": best_loss,
                "monitor": "validation_loss",
                "threshold": args.threshold,
                "config": config,
                "model_metadata": metadata,
                "seed": args.seed,
                "versions": versions,
                "rng_state": capture_rng_state(),
            }
            atomic_torch_save(checkpoint, model_dir / "best_checkpoint.pth")
            log_line(log_path, f"Novo melhor checkpoint na época {epoch}.")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                log_line(log_path, f"Early stopping acionado na época {epoch}.")
                break

    checkpoint = torch.load(
        model_dir / "best_checkpoint.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics_at_fixed_threshold, validation_predictions = run_epoch(
        model,
        validation_loader,
        criterion,
        device=device,
        threshold=float(checkpoint["threshold"]),
        optimizer=None,
        scaler=None,
        accumulation_steps=1,
        gradient_clip=None,
        mixed_precision=amp_enabled,
        description=f"{model_name} melhor validação",
    )
    threshold_selection = select_operating_threshold(
        validation_predictions,
        policy=args.threshold_policy,
        fixed_threshold=float(checkpoint["threshold"]),
    )
    validation_metrics = threshold_selection["validation_metrics"]
    validation_metrics["loss"] = validation_metrics_at_fixed_threshold["loss"]
    write_json(model_dir / "threshold_selection.json", threshold_selection)
    test_metrics, test_predictions = run_epoch(
        model,
        test_loader,
        criterion,
        device=device,
        threshold=float(threshold_selection["threshold"]),
        optimizer=None,
        scaler=None,
        accumulation_steps=1,
        gradient_clip=None,
        mixed_precision=amp_enabled,
        description=f"{model_name} teste holdout",
    )
    test_confidence_intervals = patient_bootstrap_intervals(
        test_predictions,
        threshold=float(threshold_selection["threshold"]),
        iterations=args.bootstrap_iterations,
        seed=args.seed + 10_000,
    )
    write_json(
        model_dir / "test_bootstrap_intervals.json",
        test_confidence_intervals,
    )
    validation_predictions.to_csv(
        model_dir / "validation_predictions.csv", index=False
    )
    test_predictions.to_csv(model_dir / "test_predictions.csv", index=False)
    final_results = {
        "model": model_name,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history_rows)),
        "training_duration_seconds": float(
            prior_training_seconds + time.perf_counter() - training_started
        ),
        "image_size": int(args.image_size),
        "threshold_selection": threshold_selection,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_confidence_intervals": test_confidence_intervals,
    }
    write_json(model_dir / "final_results.json", final_results)
    history_frame = pd.DataFrame(history_rows)
    plot_training_artifacts(history_frame, validation_metrics, model_dir)
    log_line(log_path, f"Execução concluída: {final_results}.")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_results


def benchmark_models(args: argparse.Namespace) -> None:

    seed_everything(args.seed)
    if args.resume_checkpoint and len(args.models) != 1:
        raise ValueError("--resume-checkpoint exige exatamente um modelo em --models.")
    if args.finalize_only and not args.resume_checkpoint:
        raise ValueError("--finalize-only exige --resume-checkpoint.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas o PyTorch instalado não possui CUDA.")
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    unknown_models = sorted(set(args.models) - set(MODEL_REGISTRY))
    if unknown_models:
        raise ValueError(
            f"Modelos desconhecidos: {unknown_models}. Opções: {sorted(MODEL_REGISTRY)}"
        )
    labels = load_labels(args.csv)
    development_manifest = discover_manifest(labels, args.development_images)
    test_manifest = discover_manifest(labels, args.test_images)
    check_patient_disjoint(development_manifest, test_manifest)
    train_manifest, validation_manifest = stratified_patient_split(
        development_manifest,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_manifest = limit_exams(
        train_manifest, args.max_train_exams, seed=args.seed
    )
    validation_manifest = limit_exams(
        validation_manifest, args.max_validation_exams, seed=args.seed + 1
    )
    test_manifest = limit_exams(
        test_manifest, args.max_test_exams, seed=args.seed + 2
    )
    check_patient_disjoint(train_manifest, validation_manifest)
    check_patient_disjoint(train_manifest, test_manifest)
    check_patient_disjoint(validation_manifest, test_manifest)
    for name, manifest in (
        ("treino", train_manifest),
        ("validação", validation_manifest),
        ("teste", test_manifest),
    ):
        if manifest.drop_duplicates("AccessionNumber")["target"].nunique() != 2:
            raise ValueError(f"A partição limitada de {name} perdeu uma classe.")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    versions = runtime_metadata()
    experiment = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "models": args.models,
        "partitions": {
            "train": manifest_summary(train_manifest),
            "validation": manifest_summary(validation_manifest),
            "test": manifest_summary(test_manifest),
        },
        "versions": versions,
        "methodology": (
            "Seleção por loss de validação; teste holdout usado uma única vez após "
            "restaurar o melhor checkpoint; métricas agregadas por exame."
        ),
    }
    write_json(run_dir / "experiment.json", experiment)
    results = []
    for model_name in args.models:
        seed_everything(args.seed)
        results.append(
            run_single_benchmark(
                model_name=model_name,
                model_spec=MODEL_REGISTRY[model_name],
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                test_manifest=test_manifest,
                run_dir=run_dir,
                args=args,
                device=device,
                versions=versions,
            )
        )
    summary_rows = []
    for result in results:
        row = {
            "model": result["model"],
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
        }
        row.update(
            {f"validation_{key}": value for key, value in result["validation"].items()}
        )
        row.update({f"test_{key}": value for key, value in result["test"].items()})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        "validation_roc_auc", ascending=False
    )
    summary.to_csv(run_dir / "benchmark_summary.csv", index=False)
    write_json(run_dir / "benchmark_summary.json", summary_rows)
    print(f"Benchmark salvo em {run_dir.resolve()}")


def dicom_pixels_to_uint8(dataset: Any) -> np.ndarray:
    """Decodifica DICOM sem mascarar erros com imagens pretas."""

    try:
        pixels = dataset.pixel_array
    except Exception as error:
        raise RuntimeError(
            "Não foi possível decodificar PixelData. Instale pylibjpeg, "
            "pylibjpeg-libjpeg e pylibjpeg-openjpeg."
        ) from error

    array = apply_modality_lut(pixels, dataset)
    try:
        array = apply_voi_lut(array, dataset)
    except Exception:
        # Nem todo arquivo possui uma LUT válida; percentis ainda são robustos.
        pass
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 2:
        array = array[0]
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("O DICOM não possui pixels finitos.")
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        raise ValueError("O DICOM não possui faixa dinâmica útil.")
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        array = 1.0 - array
    return np.round(array * 255.0).astype(np.uint8)


def preprocess_dicoms(args: argparse.Namespace) -> None:

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Diretório DICOM não encontrado: {raw_root}")

    failures: list[tuple[str, str]] = []
    converted = 0
    for exam_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        accession = canonical_id(exam_dir.name)
        output_exam = output_root / accession
        output_exam.mkdir(parents=True, exist_ok=True)
        used_names: dict[str, int] = {}
        for dicom_path in sorted(exam_dir.glob("*.dcm")):
            try:
                # Os arquivos fornecidos não possuem o preâmbulo DICM padrão.
                dataset = pydicom.dcmread(dicom_path, force=True)
                image = dicom_pixels_to_uint8(dataset)
                laterality = str(
                    getattr(
                        dataset,
                        "ImageLaterality",
                        getattr(dataset, "Laterality", "U"),
                    )
                ).upper()
                view = str(getattr(dataset, "ViewPosition", "UNK")).upper()
                base_name = re.sub(
                    r"[^A-Z0-9_-]",
                    "",
                    f"{int(accession):06d}_{laterality}{view}",
                )
                occurrence = used_names.get(base_name, 0)
                used_names[base_name] = occurrence + 1
                suffix = f"_{occurrence + 1}" if occurrence else ""
                destination = output_exam / f"{base_name}{suffix}.png"
                if not cv2.imwrite(str(destination), image):
                    raise OSError(f"Falha ao escrever {destination}")
                converted += 1
            except Exception as error:
                failures.append((str(dicom_path), str(error)))

    print(f"{converted} imagens convertidas para {output_root.resolve()}.")
    if failures:
        for path, message in failures[:10]:
            print(f"[ERRO] {path}: {message}")
        raise RuntimeError(
            f"{len(failures)} arquivos DICOM falharam; nenhuma imagem preta "
            "foi criada como substituição."
        )


def add_common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", default="DATA_DIR/train.csv")
    parser.add_argument("--model", default="resnet50_mammo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(
        command="audit",
        csv="DATA_DIR/train.csv",
        train_images="proc/train/png/train",
        validation_images="proc/test/png/test",
        model="resnet50_mammo",
        image_sample=100,
    )

    audit_parser = subparsers.add_parser(
        "audit", help="Valida CSV, imagens, separação por paciente e checkpoint."
    )
    audit_parser.add_argument("--csv", default="DATA_DIR/train.csv")
    audit_parser.add_argument("--train-images", default="proc/train/png/train")
    audit_parser.add_argument("--validation-images", default="proc/test/png/test")
    audit_parser.add_argument("--model", default="resnet50_mammo")
    audit_parser.add_argument("--image-sample", type=int, default=100)

    train_parser = subparsers.add_parser("train", help="Treina o classificador.")
    add_common_runtime_arguments(train_parser)
    train_parser.add_argument("--train-images", default="proc/train/png/train")
    train_parser.add_argument(
        "--validation-images", default="proc/test/png/test"
    )
    train_parser.add_argument("--output", default="resnet50_mammo_trained")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--max-train-exams", type=int)
    train_parser.add_argument("--max-validation-exams", type=int)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Avalia e agrega probabilidades por exame."
    )
    add_common_runtime_arguments(evaluate_parser)
    evaluate_parser.add_argument("--images", default="proc/test/png/test")
    evaluate_parser.add_argument("--max-exams", type=int)
    evaluate_parser.add_argument("--predictions")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Treina e compara CNNs com validação interna e teste holdout.",
    )
    benchmark_parser.add_argument("--csv", default="DATA_DIR/train.csv")
    benchmark_parser.add_argument(
        "--development-images", default="proc/train/png/train"
    )
    benchmark_parser.add_argument("--test-images", default="proc/test/png/test")
    benchmark_parser.add_argument(
        "--models",
        nargs="+",
        default=["efficientnet_b0", "convnext_tiny"],
        choices=sorted(MODEL_REGISTRY),
    )
    benchmark_parser.add_argument("--output-root", default="artifacts/benchmarks")
    benchmark_parser.add_argument("--run-id")
    benchmark_parser.add_argument(
        "--resume-checkpoint",
        help=(
            "Retoma modelo, otimizador e scheduler de um best_checkpoint.pth; "
            "a nova execução deve usar outro --run-id."
        ),
    )
    benchmark_parser.add_argument(
        "--finalize-only",
        action="store_true",
        help=(
            "Carrega --resume-checkpoint e executa somente validação, teste "
            "holdout e bootstrap, sem épocas adicionais de treinamento."
        ),
    )
    benchmark_parser.add_argument("--epochs", type=int, default=5)
    benchmark_parser.add_argument("--batch-size", type=int, default=8)
    benchmark_parser.add_argument("--workers", type=int, default=0)
    benchmark_parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--threshold", type=float, default=0.5)
    benchmark_parser.add_argument(
        "--threshold-policy",
        choices=["fixed", "youden"],
        default="fixed",
        help="Seleciona o ponto de corte somente na validação.",
    )
    benchmark_parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Altura e largura fornecidas ao processador do backbone.",
    )
    benchmark_parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=0,
        help="Número de reamostragens por paciente para ICs no teste.",
    )
    benchmark_parser.add_argument("--learning-rate", type=float, default=2e-5)
    benchmark_parser.add_argument("--weight-decay", type=float, default=1e-4)
    benchmark_parser.add_argument("--validation-fraction", type=float, default=0.2)
    benchmark_parser.add_argument("--patience", type=int, default=3)
    benchmark_parser.add_argument("--min-delta", type=float, default=1e-4)
    benchmark_parser.add_argument("--accumulation-steps", type=int, default=1)
    benchmark_parser.add_argument("--gradient-clip", type=float, default=1.0)
    benchmark_parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    benchmark_parser.add_argument("--max-train-exams", type=int)
    benchmark_parser.add_argument("--max-validation-exams", type=int)
    benchmark_parser.add_argument("--max-test-exams", type=int)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Converte DICOM para PNG sem ocultar falhas."
    )
    preprocess_parser.add_argument("--raw-root", default="raw_images_dir/test")
    preprocess_parser.add_argument("--output-root", default="proc/generated/png")

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        report = audit_project(
            args.csv,
            args.train_images,
            args.validation_images,
            args.model,
            image_sample=args.image_sample,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "train":
        train_model(args)
    elif args.command == "evaluate":
        evaluate_model(args)
    elif args.command == "benchmark":
        benchmark_models(args)
    elif args.command == "preprocess":
        preprocess_dicoms(args)
    else:
        raise AssertionError(f"Comando não tratado: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
