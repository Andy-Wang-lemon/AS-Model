from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from emotional_vector_model import MANDATORY_FACTORS, MODEL_FILENAME


CSV_COLUMN_RENAMES = {
    "恨（必有忿）": "恨",
    "恼（必有忿、恨）": "恼",
}
TRANSFORMER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QWEN_EMBEDDING_PROMPT = (
    "Instruct: Given a social media or diary-like text, estimate the intensity "
    "of the 43 Buddhist psychological mental-factor vectors.\nQuery: "
)
MAX_LABEL_VALUE = 0.9
RAG_K = 8
RIDGE_ALPHA = 2.0
HYBRID_WEIGHTS = {"supervised": 0.80, "examples": 0.20, "definitions": 0.0}
ENSEMBLE_COMPONENTS = [
    {
        "name": "qwen3_embedding_0_6b",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "prompt": QWEN_EMBEDDING_PROMPT,
        "batch_size": 4,
        "ridge_alpha": 2.0,
        "hybrid_weights": {"supervised": 0.80, "examples": 0.20, "definitions": 0.0},
        "ensemble_weight": 0.75,
    },
    {
        "name": "minilm_multilingual_l12",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prompt": "",
        "batch_size": 32,
        "ridge_alpha": 8.0,
        "hybrid_weights": {"supervised": 0.70, "examples": 0.30, "definitions": 0.0},
        "ensemble_weight": 0.25,
    },
]
SELECTOR_COMPONENTS = [
    {
        "name": "qwen3_embedding_0_6b_selector",
        "embedding_name": "qwen3_embedding_0_6b",
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "prompt": QWEN_EMBEDDING_PROMPT,
        "ridge_alpha": 0.5,
        "hybrid_weights": {"supervised": 1.0, "examples": 0.0, "definitions": 0.0},
        "ensemble_weight": 0.75,
    },
    {
        "name": "minilm_multilingual_l12_selector",
        "embedding_name": "minilm_multilingual_l12",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prompt": "",
        "ridge_alpha": 32.0,
        "hybrid_weights": {"supervised": 0.95, "examples": 0.05, "definitions": 0.0},
        "ensemble_weight": 0.25,
    },
]
NON_SELECTABLE_OPTIONAL_TERMS: set[str] = set()
SPREADSHEET_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _unit_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _column_index(cell_reference: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_reference.upper())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", SPREADSHEET_NS):
                text_parts = [
                    text_node.text or ""
                    for text_node in item.findall(".//main:t", SPREADSHEET_NS)
                ]
                shared_strings.append("".join(text_parts))

        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(archive.read(sheet_name))
        rows: list[list[str]] = []
        for row_node in root.findall(".//main:sheetData/main:row", SPREADSHEET_NS):
            values: list[str] = []
            for cell in row_node.findall("main:c", SPREADSHEET_NS):
                ref = cell.attrib.get("r", "")
                col_idx = _column_index(ref)
                while len(values) <= col_idx:
                    values.append("")

                cell_type = cell.attrib.get("t", "")
                if cell_type == "s":
                    raw_value = cell.findtext("main:v", default="", namespaces=SPREADSHEET_NS)
                    value = shared_strings[int(raw_value)] if raw_value else ""
                elif cell_type == "inlineStr":
                    text_parts = [
                        text_node.text or ""
                        for text_node in cell.findall(".//main:t", SPREADSHEET_NS)
                    ]
                    value = "".join(text_parts)
                else:
                    value = cell.findtext("main:v", default="", namespaces=SPREADSHEET_NS)
                values[col_idx] = value
            rows.append(values)
    return rows


def read_definitions_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, header=1)
    except ImportError:
        rows = _read_xlsx_first_sheet(path)
        if len(rows) < 2:
            raise ValueError(f"Definition workbook is empty: {path}")
        header = rows[1]
        width = len(header)
        records = []
        for row in rows[2:]:
            padded = row + [""] * max(0, width - len(row))
            records.append(padded[:width])
        return pd.DataFrame(records, columns=header)


def load_definitions(path: Path) -> pd.DataFrame:
    definitions = read_definitions_table(path)
    definitions = definitions.dropna(how="all").copy()
    definitions = definitions.rename(
        columns={
            "English Term": "english_term",
            "Chinese Term": "term",
            "Buddhist Popular Explanation": "buddhist_explanation",
            "Psychological Interpretation": "psychological_interpretation",
            "Keywords and Synonyms": "keywords",
            "Example": "example",
        }
    )
    definitions["term"] = definitions["term"].astype(str).str.strip()
    definitions = definitions[definitions["term"].ne("")]
    if len(definitions) != 43:
        raise ValueError(f"Expected 43 vector definitions, found {len(definitions)}.")
    missing = [term for term in MANDATORY_FACTORS if term not in set(definitions["term"])]
    if missing:
        raise ValueError(f"Mandatory factor(s) missing from definitions: {missing}")
    return definitions


def definition_text(row: pd.Series) -> str:
    parts = [
        row.get("english_term", ""),
        row.get("term", ""),
        row.get("buddhist_explanation", ""),
        row.get("psychological_interpretation", ""),
        row.get("keywords", ""),
        row.get("example", ""),
    ]
    return "\n".join(_clean_text(part) for part in parts if _clean_text(part))


def combined_post_text(row: pd.Series) -> str:
    english = _clean_text(row.get("英文帖子", ""))
    chinese = _clean_text(row.get("中文帖子", ""))
    if english and chinese:
        return f"{english}\n\n中文：{chinese}"
    return english or chinese


def load_training_frame(csv_path: Path, definitions: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw = raw.rename(columns=CSV_COLUMN_RENAMES)
    target_terms = definitions["term"].tolist()
    frame = pd.DataFrame({"text": raw.apply(combined_post_text, axis=1)})
    frame = frame[frame["text"].astype(str).str.len() > 0].copy()
    raw = raw.loc[frame.index].copy()

    for term in target_terms:
        if term not in raw.columns:
            raise ValueError(f"No CSV label column found for defined term: {term}")
        values = pd.to_numeric(raw[term], errors="coerce")
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS:
            values = values.fillna(0.0)
        frame[term] = values.clip(0.0, MAX_LABEL_VALUE)

    grouped = frame.groupby("text", sort=False)[target_terms].mean(numeric_only=True)
    grouped = grouped.reset_index()

    for term in MANDATORY_FACTORS + sorted(NON_SELECTABLE_OPTIONAL_TERMS & set(target_terms)):
        median = frame[term].median(skipna=True)
        if pd.isna(median):
            median = 0.5
        grouped[term] = grouped[term].fillna(float(median))

    grouped[target_terms] = grouped[target_terms].fillna(0.0).clip(0.0, MAX_LABEL_VALUE)
    return grouped


def embedding_device() -> str | None:
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return None
    return None


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    prompt: str = "",
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return _unit_normalize(embeddings)


def definition_prior(query_embeddings: np.ndarray, definition_embeddings: np.ndarray) -> np.ndarray:
    scores = query_embeddings @ definition_embeddings.T
    priors = np.zeros_like(scores, dtype=np.float32)
    for row_idx, row in enumerate(scores):
        floor = float(np.percentile(row, 20))
        ceiling = float(np.percentile(row, 95))
        if ceiling <= floor:
            continue
        scaled = (row - floor) / (ceiling - floor)
        priors[row_idx] = np.clip(scaled, 0.0, 1.0) * MAX_LABEL_VALUE
    return priors


def example_prior(
    query_embeddings: np.ndarray,
    example_embeddings: np.ndarray,
    example_targets: np.ndarray,
    *,
    exclude_self: bool,
) -> np.ndarray:
    similarities = query_embeddings @ example_embeddings.T
    priors = np.zeros((query_embeddings.shape[0], example_targets.shape[1]), dtype=np.float32)
    for row_idx, row in enumerate(similarities):
        row = row.copy()
        if exclude_self and row_idx < len(row):
            row[row_idx] = -np.inf
        top_idx = np.argsort(-row)[:RAG_K]
        top_scores = row[top_idx]
        weights = np.maximum(top_scores, 0.0)
        if float(weights.sum()) == 0.0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        priors[row_idx] = weights @ example_targets[top_idx]
    return priors


def hybrid_predict(
    regressor: Ridge,
    query_embeddings: np.ndarray,
    definition_embeddings: np.ndarray,
    example_embeddings: np.ndarray,
    example_targets: np.ndarray,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    supervised = np.clip(regressor.predict(query_embeddings), 0.0, MAX_LABEL_VALUE)
    examples = example_prior(
        query_embeddings,
        example_embeddings,
        example_targets,
        exclude_self=exclude_self,
    )
    definitions = (
        definition_prior(query_embeddings, definition_embeddings)
        if HYBRID_WEIGHTS["definitions"] > 0
        else np.zeros_like(supervised)
    )
    prediction = (
        HYBRID_WEIGHTS["supervised"] * supervised
        + HYBRID_WEIGHTS["examples"] * examples
        + HYBRID_WEIGHTS["definitions"] * definitions
    )
    return np.clip(prediction, 0.0, MAX_LABEL_VALUE)


def component_predict(
    regressor: Ridge,
    embeddings_train: np.ndarray,
    targets_train: np.ndarray,
    embeddings_eval: np.ndarray,
    *,
    hybrid_weights: dict[str, float],
) -> np.ndarray:
    supervised = np.clip(regressor.predict(embeddings_eval), 0.0, MAX_LABEL_VALUE)
    examples = example_prior(
        embeddings_eval,
        embeddings_train,
        targets_train,
        exclude_self=False,
    )
    prediction = (
        hybrid_weights["supervised"] * supervised
        + hybrid_weights["examples"] * examples
    )
    return np.clip(prediction, 0.0, MAX_LABEL_VALUE)


def ensemble_predict(
    trained_components: list[dict[str, Any]],
    targets_train: np.ndarray,
    *,
    eval_indices: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    predictions = []
    for component in trained_components:
        embeddings = component["embeddings"]
        regressor = Ridge(alpha=component["ridge_alpha"])
        regressor.fit(embeddings[train_indices], targets_train)
        component_scores = component_predict(
            regressor,
            embeddings[train_indices],
            targets_train,
            embeddings[eval_indices],
            hybrid_weights=component["hybrid_weights"],
        )
        predictions.append(component["ensemble_weight"] * component_scores)
    return np.clip(np.sum(predictions, axis=0), 0.0, MAX_LABEL_VALUE)


def selected_eight_mae_with_selector(
    y_true: np.ndarray,
    value_pred: np.ndarray,
    selector_pred: np.ndarray,
    target_terms: list[str],
) -> float:
    term_to_idx = {term: idx for idx, term in enumerate(target_terms)}
    mandatory_indices = [term_to_idx[term] for term in MANDATORY_FACTORS]
    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    errors = []
    for true_row, value_row, selector_row in zip(y_true, value_pred, selector_pred):
        pred_optional_top = np.asarray(optional_indices)[
            np.argsort(-selector_row[optional_indices])[:3]
        ]
        selected = mandatory_indices + pred_optional_top.tolist()
        errors.extend(np.abs(true_row[selected] - value_row[selected]).tolist())
    return float(np.mean(errors))


def selected_eight_within_tolerance_with_selector(
    y_true: np.ndarray,
    value_pred: np.ndarray,
    selector_pred: np.ndarray,
    target_terms: list[str],
    *,
    tolerance: float,
) -> float:
    term_to_idx = {term: idx for idx, term in enumerate(target_terms)}
    mandatory_indices = [term_to_idx[term] for term in MANDATORY_FACTORS]
    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    hits = []
    for true_row, value_row, selector_row in zip(y_true, value_pred, selector_pred):
        pred_optional_top = np.asarray(optional_indices)[
            np.argsort(-selector_row[optional_indices])[:3]
        ]
        selected = mandatory_indices + pred_optional_top.tolist()
        hits.extend((np.abs(true_row[selected] - value_row[selected]) <= tolerance + 1e-8).tolist())
    return float(np.mean(hits))


def top3_optional_recall_with_selector(
    y_true: np.ndarray,
    selector_pred: np.ndarray,
    target_terms: list[str],
) -> float:
    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    scores = []
    for true_row, selector_row in zip(y_true[:, optional_indices], selector_pred[:, optional_indices]):
        positive = np.where(true_row > 0)[0]
        if len(positive) == 0:
            continue
        true_top = set(positive[np.argsort(-true_row[positive])[:3]].tolist())
        pred_top = set(np.argsort(-selector_row)[:3].tolist())
        scores.append(len(true_top & pred_top) / min(3, len(true_top)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def top3_optional_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_terms: list[str],
) -> float:
    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    scores = []
    for true_row, pred_row in zip(y_true[:, optional_indices], y_pred[:, optional_indices]):
        positive = np.where(true_row > 0)[0]
        if len(positive) == 0:
            continue
        true_top = set(positive[np.argsort(-true_row[positive])[:3]].tolist())
        pred_top = set(np.argsort(-pred_row)[:3].tolist())
        scores.append(len(true_top & pred_top) / min(3, len(true_top)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def selected_eight_mae(y_true: np.ndarray, y_pred: np.ndarray, target_terms: list[str]) -> float:
    term_to_idx = {term: idx for idx, term in enumerate(target_terms)}
    mandatory_indices = [term_to_idx[term] for term in MANDATORY_FACTORS]
    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    errors = []
    for true_row, pred_row in zip(y_true, y_pred):
        pred_optional_top = np.asarray(optional_indices)[
            np.argsort(-pred_row[optional_indices])[:3]
        ]
        selected = mandatory_indices + pred_optional_top.tolist()
        errors.extend(np.abs(true_row[selected] - pred_row[selected]).tolist())
    return float(np.mean(errors))


def evaluate(
    embeddings: np.ndarray,
    targets: np.ndarray,
    target_terms: list[str],
    definition_embeddings: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    train_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state=random_state,
    )
    regressor = Ridge(alpha=RIDGE_ALPHA)
    regressor.fit(embeddings[train_idx], targets[train_idx])
    predictions = hybrid_predict(
        regressor,
        embeddings[test_idx],
        definition_embeddings,
        embeddings[train_idx],
        targets[train_idx],
    )
    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    mandatory_indices = [
        idx for idx, term in enumerate(target_terms) if term in MANDATORY_FACTORS
    ]
    rounded_predictions = np.round(predictions, 1)
    selected_eight_rounded_mae = selected_eight_mae(
        targets[test_idx], rounded_predictions, target_terms
    )
    selected_within = selected_eight_within_tolerance(
        targets[test_idx], rounded_predictions, target_terms, tolerance=0.1
    )
    return {
        "rows_total": int(len(targets)),
        "rows_train": int(len(train_idx)),
        "rows_test": int(len(test_idx)),
        "target_dimensions": int(len(target_terms)),
        "mandatory_dimensions": MANDATORY_FACTORS,
        "non_selectable_optional_dimensions": sorted(
            list(NON_SELECTABLE_OPTIONAL_TERMS & set(target_terms))
        ),
        "optional_dimensions": int(len(optional_indices)),
        "ridge_alpha": RIDGE_ALPHA,
        "hybrid_weights": HYBRID_WEIGHTS,
        "mandatory_mae": round(
            float(mean_absolute_error(targets[test_idx][:, mandatory_indices], predictions[:, mandatory_indices])),
            4,
        ),
        "optional_mae": round(
            float(mean_absolute_error(targets[test_idx][:, optional_indices], predictions[:, optional_indices])),
            4,
        ),
        "all_target_mae": round(float(mean_absolute_error(targets[test_idx], predictions)), 4),
        "selected_eight_mae": round(
            selected_eight_mae(targets[test_idx], predictions, target_terms), 4
        ),
        "selected_eight_mae_after_0_1_rounding": round(
            selected_eight_rounded_mae, 4
        ),
        "selected_eight_accuracy_1_minus_mae_over_0_9": round(
            1.0 - selected_eight_rounded_mae / MAX_LABEL_VALUE, 4
        ),
        "selected_eight_within_0_1_accuracy": round(
            selected_within, 4
        ),
        "optional_top3_recall": round(
            top3_optional_recall(targets[test_idx], predictions, target_terms), 4
        ),
    }


def evaluate_ensemble(
    component_embeddings: dict[str, np.ndarray],
    targets: np.ndarray,
    target_terms: list[str],
    random_state: int,
) -> dict[str, Any]:
    train_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state=random_state,
    )
    trained_components = [
        {**component, "embeddings": component_embeddings[component["name"]]}
        for component in ENSEMBLE_COMPONENTS
    ]
    selector_components = [
        {**component, "embeddings": component_embeddings[component["embedding_name"]]}
        for component in SELECTOR_COMPONENTS
    ]
    predictions = ensemble_predict(
        trained_components,
        targets[train_idx],
        eval_indices=test_idx,
        train_indices=train_idx,
    )
    selector_predictions = ensemble_predict(
        selector_components,
        targets[train_idx],
        eval_indices=test_idx,
        train_indices=train_idx,
    )

    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    mandatory_indices = [
        idx for idx, term in enumerate(target_terms) if term in MANDATORY_FACTORS
    ]
    rounded_predictions = np.round(predictions, 1)
    selected_eight_rounded_mae = selected_eight_mae_with_selector(
        targets[test_idx], rounded_predictions, selector_predictions, target_terms
    )
    selected_within = selected_eight_within_tolerance_with_selector(
        targets[test_idx],
        rounded_predictions,
        selector_predictions,
        target_terms,
        tolerance=0.1,
    )
    return {
        "rows_total": int(len(targets)),
        "rows_train": int(len(train_idx)),
        "rows_test": int(len(test_idx)),
        "target_dimensions": int(len(target_terms)),
        "mandatory_dimensions": MANDATORY_FACTORS,
        "non_selectable_optional_dimensions": sorted(
            list(NON_SELECTABLE_OPTIONAL_TERMS & set(target_terms))
        ),
        "optional_dimensions": int(len(optional_indices)),
        "architecture": "qwen3_0_6b_plus_minilm_ensemble",
        "selection_architecture": "separate_optional_top3_selector",
        "ensemble_components": [
            {
                key: value
                for key, value in component.items()
                if key not in {"prompt"}
            }
            for component in ENSEMBLE_COMPONENTS
        ],
        "selector_components": [
            {
                key: value
                for key, value in component.items()
                if key not in {"prompt"}
            }
            for component in SELECTOR_COMPONENTS
        ],
        "mandatory_mae": round(
            float(mean_absolute_error(targets[test_idx][:, mandatory_indices], predictions[:, mandatory_indices])),
            4,
        ),
        "optional_mae": round(
            float(mean_absolute_error(targets[test_idx][:, optional_indices], predictions[:, optional_indices])),
            4,
        ),
        "all_target_mae": round(float(mean_absolute_error(targets[test_idx], predictions)), 4),
        "selected_eight_mae": round(
            selected_eight_mae_with_selector(
                targets[test_idx], predictions, selector_predictions, target_terms
            ),
            4,
        ),
        "selected_eight_mae_after_0_1_rounding": round(
            selected_eight_rounded_mae, 4
        ),
        "selected_eight_accuracy_1_minus_mae_over_0_9": round(
            1.0 - selected_eight_rounded_mae / MAX_LABEL_VALUE, 4
        ),
        "selected_eight_within_0_1_accuracy": round(selected_within, 4),
        "optional_top3_recall": round(
            top3_optional_recall_with_selector(
                targets[test_idx], selector_predictions, target_terms
            ),
            4,
        ),
    }


def selected_eight_within_tolerance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_terms: list[str],
    *,
    tolerance: float,
) -> float:
    term_to_idx = {term: idx for idx, term in enumerate(target_terms)}
    mandatory_indices = [term_to_idx[term] for term in MANDATORY_FACTORS]
    optional_indices = [
        idx
        for idx, term in enumerate(target_terms)
        if term not in MANDATORY_FACTORS and term not in NON_SELECTABLE_OPTIONAL_TERMS
    ]
    hits = []
    for true_row, pred_row in zip(y_true, y_pred):
        pred_optional_top = np.asarray(optional_indices)[
            np.argsort(-pred_row[optional_indices])[:3]
        ]
        selected = mandatory_indices + pred_optional_top.tolist()
        hits.extend((np.abs(true_row[selected] - pred_row[selected]) <= tolerance + 1e-8).tolist())
    return float(np.mean(hits))


def build_definition_records(definitions: pd.DataFrame, target_terms: list[str]) -> dict[str, dict[str, str]]:
    definition_records: dict[str, dict[str, str]] = {}
    for _, row in definitions.iterrows():
        term = row["term"]
        definition_records[term] = {
            "english_term": _clean_text(row.get("english_term", "")),
            "buddhist_explanation": _clean_text(row.get("buddhist_explanation", "")),
            "psychological_interpretation": _clean_text(
                row.get("psychological_interpretation", "")
            ),
            "keywords": _clean_text(row.get("keywords", "")),
            "example": _clean_text(row.get("example", "")),
            "source": "definition_workbook",
        }

    return definition_records


def build_artifact(
    csv_path: Path,
    xlsx_path: Path,
    output_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    definitions = load_definitions(xlsx_path)
    training_frame = load_training_frame(csv_path, definitions)
    target_terms = [column for column in training_frame.columns if column != "text"]
    texts = training_frame["text"].tolist()
    targets = training_frame[target_terms].to_numpy(dtype=np.float32)

    device = embedding_device()
    component_embeddings: dict[str, np.ndarray] = {}
    for component in ENSEMBLE_COMPONENTS:
        print(f"embedding {component['name']} with {component['model_name']}")
        encoder = SentenceTransformer(component["model_name"], device=device)
        component_embeddings[component["name"]] = embed_texts(
            encoder,
            texts,
            prompt=component.get("prompt", ""),
            batch_size=int(component.get("batch_size", 32)),
        )

    definition_records = build_definition_records(definitions, target_terms)
    definition_texts = [
        "\n".join(
            value
            for value in [
                definition_records[term].get("english_term", ""),
                term,
                definition_records[term].get("buddhist_explanation", ""),
                definition_records[term].get("psychological_interpretation", ""),
                definition_records[term].get("keywords", ""),
                definition_records[term].get("example", ""),
            ]
            if value
        )
        for term in target_terms
    ]
    definition_encoder = SentenceTransformer(TRANSFORMER_MODEL_NAME, device=device)
    definition_embeddings = embed_texts(
        definition_encoder,
        definition_texts,
        prompt=QWEN_EMBEDDING_PROMPT,
        batch_size=4,
    )

    metrics = evaluate_ensemble(
        component_embeddings,
        targets,
        target_terms,
        random_state=random_state,
    )

    ensemble_models = []
    for component in ENSEMBLE_COMPONENTS:
        embeddings = component_embeddings[component["name"]]
        regressor = Ridge(alpha=float(component["ridge_alpha"]))
        regressor.fit(embeddings, targets)
        ensemble_models.append(
            {
                "name": component["name"],
                "model_name": component["model_name"],
                "prompt": component.get("prompt", ""),
                "ensemble_weight": float(component["ensemble_weight"]),
                "regressor": regressor,
                "example_embeddings": embeddings.astype(np.float32),
                "example_targets": targets.astype(np.float32),
                "hybrid_weights": component["hybrid_weights"],
                "ridge_alpha": float(component["ridge_alpha"]),
                "rag_k": RAG_K,
            }
        )

    selector_models = []
    for component in SELECTOR_COMPONENTS:
        embeddings = component_embeddings[component["embedding_name"]]
        regressor = Ridge(alpha=float(component["ridge_alpha"]))
        regressor.fit(embeddings, targets)
        selector_models.append(
            {
                "name": component["name"],
                "model_name": component["model_name"],
                "prompt": component.get("prompt", ""),
                "ensemble_weight": float(component["ensemble_weight"]),
                "regressor": regressor,
                "example_embeddings": embeddings.astype(np.float32),
                "example_targets": targets.astype(np.float32),
                "hybrid_weights": component["hybrid_weights"],
                "ridge_alpha": float(component["ridge_alpha"]),
                "rag_k": RAG_K,
            }
        )

    source_columns = list(
        pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0)
        .rename(columns=CSV_COLUMN_RENAMES)
        .columns
    )
    ignored_label_columns = [
        column for column in source_columns[3:] if column not in set(target_terms)
    ]

    artifact = {
        "target_terms": target_terms,
        "definitions": definition_records,
        "non_selectable_optional_terms": sorted(
            list(NON_SELECTABLE_OPTIONAL_TERMS & set(target_terms))
        ),
        "transformer_model_name": TRANSFORMER_MODEL_NAME,
        "embedding_prompt": QWEN_EMBEDDING_PROMPT,
        "definition_embeddings": definition_embeddings.astype(np.float32),
        "ensemble_models": ensemble_models,
        "selector_models": selector_models,
        "example_embeddings": ensemble_models[0]["example_embeddings"],
        "example_targets": targets.astype(np.float32),
        "example_texts": texts,
        "hybrid_weights": HYBRID_WEIGHTS,
        "max_label_value": MAX_LABEL_VALUE,
        "rag_k": RAG_K,
        "metrics": metrics,
        "source_files": {
            "csv": str(csv_path),
            "xlsx": str(xlsx_path),
        },
        "label_policy": {
            "mandatory": "作意, 触, 受, 想, 思 are always emitted and trained from non-null annotations with median fill only where a duplicate-group has no mandatory value.",
            "optional": "The remaining 38 Excel-defined dimensions are treated as optional; missing CSV labels are interpreted as absent/0 before aggregation.",
            "source_target_dimensions": "The 43-vector Excel definition workbook is authoritative for model targets.",
            "ignored_csv_label_columns": ignored_label_columns,
            "column_renames": CSV_COLUMN_RENAMES,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_dir / MODEL_FILENAME)
    (output_dir / "evaluation.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "vector_definitions.json").write_text(
        json.dumps(definition_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the emotional vector model.")
    parser.add_argument("--csv", default="../标注数据_202509100918.csv")
    parser.add_argument("--definitions", default="../43心所+心理学（英文版.xlsx")
    parser.add_argument("--output-dir", default="model_artifacts")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = (base_dir / args.csv).resolve()
    definitions_path = (base_dir / args.definitions).resolve()
    output_dir = (base_dir / args.output_dir).resolve()

    artifact = build_artifact(
        csv_path=csv_path,
        xlsx_path=definitions_path,
        output_dir=output_dir,
        random_state=args.random_state,
    )
    print(json.dumps(artifact["metrics"], ensure_ascii=False, indent=2))
    print(f"saved: {output_dir / MODEL_FILENAME}")


if __name__ == "__main__":
    main()
