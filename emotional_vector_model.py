from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


MANDATORY_FACTORS = ["作意", "触", "受", "想", "思"]
MODEL_FILENAME = "emotional_vector_model.joblib"
DEFAULT_NON_SELECTABLE_OPTIONAL_TERMS = ["Theta"]


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


@dataclass(frozen=True)
class Prediction:
    dimensions: list[dict[str, Any]]
    all_scores: dict[str, float]
    retrieved_examples: list[dict[str, Any]]


class EmotionalVectorModel:
    """RAG-enhanced scorer with a mandatory-five plus top-three output contract."""

    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.target_terms: list[str] = artifact["target_terms"]
        self.non_selectable_optional_terms: set[str] = set(
            artifact.get(
                "non_selectable_optional_terms",
                DEFAULT_NON_SELECTABLE_OPTIONAL_TERMS,
            )
        )
        self.optional_terms: list[str] = [
            term
            for term in self.target_terms
            if term not in MANDATORY_FACTORS
            and term not in self.non_selectable_optional_terms
        ]
        self.definitions: dict[str, dict[str, Any]] = artifact["definitions"]
        self.max_label_value: float = float(artifact.get("max_label_value", 0.9))
        self.rag_k: int = int(artifact.get("rag_k", 8))
        self.example_texts: list[str] = artifact["example_texts"]
        self.ensemble_models = self._load_ensemble_models(artifact)
        self.selector_models = self._load_ensemble_models(
            {"ensemble_models": artifact.get("selector_models", self.ensemble_models)}
        )
        self._encoders: dict[str, Any] = {}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "EmotionalVectorModel":
        if path is None:
            path = Path(__file__).resolve().parent / "model_artifacts" / MODEL_FILENAME
        artifact = joblib.load(path)
        return cls(artifact)

    def _load_ensemble_models(self, artifact: dict[str, Any]) -> list[dict[str, Any]]:
        if "ensemble_models" in artifact:
            models = artifact["ensemble_models"]
        else:
            models = [
                {
                    "name": "single",
                    "model_name": artifact["transformer_model_name"],
                    "prompt": artifact.get("embedding_prompt", ""),
                    "ensemble_weight": 1.0,
                    "regressor": artifact["regressor"],
                    "example_embeddings": artifact["example_embeddings"],
                    "example_targets": artifact["example_targets"],
                    "hybrid_weights": artifact["hybrid_weights"],
                    "rag_k": artifact.get("rag_k", 8),
                }
            ]

        loaded = []
        for component in models:
            loaded.append(
                {
                    **component,
                    "example_embeddings": np.asarray(
                        component["example_embeddings"], dtype=np.float32
                    ),
                    "example_targets": np.asarray(
                        component["example_targets"], dtype=np.float32
                    ),
                }
            )
        return loaded

    def _load_encoder(self, model_name: str):
        if model_name not in self._encoders:
            from sentence_transformers import SentenceTransformer

            self._encoders[model_name] = SentenceTransformer(model_name)
        return self._encoders[model_name]

    def _embed(self, texts: list[str], *, model_name: str, prompt: str = "") -> np.ndarray:
        encoder = self._load_encoder(model_name)
        embeddings = encoder.encode(
            texts,
            prompt=prompt,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _unit_normalize(embeddings)

    def predict(
        self,
        text: str,
        *,
        include_all_scores: bool = True,
        include_retrieval: bool = True,
    ) -> Prediction:
        text = _clean_text(text)
        if not text:
            raise ValueError("Input text is empty.")

        embedding_cache: dict[tuple[str, str], np.ndarray] = {}
        component_predictions = []
        retrieval_component = max(
            self.ensemble_models,
            key=lambda component: float(component.get("ensemble_weight", 1.0)),
        )
        retrieval_embedding = None

        for component in self.ensemble_models:
            query_embedding = self._cached_embedding(text, component, embedding_cache)
            if component is retrieval_component:
                retrieval_embedding = query_embedding

            supervised = component["regressor"].predict(query_embedding)[0]
            supervised = np.clip(supervised, 0.0, self.max_label_value)

            example_prior, _, _ = self._example_prior_for_component(
                component, query_embedding[0]
            )
            weights = component.get("hybrid_weights", {})
            component_scores = (
                float(weights.get("supervised", 1.0)) * supervised
                + float(weights.get("examples", 0.0)) * example_prior
            )
            component_predictions.append(
                float(component.get("ensemble_weight", 1.0)) * component_scores
            )

        final_scores = np.sum(component_predictions, axis=0)
        final_scores = np.clip(final_scores, 0.0, self.max_label_value)

        selector_predictions = []
        for component in self.selector_models:
            query_embedding = self._cached_embedding(text, component, embedding_cache)
            supervised = component["regressor"].predict(query_embedding)[0]
            supervised = np.clip(supervised, 0.0, self.max_label_value)
            example_prior, _, _ = self._example_prior_for_component(
                component, query_embedding[0]
            )
            weights = component.get("hybrid_weights", {})
            component_scores = (
                float(weights.get("supervised", 1.0)) * supervised
                + float(weights.get("examples", 0.0)) * example_prior
            )
            selector_predictions.append(
                float(component.get("ensemble_weight", 1.0)) * component_scores
            )
        selector_scores = np.clip(np.sum(selector_predictions, axis=0), 0.0, self.max_label_value)
        rounded_scores = np.round(final_scores, 1)

        score_map = {
            term: round(float(rounded_scores[i]), 1)
            for i, term in enumerate(self.target_terms)
        }
        selector_map = {
            term: float(selector_scores[i]) for i, term in enumerate(self.target_terms)
        }
        selected_optional = sorted(
            self.optional_terms,
            key=lambda term: (selector_map[term], term),
            reverse=True,
        )[:3]
        selected_terms = MANDATORY_FACTORS + selected_optional
        dimensions = [
            {
                "term": term,
                "english_term": self.definitions[term].get("english_term", ""),
                "type": "mandatory" if term in MANDATORY_FACTORS else "optional",
                "value": score_map[term],
            }
            for term in selected_terms
        ]

        retrieved_examples = []
        if include_retrieval and retrieval_embedding is not None:
            _, top_idx, top_scores = self._example_prior_for_component(
                retrieval_component, retrieval_embedding[0]
            )
            for idx, score in zip(top_idx[:3], top_scores[:3]):
                retrieved_examples.append(
                    {
                        "similarity": round(float(score), 4),
                        "text": self.example_texts[int(idx)][:500],
                    }
                )

        return Prediction(
            dimensions=dimensions,
            all_scores=score_map if include_all_scores else {},
            retrieved_examples=retrieved_examples,
        )

    def _cached_embedding(
        self,
        text: str,
        component: dict[str, Any],
        embedding_cache: dict[tuple[str, str], np.ndarray],
    ) -> np.ndarray:
        key = (component["model_name"], component.get("prompt", ""))
        if key not in embedding_cache:
            embedding_cache[key] = self._embed(
                [text],
                model_name=component["model_name"],
                prompt=component.get("prompt", ""),
            )
        return embedding_cache[key]

    def _example_prior_for_component(
        self, component: dict[str, Any], query_embedding: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        example_embeddings = component["example_embeddings"]
        example_targets = component["example_targets"]
        rag_k = int(component.get("rag_k", self.rag_k))
        example_scores = example_embeddings @ query_embedding
        top_idx = np.argsort(-example_scores)[:rag_k]
        top_scores = example_scores[top_idx]
        weights = np.maximum(top_scores, 0.0)
        if float(weights.sum()) == 0.0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        example_prior = weights @ example_targets[top_idx]
        return example_prior, top_idx, top_scores

    def _scale_definition_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32)
        floor = float(np.percentile(scores, 20))
        ceiling = float(np.percentile(scores, 95))
        if ceiling <= floor:
            return np.zeros_like(scores)
        scaled = (scores - floor) / (ceiling - floor)
        return np.clip(scaled, 0.0, 1.0) * self.max_label_value


def predict(text: str, model_path: str | Path | None = None) -> dict[str, Any]:
    model = EmotionalVectorModel.load(model_path)
    prediction = model.predict(text)
    return {
        "dimensions": prediction.dimensions,
        "all_scores": prediction.all_scores,
        "retrieved_examples": prediction.retrieved_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the mandatory-five plus top-three optional emotion vectors."
    )
    parser.add_argument("text", nargs="*", help="Text to score. Reads stdin if omitted.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to emotional_vector_model.joblib. Defaults to bundled artifact.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print only the eight selected dimensions.",
    )
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text:
        import sys

        text = sys.stdin.read().strip()

    model = EmotionalVectorModel.load(args.model)
    prediction = model.predict(
        text,
        include_all_scores=not args.compact,
        include_retrieval=not args.compact,
    )
    payload: dict[str, Any] = {"dimensions": prediction.dimensions}
    if not args.compact:
        payload["all_scores"] = prediction.all_scores
        payload["retrieved_examples"] = prediction.retrieved_examples
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
