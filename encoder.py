from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

@dataclass
class ProcessingResult:
    dataframe: pd.DataFrame
    warnings: List[str]

class SurveyEncoder:
    def __init__(self, config: List[Dict[str, Any]]) -> None:
        self.config = config
        self.warnings: List[str] = []

    @classmethod
    def from_json(cls, config_path: str) -> "SurveyEncoder":
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config)

    def encode_csv(self, csv_path: str) -> ProcessingResult:
        df = pd.read_csv(csv_path)
        encoded = self.encode_dataframe(df)
        return ProcessingResult(dataframe=encoded, warnings=self.warnings.copy())

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        if "Vremenska oznaka" in df.columns:
            output["timestamp"] = df["Vremenska oznaka"]

        for question in self.config:
            self._process_question(df, output, question)

        return output

    def _process_question(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        encoding = question["encoding"]

        if encoding == "one_hot":
            self._process_one_hot(df, output, question)
        elif encoding == "multi_hot":
            self._process_multi_hot(df, output, question)
        elif encoding == "binary":
            self._process_binary(df, output, question)
        elif encoding == "ordinal_passthrough":
            self._process_ordinal(df, output, question)
        elif encoding == "numeric_passthrough":
            self._process_numeric(df, output, question)
        elif encoding == "text_manual_review":
            self._process_text(df, output, question)
        elif encoding == "multi_column_ordinal":
            self._process_multi_column_ordinal(df, output, question)
        else:
            self.warnings.append(
                f"[WARN] Nepodržan encoding '{encoding}' za {question.get('question_code')}"
            )

    def _require_column(self, df: pd.DataFrame, column_name: str, context: str) -> Optional[pd.Series]:
        if column_name not in df.columns:
            self.warnings.append(f"[WARN] Nedostaje stupac '{column_name}' za {context}")
            return None
        return df[column_name]

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_for_compare(value: Any) -> str:
        text = SurveyEncoder._normalize_text(value)
        return text.casefold()

    @staticmethod
    def _slug_feature_suffix_from_label(label: str) -> str:
        text = label.casefold()
        replacements = {
            "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
            "–": "-", "—": "-", "/": "_", " ": "_",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        text = re.sub(r"[^a-z0-9_\-]+", "", text)
        text = re.sub(r"[-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    def _build_label_lookup(self, question: Dict[str, Any]) -> Dict[str, str]:
        """
        Returns mapping: normalized source answer -> output feature name
        """
        feature_labels = question.get("feature_labels", {})
        feature_names = question.get("feature_names", [])
        value_to_feature_map = question.get("value_to_feature_map", {})

        lookup: Dict[str, str] = {}

        if value_to_feature_map:
            for raw_value, feature_name in value_to_feature_map.items():
                lookup[self._normalize_for_compare(raw_value)] = feature_name
            return lookup

        for feature_name in feature_names:
            label = feature_labels.get(feature_name)
            if label is None:
                suffix = feature_name.split("__")[-1]
                lookup[self._normalize_for_compare(suffix)] = feature_name
            else:
                lookup[self._normalize_for_compare(label)] = feature_name
                match = re.match(r"^([A-ZIVX]+)\)", label.strip())
                if match:
                    token = match.group(1)
                    lookup[self._normalize_for_compare(token)] = feature_name
        return lookup

    def _process_one_hot(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        col = self._require_column(df, question["source_column"], question["question_code"])
        if col is None:
            return

        lookup = self._build_label_lookup(question)

        for feature in question["feature_names"]:
            output[feature] = 0

        other_indicator = question.get("other_indicator_feature")
        other_text = question.get("other_text_feature")
        has_other = question.get("has_other_option", False)
        if has_other and other_indicator:
            output[other_indicator] = 0
        if has_other and other_text:
            output[other_text] = pd.NA

        for idx, raw_value in col.items():
            value = self._normalize_text(raw_value)
            if value == "":
                continue

            key = self._normalize_for_compare(value)
            feature = lookup.get(key)

            if feature:
                output.at[idx, feature] = 1
                continue

            compact_key = key.rstrip(":")
            feature = lookup.get(compact_key)
            if feature:
                output.at[idx, feature] = 1
                continue

            if has_other and other_indicator:
                output.at[idx, other_indicator] = 1
                if other_text:
                    output.at[idx, other_text] = value
            else:
                self.warnings.append(
                    f"[WARN] Neprepoznata vrijednost '{value}' u {question['question_code']}"
                )

    def _process_multi_hot(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        col = self._require_column(df, question["source_column"], question["question_code"])
        if col is None:
            return

        lookup = self._build_label_lookup(question)
        for feature in question["feature_names"]:
            output[feature] = 0

        other_indicator = question.get("other_indicator_feature")
        other_text = question.get("other_text_feature")
        has_other = question.get("has_other_option", False)
        if has_other and other_indicator:
            output[other_indicator] = 0
        if has_other and other_text:
            output[other_text] = pd.NA

        for idx, raw_value in col.items():
            value = self._normalize_text(raw_value)
            if value == "":
                continue

            parts = [part.strip() for part in re.split(r",\s*", value) if part.strip()]
            unmatched_parts: List[str] = []

            for part in parts:
                key = self._normalize_for_compare(part)
                feature = lookup.get(key)
                if feature:
                    output.at[idx, feature] = 1
                else:
                    unmatched_parts.append(part)

            if unmatched_parts and has_other and other_indicator:
                output.at[idx, other_indicator] = 1
                if other_text:
                    output.at[idx, other_text] = " | ".join(unmatched_parts)
            elif unmatched_parts:
                self.warnings.append(
                    f"[WARN] Neprepoznate multi-select vrijednosti {unmatched_parts} u {question['question_code']}"
                )

    def _process_binary(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        col = self._require_column(df, question["source_column"], question["question_code"])
        if col is None:
            return

        mapping = {
            self._normalize_for_compare(k): v
            for k, v in question.get("binary_mapping", {}).items()
        }
        feature = question["feature_names"][0]
        output[feature] = pd.NA

        for idx, raw_value in col.items():
            value = self._normalize_text(raw_value)
            if value == "":
                continue
            mapped = mapping.get(self._normalize_for_compare(value))
            if mapped is None:
                self.warnings.append(
                    f"[WARN] Neprepoznata binary vrijednost '{value}' u {question['question_code']}"
                )
            else:
                output.at[idx, feature] = mapped

    def _process_ordinal(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        col = self._require_column(df, question["source_column"], question["question_code"])
        if col is None:
            return

        feature = question["feature_names"][0]
        output[feature] = pd.to_numeric(col, errors="coerce")

    def _process_numeric(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        col = self._require_column(df, question["source_column"], question["question_code"])
        if col is None:
            return

        feature = question["feature_names"][0]
        cleaned = col.astype(str).str.replace(",", ".", regex=False).str.strip()
        cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        output[feature] = pd.to_numeric(cleaned, errors="coerce")

    def _process_text(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        col = self._require_column(df, question["source_column"], question["question_code"])
        if col is None:
            return

        feature = question["feature_names"][0]
        output[feature] = col.astype("string")

        output[f"{feature}__char_count"] = output[feature].fillna("").str.len()
        output[f"{feature}__word_count"] = (
            output[feature].fillna("").str.split().apply(len).astype("Int64")
        )

    def _process_multi_column_ordinal(self, df: pd.DataFrame, output: pd.DataFrame, question: Dict[str, Any]) -> None:
        for subq in question.get("subquestions", []):
            source_column = subq.get("source_column")
            feature_name = subq["feature_name"]
            if not source_column:
                self.warnings.append(
                    f"[WARN] Subquestion bez source_column za {question['question_code']}::{subq.get('subquestion_code')}"
                )
                output[feature_name] = pd.NA
                continue

            col = self._require_column(df, source_column, f"{question['question_code']}::{subq.get('subquestion_code')}")
            if col is None:
                output[feature_name] = pd.NA
                continue

            series = col.copy()

            if subq.get("has_not_applicable"):
                na_value = self._normalize_for_compare(subq.get("not_applicable_value"))
                normalized = series.map(self._normalize_for_compare)
                series = series.where(normalized != na_value, pd.NA)

            label_map = {
                "nisam polagao/la": pd.NA,
                "dovoljan": 2,
                "dobar": 3,
                "vrlo dobar": 4,
                "izvrstan": 5,
            }

            def convert_value(v: Any) -> Any:
                if pd.isna(v):
                    return pd.NA
                txt = self._normalize_text(v)
                if txt == "":
                    return pd.NA
                txt_cf = txt.casefold()
                if txt_cf in label_map:
                    return label_map[txt_cf]
                try:
                    return int(float(txt.replace(",", ".")))
                except Exception:
                    return pd.NA

            output[feature_name] = series.map(convert_value).astype("Float64")

    def validate_config_against_csv(self, csv_path: str) -> List[str]:
        df = pd.read_csv(csv_path)
        missing = []

        for question in self.config:
            if question["encoding"] == "multi_column_ordinal":
                for subq in question.get("subquestions", []):
                    source_col = subq.get("source_column")
                    if source_col and source_col not in df.columns:
                        missing.append(
                            f"{question['question_code']}::{subq.get('subquestion_code')} -> {source_col}"
                        )
            else:
                source_col = question.get("source_column")
                if source_col and source_col not in df.columns:
                    missing.append(f"{question['question_code']} -> {source_col}")

        return missing
