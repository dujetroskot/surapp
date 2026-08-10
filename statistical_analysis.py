import pandas as pd

AAPS_EXPERT_ANSWERS = {
    1: [1, 2],
    2: [4, 5],
    3: [1, 2],
    4: [4, 5],
    5: [1, 2],
    6: [4, 5],
    7: [4, 5],
    8: [1, 2],
    9: [4, 5],
    10: [4, 5],
    11: [1, 2],
    12: [1, 2],
    13: [4, 5],
    14: [4, 5],
    15: [4, 5],
    16: [1, 2],
    17: [4, 5],
    18: [4, 5],
    19: [4, 5],
    20: [4, 5],
    21: [4, 5],
    22: [4, 5],
    23: [1, 2],
    24: [4, 5],
    25: [4, 5],
    26: [4, 5],
    27: [4, 5],
    28: [4, 5],
    29: [4, 5],
    30: [1, 2],
    31: [4, 5],
    32: ["A", "B"],
    33: ["A", "B"],
}
AAPS_CATEGORIES = {
    "kat1": {
        "name": "Konceptualno razumijevanje i primjena",
        "items": [2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 18, 21, 22, 26, 29, 32, 33],
    },
    "kat2": {
        "name": "Strategije rješavanja problema",
        "items": [3, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33],
    },
    "kat3": {
        "name": "Individualni pristup i odnos pri rješavanju zadataka",
        "items": [1, 13, 20, 22, 23, 24, 27, 28, 29, 30],
    },
}

CONCEPT_CORRECT_ANSWERS = {
    "z1": {
        "answer": "C",
        "reasoning": "V",
    },
    "z2": {
        "answer": "B",
        "reasoning": "II",
    },
    "z3": {
        "answer": "C",
        "reasoning": "V",
    },
    "z4": {
        "answer": "D",
        "reasoning": "IV",
    },
}

AAPS_COLUMN_MAP = {
    1: "q12_stav_01",
    2: "q14_stav_02",
    3: "q15_stav_03",
    4: "q17_stav_04",
    5: "q18_stav_05",
    6: "q20_stav_06",
    7: "q21_stav_07",
    8: "q22_stav_08",
    9: "q23_stav_09",
    10: "q24_stav_10",
    11: "q25_stav_11",
    12: "q27_stav_12",
    13: "q29_stav_13",
    14: "q30_stav_14",
    15: "q31_stav_15",
    16: "q33_stav_16",
    17: "q34_stav_17",
    18: "q35_stav_18",
    19: "q37_stav_19",
    20: "q38_stav_20",
    21: "q39_stav_21",
    22: "q40_stav_22",
    23: "q41_stav_23",
    24: "q42_stav_24",
    25: "q43_stav_25",
    26: "q44_stav_26",
    27: "q45_stav_27",
    28: "q46_stav_28",
    29: "q47_stav_29",
    30: "q48_stav_30",
    31: "q50_stav_31",
}


AAPS_QUESTION_32_FEATURES = {
    "A": "q51_koncept_32__a",
    "B": "q51_koncept_32__b",
    "C": "q51_koncept_32__c",
    "D": "q51_koncept_32__d",
    "E": "q51_koncept_32__e",
}

AAPS_QUESTION_33_FEATURES = {
    "A": "q52_koncept_33__a",
    "B": "q52_koncept_33__b",
    "C": "q52_koncept_33__c",
    "D": "q52_koncept_33__d",
    "E": "q52_koncept_33__e",
}


CONCEPT_FEATURES = {
    "z1": {
        "answer": {
            "A": "q54_z1_odgovor__a",
            "B": "q54_z1_odgovor__b",
            "C": "q54_z1_odgovor__c",
            "D": "q54_z1_odgovor__d",
            "E": "q54_z1_odgovor__e",
        },
        "reasoning": {
            "I": "q56_z1_obrazlozenje__i",
            "II": "q56_z1_obrazlozenje__ii",
            "III": "q56_z1_obrazlozenje__iii",
            "IV": "q56_z1_obrazlozenje__iv",
            "V": "q56_z1_obrazlozenje__v",
        },
        "confidence_answer": "q55_z1_sigurnost_odgovor",
        "confidence_reasoning": "q57_z1_sigurnost_obrazlozenje",
    },
    "z2": {
        "answer": {
            "A": "q58_z2_odgovor__a",
            "B": "q58_z2_odgovor__b",
            "C": "q58_z2_odgovor__c",
            "D": "q58_z2_odgovor__d",
            "E": "q58_z2_odgovor__e",
        },
        "reasoning": {
            "I": "q60_z2_obrazlozenje__i",
            "II": "q60_z2_obrazlozenje__ii",
            "III": "q60_z2_obrazlozenje__iii",
            "IV": "q60_z2_obrazlozenje__iv",
            "V": "q60_z2_obrazlozenje__v",
        },
        "confidence_answer": "q59_z2_sigurnost_odgovor",
        "confidence_reasoning": "q61_z2_sigurnost_obrazlozenje",
    },
    "z3": {
        "answer": {
            "A": "q62_z3_odgovor__a",
            "B": "q62_z3_odgovor__b",
            "C": "q62_z3_odgovor__c",
            "D": "q62_z3_odgovor__d",
            "E": "q62_z3_odgovor__e",
        },
        "reasoning": {
            "I": "q64_z3_obrazlozenje__i",
            "II": "q64_z3_obrazlozenje__ii",
            "III": "q64_z3_obrazlozenje__iii",
            "IV": "q64_z3_obrazlozenje__iv",
            "V": "q64_z3_obrazlozenje__v",
        },
        "confidence_answer": "q63_z3_sigurnost_odgovor",
        "confidence_reasoning": "q65_z3_sigurnost_obrazlozenje",
    },
    "z4": {
        "answer": {
            "A": "q66_z4_odgovor__a",
            "B": "q66_z4_odgovor__b",
            "C": "q66_z4_odgovor__c",
            "D": "q66_z4_odgovor__d",
            "E": "q66_z4_odgovor__e",
        },
        "reasoning": {
            "I": "q68_z4_obrazlozenje__i",
            "II": "q68_z4_obrazlozenje__ii",
            "III": "q68_z4_obrazlozenje__iii",
            "IV": "q68_z4_obrazlozenje__iv",
            "V": "q68_z4_obrazlozenje__v",
        },
        "confidence_answer": "q67_z4_sigurnost_odgovor",
        "confidence_reasoning": "q69_z4_sigurnost_obrazlozenje",
    },
}


def calculate_aaps_alignment(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for item_number, expert_values in AAPS_EXPERT_ANSWERS.items():
        aligned_col = f"aaps_{item_number:02d}_aligned"

        if item_number <= 31:
            source_col = AAPS_COLUMN_MAP[item_number]
            if source_col in result.columns:
                result[aligned_col] = result[source_col].isin(expert_values).astype(int)
            else:
                result[aligned_col] = pd.NA

        elif item_number == 32:
            result[aligned_col] = _calculate_one_hot_alignment(
                result,
                AAPS_QUESTION_32_FEATURES,
                expert_values,
            )

        elif item_number == 33:
            result[aligned_col] = _calculate_one_hot_alignment(
                result,
                AAPS_QUESTION_33_FEATURES,
                expert_values,
            )

    return result


def _calculate_one_hot_alignment(df: pd.DataFrame, feature_map: dict, correct_values: list) -> pd.Series:
    aligned = pd.Series(0, index=df.index)

    for value in correct_values:
        feature = feature_map.get(value)
        if feature in df.columns:
            aligned = aligned | (df[feature] == 1)

    return aligned.astype(int)


def calculate_category_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for category_code, category_info in AAPS_CATEGORIES.items():
        aligned_columns = [
            f"aaps_{item:02d}_aligned"
            for item in category_info["items"]
            if f"aaps_{item:02d}_aligned" in result.columns
        ]

        if aligned_columns:
            result[f"{category_code}_alignment"] = result[aligned_columns].mean(axis=1)
            result[f"{category_code}_alignment_percent"] = result[f"{category_code}_alignment"] * 100
        else:
            result[f"{category_code}_alignment"] = pd.NA
            result[f"{category_code}_alignment_percent"] = pd.NA

    all_alignment_cols = [
        f"aaps_{item:02d}_aligned"
        for item in AAPS_EXPERT_ANSWERS.keys()
        if f"aaps_{item:02d}_aligned" in result.columns
    ]

    result["aaps_total_alignment"] = result[all_alignment_cols].mean(axis=1)
    result["aaps_total_alignment_percent"] = result["aaps_total_alignment"] * 100

    return result

def calculate_concept_scores(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for task_code, task_info in CONCEPT_FEATURES.items():
        correct_answer = CONCEPT_CORRECT_ANSWERS[task_code]["answer"]
        correct_reasoning = CONCEPT_CORRECT_ANSWERS[task_code]["reasoning"]

        answer_feature = task_info["answer"].get(correct_answer)
        reasoning_feature = task_info["reasoning"].get(correct_reasoning)

        answer_correct_col = f"{task_code}_answer_correct"
        reasoning_correct_col = f"{task_code}_reasoning_correct"
        both_correct_col = f"{task_code}_both_correct"

        if answer_feature in result.columns:
            result[answer_correct_col] = (result[answer_feature] == 1).astype(int)
        else:
            result[answer_correct_col] = pd.NA

        if reasoning_feature in result.columns:
            result[reasoning_correct_col] = (result[reasoning_feature] == 1).astype(int)
        else:
            result[reasoning_correct_col] = pd.NA

        result[both_correct_col] = (
            (result[answer_correct_col] == 1) &
            (result[reasoning_correct_col] == 1)
        ).astype(int)

    answer_cols = [f"z{i}_answer_correct" for i in range(1, 5)]
    reasoning_cols = [f"z{i}_reasoning_correct" for i in range(1, 5)]
    both_cols = [f"z{i}_both_correct" for i in range(1, 5)]

    result["concept_answer_score_total"] = result[answer_cols].sum(axis=1)
    result["concept_reasoning_score_total"] = result[reasoning_cols].sum(axis=1)
    result["concept_both_score_total"] = result[both_cols].sum(axis=1)

    return result


def generate_summary(df: pd.DataFrame) -> dict:
    return {
        "num_students": len(df),
        "avg_aaps_alignment": round(df["aaps_total_alignment_percent"].mean(), 2),
        "avg_concept_answer_score": round(df["concept_answer_score_total"].mean(), 2),
        "avg_concept_reasoning_score": round(df["concept_reasoning_score_total"].mean(), 2),
        "avg_concept_both_score": round(df["concept_both_score_total"].mean(), 2),
    }


def interpret_category_alignment(avg_alignment):
    if avg_alignment is None or pd.isna(avg_alignment):
        return "Nije dostupno"

    if avg_alignment > 55:
        return "Prevladavaju stavovi u skladu sa stručnjacima"

    if avg_alignment < 45:
        return "Prevladavaju stavovi koji nisu u skladu sa stručnjacima"

    return "Nema jasnog prevladavanja"


def generate_category_stats(df):
    results = []

    for category_code, category_info in AAPS_CATEGORIES.items():
        category_name = category_info["name"]
        items = category_info["items"]

        aligned_cols = []

        for item in items:
            col = f"aaps_{item:02d}_aligned"
            if col in df.columns:
                aligned_cols.append(col)

        if not aligned_cols:
            avg_alignment = None
        else:
            avg_alignment = round(df[aligned_cols].mean(axis=1).mean() * 100, 2)

        results.append({
            "category_code": category_code,
            "category_name": category_name,
            "num_items": len(aligned_cols),
            "avg_alignment": avg_alignment,
            "dominant_attitude": interpret_category_alignment(avg_alignment),
        })

    return results


def generate_problematic_items(df: pd.DataFrame, top_n: int = 10) -> list:
    rows = []

    for item_number in AAPS_EXPERT_ANSWERS.keys():
        col = f"aaps_{item_number:02d}_aligned"

        if col not in df.columns:
            continue

        rows.append({
            "item": item_number,
            "expert_answer": str(AAPS_EXPERT_ANSWERS[item_number]),
            "alignment_percent": round(df[col].mean() * 100, 2),
        })

    rows = sorted(rows, key=lambda x: x["alignment_percent"])
    return rows[:top_n]


def generate_demographic_stats(df: pd.DataFrame) -> dict:
    result = {}

    gender_cols = {
        "Muško": "q01_spol__musko",
        "Žensko": "q01_spol__zensko",
    }

    gender_rows = []
    for gender_name, gender_col in gender_cols.items():
        if gender_col in df.columns:
            subset = df[df[gender_col] == 1]
            if not subset.empty:
                gender_rows.append({
                    "group": gender_name,
                    "num_students": len(subset),
                    "kat1": round(subset["kat1_alignment_percent"].mean(), 2),
                    "kat2": round(subset["kat2_alignment_percent"].mean(), 2),
                    "kat3": round(subset["kat3_alignment_percent"].mean(), 2),
                    "concept_score": round(subset["concept_answer_score_total"].mean(), 2),
                })

    result["gender"] = gender_rows

    school_cols = [
        col for col in df.columns
        if col.startswith("q03_srednja_skola__") and not col.endswith("_text")
    ]

    school_rows = []
    for col in school_cols:
        subset = df[df[col] == 1]
        if not subset.empty:
            school_rows.append({
                "group": col.replace("q03_srednja_skola__", ""),
                "num_students": len(subset),
                "kat1": round(subset["kat1_alignment_percent"].mean(), 2),
                "kat2": round(subset["kat2_alignment_percent"].mean(), 2),
                "kat3": round(subset["kat3_alignment_percent"].mean(), 2),
                "concept_score": round(subset["concept_answer_score_total"].mean(), 2),
            })

    result["school"] = school_rows

    return result


def generate_correlations(df: pd.DataFrame) -> list:
    target = "concept_answer_score_total"

    variables = [
        "kat1_alignment",
        "kat2_alignment",
        "kat3_alignment",
        "aaps_total_alignment",
    ]

    rows = []

    for var in variables:
        if var in df.columns and target in df.columns:
            corr = df[var].corr(df[target], method="spearman")
            rows.append({
                "variable": var,
                "correlation": round(corr, 3) if pd.notna(corr) else None,
            })

    return rows


def generate_concept_task_stats(df: pd.DataFrame) -> list:
    rows = []

    for i in range(1, 5):
        task_code = f"z{i}"
        answer_correct_col = f"{task_code}_answer_correct"
        reasoning_correct_col = f"{task_code}_reasoning_correct"
        both_correct_col = f"{task_code}_both_correct"

        confidence_answer_col = CONCEPT_FEATURES[task_code]["confidence_answer"]
        confidence_reasoning_col = CONCEPT_FEATURES[task_code]["confidence_reasoning"]

        correct_subset = df[df[answer_correct_col] == 1]
        wrong_subset = df[df[answer_correct_col] == 0]

        rows.append({
            "task": task_code.upper(),
            "answer_accuracy": round(df[answer_correct_col].mean() * 100, 2),
            "reasoning_accuracy": round(df[reasoning_correct_col].mean() * 100, 2),
            "both_accuracy": round(df[both_correct_col].mean() * 100, 2),
            "avg_confidence_correct": round(correct_subset[confidence_answer_col].mean(), 2) if not correct_subset.empty else None,
            "avg_confidence_wrong": round(wrong_subset[confidence_answer_col].mean(), 2) if not wrong_subset.empty else None,
            "correct_answer": CONCEPT_CORRECT_ANSWERS[task_code]["answer"],
            "correct_reasoning": CONCEPT_CORRECT_ANSWERS[task_code]["reasoning"],
        })

    return rows

def safe_mean(series, multiplier=1):
    if series is None or len(series) == 0:
        return None

    value = series.mean()

    if pd.isna(value):
        return None

    return round(float(value) * multiplier, 2)

def generate_alignment_group_analysis(df):
    if "aaps_total_alignment_percent" not in df.columns:
        return []

    q25 = df["aaps_total_alignment_percent"].quantile(0.25)
    q75 = df["aaps_total_alignment_percent"].quantile(0.75)

    groups = [
        {
            "group": "Nisko slaganje",
            "rule": f"≤ {round(q25, 2)}%",
            "data": df[df["aaps_total_alignment_percent"] <= q25],
        },
        {
            "group": "Srednje slaganje",
            "rule": f"{round(q25, 2)}% – {round(q75, 2)}%",
            "data": df[
                (df["aaps_total_alignment_percent"] > q25) &
                (df["aaps_total_alignment_percent"] < q75)
            ],
        },
        {
            "group": "Visoko slaganje",
            "rule": f"≥ {round(q75, 2)}%",
            "data": df[df["aaps_total_alignment_percent"] >= q75],
        },
    ]

    results = []

    for group in groups:
        group_df = group["data"]

        row = {
            "group": group["group"],
            "rule": group["rule"],
            "num_students": int(len(group_df)),
            "avg_aaps_alignment": safe_mean(group_df["aaps_total_alignment_percent"]),
            "avg_concept_answer_score": safe_mean(group_df["concept_answer_score_total"]),
            "avg_concept_reasoning_score": safe_mean(group_df["concept_reasoning_score_total"]),
            "avg_concept_both_score": safe_mean(group_df["concept_both_score_total"]),
        }

        for task in ["z1", "z2", "z3", "z4"]:
            answer_col = f"{task}_answer_correct"
            reasoning_col = f"{task}_reasoning_correct"
            both_col = f"{task}_both_correct"

            row[f"{task}_answer_accuracy"] = safe_mean(group_df[answer_col], multiplier=100) if answer_col in group_df.columns else None
            row[f"{task}_reasoning_accuracy"] = safe_mean(group_df[reasoning_col], multiplier=100) if reasoning_col in group_df.columns else None
            row[f"{task}_both_accuracy"] = safe_mean(group_df[both_col], multiplier=100) if both_col in group_df.columns else None

        results.append(row)

    return results

def run_statistical_analysis(encoded_csv_path: str) -> dict:
    df = pd.read_csv(encoded_csv_path)

    df = calculate_aaps_alignment(df)
    df = calculate_category_scores(df)
    df = calculate_concept_scores(df)

    analysis_ready_path = encoded_csv_path.replace(".csv", "_analysis_ready.csv")
    df.to_csv(analysis_ready_path, index=False, encoding="utf-8-sig")

    return {
        "summary": generate_summary(df),
        "category_stats": generate_category_stats(df),
        "problematic_items": generate_problematic_items(df),
        "demographic_stats": generate_demographic_stats(df),
        "correlations": generate_correlations(df),
        "concept_task_stats": generate_concept_task_stats(df),
        "alignment_group_analysis": generate_alignment_group_analysis(df),
        "analysis_ready_path": analysis_ready_path,
    }