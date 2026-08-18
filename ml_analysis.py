import pandas as pd
import numpy as np
import re

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

AAPS_TEXT_LIKERT_PAIRS = [
    {
        "label": "AAPS 1",
        "likert_col": "q12_stav_01",
        "text_col": "q13_stav_01_obrazlozenje",
    },
    {
        "label": "AAPS 3",
        "likert_col": "q15_stav_03",
        "text_col": "q16_stav_03_obrazlozenje",
    },
    {
        "label": "AAPS 5",
        "likert_col": "q18_stav_05",
        "text_col": "q19_stav_05_obrazlozenje",
    },
    {
        "label": "AAPS 11",
        "likert_col": "q25_stav_11",
        "text_col": "q26_stav_11_obrazlozenje",
    },
    {
        "label": "AAPS 12",
        "likert_col": "q27_stav_12",
        "text_col": "q28_stav_12_obrazlozenje",
    },
    {
        "label": "AAPS 15",
        "likert_col": "q31_stav_15",
        "text_col": "q32_stav_15_obrazlozenje",
    },
    {
        "label": "AAPS 18",
        "likert_col": "q35_stav_18",
        "text_col": "q36_stav_18_obrazlozenje",
    },
    {
        "label": "AAPS 30",
        "likert_col": "q48_stav_30",
        "text_col": "q49_stav_30_obrazlozenje",
    },
]


def clean_text_for_nlp(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    return text


def run_likert_from_text_single_question(df, pair):
    likert_col = pair["likert_col"]
    text_col = pair["text_col"]

    if likert_col not in df.columns or text_col not in df.columns:
        return {
            "label": pair["label"],
            "available": False,
            "reason": "Nedostaje Likert stupac ili tekstualno obrazloženje.",
        }

    data = df[[likert_col, text_col]].copy()
    data[likert_col] = pd.to_numeric(data[likert_col], errors="coerce")
    data[text_col] = data[text_col].apply(clean_text_for_nlp)

    data = data.dropna(subset=[likert_col])
    data = data[data[text_col].str.len() > 5]

    if len(data) < 20:
        return {
            "label": pair["label"],
            "available": False,
            "reason": "Premalo tekstualnih odgovora za treniranje modela.",
        }

    y = data[likert_col].apply(lambda value: 1 if value >= 4 else 0).astype(int)
    
    texts = data[text_col]

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        return {
            "label": pair["label"],
            "available": False,
            "reason": "Target ima samo jednu klasu.",
        }

    if class_counts.min() < 5:
        return {
            "label": pair["label"],
            "available": False,
            "reason": "Premalo primjera u jednoj od klasa.",
        }

    vectorizer = TfidfVectorizer(
        max_features=200,
        min_df=2,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Linear SVM": LinearSVC(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ),
    }

    model_results = []
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_results.append({
            "model": model_name,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
        })

        trained_models[model_name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    model_results = sorted(model_results, key=lambda x: x["f1"], reverse=True)

    best_model_name = model_results[0]["model"]
    confusion = get_confusion_matrix(best_model_name, trained_models)

    return {
        "label": pair["label"],
        "available": True,
        "likert_col": likert_col,
        "text_col": text_col,
        "num_samples": int(len(data)),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "not_agree_count": int((y == 0).sum()),
        "agree_count": int((y == 1).sum()),
        "num_tfidf_features": int(X.shape[1]),
        "best_model": best_model_name,
        "model_results": model_results,
        "confusion_matrix": confusion,
    }


def run_likert_from_text_analysis(df):
    results = []

    for pair in AAPS_TEXT_LIKERT_PAIRS:
        result = run_likert_from_text_single_question(df, pair)
        results.append(result)

    return {
        "available": True,
        "target_rule": "1 = slaganje s tvrdnjom, odnosno Likert 4 ili 5; 0 = neutralno ili neslaganje, odnosno Likert 1, 2 ili 3",
        "split": "80% trening, 20% test",
        "results": results,
    }

NON_INFORMATIVE_TEXTS = {
    "",
    ".",
    "..",
    "...",
    "/",
    "//",
    "-",
    "--",
    "?",
    "ne",
    "ne znam",
    "neznam",
    "nezz",
    "nista",
    "ništa",
    "nemam",
    "nemam pojma",
    "idk",
    "no idea",
    "ne sjecam se",
    "ne sjećam se",
}

def normalize_text_answer(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


def is_informative_text_answer(value, min_words=3, min_chars=10):
    text = normalize_text_answer(value)

    if text in NON_INFORMATIVE_TEXTS:
        return False

    cleaned = re.sub(r"[^a-zA-ZčćđšžČĆĐŠŽ0-9\s]", "", text)
    cleaned = cleaned.strip().lower()

    if cleaned in NON_INFORMATIVE_TEXTS:
        return False

    if len(cleaned.replace(" ", "")) < min_chars:
        return False

    words = [word for word in cleaned.split() if word]

    if len(words) < min_words:
        return False

    return True

def add_text_quality_flags(df, text_columns=None):
    df = df.copy()

    if text_columns is None:
        text_columns = []

        for col in df.columns:
            if df[col].dtype == "object":
                if not col.endswith("_text"):
                    text_columns.append(col)

    for col in text_columns:
        flag_col = f"{col}__is_informative"
        df[flag_col] = df[col].apply(is_informative_text_answer).astype(int)

    return df

def create_target_variable(df):
    if "concept_answer_score_total" not in df.columns:
        raise ValueError("Nedostaje stupac 'concept_answer_score_total'. Prvo treba pokrenuti statističku analizu.")

    df = df.copy()
    df["concept_result_label"] = (df["concept_answer_score_total"] >= 2).astype(int)

    return df

def get_aaps_feature_columns(df):
    selected_columns = []

    main_features = [
        "kat1_alignment_percent",
        "kat2_alignment_percent",
        "kat3_alignment_percent",
        "aaps_total_alignment_percent",
    ]

    for col in main_features:
        if col in df.columns:
            selected_columns.append(col)

    for col in df.columns:
        if "_stav_" in col:
            selected_columns.append(col)

    for col in df.columns:
        if col.startswith("q51_koncept_32__") or col.startswith("q52_koncept_33__"):
            selected_columns.append(col)

    return list(dict.fromkeys(selected_columns))


def get_demographic_feature_columns(df):
    selected_columns = []

    demographic_prefixes = [
        "q01_spol__",
        "q03_srednja_skola__",
    ]

    for col in df.columns:
        if any(col.startswith(prefix) for prefix in demographic_prefixes):
            if not col.endswith("_text"):
                selected_columns.append(col)

    return list(dict.fromkeys(selected_columns))

def clean_feature_matrix(df, selected_columns):
    if not selected_columns:
        raise ValueError("Nije pronađena nijedna značajka za treniranje modela.")

    X = df[selected_columns].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.dropna(axis=1, how="all")

    selected_columns = list(X.columns)

    if not selected_columns:
        raise ValueError("Nakon uklanjanja praznih stupaca nije ostala nijedna značajka.")

    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    if X.isna().sum().sum() > 0:
        raise ValueError("Feature matrica i dalje sadrži NaN vrijednosti.")

    return X, selected_columns

def select_features(df, feature_set="aaps_demographic"):
    aaps_columns = get_aaps_feature_columns(df)

    if feature_set == "aaps_only":
        selected_columns = aaps_columns

    elif feature_set == "aaps_demographic":
        demographic_columns = get_demographic_feature_columns(df)
        selected_columns = aaps_columns + demographic_columns

    else:
        raise ValueError(f"Nepoznat feature_set: {feature_set}")

    selected_columns = list(dict.fromkeys(selected_columns))

    return clean_feature_matrix(df, selected_columns)

def train_models(X, y):
    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError("Ciljna varijabla ima samo jednu klasu. Model se ne može trenirati.")

    use_stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=use_stratify
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(max_iter=5000, class_weight="balanced", random_state=42))
        ]),
    }

    train_class_counts = y_train.value_counts()
    min_class_count = int(train_class_counts.min())

    if min_class_count >= 5:
        n_splits = 5
    elif min_class_count >= 3:
        n_splits = 3
    else:
        n_splits = 2

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    results = []
    trained_models = {}

    for model_name, model in models.items():
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "model": model_name,
            "cv_f1": round(float(cv_scores.mean()), 3),
            "cv_f1_std": round(float(cv_scores.std()), 3),
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
        })

        trained_models[model_name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    results = sorted(results, key=lambda x: x["cv_f1"], reverse=True)

    return results, trained_models, X_train, X_test, y_train, y_test

def get_confusion_matrix(best_model_name, trained_models):
    y_test = trained_models[best_model_name]["y_test"]
    y_pred = trained_models[best_model_name]["y_pred"]

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


def get_random_forest_feature_importance(trained_models, feature_names, top_n=10):
    rf_model = trained_models["Random Forest"]["model"]

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(by="importance", ascending=False).head(top_n)

    importance_df["importance"] = importance_df["importance"].round(4)

    return importance_df.to_dict(orient="records")


def generate_ml_summary(results, feature_importance):
    best_model = results[0]

    if feature_importance:
        top_feature = feature_importance[0]["feature"]
    else:
        top_feature = "nije dostupno"

    summary = (
        f"Najbolji rezultat ostvario je model {best_model['model']} "
        f"s F1 vrijednošću {best_model['f1']} i accuracy vrijednošću {best_model['accuracy']}. "
        f"Prema Random Forest modelu, najvažnija značajka za predikciju bila je '{top_feature}'."
    )

    return summary

def run_single_feature_set_analysis(df, y, feature_set_key, feature_set_name):
    X, feature_names = select_features(df, feature_set=feature_set_key)

    results, trained_models, X_train, X_test, y_train, y_test = train_models(X, y)

    best_model_name = results[0]["model"]

    cm = get_confusion_matrix(best_model_name, trained_models)

    feature_importance = get_random_forest_feature_importance(
        trained_models=trained_models,
        feature_names=feature_names,
        top_n=10
    )

    summary = generate_ml_summary(results, feature_importance)

    return {
        "feature_set_key": feature_set_key,
        "feature_set_name": feature_set_name,
        "num_features": int(len(feature_names)),
        "model_results": results,
        "best_model": best_model_name,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
        "summary": summary,
    }

def run_single_task_prediction(df, target_col, task_name, feature_set_key, feature_set_name):
    if target_col not in df.columns:
        return {
            "task": task_name,
            "feature_set": feature_set_name,
            "skipped": True,
            "reason": f"Nedostaje stupac {target_col}."
        }

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        return {
            "task": task_name,
            "feature_set": feature_set_name,
            "skipped": True,
            "reason": "Ciljna varijabla ima samo jednu klasu."
        }

    if class_counts.min() < 5:
        return {
            "task": task_name,
            "feature_set": feature_set_name,
            "skipped": True,
            "reason": "Premalo primjera u jednoj od klasa."
        }

    X, feature_names = select_features(df, feature_set=feature_set_key)

    results, trained_models, X_train, X_test, y_train, y_test = train_models(X, y)

    best_model_name = results[0]["model"]
    cm = get_confusion_matrix(best_model_name, trained_models)

    return {
        "task": task_name,
        "target_col": target_col,
        "feature_set": feature_set_name,
        "skipped": False,
        "num_features": int(len(feature_names)),
        "wrong_count": int((y == 0).sum()),
        "correct_count": int((y == 1).sum()),
        "best_model": best_model_name,
        "accuracy": results[0]["accuracy"],
        "precision": results[0]["precision"],
        "recall": results[0]["recall"],
        "f1": results[0]["f1"],
        "confusion_matrix": cm,
    }

def run_task_prediction_analysis(df):
    tasks = [
        ("z1_answer_correct", "Zadatak 1"),
        ("z2_answer_correct", "Zadatak 2"),
        ("z3_answer_correct", "Zadatak 3"),
        ("z4_answer_correct", "Zadatak 4"),
    ]

    feature_sets = [
        ("aaps_only", "AAPS-only"),
        ("aaps_demographic", "AAPS + demografija"),
    ]

    task_results = []

    for target_col, task_name in tasks:
        for feature_set_key, feature_set_name in feature_sets:
            result = run_single_task_prediction(
                df=df,
                target_col=target_col,
                task_name=task_name,
                feature_set_key=feature_set_key,
                feature_set_name=feature_set_name
            )

            task_results.append(result)

    return task_results

def get_aaps_text_columns(df):
    text_columns = []

    for col in df.columns:
        if "stav" in col and "obrazlozenje" in col:
            if not col.endswith("__char_count") and not col.endswith("__word_count"):
                text_columns.append(col)

    return text_columns


def combine_text_columns(df, text_columns):
    if not text_columns:
        return pd.Series([""] * len(df))

    text_df = df[text_columns].fillna("").astype(str)

    return text_df.apply(lambda row: " ".join(row.values), axis=1)


def run_nlp_text_prediction_analysis(df):
    if "concept_result_label" not in df.columns:
        df = create_target_variable(df)

    y = df["concept_result_label"]

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        return {
            "available": False,
            "reason": "Target ima samo jednu klasu. NLP model se ne može trenirati.",
        }

    text_columns = get_aaps_text_columns(df)

    if not text_columns:
        return {
            "available": False,
            "reason": "Nisu pronađena tekstualna AAPS obrazloženja.",
        }

    texts = combine_text_columns(df, text_columns)

    use_stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.25,
        random_state=42,
        stratify=use_stratify
    )

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=300,
                min_df=3,
                ngram_range=(1, 2)
            )),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=300,
                min_df=3,
                ngram_range=(1, 2)
            )),
            ("model", LinearSVC(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            ))
        ]),
    }

    train_class_counts = y_train.value_counts()
    min_class_count = int(train_class_counts.min())

    if min_class_count >= 5:
        n_splits = 5
    elif min_class_count >= 3:
        n_splits = 3
    else:
        n_splits = 2

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    results = []
    trained_models = {}

    for model_name, model in models.items():
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "model": model_name,

            # Rezultat korišten za izbor modela
            "cv_f1": round(float(cv_scores.mean()), 3),
            "cv_f1_std": round(float(cv_scores.std()), 3),

            # Konačna evaluacija na testnom skupu
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 3),
        })

        trained_models[model_name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    # Model se bira prema F1 rezultatu iz cross-validacije na trening skupu,
    # a ne prema F1 rezultatu na testnom skupu.
    results = sorted(results, key=lambda x: x["cv_f1"], reverse=True)

    best_model_name = results[0]["model"]
    confusion = get_confusion_matrix(best_model_name, trained_models)

    best_pipeline = trained_models[best_model_name]["model"]

    vectorizer = best_pipeline.named_steps["tfidf"]
    classifier = best_pipeline.named_steps["model"]

    feature_names = vectorizer.get_feature_names_out()
    num_tfidf_features = len(feature_names)

    top_terms = []

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]

        top_positive_indices = coefficients.argsort()[-10:][::-1]
        top_negative_indices = coefficients.argsort()[:10]

        top_terms = {
            "higher_result_terms": [
                {
                    "term": feature_names[i],
                    "weight": round(float(coefficients[i]), 4)
                }
                for i in top_positive_indices
            ],
            "lower_result_terms": [
                {
                    "term": feature_names[i],
                    "weight": round(float(coefficients[i]), 4)
                }
                for i in top_negative_indices
            ],
        }

    return {
        "available": True,
        "text_columns": text_columns,
        "num_text_columns": len(text_columns),
        "num_tfidf_features": int(num_tfidf_features),
        "target": "concept_result_label",
        "target_rule": "0 ili 1 točan odgovor = niži rezultat; 2, 3 ili 4 točna odgovora = viši rezultat",
        "model_results": results,
        "best_model": best_model_name,
        "confusion_matrix": confusion,
        "top_terms": top_terms,
    }

def run_ml_analysis(analysis_ready_csv_path):
    csv_path = Path(analysis_ready_csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Datoteka ne postoji: {analysis_ready_csv_path}")

    df = pd.read_csv(csv_path)

    df = create_target_variable(df)

    y = df["concept_result_label"]

    class_distribution = {
        "lower_result": int((y == 0).sum()),
        "higher_result": int((y == 1).sum()),
    }

    aaps_only_results = run_single_feature_set_analysis(
        df=df,
        y=y,
        feature_set_key="aaps_only",
        feature_set_name="AAPS-only"
    )

    aaps_demographic_results = run_single_feature_set_analysis(
        df=df,
        y=y,
        feature_set_key="aaps_demographic",
        feature_set_name="AAPS + demografija"
    )

    feature_set_results = [
        aaps_only_results,
        aaps_demographic_results,
    ]

    best_overall = max(
        feature_set_results,
        key=lambda item: item["model_results"][0]["f1"]
    )

    task_prediction_results = run_task_prediction_analysis(df)
    nlp_text_prediction = run_nlp_text_prediction_analysis(df)
    likert_from_text = run_likert_from_text_analysis(df)

    return {
        "target_description": "Predikcija nižeg ili višeg rezultata na konceptualnim zadacima",
        "target_rule": "0 ili 1 točan odgovor = niži rezultat; 2, 3 ili 4 točna odgovora = viši rezultat",
        "num_samples": int(len(df)),
        "class_distribution": class_distribution,
        "feature_set_results": feature_set_results,
        "task_prediction_results": task_prediction_results,
        "nlp_text_prediction": nlp_text_prediction,
        "likert_from_text": likert_from_text,
        "best_overall_feature_set": best_overall["feature_set_name"],
        "best_overall_model": best_overall["best_model"],

        "num_features": aaps_demographic_results["num_features"],
        "model_results": aaps_demographic_results["model_results"],
        "best_model": aaps_demographic_results["best_model"],
        "confusion_matrix": aaps_demographic_results["confusion_matrix"],
        "feature_importance": aaps_demographic_results["feature_importance"],
        "summary": (
            f"Najbolja ukupna kombinacija bila je {best_overall['feature_set_name']} "
            f"s modelom {best_overall['best_model']}. "
            f"Za osnovni prikaz koristi se model s AAPS i demografskim značajkama."
        ),
    }