from django.shortcuts import render
import pandas as pd
import json
from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO
from pathlib import Path
from django.conf import settings
from statistical_analysis import run_statistical_analysis
from encoder import SurveyEncoder
from ml_analysis import run_ml_analysis

def upload_view(request):
    if request.method == "POST":
        config_file = request.FILES["config"]
        answers_file = request.FILES["answers"]

        config = json.load(config_file)
        df = pd.read_csv(answers_file)

        encoder = SurveyEncoder(config)
        encoded_df = encoder.encode_dataframe(df)

        encoded_path = Path(settings.BASE_DIR) / "encoded.csv"
        encoded_df.to_csv(encoded_path, index=False, encoding="utf-8-sig")

        buffer = BytesIO()
        encoded_df.to_csv(buffer, index=False)
        buffer.seek(0)

        response = HttpResponse(buffer, content_type="text/csv")
        response['Content-Disposition'] = 'attachment; filename="encoded.csv"'
        return response

    return render(request, "main/upload.html")

def statistics_view(request):
    encoded_path = Path(settings.BASE_DIR) / "encoded.csv"

    if not encoded_path.exists():
        return render(request, "main/statistics.html", {
            "error": "Encoded CSV još nije generiran."
        })

    analysis = run_statistical_analysis(str(encoded_path))

    return render(request, "main/statistics.html", {
        "analysis": analysis
    })

def ml_view(request):
    analysis_ready_path = Path(settings.BASE_DIR) / "encoded_analysis_ready.csv"

    if not analysis_ready_path.exists():
        return render(request, "main/ml.html", {
            "error": "Datoteka encoded_analysis_ready.csv još nije generirana."
        })

    try:
        ml_results = run_ml_analysis(str(analysis_ready_path))

        return render(request, "main/ml.html", {
            "ml": ml_results
        })

    except Exception as e:
        return render(request, "main/ml.html", {
            "error": f"Greška pri izvođenju ML analize: {str(e)}"
        })