import os
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# 2. STRUKTURA WYJŚCIOWA (Pydantic Schema)
# ---------------------------------------------------------------------------
class ImprovementItem(BaseModel):
    title: str = Field(description="Krótkie, chwytliwe nazwanie błędu/zmiany (max 5 słów).")
    why_it_matters: str = Field(description="Dlaczego to szkodzi w odbiorze profilu (1 zdanie).")
    quick_fix: str = Field(description="Prosta, konkretna instrukcja co zrobić od razu (1-2 zdania).")


class ProfileExecutiveSummary(BaseModel):
    overall_action_plan_summary: str = Field(
        description=(
            "Ogólne podsumowanie i synteza całego profilu w 3-5 zdaniach. "
            "Opisz wprost: co bezwzględnie zostawić (mocne strony), co natychmiast wyrzucić/zmienić "
            "oraz jaki jest ogólny cel i kierunek poprawek."
        )
    )
    one_great_thing: str = Field(
        description="Jedna najlepsza rzecz, którą użytkownik zrobił dobrze na profilu. Pochwal i wytłumacz dlaczego to działa."
    )
    top_5_improvements: list[ImprovementItem] = Field(
        description="Dokładnie 5 najważniejszych i najłatwiejszych do wdrożenia poprawek, ułożonych od najbardziej kluczowych."
    )


# ---------------------------------------------------------------------------
# 3. SILNIK SYNTEZY RAPORTÓW
# ---------------------------------------------------------------------------
def generate_summary_for_reports(reports_dir: str, output_dir: str):
    if API_KEY == "TUTAJ_WKLEJ_SWOJ_KLUCZ_API_GEMINI":
        print("❌ BŁĄD: Podaj swój klucz API na początku pliku!")
        return

    client = genai.Client(api_key=API_KEY)
    reports_path = Path(reports_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_files = list(reports_path.glob("*_report.json"))

    if not report_files:
        print(f"⚠️ Nie znaleziono plików raportów w: {reports_path.resolve()}")
        print("ℹ️ Uruchom najpierw skrypt analyze_profile.py!")
        return

    print(f"🚀 Znaleziono {len(report_files)} raportów z audytu. Generuję syntezę przez Gemini API...\n")

    # Wczytanie wszystkich zgromadzonych raportów audytu
    combined_audits = []
    for rep in report_files:
        with open(rep, "r", encoding="utf-8") as f:
            combined_audits.append(json.load(f))

    prompt = f"""
Przeanalizuj poniższe szczegółowe dane z audytu profilu na aplikacji randkowej:

{json.dumps(combined_audits, ensure_ascii=False, indent=2)}

Twoim zadaniem jest stworzenie krótkiego, potężnego podsumowania wygenerowanego bezpośrednio dla użytkownika.

Wymagania:
1. W polu 'overall_action_plan_summary' napisz zwięzłą syntezę (3-5 zdań), w której podsumujesz cały profil: co jest dobrą bazą i należy zostawić, co bezwzględnie usunąć lub zmienić, oraz jaki jest główny priorytet zmian.
2. W polu 'one_great_thing' znajdź DOKŁADNIE 1 rzecz, która na tym profilu jest najwyższej jakości lub wyróżnia go na plus.
3. W polu 'top_5_improvements' wybierz DOKŁADNIE 5 najważniejszych, a zarazem NAJŁATWIEJSZYCH do natychmiastowego zastosowania zmian.
4. Mów językiem korzyści, bezpośrednio, bez akademickiego lania wody.
5. Skup się na zmianach dających najwyższy zwrot z inwestycji czasu.
"""

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProfileExecutiveSummary,
                temperature=0.2,
            ),
        )

        final_data = ProfileExecutiveSummary.model_validate_json(response.text)

        # Drukowanie w konsoli
        print("============================================================")
        print("🌟 KOŃCOWE PODSUMOWANIE DLA UŻYTKOWNIKA")
        print("============================================================")
        print(f"\n📋 OGÓLNY PLAN DZIAŁANIA I SYNTEZA:\n{final_data.overall_action_plan_summary}\n")
        print("------------------------------------------------------------")
        print(f"✅ JEDNA RZECZ, KTÓRA JEST ŚWIETNA:\n{final_data.one_great_thing}\n")
        print("------------------------------------------------------------")
        print("🛠️ 5 NAJWAŻNIEJSZYCH I NAJPROSTSZYCH POPRAWEK:")
        print("------------------------------------------------------------")
        for idx, item in enumerate(final_data.top_5_improvements, 1):
            print(f"{idx}. {item.title.upper()}")
            print(f"   • Wpływ: {item.why_it_matters}")
            print(f"   • 💡 Szybka zmiana: {item.quick_fix}\n")

        # Zapis do pliku
        final_file = output_path / "final_profile_summary.json"
        with open(final_file, "w", encoding="utf-8") as f:
            f.write(final_data.model_dump_json(indent=2))

        print(f"📁 Zapisano gotowy plik podsumowania w: {final_file.resolve()}")

    except Exception as e:
        print(f"❌ Błąd podczas generowania podsumowania: {e}")


if __name__ == "__main__":
    generate_summary_for_reports(
        reports_dir="./audit_reports",
        output_dir="./final_summary"
    )