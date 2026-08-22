import os
import gc
import json
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5.6-terra"

# ---------------------------------------------------------------------------
# 1. SCHEMAT DLA MODELU — WYŁĄCZNIE część generatywna/kreatywna.
#    Dane per-zdjęcie NIE wchodzą tu w ogóle.
# ---------------------------------------------------------------------------
class ImprovementItem(BaseModel):
    title: str = Field(description="Krótkie, chwytliwe nazwanie błędu/zmiany (max 5 słów).")
    why_it_matters: str = Field(description="Dlaczego to szkodzi w odbiorze profilu (1 zdanie).")
    quick_fix: str = Field(description="Prosta, konkretna instrukcja co zrobić od razu (1-2 zdania).")


class BioSuggestion(BaseModel):
    improved_bio: str = Field(
        description="Gotowe do wklejenia, poprawione bio w TYM SAMYM języku co oryginał. 2-4 zdania, naturalny ton, bez emoji-spamu, z co najmniej jednym konkretnym punktem zaczepienia (pytaniem lub zaproszeniem do rozmowy)."
    )
    why_better: str = Field(
        description="1-2 zdania: co konkretnie poprawiono względem oryginału i dlaczego to zadziała lepiej."
    )


class LLMGeneratedSummary(BaseModel):
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
    top_5_improvements: List[ImprovementItem] = Field(
        description="Dokładnie 5 najważniejszych i najłatwiejszych do wdrożenia poprawek, ułożonych od najbardziej kluczowych."
    )
    bio_suggestion: BioSuggestion = Field(
        description="Poprawiona propozycja bio użytkownika."
    )


# ---------------------------------------------------------------------------
# 2. POMOCNICZE: mapowanie akcji zdjęcia na kolor dla frontendu
# ---------------------------------------------------------------------------
_ACTION_COLOR_MAP = {
    "Zostaw": "green",
    "Usuń": "red",
    "Podobne": "orange",
    "Info": "blue",
}


def _action_to_color(action: str) -> str:
    return _ACTION_COLOR_MAP.get(action, "orange")


def _build_photo_by_photo(audit_data: dict) -> List[dict]:
    """Składa gotową do wyrenderowania listę kart zdjęć 1:1 z danych
    audytu (Audit.py) — bez udziału modelu, żeby uniknąć rozjazdu między
    tym, co faktycznie widać na zdjęciu, a tym, co napisze LLM."""
    photo_evaluations = audit_data.get("photo_evaluations", [])
    cards = []
    for p in photo_evaluations:
        action = p.get("action", "Podobne")
        cards.append({
            "photo_index": p.get("photo_index"),
            "action": action,
            "action_color": _action_to_color(action),
            "tags": [
                {"label": t.get("label", ""), "is_positive": t.get("is_positive", False)}
                for t in p.get("tags", [])
            ],
        })
    cards.sort(key=lambda c: c["photo_index"] if c["photo_index"] is not None else 0)
    return cards


# ---------------------------------------------------------------------------
# 3. SILNIK SYNTEZY RAPORTÓW (GPT-5.6-TERRA)
# ---------------------------------------------------------------------------
def generate_summary_for_reports(reports_dir: str, output_dir: str):
    if not API_KEY or API_KEY == "TUTAJ_WKLEJ_SWOJ_KLUCZ_API":
        print("❌ BŁĄD: Ustaw OPENAI_API_KEY w pliku .env!")
        return

    client = OpenAI(api_key=API_KEY)
    reports_path = Path(reports_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_files = list(reports_path.glob("*_report.json"))

    if not report_files:
        print(f"⚠️ Nie znaleziono plików raportów w: {reports_path.resolve()}")
        print("ℹ️ Uruchom najpierw skrypt analyze_profile.py!")
        return

    print(f"🚀 Znaleziono {len(report_files)} raportów z audytu. Generuję syntezę przez OpenAI API ({MODEL_NAME})...\n")

    combined_audits = []
    for rep in report_files:
        with open(rep, "r", encoding="utf-8") as f:
            combined_audits.append(json.load(f))

    primary_audit = combined_audits[0] if combined_audits else {}
    photo_by_photo = _build_photo_by_photo(primary_audit)
    bio_analysis = primary_audit.get("bio_analysis") or {}
    original_bio_text = bio_analysis.get("raw_bio_text") or ""
    bio_language = bio_analysis.get("language") or "pl"

    prompt = f"""
Przeanalizuj poniższe szczegółowe dane z audytu profilu na aplikacji randkowej:

{json.dumps(combined_audits, ensure_ascii=False, indent=2)}

Twoim zadaniem jest stworzenie krótkiego, potężnego podsumowania wygenerowanego bezpośrednio dla użytkownika, ORAZ propozycji poprawionego bio.

Wymagania:
1. W polu 'overall_action_plan_summary' napisz zwięzłą syntezę (3-5 zdań), w której podsumujesz cały profil: co jest dobrą bazą i należy zostawić, co bezwzględnie usunąć lub zmienić, oraz jaki jest główny priorytet zmian. Uwzględnij WSZYSTKIE dostępne dane: oceny poszczególnych zdjęć (photo_evaluations), globalne tagi (global_tags), analizę bio (bio_analysis) i wykryte reguły (detected_issues).
2. W polu 'one_great_thing' znajdź DOKŁADNIE 1 rzecz, która na tym profilu jest najwyższej jakości lub wyróżnia go na plus.
3. W polu 'top_5_improvements' wybierz DOKŁADNIE 5 najważniejszych, a zarazem NAJŁATWIEJSZYCH do natychmiastowego zastosowania zmian — czerp zarówno z detected_issues, jak i z profile_summary/bio_analysis.
4. W polu 'bio_suggestion' napisz gotową, poprawioną wersję bio użytkownika:
   - Oryginalne bio: "{original_bio_text}"
   - Język oryginału: {bio_language} — napisz poprawkę w TYM SAMYM języku.
   - Uwzględnij wykryte zainteresowania (interests_and_passions) i uwagi z bio_analysis.bio_suggestions, jeśli są dostępne.
   - Bio ma brzmieć naturalnie, jak coś, co realny użytkownik faktycznie by o sobie napisał — nie jak reklama.
5. Mów językiem korzyści, bezpośrednio, bez akademickiego lania wody.
6. Skup się na zmianach dających najwyższy zwrot z inwestycji czasu.
"""

    del combined_audits
    gc.collect()

    try:
        response = client.responses.parse(
            model=MODEL_NAME,
            input=[{
                "role": "user",
                "content": prompt
            }],
            text_format=LLMGeneratedSummary,
            reasoning={"effort": "medium"},
        )

        llm_data = response.output_parsed
        del response

        final_payload = {
            "profile_name": primary_audit.get("profile_name", "Profil"),
            "overall_score": primary_audit.get("overall_score", 0),
            "total_photos_analyzed": primary_audit.get("total_photos_analyzed", len(photo_by_photo)),
            "overview": {
                "overall_action_plan_summary": llm_data.overall_action_plan_summary,
                "one_great_thing": llm_data.one_great_thing,
                "top_5_improvements": [item.model_dump() for item in llm_data.top_5_improvements],
            },
            "photo_by_photo": photo_by_photo,
            "bio": {
                "original_bio": original_bio_text,
                "improved_bio": llm_data.bio_suggestion.improved_bio,
                "why_better": llm_data.bio_suggestion.why_better,
            },
        }

        print("============================================================")
        print("🌟 KOŃCOWE PODSUMOWANIE DLA UŻYTKOWNIKA")
        print("============================================================")
        print(f"\n📈 WYNIK OGÓLNY: {final_payload['overall_score']}/100\n")
        print(f"📋 OGÓLNY PLAN DZIAŁANIA I SYNTEZA:\n{llm_data.overall_action_plan_summary}\n")
        print("------------------------------------------------------------")
        print(f"✅ JEDNA RZECZ, KTÓRA JEST ŚWIETNA:\n{llm_data.one_great_thing}\n")
        print("------------------------------------------------------------")
        print("🛠️ 5 NAJWAŻNIEJSZYCH I NAJPROSTSZYCH POPRAWEK:")
        print("------------------------------------------------------------")
        for idx, item in enumerate(llm_data.top_5_improvements, 1):
            print(f"{idx}. {item.title.upper()}")
            print(f"   • Wpływ: {item.why_it_matters}")
            print(f"   • 💡 Szybka zmiana: {item.quick_fix}\n")
        print("------------------------------------------------------------")
        print("🖼️ ZDJĘCIA:")
        print("------------------------------------------------------------")
        for p in photo_by_photo:
            print(f"  Zdjęcie {p['photo_index']}: {p['action']}")
        print("------------------------------------------------------------")
        print("✍️ PROPOZYCJA POPRAWIONEGO BIO:")
        print("------------------------------------------------------------")
        print(f"  Oryginał: {original_bio_text or '(brak)'}")
        print(f"  Poprawione: {llm_data.bio_suggestion.improved_bio}")
        print(f"  Dlaczego lepsze: {llm_data.bio_suggestion.why_better}")

        final_file = output_path / "final_profile_summary.json"
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)

        print(f"\n📁 Zapisano gotowy plik podsumowania w: {final_file.resolve()}")

    except Exception as e:
        print(f"❌ Błąd podczas generowania podsumowania: {e}")
    finally:
        del client
        gc.collect()


if __name__ == "__main__":
    generate_summary_for_reports(
        reports_dir="./audit_reports",
        output_dir="./final_summary"
    )