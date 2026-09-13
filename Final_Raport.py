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

LANGUAGE_NAMES = {
    'pl': 'polskim',
    'en': 'angielskim (English)',
}

# ---------------------------------------------------------------------------
# Odpowiedzi z ankiety na /raport (patrz frontend: src/pages/Raport.jsx,
# GOAL_VALUES / ATTRACT_KEYS / SELF_KEYS) - mapujemy kody na czytelny opis,
# żeby model dostał gotowe zdanie zamiast surowego identyfikatora.
# ---------------------------------------------------------------------------
RELATIONSHIP_GOAL_LABELS = {
    'pl': {
        'serious': 'poważnej, długoterminowej relacji',
        'dating': 'randkowania i poznawania nowych osób',
        'casual': 'czegoś luźnego, bez większych zobowiązań',
        'nostrings': 'czegoś całkowicie niezobowiązującego',
        'unsure': 'jeszcze nie wie, czego dokładnie szuka',
    },
    'en': {
        'serious': 'a serious, long-term relationship',
        'dating': 'dating around and meeting new people',
        'casual': 'something casual, without big commitments',
        'nostrings': 'something completely no-strings-attached',
        'unsure': 'is not sure yet what they are looking for',
    },
}

# Klucze zgodne z i18n_complete.json -> raport.step2.options (kogo chce przyciągać).
ATTRACT_TRAIT_LABELS = {
    'pl': {
        'intelligent': 'inteligentne', 'funny': 'zabawne', 'active': 'aktywne',
        'family': 'rodzinne', 'social': 'towarzyskie', 'calm': 'spokojne',
        'ambitious': 'ambitne', 'other': 'inne',
    },
    'en': {
        'intelligent': 'intelligent', 'funny': 'funny', 'active': 'active',
        'family': 'family-oriented', 'social': 'sociable', 'calm': 'calm',
        'ambitious': 'ambitious', 'other': 'other people in general',
    },
}

# Klucze zgodne z i18n_complete.json -> raport.step3.options (jak chce być odbierany).
SELF_TRAIT_LABELS = {
    'pl': {
        'confident': 'pewny siebie', 'funny': 'zabawny', 'intelligent': 'inteligentny',
        'warm': 'ciepły', 'attractive': 'atrakcyjny', 'interesting': 'ciekawy',
        'mature': 'dojrzały', 'active': 'aktywny',
    },
    'en': {
        'confident': 'confident', 'funny': 'funny', 'intelligent': 'intelligent',
        'warm': 'warm', 'attractive': 'attractive', 'interesting': 'interesting',
        'mature': 'mature', 'active': 'active',
    },
}


def _describe_goal(code: str, language: str) -> Optional[str]:
    labels = RELATIONSHIP_GOAL_LABELS.get(language, RELATIONSHIP_GOAL_LABELS['pl'])
    return labels.get((code or '').strip())


def _describe_traits(codes_csv: str, mapping: dict, language: str) -> List[str]:
    if not codes_csv:
        return []
    labels = mapping.get(language, mapping['pl'])
    codes = [c.strip() for c in codes_csv.split(',') if c.strip()]
    return [labels.get(c, c) for c in codes]


def _build_preferences_block(relationship_goal: str, attract_traits: str, self_traits: str, language: str) -> str:
    """Składa czytelny opis odpowiedzi z ankiety /raport do wstrzyknięcia w prompt.
    Zwraca pusty string, gdy użytkownik nic nie wybrał (np. stare wywołania API
    sprzed dodania ankiety) - wtedy prompt po prostu pomija tę sekcję."""
    lines = []
    goal_desc = _describe_goal(relationship_goal, language)
    attract_list = _describe_traits(attract_traits, ATTRACT_TRAIT_LABELS, language)
    self_list = _describe_traits(self_traits, SELF_TRAIT_LABELS, language)

    if language == 'pl':
        if goal_desc:
            lines.append(f"- Szuka na aplikacji: {goal_desc}.")
        if attract_list:
            lines.append(f"- Chce przyciągać osoby, które są: {', '.join(attract_list)}.")
        if self_list:
            lines.append(f"- Chce, żeby profil sprawiał wrażenie, że jest: {', '.join(self_list)}.")
    else:
        if goal_desc:
            lines.append(f"- Looking for: {goal_desc}.")
        if attract_list:
            lines.append(f"- Wants to attract people who are: {', '.join(attract_list)}.")
        if self_list:
            lines.append(f"- Wants the profile to come across as: {', '.join(self_list)}.")

    return "\n".join(lines)


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
def generate_summary_for_reports(
    reports_dir: str,
    output_dir: str,
    language: str = 'pl',
    relationship_goal: str = '',
    attract_traits: str = '',
    self_traits: str = '',
):
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
    response_language_name = LANGUAGE_NAMES.get(language, LANGUAGE_NAMES['pl'])
    preferences_block = _build_preferences_block(relationship_goal, attract_traits, self_traits, language)

    if preferences_block:
        if language == 'pl':
            preferences_section = f"""
Użytkownik dodatkowo odpowiedział na krótką ankietę o swoich celach randkowych:
{preferences_block}

WAŻNE: uwzględnij te preferencje przy pisaniu 'overall_action_plan_summary', doborze
'top_5_improvements' oraz - przede wszystkim - w treści 'improved_bio'. Ton, słownictwo
i zaproszenie do rozmowy w poprawionym bio mają pasować do powyższego celu i cech, które
użytkownik chce przekazać (np. poważny cel = cieplejszy, dojrzalszy ton i pytanie
zachęcające do głębszej rozmowy; coś luźnego/na szybko = lżejszy, żartobliwy ton).
"""
        else:
            preferences_section = f"""
The user additionally answered a short survey about their dating goals:
{preferences_block}

IMPORTANT: take these preferences into account when writing 'overall_action_plan_summary',
choosing 'top_5_improvements', and - most importantly - in the 'improved_bio' content. The
tone, vocabulary, and conversation hook in the improved bio should match the goal and traits
above (e.g. a serious goal = warmer, more mature tone and a question inviting deeper
conversation; something casual = a lighter, more playful tone).
"""
    else:
        preferences_section = ""

    prompt = f"""
Przeanalizuj poniższe szczegółowe dane z audytu profilu na aplikacji randkowej:

{json.dumps(combined_audits, ensure_ascii=False, indent=2)}
{preferences_section}
Twoim zadaniem jest stworzenie krótkiego, potężnego podsumowania wygenerowanego bezpośrednio dla użytkownika, ORAZ propozycji poprawionego bio.

⚠️ WAŻNA ZASADA JĘZYKOWA (przestrzegaj rygorystycznie, dotyczy każdego pola z osobna):
- Pola 'overall_action_plan_summary', 'one_great_thing', 'top_5_improvements' (title, why_it_matters, quick_fix) oraz 'why_better' MUSZĄ być napisane w języku {response_language_name} - to są komentarze/wyjaśnienia DLA użytkownika czytającego stronę w tym języku.
- Pole 'improved_bio' jest WYJĄTKIEM od powyższej zasady: MUSI pozostać w oryginalnym języku bio użytkownika ({bio_language}), niezależnie od języka {response_language_name} - bo użytkownik wklei ten tekst bezpośrednio na swój profil randkowy, który jest w tamtym języku.

Wymagania:
1. W polu 'overall_action_plan_summary' napisz zwięzłą syntezę (3-5 zdań), w której podsumujesz cały profil: co jest dobrą bazą i należy zostawić, co bezwzględnie usunąć lub zmienić, oraz jaki jest główny priorytet zmian. Uwzględnij WSZYSTKIE dostępne dane: oceny poszczególnych zdjęć (photo_evaluations), globalne tagi (global_tags), analizę bio (bio_analysis) i wykryte reguły (detected_issues).
2. W polu 'one_great_thing' znajdź DOKŁADNIE 1 rzecz, która na tym profilu jest najwyższej jakości lub wyróżnia go na plus.
3. W polu 'top_5_improvements' wybierz DOKŁADNIE 5 najważniejszych, a zarazem NAJŁATWIEJSZYCH do natychmiastowego zastosowania zmian — czerp zarówno z detected_issues, jak i z profile_summary/bio_analysis.
4. W polu 'bio_suggestion' napisz gotową, poprawioną wersję bio użytkownika:
   - Oryginalne bio: "{original_bio_text}"
   - Język oryginału: {bio_language} — napisz 'improved_bio' w TYM SAMYM języku (patrz zasada językowa powyżej).
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