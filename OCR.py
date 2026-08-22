import os
import gc
import json
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5.6-terra"

MAX_DIMENSION = 1024
JPEG_QUALITY = 82

class PhotoTag(BaseModel):
    label: str = Field(description="Krótka etykieta cechy np. 'Wyraźnie widoczna twarz', 'Słabe światło'.")
    is_positive: bool = Field(description="True dla zalety, False dla wady.")

class PhotoEvaluation(BaseModel):
    photo_index: int = Field(description="Numer obrazu (1-based) DOKŁADNIE zgodny z kolejnością, w jakiej obrazy zostały przesłane. Nigdy nie pomijaj numerów.")
    action: str = Field(description="Rekomendacja: 'Zostaw', 'Usuń', 'Podobne' lub 'Info' (dla ekranów, które NIE są zdjęciem osoby - np. czysty ekran z opisem/bio/statystykami/promptami).")
    tags: List[PhotoTag] = Field(description="Etykiety oceniające zdjęcie. Dla obrazów typu 'Info' użyj tagów opisujących zawartość ekranu (np. 'Ekran z opisem profilu'), NIGDY tagów o twarzy/ubiorze/świetle.")

class BioAnalysis(BaseModel):
    has_bio: bool = Field(description="Czy użytkownik posiada opis/bio.")
    raw_bio_text: str = Field(description="Dokładna treść opisu przepisana z profilu (lub pusty string if brak).")
    character_count: int = Field(description="Liczba znaków w opisach/bio.")
    word_count: int = Field(description="Liczba słów w opisie.")
    emoji_count: int = Field(description="Liczba użytych emoji.")
    language: str = Field(description="Język opisu (np. 'pl', 'en', 'brak').")
    bio_evaluation: str = Field(description="Ocena jakości bio np. 'Za krótki', 'Optymalny', 'Brak opisu', 'Przeładowany emoji'.")
    bio_suggestions: List[str] = Field(description="Wskazówki jak ulepszyć opis lub co warto dodać.")

class RawVisualObservation(BaseModel):
    photo_number: int
    photo_type: str
    subject_details: str
    clothing_and_grooming: str
    lighting_and_quality: str
    background_and_environment: str
    framing_and_angle: str
    missing_details: str

class ProfileChecklist(BaseModel):
    has_clear_full_body_photo: bool
    has_clear_unobscured_face_photo: bool
    has_genuine_smile: bool
    has_hobby_or_activity_photo: bool
    has_filled_bio: bool
    critical_missing_elements: List[str]

class CategoryBadge(BaseModel):
    category: str
    value: str

class PromptAnswer(BaseModel):
    prompt: str
    answer: str

class ExtractedRule(BaseModel):
    rule_name: str
    section: str
    severity: str
    full_description: str
    what_to_look_for: List[str]
    why_it_matters: str
    quick_fix: str
    positive_impact: Optional[str] = None

class ProfileSummary(BaseModel):
    photos_count_evaluation: str
    top_3_pros: List[str]
    top_3_cons: List[str]
    single_biggest_improvement: str
    recommended_photo_order: List[int]
    photo_action_plan: str

class SingleProfileFullAudit(BaseModel):
    document_title: str = Field(default="Audyt Profilu")
    overall_score: int = Field(description="Ocena jakości profilu 0-100.")
    total_photos_analyzed: int = Field(description="Liczba analizowanych zdjęć.")
    photo_evaluations: List[PhotoEvaluation]
    global_tags: List[PhotoTag]
    bio_analysis: BioAnalysis = Field(description="Szczegółowa detekcja i analiza opisu (bio).")
    raw_visual_observations: List[RawVisualObservation]
    profile_checklist: ProfileChecklist
    categories_and_badges: List[CategoryBadge]
    interests_and_passions: List[str]
    prompts_and_answers: List[PromptAnswer]
    one_great_thing: str
    extracted_rules: List[ExtractedRule]
    summary: ProfileSummary

def _load_compressed_jpeg_bytes(path: Path) -> Optional[bytes]:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            return buffer.getvalue()
    except Exception as e:
        print(f"❌ Błąd podczas przetwarzania pliku {path.name}: {e}")
        return None


def _reconcile_photo_evaluations(result: "SingleProfileFullAudit", num_images: int) -> None:
    """Zabezpieczenie na wypadek, gdyby model mimo instrukcji pominął/przesunął
    numerację obrazów (np. pomijając ekran z samym bio). Działa w miejscu
    (mutuje result.photo_evaluations), żeby dalsza część kodu (Final_Raport.py,
    frontend) zawsze mogła bezpiecznie zakładać: photo_index i-tej oceny
    odpowiada i-temu przesłanemu plikowi, 1:1, bez luk."""
    evaluations = result.photo_evaluations

    if len(evaluations) != num_images:
        print(
            f"⚠️ UWAGA: model zwrócił {len(evaluations)} ocen zdjęć, "
            f"a przesłano {num_images} obrazów. Wymuszam wyrównanie kolejności - "
            f"sprawdź logi/wynik ręcznie, bo część ocen może nie odpowiadać "
            f"właściwym zdjęciom."
        )

    # Ufamy KOLEJNOŚCI, w jakiej model zwrócił listę (to zwykle odpowiada
    # kolejności przesłanych obrazów, nawet gdy model pomyli się w numeracji
    # photo_index) i wymuszamy numerację 1..N zgodną z pozycją w liście.
    for position, evaluation in enumerate(evaluations, start=1):
        evaluation.photo_index = position

    if len(evaluations) < num_images:
        # Model pominął jakiś obraz - dopełniamy listę, żeby liczba ocen
        # zawsze zgadzała się z liczbą przesłanych plików (frontend indeksuje
        # zdjęcia po pozycji, więc brakujący wpis psuje wszystko co po nim).
        for position in range(len(evaluations) + 1, num_images + 1):
            evaluations.append(
                PhotoEvaluation(
                    photo_index=position,
                    action="Info",
                    tags=[PhotoTag(
                        label="Nie udało się jednoznacznie ocenić tego obrazu automatycznie",
                        is_positive=True,
                    )],
                )
            )
    elif len(evaluations) > num_images:
        del evaluations[num_images:]

    result.total_photos_analyzed = num_images


def process_single_profile(client: OpenAI, image_paths: List[Path]) -> Optional[SingleProfileFullAudit]:
    print(f"🔄 Rozpoczynanie analizy dla {len(image_paths)} zdjęć...")

    image_parts = []
    for idx, path in enumerate(image_paths, 1):
        print(f"   📸 Ładowanie zdjęcia {idx}/{len(image_paths)}: {path.name}")
        jpeg_bytes = _load_compressed_jpeg_bytes(path)
        if jpeg_bytes is not None:
            b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
            image_parts.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64_image}",
            })
            del jpeg_bytes

    if not image_parts:
        print("❌ Nie udało się przetworzyć żadnego zdjęcia.")
        return None

    num_images = len(image_parts)

    prompt = f"""
Jesteś profesjonalnym, wymagającym i bezstronnym audytorem profili w aplikacjach randkowych (Tinder, Bumble, Hinge).

Otrzymasz ZAŁĄCZONY ZESTAW DOKŁADNIE {num_images} OBRAZÓW przedstawiających JEDEN PEŁNY PROFIL UŻYTKOWNIKA, w konkretnej, ustalonej kolejności (obraz 1, obraz 2, ..., obraz {num_images}). Przeanalizuj WSZYSTKIE załączone obrazy jako jedną spójną całość.

ZASADY ANALIZY:
1. Opieraj się WYŁĄCZNIE na informacjach widocznych na obrazach. Nie zgaduj danych, jeśli nie wynikają z tekstu/zdjęć.
2. Wykryj i przeanalizuj OPIS PROFILU (BIO):
   - Odczytaj dokładną treść tekstu (raw_bio_text). Jeśli opisu brak, ustaw has_bio na false i wskaż to w opisie.
   - Zlicz dokładnie liczbę znaków (character_count), liczbę słów (word_count) oraz użyte emoji (emoji_count).
   - Określ język (language) oraz oszacuj jakość opisu (bio_evaluation) wraz ze wskazówkami poprawy (bio_suggestions).
3. Przeanalizuj KAŻDY z {num_images} obrazów, jeden po drugim i przydziel mu akcję oraz odpowiednie tagi.
4. Wylicz ogólną punktację profilu w skali 0–100 (overall_score).

⚠️ KRYTYCZNIE WAŻNA ZASADA DOTYCZĄCA `photo_evaluations` (rygorystycznie przestrzegaj):
- Tablica `photo_evaluations` MUSI zawierać DOKŁADNIE {num_images} elementów - po jednym na KAŻDY przesłany obraz, BEZ WYJĄTKU.
- `photo_index` musi być równy numerowi kolejnemu obrazu zgodnie z kolejnością przesyłania: pierwszy przesłany obraz = photo_index 1, drugi = photo_index 2, itd. aż do {num_images}. Nie wolno pomijać, łączyć ani przestawiać numerów.
- NIGDY nie pomijaj obrazu tylko dlatego, że nie przedstawia on bezpośrednio osoby (np. jest to ekran z samym tekstem opisu/bio, listą zainteresowań, statystykami typu wzrost/znak zodiaku, promptami/pytaniami). Taki obraz TEŻ dostaje swój wpis w `photo_evaluations`, tylko z action="Info" i tagami opisującymi WYŁĄCZNIE zawartość tego ekranu (np. "Ekran z opisem profilu", "Lista zainteresowań") - nigdy nie pisz dla niego o twarzy, ubiorze, świetle czy kadrze, bo to nie jest zdjęcie osoby.
- Przed zwróceniem odpowiedzi PRZELICZ swoje `photo_evaluations` i upewnij się, że ich liczba to dokładnie {num_images}, a numery photo_index to kolejno 1, 2, 3, ..., {num_images} bez luk i powtórzeń.

WYMAGANY FORMAT WYJŚCIOWY (JSON):

{
  "document_title": "Audyt Profilu - [Imię/Identyfikator]",
  "overall_score": 71,
  "total_photos_analyzed": 5,
  "bio_analysis": {
    "has_bio": true,
    "raw_bio_text": "Fani kawy i górskich wędrówek łączcie się! ☕🏔️ Szukam kogoś na niedzielny spacer i dobry sernik.",
    "character_count": 92,
    "word_count": 15,
    "emoji_count": 2,
    "language": "pl",
    "bio_evaluation": "Optymalny, ciepły i zachęcający do interakcji",
    "bio_suggestions": [
      "Możesz dodać jeden konkretny luźny temat do dyskusji (np. ulubiony szlak)",
      "Rozważ dodanie wymarzonego celu podróży"
    ]
  },
  "photo_evaluations": [
    {
      "photo_index": 1,
      "action": "Zostaw",
      "tags": [
        {"label": "Wyraźnie widoczna twarz", "is_positive": true},
        {"label": "Słabe światło", "is_positive": false}
      ]
    },
    {
      "photo_index": 2,
      "action": "Info",
      "tags": [
        {"label": "Ekran z opisem profilu (bio) - nie jest to zdjęcie osoby", "is_positive": true}
      ]
    }
  ],
  "global_tags": [
    {"label": "Dobry naturalny uśmiech", "is_positive": true},
    {"label": "Krótki, ale konkretny opis", "is_positive": true}
  ],
  "raw_visual_observations": [
    {
      "photo_number": 1,
      "photo_type": "portret",
      "subject_details": "Opis widocznej osoby...",
      "clothing_and_grooming": "Opis ubioru...",
      "lighting_and_quality": "Jakość światła...",
      "background_and_environment": "Tło...",
      "framing_and_angle": "Kadr...",
      "missing_details": "Braki na zdjęciu..."
    }
  ],
  "profile_checklist": {
    "has_clear_full_body_photo": false,
    "has_clear_unobscured_face_photo": true,
    "has_genuine_smile": true,
    "has_hobby_or_activity_photo": false,
    "has_filled_bio": true,
    "critical_missing_elements": [
      "Brak zdjęcia całej sylwetki"
    ]
  },
  "categories_and_badges": [
    {"category": "Wzrost", "value": "180 cm"}
  ],
  "interests_and_passions": ["Kawa", "Góry"],
  "prompts_and_answers": [],
  "one_great_thing": "Bardzo naturalny uśmiech na pierwszym zdjęciu.",
  "extracted_rules": [
    {
      "rule_name": "Brak zdjęcia całej sylwetki",
      "section": "Różnorodność Zdjęć",
      "severity": "CRITICAL",
      "full_description": "Brak zdjęcia pokazaującego sylwetkę.",
      "what_to_look_for": ["brak_sylwetki"],
      "why_it_matters": "Obniża zaufanie oglądającego.",
      "quick_fix": "Dodaj zdjęcie w pełnym kadrze.",
      "positive_impact": "Zwiększa szansę na parę."
    }
  ],
  "summary": {
    "photos_count_evaluation": "Ocena liczby zdjęć",
    "top_3_pros": ["Zaleta 1", "Zaleta 2", "Zaleta 3"],
    "top_3_cons": ["Wada 1", "Wada 2", "Wada 3"],
    "single_biggest_improvement": "Kluczowa zmiana",
    "recommended_photo_order": [1, 2],
    "photo_action_plan": "Plan działania dla zdjęć"
  }
}
"""

    contents = [{
        "role": "user",
        "content": [*image_parts, {"type": "input_text", "text": prompt}],
    }]

    print(f"🚀 Wysyłanie zapytania do API ({MODEL_NAME})...")
    try:
        response = client.responses.parse(
            model=MODEL_NAME,
            input=contents,
            text_format=SingleProfileFullAudit,
            reasoning={"effort": "medium"},
        )
        result = response.output_parsed
        print("✅ Pomyślnie odebrano odpowiedź z API OpenAI.")
        _reconcile_photo_evaluations(result, num_images)
        del response
        return result
    except Exception as e:
        print(f"❌ Błąd podczas wywołania API OpenAI: {e}")
        return None
    finally:
        del contents, image_parts
        gc.collect()


def run_batch_extraction(input_dir: str = "./photos", output_dir: str = "./OCR_result"):
    print("=" * 50)
    print("🚀 Inicjalizacja skryptu OCR.py")
    print("=" * 50)

    if not API_KEY or API_KEY == "TUTAJ_WKLEJ_SWOJ_KLUCZ_API":
        print("❌ BŁĄD: Brak poprawnego klucza OPENAI_API_KEY w pliku .env!")
        return

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in supported_extensions])

    if not images:
        print(f"⚠️ Brak zdjęć w katalogu wejściowym: {input_path.resolve()}")
        print("   Wrzuć zdjęcia (.jpg, .png, .webp) do wyżej wymienionego folderu i uruchom ponownie.")
        return

    print(f"📁 Znaleziono {len(images)} plików w katalogu {input_path.name}:")
    for img in images:
        print(f"   - {img.name}")

    client = OpenAI(api_key=API_KEY)
    result = process_single_profile(client, images)

    if result:
        output_file = output_path / "FULL_PROFILE_AUDIT.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        print(f"🎉 Sukces! Wynik zapisano w pliku: {output_file.resolve()}")

    del client
    gc.collect()


if __name__ == "__main__":
    run_batch_extraction()