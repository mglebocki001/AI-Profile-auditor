import os
import json
import time
from pathlib import Path
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# 1. PRECYZYJNY SCHEMAT DANYCH (Rozróżnienie UI aplikacji randkowych)
# ---------------------------------------------------------------------------

class ProfileBioSection(BaseModel):
    raw_bio_text: Optional[str] = Field(
        default=None, 
        description="Rzeczywisty opis pisany przez użytkownika (Bio/Prompt). NIE umieszczaj tu tagów ani kategorii."
    )
    categories_and_badges: List[str] = Field(
        default_factory=list, 
        description="Sztywne kategorie/plakietki z interfejsu aplikacji (np. Wzrost, Zodiak, Wykształcenie, Stanowisko, Alkohol, Papierosy, Dzieci)."
    )
    interests_and_passions: List[str] = Field(
        default_factory=list, 
        description="Zainteresowania, hobby, tagi i pasje wybierane z listy (np. 'Kawa', 'Muzyka', 'Podróże', 'Gym')."
    )

class VisualRuleDetail(BaseModel):
    rule_name: str = Field(
        description="Nazwa wykrytej cechy wizualnej, błędu lub zalety na zdjęciach."
    )
    category: str = Field(
        description="Kategoria cechy wizualnej (np. Jakość i Oświetlenie, Ekspozycja Twarzy, Mowa Ciała, Otoczenie, Sylwetka)."
    )
    full_description: str = Field(
        description="Dokładny opis elementu widocznego na kadrze zdjęć."
    )
    what_to_look_for: List[str] = Field(
        default_factory=list, 
        description="Detale wizualne na zdjęciu (np. 'ciemne okulary', 'bałagan w tle', 'uśmiech', 'zdjęcie w grupie')."
    )
    photo_index: Optional[int] = Field(
        default=None,
        description="Numer zdjęcia w kolejności, na którym wykryto daną cechę (np. 1, 2, 3)."
    )

class SingleProfileFullAudit(BaseModel):
    profile_name: str = Field(description="Imię i wiek użytkownika z profilu (np. 'Ania, 25' lub 'Nieznany').")
    total_photos_analyzed: int = Field(description="Liczba przeanalizowanych zdjęć należących do tego profilu.")
    one_great_thing: str = Field(
        description="JEDNA KONKRETNA, NAJWIĘKSZA ZALETA WYKRYTA NA ZDJĘCIACH LUB W OPISIE PROFILU. Musi opisywać rzeczywisty atut wizualny lub tekstowy użytkownika."
    )
    profile_text_data: ProfileBioSection = Field(description="Drobiazgowo rozdzielone dane tekstowe z profilu.")
    extracted_rules: List[VisualRuleDetail] = Field(description="Zbiorcza lista wszystkich cech wizualnych ze wszystkich zdjęć razem.")


# ---------------------------------------------------------------------------
# 2. LOGIKA PRZETWARZANIA PACZKI ZDJĘĆ W JEDNYM AUDYCIE
# ---------------------------------------------------------------------------

def process_single_profile(client: genai.Client, image_paths: List[Path]) -> Optional[SingleProfileFullAudit]:
    """Przesyła WSZYSTKIE zdjęcia jednego profilu równolegle do Gemini i zwraca JEDEN audyt."""
    print(f"🔄 Analizowanie pełnego profilu z {len(image_paths)} zdjęć...")
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"❌ Błąd podczas otwierania pliku {path.name}: {e}")

    if not images:
        return None

    # Ścisłe instrukcje eliminujące generyczne pochwały i mylenie opisu z kategoriami
    prompt = """
Jesteś profesjonalnym, wymagającym i bezstronnym audytorem profili w aplikacjach randkowych (Tinder, Bumble, Hinge).

Otrzymasz ZAŁĄCZONY ZESTAW OBRAZÓW przedstawiających JEDEN PEŁNY PROFIL UŻYTKOWNIKA. Przeanalizuj WSZYSTKIE załączone obrazy jako jedną spójną całość.

ZASADY ANALIZY:
1. Opieraj się WYŁĄCZNIE na informacjach widocznych na obrazach. Nie zgaduj wieku, wzrostu, zawodu, majątku ani charakteru, jeśli nie wynikają z tekstu/zdjęć.
2. Policz dokładnie wszystkie przesłane zdjęcia profilowe i zapisz ich łączną liczbę.
3. Zanim przejdzesz do audytu, ODDZIELNIE wypisz czyste, surowe obserwacje wizualne (co dokładnie jest widoczne na każdym zdjęciu oraz CZEGO NA NIM BRAKUJE).
4. Odróżniaj czystą obserwację (co widać / czego brakuje) od interpretacji i oceny (jaki ma to wpływ na odbiorcę).
5. Oceniaj skuteczność profilu, jakość i odbiór – nigdy nie oceniaj atrakcyjności w sposób obraźliwy.
6. Unikaj pustych ogólników. Każdą ocenę poprzyj konkretnym elementem obrazu, tekstu lub ich BRAKIEM.
7. Numeruj zdjęcia zgodnie z kolejnością przesłania: Zdjęcie 1, Zdjęcie 2, Zdjęcie 3 itd.
8. PRZEPROWADŹ KONTROLĘ BRAKÓW: Zidentyfikuj brak kluczowych elementów profilu (np. brak zdjęcia całej sylwetki, brak naturalnego uśmiechu, brak widocznych oczu przez okulary przeciwsłoneczne, brak interakcji z otoczeniem, pusty opis).

WYMAGANY FORMAT WYJŚCIOWY:
Zwróć wynik WYŁĄCZNIE w poprawnym formacie JSON. 
ZASADA KRYTYCZNA JSON: Nie umieszczaj wiszących przecinków (trailing commas) na końcu tablic ani obiektów!

Struktura JSON musi wyglądać dokładnie tak:

{
  "document_title": "Audyt Profilu - [Imię/Identyfikator]",
  "total_photos_count": 5,
  "raw_visual_observations": [
    {
      "photo_number": 1,
      "photo_type": "np. portret, selfie, cała sylwetka, zdjęcie grupowe, zdjęcie ze zwierzęciem",
      "subject_details": "Opis widocznej osoby: płeć/wygląd, ekspozycja twarzy, oczy, fryzura, mimika (uśmiech, powaga), okulary/nakrycie głowy, pozycja ciała",
      "clothing_and_grooming": "Szczegółowy opis ubioru, schludności, dodatków (np. czarny t-shirt, zegarek, koszula w kratę)",
      "lighting_and_quality": "Kierunek i jakość światła, ostrość, jakość techniczna, szum, ewentualne filtry",
      "background_and_environment": "Opis tła i otoczenia, poziom porządku/bałaganu, rozpraszające elementy",
      "framing_and_angle": "Kadr (bliski, średni, szeroki), perspektywa, odległość od aparatu",
      "missing_details": "Wskazanie braków na TYM konkretnym zdjęciu (np. zasłonięte oczy, brak uśmiechu, obcięta głowa, pozbawione kontekstu neutralne tło)"
    }
  ],
  "profile_checklist": {
    "has_clear_full_body_photo": true,
    "has_clear_unobscured_face_photo": true,
    "has_genuine_smile": false,
    "has_hobby_or_activity_photo": false,
    "has_filled_bio": true,
    "critical_missing_elements": [
      "Wymień najważniejsze elementy, których BEZWZGLĘDNIE brakuje na profilu (np. Brak zdjęcia całej sylwetki, Brak zdjęcia z pasją/aktywnością)"
    ]
  },
  "raw_bio_text": "Dokładny ręczny tekst napisany przez użytkownika (zachowaj pisownię i emoji). Puste, jeśli brak.",
  "categories_and_badges": [
    {"category": "Wzrost", "value": "180 cm"}
  ],
  "interests_and_passions": ["Kawa", "Podróże"],
  "prompts_and_answers": [
    {"prompt": "Treść pytania", "answer": "Treść odpowiedzi"}
  ],
  "one_great_thing": "DOKŁADNIE JEDNA najmocniejsza zaleta ze wskazaniem źródła. Brak ogólników!",
  "extracted_rules": [
    {
      "rule_name": "Krótka nazwa reguły lub krytycznego braku (np. Brak zdjęcia całej sylwetki)",
      "section": "Kategoria (np. Liczba i Różnorodność Zdjęć, Ekspozycja Twarzy, Jakość Zdjęć, Bio)",
      "severity": "CRITICAL / HIGH / MEDIUM / LOW",
      "full_description": "Dokładny opis braku lub problemu na podstawie surowych danych.",
      "what_to_look_for": ["brak_sylwetki", "okulary_sloneczne", "puste_bio"],
      "why_it_matters": "Wyjaśnienie, dlaczego brak tego elementu budzi niepewność lub obniża atrakcyjność profilu.",
      "quick_fix": "Instrukcja, jakie zdjęcie/tekst należy dodać, aby uzupełnić tę lukę.",
      "positive_impact": null
    }
  ],
  "summary": {
    "photos_count_evaluation": "Ocena ilości zdjęć.",
    "top_3_pros": ["Zaleta 1", "Zaleta 2", "Zaleta 3"],
    "top_3_cons": ["Kluczowy brak 1", "Kluczowy brak 2", "Kluczowy brak 3"],
    "single_biggest_improvement": "Jedna zmiana/dodatek o największym potencjale (często uzupełnienie najważniejszego braku)",
    "recommended_photo_order": [1, 3, 2, 4],
    "photo_action_plan": "Wskazówki, które zdjęcia usunąć, przenieść lub DODAĆ (by uzupełnić braki)."
  }
}
"""


    contents = [*images, prompt]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SingleProfileFullAudit,
                    temperature=0.1
                ),
            )
            
            return SingleProfileFullAudit.model_validate_json(response.text)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                wait_time = 10
                print(f"⏳ Limit API (429/Quota). Czekam {wait_time} s przed próbą {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"❌ Błąd podczas wywołania API dla profilu: {e}")
                return None

    return None


def run_batch_extraction(input_dir: str, output_dir: str):
    """Zbiega wszystkie zdjęcia z folderu i generuje z nich JEDEN plik audytu JSON."""
    if API_KEY == "TUTAJ_WKLEJ_SWOJ_KLUCZ_API_GEMINI":
        print("❌ BŁĄD: Musisz wkleić swój klucz API na początku pliku!")
        return

    client = genai.Client(api_key=API_KEY)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in supported_extensions])

    if not images:
        print(f"⚠️ Nie znaleziono plików graficznych w katalogu: {input_path.resolve()}")
        return

    print(f"🚀 Znaleziono łącznie {len(images)} obrazów. Łączę je w JEDEN audyt profilu...\n")

    # Przetwarzanie całej paczki obrazów równolegle w jednym zapytaniu
    result = process_single_profile(client, images)

    if result:
        output_file = output_path / "FULL_PROFILE_AUDIT.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
            
        print("="*65)
        print("✅ GENIALNIE! Wygenerowano JEDEN skumulowany raport audytu.")
        print(f"📁 Plik zapisano w: {output_file.name}")
        print(f"👤 Wykryty profil: {result.profile_name}")
        print(f"🌟 Główna zaleta profilu: {result.one_great_thing}")
        print(f"📝 Długość opisu (Bio): {len(result.profile_text_data.raw_bio_text or '')} znaków")
        print(f"🏷️ Wykryte kategorie systemowe: {len(result.profile_text_data.categories_and_badges)}")
        print(f"🎯 Wykryte zainteresowania: {len(result.profile_text_data.interests_and_passions)}")
        print(f"🔍 Wszystkich zebranych reguł wizualnych: {len(result.extracted_rules)}")
        print("="*65 + "\n")


if __name__ == "__main__":
    INPUT_FOLDER = "./photos"       # Włóż tutaj WSZYSTKIE zdjęcia danego profilu
    OUTPUT_FOLDER = "./OCR_result"   # Tu powstanie JEDEN plik FULL_PROFILE_AUDIT.json
    
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    run_batch_extraction(INPUT_FOLDER, OUTPUT_FOLDER)