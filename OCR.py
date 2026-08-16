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

# Zdjęcia z telefonu potrafią mieć 3000-4000px i po dekompresji do surowej
# bitmapy zajmować kilkadziesiąt MB KAŻDE (np. 4032x3024 RGB ≈ 36 MB).
# Przy 8-10 zdjęciach jednego profilu trzymanych naraz w pamięci to już
# 300-400 MB tylko na same bitmapy — zanim jeszcze cokolwiek wyślemy do API.
# Zamiast tego: każde zdjęcie skalujemy i kompresujemy do JPEG w locie,
# a surową bitmapę zwalniamy natychmiast po kompresji.
MAX_DIMENSION = 1024
JPEG_QUALITY = 82


# ---------------------------------------------------------------------------
# 1. PRECYZYJNY SCHEMAT DANYCH (bez zmian)
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
# 2. KOMPRESJA ZDJĘĆ (bez zmian — nadal kluczowa dla RAM)
# ---------------------------------------------------------------------------

def _load_compressed_jpeg_bytes(path: Path) -> Optional[bytes]:
    """Otwiera zdjęcie, skaluje do max MAX_DIMENSION px i zwraca skompresowane
    bajty JPEG. Oryginalna, surowa bitmapa jest zwalniana natychmiast po
    wyjściu z bloku `with` — nie zostaje w pamięci."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            data = buffer.getvalue()
        return data
    except Exception as e:
        print(f"❌ Błąd podczas otwierania/kompresji pliku {path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. LOGIKA PRZETWARZANIA PACZKI ZDJĘĆ W JEDNYM AUDYCIE
# ---------------------------------------------------------------------------

def process_single_profile(client: OpenAI, image_paths: List[Path]) -> Optional[SingleProfileFullAudit]:
    """Przesyła WSZYSTKIE zdjęcia jednego profilu do GPT-5.6 Terra (Responses API,
    wejście multimodalne) i zwraca JEDEN audyt. Zdjęcia są kompresowane pojedynczo
    (nie wszystkie naraz jako pełnowymiarowe obiekty PIL) — w pamięci trzymane są
    tylko lekkie, skompresowane bajty JPEG zakodowane w Base64."""
    print(f"🔄 Analizowanie pełnego profilu z {len(image_paths)} zdjęć...")

    image_parts = []
    for path in image_paths:
        jpeg_bytes = _load_compressed_jpeg_bytes(path)
        if jpeg_bytes is not None:
            b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
            image_parts.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64_image}",
            })
        del jpeg_bytes

    if not image_parts:
        return None

    prompt = """
Jesteś profesjonalnym, wymagającym i bezstronnym audytorem profili w aplikacjach randkowych (Tinder, Bumble, Hinge).

Otrzymasz ZAŁĄCZONY ZESTAW OBRAZÓW przedstawiających JEDEN PEŁNY PROFIL UŻYTKOWNIKA. Przeanalizuj WSZYSTKIE załączone obrazy jako jedną spójną całość.

ZASADY ANALIZY:
1. Opieraj się WYŁĄCZNIE na informacjach widocznych na obrazach. Nie zgaduj wieku, wzrostu, zawodu, majątku ani charakteru, jeśli nie wynikają z tekstu/zdjęć.
2. Policz dokładnie wszystkie przesłane zdjęcia profilowe i zapisz ich łączną liczbę.
3. Zanim przejdziesz do audytu, ODDZIELNIE wypisz czyste, surowe obserwacje wizualne (co dokładnie jest widoczne na każdym zdjęciu oraz CZEGO NA NIM BRAKUJE).
4. Odróżniaj czystą obserwację (co widać / czego brakuje) od interpretacji i oceny (jaki ma to wpływ na odbiorcę).
5. Oceniaj skuteczność profilu, jakość i odbiór – nigdy nie oceniaj atrakcyjności w sposób obraźliwy.
6. Unikaj pustych ogólników. Każdą ocenę poprzyj konkretnym elementem obrazu, tekstu lub ich BRAKIEM.
7. Numeruj zdjęcia zgodnie z kolejnością przesłania: Zdjęcie 1, Zdjęcie 2, Zdjęcie 3 itd.
8. PRZEPROWADŹ KONTROLĘ BRAKÓW: Zidentyfikuj brak kluczowych elementów profilu (np. brak zdjęcia całej sylwetki, brak naturalnego uśmiechu, brak widocznych oczu przez okulary przeciwsłoneczne, brak interakcji z otoczeniem, pusty opis).

Zwróć ustrukturyzowaną analizę zgodną z podanym schematem danych. Nie zgaduj pól, których nie da się wywnioskować z obrazów — pozostaw je puste/domyślne.
"""

    contents = [{
        "role": "user",
        "content": [*image_parts, {"type": "input_text", "text": prompt}],
    }]

    max_retries = 3
    result: Optional[SingleProfileFullAudit] = None
    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=MODEL_NAME,
                input=contents,
                text_format=SingleProfileFullAudit,
                reasoning={"effort": "low"},
            )

            result = response.output_parsed
            del response
            break

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                wait_time = 10
                print(f"⏳ Limit API (429/Rate limit). Czekam {wait_time} s przed próbą {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"❌ Błąd podczas wywołania API dla profilu: {e}")
                break

    # Zwalniamy skompresowane bajty obrazów i listę contents niezależnie od wyniku
    del contents, image_parts
    gc.collect()

    return result


def run_batch_extraction(input_dir: str, output_dir: str):
    """Zbiera wszystkie zdjęcia z folderu i generuje z nich JEDEN plik audytu JSON."""
    if not API_KEY or API_KEY == "TUTAJ_WKLEJ_SWOJ_KLUCZ_API":
        print("❌ BŁĄD: Musisz ustawić OPENAI_API_KEY w pliku .env!")
        return

    client = OpenAI(api_key=API_KEY)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in supported_extensions])

    if not images:
        print(f"⚠️ Nie znaleziono plików graficznych w katalogu: {input_path.resolve()}")
        return

    print(f"🚀 Znaleziono łącznie {len(images)} obrazów. Łączę je w JEDEN audyt profilu...\n")

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

    del client
    gc.collect()


if __name__ == "__main__":
    INPUT_FOLDER = "./photos"       # Włóż tutaj WSZYSTKIE zdjęcia danego profilu
    OUTPUT_FOLDER = "./OCR_result"   # Tu powstanie JEDEN plik FULL_PROFILE_AUDIT.json

    os.makedirs(INPUT_FOLDER, exist_ok=True)
    run_batch_extraction(INPUT_FOLDER, OUTPUT_FOLDER)
