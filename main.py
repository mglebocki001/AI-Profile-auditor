import shutil
import os
import json
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Importy logiki z Twoich skryptów
from OCR import run_batch_extraction
from Audit import ProfileAnalyzerEngine
from Final_Raport import generate_summary_for_reports

# ---------------------------------------------------------------------------
# 1. KONFIGURACJA SERWERA & KATALOGÓW
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Dating Profile Analyzer API",
    description="Automatyczny audyt profili randkowych z użyciem Gemini Vision & Pydantic",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
PHOTOS_DIR = BASE_DIR / "photos"
OCR_RESULT_DIR = BASE_DIR / "OCR_result"
AUDIT_REPORTS_DIR = BASE_DIR / "audit_reports"
FINAL_SUMMARY_DIR = BASE_DIR / "final_summary"

DIRS_TO_CLEAN = [PHOTOS_DIR, OCR_RESULT_DIR, AUDIT_REPORTS_DIR, FINAL_SUMMARY_DIR]


# ---------------------------------------------------------------------------
# 2. HELPER: CZYSZCZENIE KATALOGÓW ROBOCZYCH
# ---------------------------------------------------------------------------
def clean_working_directories():
    """Usuwa stare pliki z poprzednich zapytań, aby zapobiec nakładaniu się wyników."""
    for folder in DIRS_TO_CLEAN:
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"⚠️ Nie udało się usunąć {item}: {e}")
    print("🧹 Wyczyszczono katalogi robocze z poprzednich zapytań.")


# ---------------------------------------------------------------------------
# 3. ENDPOINT API DO ANALIZY
# ---------------------------------------------------------------------------
@app.post("/analyze-profile")
async def analyze_profile_endpoint(
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="Brak przesłanych plików obrazów.")

    # Krok 0: Czyszczenie środowiska
    clean_working_directories()

    # Krok 1: Zapisanie przesłanych obrazów do folderu ./photos
    saved_file_paths = []
    for image in files:
        file_path = PHOTOS_DIR / image.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        saved_file_paths.append(file_path)

    print(f"📥 Zapisano {len(saved_file_paths)} obrazów w folderze ./photos")

    # Krok 2: Uruchomienie OCR (Gemini Vision API - Zbiorcza analiza w 1 pliku)
    try:
        run_batch_extraction(input_dir=str(PHOTOS_DIR), output_dir=str(OCR_RESULT_DIR))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas ekstrakcji OCR: {str(e)}")

    # Krok 3: Uruchomienie silnika audytu (Audit vs rules.json)
    try:
        analyzer = ProfileAnalyzerEngine(rules_db_path="rules.json")
        ocr_files = list(OCR_RESULT_DIR.glob("*.json"))

        if not ocr_files:
            raise HTTPException(status_code=422, detail="Model Vision AI nie wyekstrahował danych z podanych obrazów.")

        for file in ocr_files:
            audit = analyzer.analyze_extracted_json(str(file))
            out_file = AUDIT_REPORTS_DIR / f"{file.stem}_audit_report.json"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(audit.model_dump_json(indent=2))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas audytu reguł: {str(e)}")

    # Krok 4: Generowanie końcowego raportu podsumowującego
    try:
        generate_summary_for_reports(
            reports_dir=str(AUDIT_REPORTS_DIR),
            output_dir=str(FINAL_SUMMARY_DIR)
        )

        final_file = FINAL_SUMMARY_DIR / "final_profile_summary.json"
        
        if not final_file.exists():
            raise HTTPException(status_code=500, detail="Błąd: Plik finalnego raportu nie został utworzony.")

        with open(final_file, "r", encoding="utf-8") as f:
            final_data = json.load(f)

        return {
            "status": "success",
            "message": "Profil został pomyślnie przeanalizowany.",
            "data": final_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas generowania finalnego raportu: {str(e)}")


# ---------------------------------------------------------------------------
# 4. NOWOCZESNY INTERFEJS GRAFICZNY (ZAMIAST SUROWEGO JSONA)
# ---------------------------------------------------------------------------
@app.get("/test", response_class=HTMLResponse)
def test_interface():
    return """
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Dating Profile Audit</title>
        <style>
            :root {
                --bg-color: #0f172a;
                --card-bg: #1e293b;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --accent-green: #10b981;
                --accent-red: #f43f5e;
                --accent-blue: #3b82f6;
                --border-color: #334155;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-main);
                margin: 0;
                padding: 20px;
                display: flex;
                justify-content: center;
            }

            .container {
                width: 100%;
                max-width: 680px;
            }

            .header {
                text-align: center;
                margin-bottom: 25px;
            }

            .header h1 {
                font-size: 26px;
                font-weight: 800;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #a855f7, #ec4899);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .header p {
                color: var(--text-muted);
                font-size: 14px;
                margin: 0;
            }

            .upload-card {
                background: var(--card-bg);
                border: 2px dashed var(--border-color);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                margin-bottom: 25px;
            }

            .upload-card:hover {
                border-color: #6366f1;
            }

            input[type="file"] {
                display: none;
            }

            .file-label {
                display: inline-block;
                padding: 10px 20px;
                background: #334155;
                color: white;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 500;
                font-size: 14px;
                margin-bottom: 12px;
            }

            .file-label:hover {
                background: #475569;
            }

            #file-count {
                display: block;
                color: var(--text-muted);
                font-size: 13px;
                margin-bottom: 15px;
            }

            .submit-btn {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #6366f1, #8b5cf6);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
            }

            .submit-btn:disabled {
                background: #475569;
                cursor: not-allowed;
            }

            .loader {
                display: none;
                text-align: center;
                padding: 30px;
            }

            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid rgba(255,255,255,0.1);
                border-top: 4px solid #6366f1;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .results-section {
                display: none;
            }

            .section-title {
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: var(--text-muted);
                margin-bottom: 12px;
                font-weight: 700;
            }

            .card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 18px;
                margin-bottom: 15px;
                border-left: 5px solid;
                display: flex;
                gap: 15px;
                align-items: flex-start;
            }

            .card.summary {
                border-color: var(--accent-blue);
                background: rgba(59, 130, 246, 0.08);
            }

            .card.great {
                border-color: var(--accent-green);
                background: rgba(16, 185, 129, 0.08);
            }

            .card.improvement {
                border-color: var(--accent-red);
                background: rgba(244, 63, 94, 0.08);
            }

            .icon {
                font-size: 22px;
                line-height: 1;
                margin-top: 2px;
            }

            .card-content {
                flex: 1;
            }

            .card-title {
                font-weight: 700;
                font-size: 16px;
                margin-bottom: 6px;
            }

            .summary .card-title {
                color: var(--accent-blue);
            }

            .great .card-title {
                color: var(--accent-green);
            }

            .improvement .card-title {
                color: var(--accent-red);
            }

            .card-text {
                font-size: 14px;
                color: #cbd5e1;
                line-height: 1.5;
                margin-bottom: 6px;
            }

            .quick-fix {
                font-size: 13px;
                color: #f1f5f9;
                background: rgba(255, 255, 255, 0.06);
                padding: 8px 12px;
                border-radius: 6px;
                margin-top: 8px;
            }

            .quick-fix-tag {
                font-weight: 600;
                color: #fbbf24;
            }
        </style>
    </head>
    <body>

    <div class="container">
        <div class="header">
            <h1>Audyt Profilu Randkowego</h1>
            <p>Szybki raport jakości Twoich zdjęć oraz opisu</p>
        </div>

        <div class="upload-card">
            <form id="auditForm">
                <label for="fileInput" class="file-label">📁 Wybierz zdjęcia profilu</label>
                <input type="file" id="fileInput" name="files" multiple accept="image/*" required>
                <span id="file-count">Nie wybrano plików</span>
                <button type="submit" class="submit-btn" id="submitBtn">Rozpocznij Audyt</button>
            </form>
        </div>

        <div class="loader" id="loader">
            <div class="spinner"></div>
            <p style="color: var(--text-muted); font-size: 14px;">Generowanie raportu przez AI...</p>
        </div>

        <div class="results-section" id="results">
            <!-- 0. OGÓLNE PODSUMOWANIE PROFILU -->
            <div class="section-title">Ogólny plan działania</div>
            <div class="card summary">
                <div class="icon">📌</div>
                <div class="card-content">
                    <div class="card-title">Podsumowanie audytu</div>
                    <div class="card-text" id="summary-text"></div>
                </div>
            </div>

            <!-- 1. ZIELONY PTASZEK (ZALETA) -->
            <div class="section-title" style="margin-top: 25px;">Co masz zrobione świetnie</div>
            <div class="card great">
                <div class="icon">✅</div>
                <div class="card-content">
                    <div class="card-title">Mocna strona profilu</div>
                    <div class="card-text" id="great-thing-text"></div>
                </div>
            </div>

            <!-- 2. BŁĘDY DO POPRAWY -->
            <div class="section-title" style="margin-top: 25px;">Kluczowe punkty do poprawy</div>
            <div id="improvements-list"></div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileCount = document.getElementById('file-count');
        const form = document.getElementById('auditForm');
        const loader = document.getElementById('loader');
        const results = document.getElementById('results');
        const submitBtn = document.getElementById('submitBtn');

        fileInput.addEventListener('change', () => {
            const count = fileInput.files.length;
            fileCount.textContent = count > 0 ? `Wybrano plików: ${count}` : 'Nie wybrano plików';
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (fileInput.files.length === 0) return;

            loader.style.display = 'block';
            results.style.display = 'none';
            submitBtn.disabled = true;

            const formData = new FormData();
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('files', fileInput.files[i]);
            }

            try {
                const response = await fetch('/analyze-profile', {
                    method: 'POST',
                    body: formData
                });

                const res = await response.json();

                if (res.status === 'success' && res.data) {
                    renderResults(res.data);
                } else {
                    alert('Błąd podczas analizy: ' + (res.detail || 'Nieznany błąd'));
                }
            } catch (err) {
                alert('Wystąpił błąd połączenia z serwerem: ' + err);
            } finally {
                loader.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function renderResults(data) {
            // 0. Ogólne Podsumowanie
            document.getElementById('summary-text').innerText = data.overall_action_plan_summary || 'Brak ogólnego podsumowania.';

            // 1. Ptaszek (Zielony)
            document.getElementById('great-thing-text').innerText = data.one_great_thing || 'Twój profil ma dobre podstawy.';

            // 2. Poprawki (Czerwone błędy)
            const listContainer = document.getElementById('improvements-list');
            listContainer.innerHTML = '';

            const improvements = data.top_5_improvements || [];

            improvements.forEach((item, index) => {
                const card = document.createElement('div');
                card.className = 'card improvement';

                card.innerHTML = `
                    <div class="icon">❌</div>
                    <div class="card-content">
                        <div class="card-title">${index + 1}. ${escapeHtml(item.title)}</div>
                        <div class="card-text">${escapeHtml(item.why_it_matters)}</div>
                        <div class="quick-fix">
                            <span class="quick-fix-tag">💡 Rozwiązanie:</span> ${escapeHtml(item.quick_fix)}
                        </div>
                    </div>
                `;

                listContainer.appendChild(card);
            });

            results.style.display = 'block';
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
        }
    </script>

    </body>
    </html>
    """


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Dating Profile AI Analyzer"}