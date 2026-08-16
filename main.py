import shutil
import os
import gc
import json
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from OCR import run_batch_extraction
from Audit import get_engine
from Final_Raport import generate_summary_for_reports

app = FastAPI(title="AI Dating Profile Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
PHOTOS_DIR = Path("/opt/render/project/src/photos")
OCR_RESULT_DIR = BASE_DIR / "OCR_result"
AUDIT_REPORTS_DIR = BASE_DIR / "audit_reports"
FINAL_SUMMARY_DIR = BASE_DIR / "final_summary"

# Tworzenie katalogów roboczych
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
OCR_RESULT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Serwowanie zdjęć tymczasowych
app.mount("/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

DIRS_TO_CLEAN = [PHOTOS_DIR, OCR_RESULT_DIR, AUDIT_REPORTS_DIR, FINAL_SUMMARY_DIR]

def clean_working_directories():
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

@app.post("/analyze-profile")
async def analyze_profile_endpoint(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Brak przesłanych plików.")

    clean_working_directories()

    saved_filenames = []
    for idx, image in enumerate(files):
        ext = Path(image.filename).suffix or ".jpg"
        filename = f"photo_{idx+1}{ext}"
        file_path = PHOTOS_DIR / filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        saved_filenames.append(filename)

    # 1. Ekstrakcja danych i OCR
    try:
        run_batch_extraction(input_dir=str(PHOTOS_DIR), output_dir=str(OCR_RESULT_DIR))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas ekstrakcji OCR: {str(e)}")

    # 2. Generowanie raportu końcowego
    try:
        generate_summary_for_reports(reports_dir=str(AUDIT_REPORTS_DIR), output_dir=str(FINAL_SUMMARY_DIR))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas generowania syntezy OpenAI: {str(e)}")

    final_file = FINAL_SUMMARY_DIR / "final_profile_summary.json"
    
    if not final_file.exists():
        raise HTTPException(
            status_code=500, 
            detail="Plik final_profile_summary.json nie został wygenerowany. Sprawdź logi serwera i klucz OPENAI_API_KEY."
        )

    with open(final_file, "r", encoding="utf-8") as f:
        final_data = json.load(f)

    return {
        "status": "success",
        "photos": saved_filenames,
        "data": final_data
    }

@app.get("/test", response_class=HTMLResponse)
def test_interface():
    return """
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <title>Audyt Profilu - AI Dating Analyzer</title>
        <style>
            body {
                background-color: #0d0f17;
                color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                margin: 0;
                padding: 40px 20px;
                display: flex;
                justify-content: center;
            }
            .container { width: 100%; max-width: 1100px; }
            
            .upload-box {
                background: #161a23;
                border: 2px dashed #2a3142;
                border-radius: 12px;
                padding: 30px;
                text-align: center;
                margin-bottom: 30px;
            }
            .file-btn {
                background: #2563eb;
                color: #fff;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                display: inline-block;
            }
            .submit-btn {
                background: #10b981;
                color: #fff;
                border: none;
                padding: 10px 24px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 15px;
            }

            #loading {
                display: none;
                margin-top: 15px;
                color: #eab308;
                font-weight: 600;
            }

            /* Galeria zdjęć */
            .gallery-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
                gap: 16px;
                margin-bottom: 30px;
            }
            .photo-card {
                background: #121620;
                border-radius: 14px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding-bottom: 12px;
            }
            .photo-card img {
                width: 100%;
                height: 260px;
                object-fit: cover;
                border-radius: 12px 12px 0 0;
            }
            
            .action-status {
                margin-top: 12px;
                font-weight: 700;
                font-size: 15px;
            }
            .action-status.Zostaw { color: #22c55e; }
            .action-status.Usuń { color: #f43f5e; }
            .action-status.Podobne { color: #eab308; }

            .tags-list {
                margin-top: 10px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                padding: 0 10px;
                width: 100%;
                box-sizing: border-box;
            }
            .tag-item {
                font-size: 12px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .tag-item.positive { color: #22c55e; }
            .tag-item.negative { color: #f43f5e; }

            /* Sekcja szczegółów i wyników */
            .card-panel {
                background: #121620;
                border: 1px solid #1e2433;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 20px;
            }
            .score-card {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .score-card .title { font-size: 18px; color: #94a3b8; font-weight: 600; }
            .score-card .val { font-size: 32px; font-weight: 800; color: #10b981; }

            .section-title { font-size: 18px; color: #60a5fa; margin-bottom: 12px; font-weight: 700; }
            .text-box { font-size: 15px; line-height: 1.6; color: #cbd5e1; }

            .improvement-item {
                background: #1a202c;
                border-left: 4px solid #f59e0b;
                padding: 12px 16px;
                margin-bottom: 10px;
                border-radius: 4px;
            }
            .improvement-item h4 { margin: 0 0 6px 0; color: #f3f4f6; }
            .improvement-item p { margin: 2px 0; font-size: 13px; color: #9ca3af; }
        </style>
    </head>
    <body>

    <div class="container">
        <div class="upload-box">
            <form id="auditForm">
                <label for="fileInput" class="file-btn">📁 Wybierz zdjęcia profilu</label>
                <input type="file" id="fileInput" multiple accept="image/*" style="display:none;">
                <br>
                <button type="submit" class="submit-btn" id="subBtn">Rozpocznij Analizę</button>
                <div id="loading">⏳ Trwa analiza zdjęcia i generowanie raportu... Proszę czekać.</div>
            </form>
        </div>

        <div id="results" style="display:none;">
            <!-- Ocena ogólna -->
            <div class="card-panel score-card">
                <div class="title">Ocena jakości profilu</div>
                <div class="val" id="scoreVal">0/100</div>
            </div>

            <!-- Siatka zdjęć ze statusami -->
            <div class="gallery-grid" id="galleryGrid"></div>

            <!-- Podsumowanie planu działania -->
            <div class="card-panel">
                <div class="section-title">📋 Ogólny Plan Działania</div>
                <div class="text-box" id="actionPlan"></div>
            </div>

            <!-- Jedna świetna rzecz -->
            <div class="card-panel">
                <div class="section-title">✅ Największa Zaleta Profilu</div>
                <div class="text-box" id="oneGreatThing"></div>
            </div>

            <!-- Sugestia Bio -->
            <div class="card-panel">
                <div class="section-title">✍️ Sugestia Poprawionego Bio</div>
                <p><strong>Oryginalne Bio:</strong> <span id="origBio" style="color: #9ca3af;"></span></p>
                <p><strong>Propozycja zmian:</strong></p>
                <div class="text-box" id="improvedBio" style="background: #1a2332; padding: 12px; border-radius: 8px; color: #34d399;"></div>
                <p style="font-size: 13px; color: #9ca3af; margin-top: 8px;"><strong>Dlaczego działa lepiej:</strong> <span id="whyBetterBio"></span></p>
            </div>

            <!-- Top 5 poprawek -->
            <div class="card-panel">
                <div class="section-title">🛠️ Top 5 Rekomendowanych Poprawek</div>
                <div id="improvementsList"></div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('auditForm');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) return;

            loading.style.display = 'block';
            document.getElementById('results').style.display = 'none';

            const formData = new FormData();
            for (let f of fileInput.files) formData.append('files', f);

            try {
                const res = await fetch('/analyze-profile', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok && data.status === 'success') {
                    renderResults(data);
                } else {
                    alert('Błąd: ' + (data.detail || 'Wystąpił błąd podczas analizy.'));
                }
            } catch (err) {
                alert('Błąd połączenia z serwerem: ' + err.message);
            } finally {
                loading.style.display = 'none';
            }
        });

        function renderResults(resData) {
            const payload = resData.data;

            // Wynik punktowy
            document.getElementById('scoreVal').innerText = `${payload.overall_score || 0}/100`;

            // Karty zdjęć (photo_by_photo)
            const grid = document.getElementById('galleryGrid');
            grid.innerHTML = '';
            const photoCards = payload.photo_by_photo || [];

            photoCards.forEach((pCard) => {
                const photoName = resData.photos[pCard.photo_index - 1] || `photo_${pCard.photo_index}.jpg`;
                const card = document.createElement('div');
                card.className = 'photo-card';

                let tagsHtml = '';
                (pCard.tags || []).forEach(t => {
                    const icon = t.is_positive ? '✓' : '▲';
                    const cls = t.is_positive ? 'positive' : 'negative';
                    tagsHtml += `<div class="tag-item ${cls}">${icon} ${t.label}</div>`;
                });

                card.innerHTML = `
                    <img src="/photos/${photoName}">
                    <div class="action-status ${pCard.action}">${pCard.action}</div>
                    <div class="tags-list">${tagsHtml}</div>
                `;
                grid.appendChild(card);
            });

            // Sekcje opisu / overview
            const overview = payload.overview || {};
            document.getElementById('actionPlan').innerText = overview.overall_action_plan_summary || 'Brak danych';
            document.getElementById('oneGreatThing').innerText = overview.one_great_thing || 'Brak danych';

            // Bio
            const bioData = payload.bio || {};
            document.getElementById('origBio').innerText = bioData.original_bio || '(Brak opisu)';
            document.getElementById('improvedBio').innerText = bioData.improved_bio || 'Nie wygenerowano propozycji.';
            document.getElementById('whyBetterBio').innerText = bioData.why_better || '-';

            // Lista Top 5 poprawek
            const impDiv = document.getElementById('improvementsList');
            impDiv.innerHTML = '';
            (overview.top_5_improvements || []).forEach(item => {
                impDiv.innerHTML += `
                    <div class="improvement-item">
                        <h4>${item.title}</h4>
                        <p><strong>Dlaczego to ważne:</strong> ${item.why_it_matters}</p>
                        <p><strong>Szybka zmiana:</strong> ${item.quick_fix}</p>
                    </div>
                `;
            });

            document.getElementById('results').style.display = 'block';
        }
    </script>
    </body>
    </html>
    """