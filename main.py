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

# 2. Utworzenie folderów (jeśli nie istnieją), co zapobiega błędom przy starcie
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
OCR_RESULT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(OCR_RESULT_DIR, exist_ok=True)
os.makedirs(AUDIT_REPORTS_DIR, exist_ok=True)
os.makedirs(FINAL_SUMMARY_DIR, exist_ok=True)

# Serwowanie zdjęć tymczasowych na potrzeby podglądu w interfejsie
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
        raise HTTPException(status_code=400, detail="Brak plików.")

    clean_working_directories()

    saved_filenames = []
    for idx, image in enumerate(files):
        ext = Path(image.filename).suffix or ".jpg"
        filename = f"photo_{idx+1}{ext}"
        file_path = PHOTOS_DIR / filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        saved_filenames.append(filename)

    run_batch_extraction(input_dir=str(PHOTOS_DIR), output_dir=str(OCR_RESULT_DIR))
    generate_summary_for_reports(reports_dir=str(AUDIT_REPORTS_DIR), output_dir=str(FINAL_SUMMARY_DIR))

    final_file = FINAL_SUMMARY_DIR / "final_profile_summary.json"
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
        <title>Audyt Zdjęć Profilu</title>
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
            
            /* Galeria w stylu ze zrzutu ekranu */
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
                border-radius: 12px;
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

            /* Dolna sekcja wyników */
            .global-tags {
                display: flex;
                flex-wrap: wrap;
                gap: 18px;
                margin-bottom: 25px;
                font-size: 14px;
                font-weight: 600;
            }
            .score-card {
                background: #121620;
                border: 1px solid #1e2433;
                border-radius: 12px;
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .score-card .title { font-size: 16px; color: #94a3b8; font-weight: 600; }
            .score-card .val { font-size: 28px; font-weight: 800; color: #ffffff; }
        </style>
    </head>
    <body>

    <div class="container">
        <div class="upload-box">
            <form id="auditForm">
                <label for="fileInput" class="file-btn">📁 Wybierz zdjęcia</label>
                <input type="file" id="fileInput" multiple accept="image/*" style="display:none;">
                <br>
                <button type="submit" class="submit-btn" id="subBtn">Rozpocznij Analizę</button>
            </form>
        </div>

        <div id="results" style="display:none;">
            <!-- Siatka zdjęć ze statusami -->
            <div class="gallery-grid" id="galleryGrid"></div>

            <!-- Globalne tagi pod zdjęciami -->
            <div class="global-tags" id="globalTags"></div>

            <!-- Ocena ogólna -->
            <div class="score-card">
                <div class="title">Ocena jakości profilu</div>
                <div class="val" id="scoreVal">0/100</div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('auditForm');
        const fileInput = document.getElementById('fileInput');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) return;

            const formData = new FormData();
            for (let f of fileInput.files) formData.append('files', f);

            const res = await fetch('/analyze-profile', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.status === 'success') {
                renderResults(data);
            }
        });

        function renderResults(resData) {
            const grid = document.getElementById('galleryGrid');
            grid.innerHTML = '';

            const evaluations = resData.data.photo_evaluations || [];
            const photos = resData.photos || [];

            photos.forEach((photoName, idx) => {
                const evalData = evaluations[idx] || { action: 'Zostaw', tags: [] };
                
                const card = document.createElement('div');
                card.className = 'photo-card';

                let tagsHtml = '';
                (evalData.tags || []).forEach(t => {
                    const icon = t.is_positive ? '✓' : '▲';
                    const cls = t.is_positive ? 'positive' : 'negative';
                    tagsHtml += `<div class="tag-item ${cls}">${icon} ${t.label}</div>`;
                });

                card.innerHTML = `
                    <img src="/photos/${photoName}">
                    <div class="action-status ${evalData.action}">${evalData.action}</div>
                    <div class="tags-list">${tagsHtml}</div>
                `;
                grid.appendChild(card);
            });

            // Globalne tagi
            const globalTagsDiv = document.getElementById('globalTags');
            globalTagsDiv.innerHTML = '';
            (resData.data.global_tags || []).forEach(t => {
                const icon = t.is_positive ? '✓' : '▲';
                const cls = t.is_positive ? 'positive' : 'negative';
                globalTagsDiv.innerHTML += `<div class="tag-item ${cls}">${icon} ${t.label}</div>`;
            });

            // Ocena punktowa
            document.getElementById('scoreVal').innerText = `${resData.data.overall_score}/100`;
            document.getElementById('results').style.display = 'block';
        }
    </script>
    </body>
    </html>
    """