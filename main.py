import os
import base64
import requests
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # ⬅️ sabse pehle

# ---------- FastAPI App ----------
app = FastAPI(title="Luxand.cloud Face Recognition API", version="1.0.0")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # ⬅️ * ke saath False hona chahiye
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Configuration ----------
API_TOKEN = os.getenv("API_TOKEN")
LUXAND_API_BASE = "https://api.luxand.cloud"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not API_TOKEN:
    raise ValueError("API_TOKEN not found in environment variables")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Local Profile Images Storage ----------
PROFILE_IMAGES_DIR = Path("profile_images")
PROFILE_IMAGES_DIR.mkdir(exist_ok=True)

def save_profile_image(person_uuid: str, image_data: bytes, content_type: str) -> str:
    ext = "jpg" if "jpeg" in content_type.lower() else "png"
    file_path = PROFILE_IMAGES_DIR / f"{person_uuid}.{ext}"
    with open(file_path, "wb") as f:
        f.write(image_data)
    return str(file_path)

def get_profile_image_url(person_uuid: str) -> Optional[str]:
    for ext in ["jpg", "jpeg", "png"]:
        file_path = PROFILE_IMAGES_DIR / f"{person_uuid}.{ext}"
        if file_path.exists():
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
            return f"data:{mime};base64,{b64}"
    return None

def delete_profile_image(person_uuid: str):
    for ext in ["jpg", "jpeg", "png"]:
        file_path = PROFILE_IMAGES_DIR / f"{person_uuid}.{ext}"
        if file_path.exists():
            file_path.unlink()

# ---------- Pydantic Models ----------
class TrainResponse(BaseModel):
    status: str
    person_id: str
    name: str
    faces_added: int
    message: str

class VerifyResponse(BaseModel):
    success: bool
    matched: bool
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    message: str

class PersonInfo(BaseModel):
    uuid: str
    name: str
    image_url: Optional[str] = None

class DeleteRequest(BaseModel):
    uuids: List[str]

# ---------- Helper Functions ----------
def verify_api_connection():
    try:
        url = f"{LUXAND_API_BASE}/subject"
        headers = {"token": API_TOKEN}
        response = requests.get(url, headers=headers)
        if response.status_code == 401:
            raise HTTPException(401, "Invalid API token")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"API connection error: {str(e)}")
        raise HTTPException(502, f"Cannot connect to Luxand API: {str(e)}")

async def send_telegram_notification(photo_bytes: bytes, filename: str, result_text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials missing, skipping notification")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": (filename, photo_bytes, "image/jpeg")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": result_text}
    try:
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
        logger.info("Telegram notification sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {str(e)}")

# ---------- Health ----------
@app.get("/health")
async def health():
    verify_api_connection()
    return {"status": "ok", "api_configured": True}

# ---------- Training (saves first image) ----------
@app.post("/train", response_model=TrainResponse)
async def train_face(
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(400, "No files uploaded")
    
    allowed = ["image/jpeg", "image/png", "image/jpg"]
    for f in files:
        if f.content_type not in allowed:
            raise HTTPException(400, f"Unsupported type: {f.content_type}")
    
    # 1. Create person with first image
    first = files[0]
    first_data = await first.read()
    person_files = {"photos": ("photo.jpg", first_data, first.content_type)}
    person_data = {"name": name, "store": "1"}
    
    try:
        resp = requests.post(
            f"{LUXAND_API_BASE}/v2/person",
            headers={"token": API_TOKEN},
            data=person_data,
            files=person_files
        )
        resp.raise_for_status()
        person_uuid = resp.json().get("uuid")
        if not person_uuid:
            raise HTTPException(502, "No UUID returned")
    except Exception as e:
        logger.error(f"Person creation failed: {e}")
        raise HTTPException(502, f"Failed to create person: {str(e)}")
    
    # Save the first image as profile picture
    save_profile_image(person_uuid, first_data, first.content_type)
    
    # 2. Add remaining images
    faces_added = 1
    for idx, file in enumerate(files[1:], 2):
        try:
            img_data = await file.read()
            add_files = {"photo": (file.filename, img_data, file.content_type)}
            add_resp = requests.post(
                f"{LUXAND_API_BASE}/v2/person/{person_uuid}",
                headers={"token": API_TOKEN},
                data={"store": "1"},
                files=add_files
            )
            add_resp.raise_for_status()
            faces_added += 1
        except Exception as e:
            logger.warning(f"Failed to add face {idx}: {e}")
    
    return TrainResponse(
        status="success",
        person_id=person_uuid,
        name=name,
        faces_added=faces_added,
        message=f"Trained with {faces_added} images"
    )


# ---------- Verification (with Telegram) ----------
@app.post("/verify", response_model=VerifyResponse)
async def verify_face(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    allowed = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    
    image_data = await file.read()
    files = {"photo": (file.filename, image_data, file.content_type)}
    
    try:
        response = requests.post(
            f"{LUXAND_API_BASE}/photo/search/v2",
            headers={"token": API_TOKEN},
            files=files
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Recognition raw result: {result}")
    except Exception as e:
        logger.error(f"Recognition request failed: {e}")
        raise HTTPException(502, f"Face recognition failed: {str(e)}")
    
    # Parse Luxand response (list of matches)
    if isinstance(result, list):
        if len(result) > 0:
            best = result[0]
            confidence_val = best.get("probability", 0)
            person_name = best.get("name", "Unknown")
            
            response_data = VerifyResponse(
                success=True,
                matched=True,
                person_id=best.get("uuid"),
                person_name=person_name,
                confidence=confidence_val,
                message=f"✨ Match Found! Identified as {person_name} with {confidence_val}% confidence."
            )
        else:
            response_data = VerifyResponse(
                success=True,
                matched=False,
                message="🔍 Scan Complete: No matching face found in the secure database."
            )
    elif isinstance(result, dict) and "error" in result:
        error_msg = result["error"]
        if "no faces" in error_msg.lower() or "not found" in error_msg.lower():
            response_data = VerifyResponse(
                success=True, 
                matched=False, 
                message="⚠️ Vision Alert: No face detected in the provided image."
            )
        else:
            raise HTTPException(502, f"Luxand API error: {error_msg}")
    else:
        raise HTTPException(502, f"Unexpected response format: {type(result)}")
    
    # --- Rich Formatting for Telegram Notification ---
    if response_data.matched:
        caption = (
            f"🤖 AI FACIAL RECOGNITION\n"
            f"━═━═━═━═━═━═━═━═━═━\n"
            f"🟢 Status: Access Granted / Match Found\n"
            f"👤 Identity: `{response_data.person_name}`\n"
            f"🆔 ID: `{response_data.person_id}`\n"
            f"🎯 Confidence: `{response_data.confidence}%`\n"
            f"━═━═━═━═━═━═━═━═━═━\n"
            f"⚡ System processed successfully."
        )
    else:
        caption = (
            f"🤖 AI FACIAL RECOGNITION\n"
            f"━═━═━═━═━═━═━═━═━═━\n"
            f"🔴 Status: Access Denied / Unrecognized\n"
            f"📝 Details: {response_data.message}\n"
            f"━═━═━═━═━═━═━═━═━═━\n"
            f"⚠️ Security review recommended."
        )
        
    background_tasks.add_task(send_telegram_notification, image_data, file.filename, caption)
    
    return response_data



# ---------- List persons with local profile images ----------
@app.get("/persons_with_images", response_model=List[PersonInfo])
async def list_persons_with_images():
    verify_api_connection()
    resp = requests.get(f"{LUXAND_API_BASE}/subject", headers={"token": API_TOKEN})
    resp.raise_for_status()
    subjects = resp.json()
    if not isinstance(subjects, list):
        subjects = []
    
    result = []
    for sub in subjects:
        uuid = sub.get("uuid")
        name = sub.get("name", "Unknown")
        image_url = get_profile_image_url(uuid)
        result.append(PersonInfo(uuid=uuid, name=name, image_url=image_url))
    return result

# ---------- Bulk delete (using correct endpoint) ----------
@app.post("/delete_persons")
async def delete_persons(request: DeleteRequest):
    verify_api_connection()
    deleted = []
    failed = []
    for uuid in request.uuids:
        try:
            # Use the correct Luxand endpoint
            resp = requests.delete(
                f"{LUXAND_API_BASE}/subject/{uuid}",
                headers={"token": API_TOKEN}
            )
            if resp.status_code == 404:
                # Fallback to older endpoint if needed
                resp = requests.delete(
                    f"{LUXAND_API_BASE}/v2/person/{uuid}",
                    headers={"token": API_TOKEN}
                )
            resp.raise_for_status()
            deleted.append(uuid)
            delete_profile_image(uuid)   # remove local image
        except Exception as e:
            logger.error(f"Failed to delete {uuid}: {e}")
            failed.append(uuid)
    return {"deleted": deleted, "failed": failed}

# ---------- Serve HTML Frontend ----------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition Studio | Luxand.cloud</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <style>
        .profile-img {
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 50%;
            border: 2px solid #3b82f6;
        }
        .person-card {
            transition: all 0.2s;
        }
        .person-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body class="bg-gradient-to-br from-gray-900 to-gray-800 min-h-screen text-white">
    <div class="container mx-auto px-4 py-12 max-w-6xl">
        <div class="text-center mb-12">
            <h1 class="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">Face Recognition Studio</h1>
            <p class="text-gray-300 mt-3">Train with multiple photos, verify faces, manage users & get Telegram alerts</p>
        </div>

        <div class="grid lg:grid-cols-2 gap-8 mb-12">
            <!-- Train Card -->
            <div class="bg-gray-800/60 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-gray-700">
                <div class="flex items-center gap-3 mb-6">
                    <div class="p-2 bg-blue-500/20 rounded-lg">
                        <svg class="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
                    </div>
                    <h2 class="text-2xl font-semibold">Register Person</h2>
                </div>
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-1">Person Name</label>
                        <input type="text" id="trainName" placeholder="e.g., himanshu" class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-1">Select Photos (5-6 recommended)</label>
                        <input type="file" id="trainFiles" multiple accept="image/jpeg,image/png,image/jpg" class="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer">
                    </div>
                    <button id="trainBtn" class="w-full bg-blue-600 hover:bg-blue-700 transition-all duration-200 py-3 rounded-lg font-medium flex items-center justify-center gap-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                        Train Model
                    </button>
                </div>
                <div id="trainResult" class="mt-4 text-sm"></div>
            </div>

            <!-- Verify Card -->
            <div class="bg-gray-800/60 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-gray-700">
                <div class="flex items-center gap-3 mb-6">
                    <div class="p-2 bg-green-500/20 rounded-lg">
                        <svg class="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    </div>
                    <h2 class="text-2xl font-semibold">Verify Face</h2>
                </div>
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-300 mb-1">Upload Face Image</label>
                        <input type="file" id="verifyFile" accept="image/jpeg,image/png,image/jpg" class="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-green-600 file:text-white hover:file:bg-green-700 cursor-pointer">
                    </div>
                    <button id="verifyBtn" class="w-full bg-green-600 hover:bg-green-700 transition-all duration-200 py-3 rounded-lg font-medium flex items-center justify-center gap-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                        Verify Identity
                    </button>
                </div>
                <div id="verifyResult" class="mt-4 text-sm"></div>
            </div>
        </div>

        <!-- Registered Users Section with Delete -->
        <div class="bg-gray-800/40 rounded-2xl p-6 border border-gray-700">
            <div class="flex flex-wrap items-center justify-between gap-4 mb-6">
                <h2 class="text-2xl font-semibold flex items-center gap-2">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
                    Registered Persons
                </h2>
                <div class="flex gap-3">
                    <button id="refreshPersonsBtn" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition flex items-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                        Refresh
                    </button>
                    <button id="deleteSelectedBtn" class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition flex items-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                        Delete Selected
                    </button>
                </div>
            </div>
            <div id="personsGrid" class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                <div class="col-span-full text-center py-8 text-gray-400">Click "Refresh" to load registered users</div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = "";

        async function loadPersons() {
            const grid = document.getElementById("personsGrid");
            grid.innerHTML = `<div class="col-span-full text-center py-8"><div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div><p class="mt-2">Loading...</p></div>`;
            try {
                const res = await fetch(`${API_BASE}/persons_with_images`);
                if (!res.ok) throw new Error("Failed to fetch");
                const persons = await res.json();
                if (!persons.length) {
                    grid.innerHTML = `<div class="col-span-full text-center py-8 text-gray-400">No registered users yet. Train a person first.</div>`;
                    return;
                }
                grid.innerHTML = persons.map(p => `
                    <div class="person-card bg-gray-800/80 rounded-xl p-4 text-center border border-gray-700">
                        <div class="relative">
                            <input type="checkbox" class="delete-checkbox absolute top-0 left-0 w-5 h-5 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500" data-uuid="${p.uuid}">
                            <img src="${p.image_url || 'https://via.placeholder.com/80?text=No+Face'}" class="profile-img mx-auto mt-2" alt="profile">
                        </div>
                        <h3 class="font-medium mt-3 truncate" title="${p.name}">${p.name}</h3>
                        <p class="text-xs text-gray-400 truncate">${p.uuid.substring(0,8)}...</p>
                    </div>
                `).join('');
            } catch (err) {
                console.error(err);
                grid.innerHTML = `<div class="col-span-full text-center py-8 text-red-400">Failed to load persons: ${err.message}</div>`;
            }
        }

        async function deleteSelected() {
            const checkboxes = document.querySelectorAll(".delete-checkbox:checked");
            if (checkboxes.length === 0) {
                Swal.fire("No Selection", "Please select at least one person to delete", "warning");
                return;
            }
            const confirm = await Swal.fire({
                title: "Delete Persons?",
                text: `You are about to delete ${checkboxes.length} person(s). This action cannot be undone.`,
                icon: "warning",
                showCancelButton: true,
                confirmButtonColor: "#d33",
                confirmButtonText: "Yes, delete them"
            });
            if (!confirm.isConfirmed) return;

            const uuids = Array.from(checkboxes).map(cb => cb.dataset.uuid);
            try {
                const res = await fetch(`${API_BASE}/delete_persons`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ uuids })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Delete failed");
                Swal.fire("Deleted!", `${data.deleted.length} person(s) removed.`, "success");
                loadPersons(); // refresh list
            } catch (err) {
                Swal.fire("Error", err.message, "error");
            }
        }

        async function trainPerson() {
            const name = document.getElementById("trainName").value.trim();
            const files = document.getElementById("trainFiles").files;
            const btn = document.getElementById("trainBtn");
            const resultDiv = document.getElementById("trainResult");
            
            if (!name) { Swal.fire("Error", "Please enter a name", "error"); return; }
            if (files.length === 0) { Swal.fire("Error", "Select at least one image", "error"); return; }

            btn.disabled = true;
            btn.innerHTML = `<div class="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div> Training...`;
            resultDiv.innerHTML = "";

            const formData = new FormData();
            formData.append("name", name);
            for (let i = 0; i < files.length; i++) formData.append("files", files[i]);

            try {
                const res = await fetch(`${API_BASE}/train`, { method: "POST", body: formData });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Training failed");
                Swal.fire("Success!", `Person "${data.name}" trained with ${data.faces_added} images.`, "success");
                resultDiv.innerHTML = `<div class="text-green-400">✅ ${data.message}</div>`;
                document.getElementById("trainName").value = "";
                document.getElementById("trainFiles").value = "";
                loadPersons();
            } catch (err) {
                Swal.fire("Training Error", err.message, "error");
                resultDiv.innerHTML = `<div class="text-red-400">❌ ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg> Train Model`;
            }
        }

        async function verifyFace() {
            const fileInput = document.getElementById("verifyFile");
            const file = fileInput.files[0];
            const btn = document.getElementById("verifyBtn");
            const resultDiv = document.getElementById("verifyResult");
            
            if (!file) { Swal.fire("Error", "Please select an image", "error"); return; }

            btn.disabled = true;
            btn.innerHTML = `<div class="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div> Verifying...`;
            resultDiv.innerHTML = "";

            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch(`${API_BASE}/verify`, { method: "POST", body: formData });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Verification failed");
                
                if (data.matched) {
                    resultDiv.innerHTML = `<div class="bg-green-900/50 border border-green-500 rounded-lg p-3">
                        <span class="font-bold text-green-300">✅ Match Found!</span><br>
                        Person: <span class="font-mono">${data.person_name}</span><br>
                        Confidence: ${data.confidence}%<br>
                        UUID: ${data.person_id?.slice(0,8)}...
                    </div>`;
                    Swal.fire("Match Found!", `Welcome ${data.person_name} (${data.confidence}% confidence)`, "success");
                } else {
                    resultDiv.innerHTML = `<div class="bg-red-900/50 border border-red-500 rounded-lg p-3">
                        <span class="font-bold text-red-300">❌ No Match</span><br>
                        ${data.message}
                    </div>`;
                    Swal.fire("No Match", "Face not recognized in database", "error");
                }
            } catch (err) {
                Swal.fire("Verification Error", err.message, "error");
                resultDiv.innerHTML = `<div class="text-red-400">❌ ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg> Verify Identity`;
            }
        }

        document.getElementById("trainBtn").addEventListener("click", trainPerson);
        document.getElementById("verifyBtn").addEventListener("click", verifyFace);
        document.getElementById("refreshPersonsBtn").addEventListener("click", loadPersons);
        document.getElementById("deleteSelectedBtn").addEventListener("click", deleteSelected);
        loadPersons();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_PAGE)
