import os
import json
import base64
import requests
import logging
import uuid as uuid_lib
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------- FastAPI App ----------
app = FastAPI(title="Face++ Face Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Configuration ----------
FACE_API_KEY    = os.getenv("FACE_API_KEY", "").strip()
FACE_API_SECRET = os.getenv("FACE_API_SECRET", "").strip()
FACE_REGION     = os.getenv("FACE_REGION", "us").strip()   # 'us' or 'cn'
FACE_API_BASE   = f"https://api-{FACE_REGION}.faceplusplus.com/facepp/v3"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Confidence threshold – matches below this are treated as "no match"
# Face++ uses 0-100 scale. 1e-5 FAR threshold is ~73-76 for US server.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "73"))

if not FACE_API_KEY or not FACE_API_SECRET:
    raise ValueError("FACE_API_KEY and FACE_API_SECRET must be set in .env")

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Local Storage ----------
PROFILE_IMAGES_DIR = Path("profile_images")
PROFILE_IMAGES_DIR.mkdir(exist_ok=True)

PERSONS_DB_PATH = Path("persons.json")


def load_persons_db() -> Dict:
    if not PERSONS_DB_PATH.exists():
        return {}
    with open(PERSONS_DB_PATH, "r") as f:
        return json.load(f)


def save_persons_db(db: Dict):
    with open(PERSONS_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


# ---------- Face++ Core Helper ----------
def facepp_post(endpoint: str, data: dict = None, files: dict = None) -> dict:
    """POST to Face++ with api_key/api_secret injected. Returns parsed JSON."""
    url = f"{FACE_API_BASE}/{endpoint}"
    payload = dict(data or {})
    payload["api_key"]    = FACE_API_KEY
    payload["api_secret"] = FACE_API_SECRET

    logger.info(f"→ Face++ POST {endpoint}")
    try:
        resp = requests.post(url, data=payload, files=files, timeout=30)
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, f"Network error reaching Face++: {e}")

    logger.info(f"← {resp.status_code}")
    if resp.status_code != 200:
        logger.error(f"Face++ error body: {resp.text}")
        try:
            msg = resp.json().get("error_message", resp.text)
        except Exception:
            msg = resp.text
        raise HTTPException(502, f"Face++ API error: {msg}")

    return resp.json()


# ---------- Face Detection ----------
def detect_largest_face(image_bytes: bytes) -> Optional[str]:
    """
    Detect faces in image and return face_token of the largest face.
    Returns None if no face found.
    """
    files  = {"image_file": ("image.jpg", image_bytes, "image/jpeg")}
    result = facepp_post("detect", data={"return_landmark": "0", "return_attributes": "none"}, files=files)
    faces  = result.get("faces", [])
    if not faces:
        return None
    largest = max(
        faces,
        key=lambda f: f.get("face_rectangle", {}).get("width", 0)
                    * f.get("face_rectangle", {}).get("height", 0)
    )
    return largest.get("face_token")


# ---------- FaceSet Management ----------
def create_faceset(outer_id: str, face_tokens: List[str]) -> str:
    """Create a FaceSet with outer_id and add face_tokens. Returns faceset_token."""
    res = facepp_post("faceset/create", data={"outer_id": outer_id})
    faceset_token = res.get("faceset_token")
    if not faceset_token:
        raise HTTPException(502, "Face++ did not return a faceset_token on create")

    if face_tokens:
        add_res = facepp_post(
            "faceset/addface",
            data={
                "faceset_token": faceset_token,
                "face_tokens": ",".join(face_tokens),
            },
        )
        added = add_res.get("face_added", 0)
        if added < len(face_tokens):
            logger.warning(f"Only {added}/{len(face_tokens)} faces added to faceset")

    return faceset_token


def delete_faceset(faceset_token: str):
    """Delete a FaceSet by its token (does not raise if already gone)."""
    try:
        facepp_post("faceset/delete", data={"faceset_token": faceset_token, "check_empty": "0"})
    except HTTPException as e:
        logger.warning(f"delete_faceset warning: {e.detail}")


# ---------- Face Search (one faceset at a time) ----------
def search_across_facesets(face_token: str, db: Dict) -> Optional[Dict]:
    """
    Search face_token in every stored faceset individually.
    The Face++ /search endpoint only accepts a SINGLE faceset_token per call.
    We iterate over all persons and keep the highest-confidence result that
    is above the CONFIDENCE_THRESHOLD.

    Returns a dict with keys:
        matched_uuid, matched_name, confidence
    or None if no qualifying match found.
    """
    best_confidence = -1.0
    best_uuid       = None
    best_name       = None

    for uuid, entry in db.items():
        faceset_token = entry.get("faceset_token")
        if not faceset_token:
            logger.warning(f"Person {uuid} has no faceset_token, skipping")
            continue

        try:
            result = facepp_post(
                "search",
                data={
                    "face_token":         face_token,
                    "faceset_token":      faceset_token,   # singular – required by Face++ v3
                    "return_result_count": "1",
                },
            )
            # Face++ search response:
            # {
            #   "results": [
            #     {"face_token": "...", "confidence": 97.076, "user_id": ""}
            #   ],
            #   "thresholds": {"1e-3": 62.3, "1e-5": 73.9, "1e-4": 69.1},
            #   ...
            # }
            candidates = result.get("results", [])
            if candidates:
                top        = candidates[0]
                confidence = top.get("confidence", 0)
                logger.info(f"Person '{entry.get('name')}' → confidence {confidence}")
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_uuid       = uuid
                    best_name       = entry.get("name", "Unknown")

        except HTTPException as e:
            logger.warning(f"Search failed for faceset {faceset_token}: {e.detail}")
            continue

    if best_confidence >= CONFIDENCE_THRESHOLD:
        return {
            "matched_uuid": best_uuid,
            "matched_name": best_name,
            "confidence":   round(best_confidence, 3),
        }
    return None


# ---------- Profile Image Helpers ----------
def save_profile_image(person_uuid: str, image_data: bytes, content_type: str):
    ext       = "jpg" if "jpeg" in content_type.lower() or "jpg" in content_type.lower() else "png"
    file_path = PROFILE_IMAGES_DIR / f"{person_uuid}.{ext}"
    file_path.write_bytes(image_data)


def get_profile_image_b64(person_uuid: str) -> Optional[str]:
    for ext in ["jpg", "jpeg", "png"]:
        file_path = PROFILE_IMAGES_DIR / f"{person_uuid}.{ext}"
        if file_path.exists():
            b64  = base64.b64encode(file_path.read_bytes()).decode()
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
            return f"data:{mime};base64,{b64}"
    return None


def delete_profile_image(person_uuid: str):
    for ext in ["jpg", "jpeg", "png"]:
        p = PROFILE_IMAGES_DIR / f"{person_uuid}.{ext}"
        if p.exists():
            p.unlink()


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
    person_id: Optional[str]   = None
    person_name: Optional[str] = None
    confidence: Optional[float] = None
    message: str


class PersonInfo(BaseModel):
    uuid: str
    name: str
    image_url: Optional[str] = None


class DeleteRequest(BaseModel):
    uuids: List[str]


# ---------- Telegram ----------
async def send_telegram_notification(photo_bytes: bytes, filename: str, caption: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url   = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": (filename, photo_bytes, "image/jpeg")}
    data  = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=data, files=files, timeout=15)
        resp.raise_for_status()
        logger.info("Telegram notification sent")
    except Exception as e:
        logger.error(f"Telegram failed: {e}")


# ──────────────────────────────────────────────
#  ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    """Validate credentials and connectivity."""
    try:
        facepp_post("faceset/getfacesets", data={"page": "1", "size": "1"})
        return {"status": "ok", "region": FACE_REGION, "api_configured": True}
    except HTTPException as e:
        raise HTTPException(503, f"Face++ unreachable: {e.detail}")


# ---------- Train ----------
@app.post("/train", response_model=TrainResponse)
async def train_face(
    name:  str              = Form(...),
    files: List[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    allowed = {"image/jpeg", "image/png", "image/jpg"}
    for f in files:
        if f.content_type not in allowed:
            raise HTTPException(400, f"Unsupported file type: {f.content_type}")

    person_uuid        = str(uuid_lib.uuid4())
    face_tokens        = []
    first_image_data   = None
    first_content_type = None

    for idx, upload in enumerate(files):
        image_data = await upload.read()
        if idx == 0:
            first_image_data   = image_data
            first_content_type = upload.content_type

        token = detect_largest_face(image_data)
        if token:
            face_tokens.append(token)
        else:
            logger.warning(f"No face detected in {upload.filename}, skipping")

    if not face_tokens:
        raise HTTPException(400, "No faces detected in any of the uploaded images.")

    # Create FaceSet in Face++ (outer_id = our UUID for easy reference)
    faceset_token = create_faceset(person_uuid, face_tokens)

    # Persist profile picture
    if first_image_data:
        save_profile_image(person_uuid, first_image_data, first_content_type)

    # Persist person record
    db = load_persons_db()
    db[person_uuid] = {
        "name":          name,
        "faceset_token": faceset_token,   # needed for /search calls
        "face_count":    len(face_tokens),
    }
    save_persons_db(db)
    logger.info(f"Registered '{name}' as {person_uuid} with {len(face_tokens)} faces")

    return TrainResponse(
        status="success",
        person_id=person_uuid,
        name=name,
        faces_added=len(face_tokens),
        message=f"Successfully trained '{name}' with {len(face_tokens)} face image(s).",
    )


# ---------- Verify ----------
@app.post("/verify", response_model=VerifyResponse)
async def verify_face(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    image_data = await file.read()

    # ── Step 1: detect face ──────────────────────────────────────────────────
    face_token = detect_largest_face(image_data)

    if not face_token:
        resp = VerifyResponse(
            success=True,
            matched=False,
            message="⚠️ No face detected in the provided image. Please upload a clear, front-facing photo.",
        )
        caption = (
            "🤖 *AI FACIAL RECOGNITION*\n"
            "━═━═━═━═━═━═━═━═━═━\n"
            "🔴 *Status:* No Face Detected\n"
            f"📝 {resp.message}\n"
            "━═━═━═━═━═━═━═━═━═━"
        )
        background_tasks.add_task(send_telegram_notification, image_data, file.filename, caption)
        return resp

    # ── Step 2: load DB ──────────────────────────────────────────────────────
    db = load_persons_db()
    if not db:
        resp = VerifyResponse(
            success=True,
            matched=False,
            message="🔍 No registered persons in the database. Please train the system first.",
        )
        caption = (
            "🤖 *AI FACIAL RECOGNITION*\n"
            "━═━═━═━═━═━═━═━═━═━\n"
            "🔴 *Status:* No Registered Users\n"
            f"📝 {resp.message}\n"
            "━═━═━═━═━═━═━═━═━═━"
        )
        background_tasks.add_task(send_telegram_notification, image_data, file.filename, caption)
        return resp

    # ── Step 3: search each FaceSet ──────────────────────────────────────────
    match = search_across_facesets(face_token, db)

    if match:
        matched_uuid = match["matched_uuid"]
        matched_name = match["matched_name"]
        confidence   = match["confidence"]

        resp = VerifyResponse(
            success=True,
            matched=True,
            person_id=matched_uuid,
            person_name=matched_name,
            confidence=confidence,
            message=f"✨ Match Found! Identified as '{matched_name}' with {confidence}% confidence.",
        )
        caption = (
            "🤖 *AI FACIAL RECOGNITION*\n"
            "━═━═━═━═━═━═━═━═━═━\n"
            "🟢 *Status:* Access Granted / Match Found\n"
            f"👤 *Identity:* `{matched_name}`\n"
            f"🆔 *ID:* `{matched_uuid[:8]}...`\n"
            f"🎯 *Confidence:* `{confidence}%`\n"
            "━═━═━═━═━═━═━═━═━═━\n"
            "⚡ System processed successfully."
        )
    else:
        resp = VerifyResponse(
            success=True,
            matched=False,
            message="🔍 Scan Complete: No matching face found in the secure database.",
        )
        caption = (
            "🤖 *AI FACIAL RECOGNITION*\n"
            "━═━═━═━═━═━═━═━═━═━\n"
            "🔴 *Status:* Access Denied / Unrecognized\n"
            f"📝 {resp.message}\n"
            "━═━═━═━═━═━═━═━═━═━\n"
            "⚠️ Security review recommended."
        )

    background_tasks.add_task(send_telegram_notification, image_data, file.filename, caption)
    return resp


# ---------- List Persons ----------
@app.get("/persons_with_images", response_model=List[PersonInfo])
async def list_persons_with_images():
    db = load_persons_db()
    return [
        PersonInfo(
            uuid=uuid,
            name=entry.get("name", "Unknown"),
            image_url=get_profile_image_b64(uuid),
        )
        for uuid, entry in db.items()
    ]


# ---------- Delete Persons ----------
@app.post("/delete_persons")
async def delete_persons(request: DeleteRequest):
    db      = load_persons_db()
    deleted = []
    failed  = []

    for uuid in request.uuids:
        entry = db.get(uuid)
        if not entry:
            failed.append(uuid)
            continue
        try:
            faceset_token = entry.get("faceset_token")
            if faceset_token:
                delete_faceset(faceset_token)
            del db[uuid]
            delete_profile_image(uuid)
            deleted.append(uuid)
            logger.info(f"Deleted person {uuid}")
        except Exception as e:
            logger.error(f"Failed to delete {uuid}: {e}")
            failed.append(uuid)

    save_persons_db(db)
    return {"deleted": deleted, "failed": failed}


# ──────────────────────────────────────────────
#  FRONTEND
# ──────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition Studio | Face++</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <style>
        .profile-img {
            width: 80px; height: 80px;
            object-fit: cover; border-radius: 50%;
            border: 2px solid #3b82f6;
        }
        .person-card { transition: all 0.2s; }
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
        <p class="text-gray-300 mt-3">Train with multiple photos, verify faces, manage users &amp; get Telegram alerts</p>
    </div>

    <div class="grid lg:grid-cols-2 gap-8 mb-12">
        <!-- Train -->
        <div class="bg-gray-800/60 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-gray-700">
            <div class="flex items-center gap-3 mb-6">
                <div class="p-2 bg-blue-500/20 rounded-lg">
                    <svg class="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/>
                    </svg>
                </div>
                <h2 class="text-2xl font-semibold">Register Person</h2>
            </div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-1">Person Name</label>
                    <input type="text" id="trainName" placeholder="e.g., Himanshu"
                        class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-white">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-1">Select Photos (5-6 recommended)</label>
                    <input type="file" id="trainFiles" multiple accept="image/jpeg,image/png,image/jpg"
                        class="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer">
                </div>
                <button id="trainBtn"
                    class="w-full bg-blue-600 hover:bg-blue-700 transition-all duration-200 py-3 rounded-lg font-medium flex items-center justify-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
                    </svg>
                    Train Model
                </button>
            </div>
            <div id="trainResult" class="mt-4 text-sm"></div>
        </div>

        <!-- Verify -->
        <div class="bg-gray-800/60 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-gray-700">
            <div class="flex items-center gap-3 mb-6">
                <div class="p-2 bg-green-500/20 rounded-lg">
                    <svg class="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                </div>
                <h2 class="text-2xl font-semibold">Verify Face</h2>
            </div>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-1">Upload Face Image</label>
                    <input type="file" id="verifyFile" accept="image/jpeg,image/png,image/jpg"
                        class="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-green-600 file:text-white hover:file:bg-green-700 cursor-pointer">
                </div>
                <button id="verifyBtn"
                    class="w-full bg-green-600 hover:bg-green-700 transition-all duration-200 py-3 rounded-lg font-medium flex items-center justify-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                    </svg>
                    Verify Identity
                </button>
            </div>
            <div id="verifyResult" class="mt-4 text-sm"></div>
        </div>
    </div>

    <!-- Registered Users -->
    <div class="bg-gray-800/40 rounded-2xl p-6 border border-gray-700">
        <div class="flex flex-wrap items-center justify-between gap-4 mb-6">
            <h2 class="text-2xl font-semibold flex items-center gap-2">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"/>
                </svg>
                Registered Persons
            </h2>
            <div class="flex gap-3">
                <button id="refreshPersonsBtn" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition flex items-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                    </svg>
                    Refresh
                </button>
                <button id="deleteSelectedBtn" class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition flex items-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
                    </svg>
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
    grid.innerHTML = `<div class="col-span-full text-center py-8">
        <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
        <p class="mt-2 text-gray-400">Loading...</p></div>`;
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
                    <input type="checkbox" class="delete-checkbox absolute top-0 left-0 w-5 h-5 rounded
                        border-gray-600 bg-gray-700 text-blue-500 focus:ring-blue-500" data-uuid="${p.uuid}">
                    <img src="${p.image_url || 'https://via.placeholder.com/80?text=No+Face'}"
                        class="profile-img mx-auto mt-2" alt="profile">
                </div>
                <h3 class="font-medium mt-3 truncate" title="${p.name}">${p.name}</h3>
                <p class="text-xs text-gray-400 truncate">${p.uuid.substring(0,8)}...</p>
            </div>`).join('');
    } catch (err) {
        grid.innerHTML = `<div class="col-span-full text-center py-8 text-red-400">Failed to load: ${err.message}</div>`;
    }
}

async function deleteSelected() {
    const checkboxes = document.querySelectorAll(".delete-checkbox:checked");
    if (!checkboxes.length) { Swal.fire("No Selection", "Select at least one person to delete", "warning"); return; }
    const confirmed = await Swal.fire({
        title: "Delete Persons?",
        text: `You are about to delete ${checkboxes.length} person(s). This cannot be undone.`,
        icon: "warning", showCancelButton: true,
        confirmButtonColor: "#d33", confirmButtonText: "Yes, delete them"
    });
    if (!confirmed.isConfirmed) return;
    const uuids = Array.from(checkboxes).map(cb => cb.dataset.uuid);
    try {
        const res = await fetch(`${API_BASE}/delete_persons`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({uuids})
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Delete failed");
        Swal.fire("Deleted!", `${data.deleted.length} person(s) removed.`, "success");
        loadPersons();
    } catch (err) { Swal.fire("Error", err.message, "error"); }
}

async function trainPerson() {
    const name  = document.getElementById("trainName").value.trim();
    const files = document.getElementById("trainFiles").files;
    const btn   = document.getElementById("trainBtn");
    const resDiv = document.getElementById("trainResult");

    if (!name)          { Swal.fire("Error", "Please enter a name", "error"); return; }
    if (!files.length)  { Swal.fire("Error", "Select at least one image", "error"); return; }

    btn.disabled = true;
    btn.innerHTML = `<div class="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div> Training...`;
    resDiv.innerHTML = "";

    const fd = new FormData();
    fd.append("name", name);
    for (let f of files) fd.append("files", f);

    try {
        const res  = await fetch(`${API_BASE}/train`, {method: "POST", body: fd});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Training failed");
        Swal.fire("Success!", `"${data.name}" registered with ${data.faces_added} face(s).`, "success");
        resDiv.innerHTML = `<div class="text-green-400">✅ ${data.message}</div>`;
        document.getElementById("trainName").value  = "";
        document.getElementById("trainFiles").value = "";
        loadPersons();
    } catch (err) {
        Swal.fire("Training Error", err.message, "error");
        resDiv.innerHTML = `<div class="text-red-400">❌ ${err.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg> Train Model`;
    }
}

async function verifyFace() {
    const file = document.getElementById("verifyFile").files[0];
    const btn  = document.getElementById("verifyBtn");
    const resDiv = document.getElementById("verifyResult");

    if (!file) { Swal.fire("Error", "Please select an image", "error"); return; }

    btn.disabled = true;
    btn.innerHTML = `<div class="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div> Verifying...`;
    resDiv.innerHTML = "";

    const fd = new FormData();
    fd.append("file", file);

    try {
        const res  = await fetch(`${API_BASE}/verify`, {method: "POST", body: fd});
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Verification failed");

        if (data.matched) {
            resDiv.innerHTML = `
                <div class="bg-green-900/50 border border-green-500 rounded-lg p-3">
                    <span class="font-bold text-green-300">✅ Match Found!</span><br>
                    <span class="text-gray-300">Person:</span>
                    <span class="font-mono text-white font-semibold"> ${data.person_name}</span><br>
                    <span class="text-gray-300">Confidence:</span>
                    <span class="text-green-300 font-semibold"> ${data.confidence}%</span><br>
                    <span class="text-gray-300">ID:</span>
                    <span class="text-xs text-gray-400"> ${data.person_id ? data.person_id.slice(0,8) : 'N/A'}...</span>
                </div>`;
            Swal.fire("Match Found!", `Welcome, ${data.person_name}! (${data.confidence}% confidence)`, "success");
        } else {
            resDiv.innerHTML = `
                <div class="bg-red-900/50 border border-red-500 rounded-lg p-3">
                    <span class="font-bold text-red-300">❌ No Match</span><br>
                    <span class="text-gray-300">${data.message}</span>
                </div>`;
            Swal.fire("No Match", "Face not recognised in the database.", "error");
        }
    } catch (err) {
        Swal.fire("Verification Error", err.message, "error");
        resDiv.innerHTML = `<div class="text-red-400">❌ ${err.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg> Verify Identity`;
    }
}

document.getElementById("trainBtn").addEventListener("click", trainPerson);
document.getElementById("verifyBtn").addEventListener("click", verifyFace);
document.getElementById("refreshPersonsBtn").addEventListener("click", loadPersons);
document.getElementById("deleteSelectedBtn").addEventListener("click", deleteSelected);

loadPersons();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=HTML_PAGE)
