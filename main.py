import os
os.environ["GLOG_minloglevel"] = "2"

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Maksimalkan thread OpenCV
cv2.setNumThreads(0)

# ==========================================
# 1. MULTI-THREADED CAMERA READER
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # MJPG codec mempercepat transfer USB webcam ke 60 FPS di resolusi HD
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.002)

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# ==========================================
# 2. INISIALISASI MEDIAPIPE TASKS
# ==========================================
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
    output_face_blendshapes=True,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_hand_skeleton(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


# ==========================================
# 3. PRE-CACHE EMOJI PNG DI RAM
# ==========================================
def load_and_prepare_emoji(path, size):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3] / 255.0)[:, :, None].astype(np.float32)
        return (bgr, alpha, size)
    return (img, None, size)

EMOJI_SPECS = {
    # Emoji Tangan (lowercase semua sesuai permintaan)
    "OK_HAND":   ("ok_hand.png", 110),
    "THUMBS_UP": ("thumbs_up.png", 110),
    "OPEN_PALM": ("open_palm.png", 110),
    "PEACE":     ("peace.png", 110),
    "ROCK_ON":   ("rock_on.png", 110),
    "TRIANGLE":  ("triangle.png", 140),
    # Emoji Ekspresi Wajah
    "SHOCKED":   ("shocked.png", 110),
    "SMILE":     ("smile.png", 110),
    "WINKED":    ("wink.png", 110),
}

emojis = {k: load_and_prepare_emoji(path, sz) for k, (path, sz) in EMOJI_SPECS.items()}

def overlay_cached_png(background, cached_data, center_x, center_y, fallback_text=""):
    bg_h, bg_w, _ = background.shape

    if cached_data is None:
        if fallback_text:
            cx = max(50, min(bg_w - 100, center_x - 45))
            cy = max(30, min(bg_h - 20, center_y))
            cv2.putText(background, fallback_text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        return background

    bgr, alpha, size = cached_data
    half = size // 2

    center_x = max(half + 5, min(bg_w - half - 5, center_x))
    center_y = max(half + 5, min(bg_h - half - 5, center_y))

    x1, y1 = int(center_x - half), int(center_y - half)
    x2, y2 = x1 + size, y1 + size

    if alpha is not None:
        roi = background[y1:y2, x1:x2].astype(np.float32)
        background[y1:y2, x1:x2] = (alpha * bgr + (1.0 - alpha) * roi).astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = bgr

    return background


# ==========================================
# 4. FUNGSI FILTER
# ==========================================
SEPIA_KERNEL = np.array([
    [0.131, 0.534, 0.272],
    [0.168, 0.686, 0.349],
    [0.189, 0.769, 0.393]
], dtype=np.float32)

def apply_bw_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def apply_sepia_filter(frame):
    return cv2.transform(frame, SEPIA_KERNEL)


# ==========================================
# 5. LOGIKA DETEKSI GESTUR (SIMETRIS KIRI & KANAN)
# ==========================================
def get_finger_extension_ratio(lm, tip_idx, pip_idx, mcp_idx):
    """Mengembalikan rasio kelurusan jari relatif terhadap sendi tengahnya."""
    wrist = lm[0]
    dist_tip = math.hypot(lm[tip_idx].x - wrist.x, lm[tip_idx].y - wrist.y)
    dist_pip = math.hypot(lm[pip_idx].x - wrist.x, lm[pip_idx].y - wrist.y)
    if dist_pip < 1e-5:
        return 0.0
    return dist_tip / dist_pip

def analyze_hand(lm):
    palm_size = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
    if palm_size < 1e-5:
        palm_size = 0.1

    # Rasio ekstensi 4 jari utama (> 1.12 artinya lurus, < 1.05 artinya terlipat)
    idx_ratio = get_finger_extension_ratio(lm, 8, 6, 5)
    mid_ratio = get_finger_extension_ratio(lm, 12, 10, 9)
    rng_ratio = get_finger_extension_ratio(lm, 16, 14, 13)
    pnk_ratio = get_finger_extension_ratio(lm, 20, 18, 17)

    index_up  = idx_ratio > 1.12
    middle_up = mid_ratio > 1.12
    ring_up   = rng_ratio > 1.10
    pinky_up  = pnk_ratio > 1.10

    # Status Jempol (tidak butuh di-spread lebar; cukup ujung jempol lebih jauh dari pangkal telunjuk/kelingking)
    # Berlaku simetris untuk tangan kiri maupun kanan
    thumb_tip_to_pinky_mcp = math.hypot(lm[4].x - lm[17].x, lm[4].y - lm[17].y) / palm_size
    thumb_ip_to_pinky_mcp  = math.hypot(lm[3].x - lm[17].x, lm[3].y - lm[17].y) / palm_size
    thumb_open = thumb_tip_to_pinky_mcp > (thumb_ip_to_pinky_mcp * 1.08)

    thumb_up   = (lm[4].y < lm[3].y) and (lm[4].y < lm[5].y) and thumb_open
    pinch_ratio = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / palm_size

    # --- A. CEK TOGGLE FILTER (Simetris untuk tangan kiri & kanan) ---
    filter_cmd = None
    # 3 Jari (Telunjuk + Tengah + Manis lurus, Kelingking terlipat) -> B&W
    if index_up and middle_up and ring_up and not pinky_up:
        filter_cmd = "BW"
    # 2 Jari (Manis + Kelingking lurus, Telunjuk terlipat, Tengah ikut terlipat/lebih rendah dari manis) -> Sepia
    elif (idx_ratio < 1.05) and (mid_ratio < 1.12) and ring_up and pinky_up:
        filter_cmd = "SEPIA"
    # 1 Jari (Hanya Telunjuk lurus, 3 jari lain terlipat) -> Original
    elif index_up and not middle_up and not ring_up and not pinky_up and pinch_ratio > 0.45:
        filter_cmd = "ORIGINAL"

    # --- B. CEK 5 EMOJI SINGLE-HAND ---
    gesture, pos = None, None

    # 1. OK_HAND (Jempol & telunjuk bertemu, 3 jari lain tegak)
    if pinch_ratio < 0.35 and middle_up and ring_up and pinky_up:
        gesture, pos = "OK_HAND", (lm[8].x, lm[8].y)

    # 2. OPEN_PALM (5 jari tegak semua — jari rapat pun langsung terdeteksi!)
    elif index_up and middle_up and ring_up and pinky_up and thumb_open:
        gesture, pos = "OPEN_PALM", (lm[9].x, lm[9].y)

    # 3. PEACE (Telunjuk & jari tengah tegak)
    elif index_up and middle_up and not ring_up and not pinky_up:
        gesture, pos = "PEACE", (lm[8].x, lm[8].y)

    # 4. ROCK_ON (Telunjuk & kelingking tegak)
    elif index_up and pinky_up and not middle_up and not ring_up:
        gesture, pos = "ROCK_ON", (lm[9].x, lm[9].y)

    # 5. THUMBS_UP (Hanya jempol ke atas, 4 jari lain terlipat)
    elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        gesture, pos = "THUMBS_UP", (lm[4].x, lm[4].y)

    return filter_cmd, gesture, pos

def detect_face_expression(blendshapes):
    scores = {b.category_name: b.score for b in blendshapes}
    jaw_open = scores.get("jawOpen", 0)
    smile    = (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) * 0.5
    blink_l  = scores.get("eyeBlinkLeft", 0)
    blink_r  = scores.get("eyeBlinkRight", 0)

    if jaw_open > 0.45:
        return "SHOCKED"
    if (blink_l > 0.45 and blink_r < 0.25) or (blink_r > 0.45 and blink_l < 0.25):
        return "WINKED"
    if smile > 0.42:
        return "SMILE"
    return None


# ==========================================
# 6. MAIN LOOP (PARALLEL INFERENCE)
# ==========================================
def main():
    current_filter = "ORIGINAL"
    last_toggle_time = 0.0
    TOGGLE_COOLDOWN = 0.6

    last_face_expr = None
    last_forehead_pos = None
    face_hold_frames = 0
    MAX_HOLD_FRAMES = 8

    prev_time = time.perf_counter()
    fps_smooth = 0.0
    frame_count = 0

    cam = ThreadedCamera(src=0, width=1280, height=720)
    # Executor 2 worker untuk menjalankan Face & Hand AI secara paralel di multi-core Ryzen 5
    executor = ThreadPoolExecutor(max_workers=2)
    time.sleep(0.3)

    face_results = None

    try:
        while True:
            success, frame = cam.read()
            if not success:
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Resolusi kecil khusus inferensi AI (480px lebar cukup untuk 30-60 FPS stabil)
            mp_w = 480
            mp_h = int(mp_w * (h / w))
            small_rgb = cv2.cvtColor(cv2.resize(frame, (mp_w, mp_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)

            # Jalankan Hand Detector setiap frame, dan Face Detector setiap 2 frame secara PARALEL
            future_hand = executor.submit(hand_detector.detect, mp_image)
            future_face = executor.submit(face_detector.detect, mp_image) if (frame_count % 2 == 1) else None

            hand_results = future_hand.result()
            if future_face is not None:
                face_results = future_face.result()

            # --- DETEKSI WAJAH ---
            if face_results and face_results.face_landmarks and face_results.face_blendshapes:
                face_lm = face_results.face_landmarks[0]
                forehead_now = (int(face_lm[10].x * w), int(face_lm[10].y * h))
                expr_now = detect_face_expression(face_results.face_blendshapes[0])

                if expr_now is not None:
                    last_face_expr = expr_now
                    last_forehead_pos = forehead_now
                    face_hold_frames = MAX_HOLD_FRAMES
                elif face_hold_frames > 0:
                    last_forehead_pos = forehead_now
                    face_hold_frames -= 1
                else:
                    last_face_expr = None
            else:
                if face_hold_frames > 0:
                    face_hold_frames -= 1
                else:
                    last_face_expr = None

            # --- DETEKSI TANGAN ---
            detected_hands_data = []
            triangle_center = None
            now = time.time()

            if hand_results and hand_results.hand_landmarks:
                all_hands = hand_results.hand_landmarks

                # Cek 2 Tangan: Triangle
                if len(all_hands) == 2:
                    lm1, lm2 = all_hands[0], all_hands[1]
                    palm1 = math.hypot(lm1[9].x - lm1[0].x, lm1[9].y - lm1[0].y)
                    palm2 = math.hypot(lm2[9].x - lm2[0].x, lm2[9].y - lm2[0].y)
                    avg_palm = max((palm1 + palm2) * 0.5, 0.05)

                    index_dist = math.hypot(lm1[8].x - lm2[8].x, lm1[8].y - lm2[8].y) / avg_palm
                    thumb_dist = math.hypot(lm1[4].x - lm2[4].x, lm1[4].y - lm2[4].y) / avg_palm

                    if index_dist < 0.45 and thumb_dist < 0.45:
                        cx = int(((lm1[8].x + lm2[8].x + lm1[4].x + lm2[4].x) * 0.25) * w)
                        cy = int(((lm1[8].y + lm2[8].y + lm1[4].y + lm2[4].y) * 0.25) * h)
                        triangle_center = (cx, cy)

                for lm in all_hands:
                    filter_cmd, gesture, pos = analyze_hand(lm)
                    detected_hands_data.append((lm, gesture, pos))

                    if filter_cmd is not None and (now - last_toggle_time) > TOGGLE_COOLDOWN:
                        if current_filter != filter_cmd:
                            current_filter = filter_cmd
                            last_toggle_time = now

            # --- 1. GAMBAR EMOJI WAJAH ---
            if last_face_expr and last_forehead_pos:
                frame = overlay_cached_png(
                    frame, emojis[last_face_expr],
                    last_forehead_pos[0], last_forehead_pos[1] - 70,
                    fallback_text=last_face_expr
                )

            # --- 2. GAMBAR SKELETON & EMOJI TANGAN ---
            if triangle_center:
                for lm, _, _ in detected_hands_data:
                    draw_hand_skeleton(frame, lm, w, h)
                frame = overlay_cached_png(
                    frame, emojis["TRIANGLE"],
                    triangle_center[0], triangle_center[1],
                    fallback_text="TRIANGLE"
                )
            else:
                for lm, gesture, pos in detected_hands_data:
                    draw_hand_skeleton(frame, lm, w, h)
                    if gesture:
                        cx, cy = int(pos[0] * w), int(pos[1] * h)
                        frame = overlay_cached_png(
                            frame, emojis[gesture],
                            cx, cy - 45,
                            fallback_text=gesture
                        )

            # --- 3. TERAPKAN FILTER KE SELURUH FRAME ---
            if current_filter == "BW":
                frame = apply_bw_filter(frame)
            elif current_filter == "SEPIA":
                frame = apply_sepia_filter(frame)

            # --- 4. HITUNG & TAMPILKAN FPS + HUD ---
            curr_time = time.perf_counter()
            instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            fps_smooth = (0.9 * fps_smooth + 0.1 * instant_fps) if fps_smooth > 0 else instant_fps

            cv2.putText(frame, f"Filter: {current_filter} | 1(Telunjuk):Orig | 2(Manis+Kelingking):Sepia | 3:B&W",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow("Hand + Face Emoji AR (Parallel)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        executor.shutdown(wait=False)
        cam.release()
        hand_detector.close()
        face_detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()