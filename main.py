import customtkinter as ctk
from PIL import Image
import cv2
import mediapipe as mp
import numpy as np

ctk.set_appearance_mode("dark")

# =========================
# Theme
# =========================

PRIMARY       = "#14B8A6"
PRIMARY_HOVER = "#0F9B8E"
BG            = "#0F172A"
CARD          = "#1E293B"
TEXT          = "#F8FAFC"

# =========================
# MediaPipe Setup
# =========================

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.55
)

# =========================
# Data
# =========================

letters = {
    letter: {
        "description": f"ASL sign for the letter {letter}",
        "image": f"images/{letter}.jpg"
    }
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
}

letter_keys    = list(letters.keys())
current_letter = 0

# =========================
# Motion tracking for J
# =========================

prev_index_tip = None
j_motion_frames = 0
J_MOTION_THRESHOLD = 8

# =========================
# Helper Functions
# =========================

def dist2d(lm, a, b):
    return np.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)

def dist3d(lm, a, b):
    return np.sqrt(
        (lm[a].x - lm[b].x)**2 +
        (lm[a].y - lm[b].y)**2 +
        (lm[a].z - lm[b].z)**2
    )

def classify_asl(lm):
    if len(lm) < 21:
        return None

    def finger_extended(tip, pip, mcp):
        wrist = lm[0]
        tip_dist = np.hypot(lm[tip].x - wrist.x, lm[tip].y - wrist.y)
        pip_dist = np.hypot(lm[pip].x - wrist.x, lm[pip].y - wrist.y)
        return tip_dist > pip_dist * 1.08

    idx_ext = finger_extended(8, 6, 5)
    mid_ext = finger_extended(12, 10, 9)
    rng_ext = finger_extended(16, 14, 13)
    pnk_ext = finger_extended(20, 18, 17)
    thm_ext = dist2d(lm, 4, 5) > 0.085

    palm_size = dist2d(lm, 0, 9) + 1e-6

    d_thumb_idx = dist2d(lm, 4, 8) / palm_size
    d_thumb_mid = dist2d(lm, 4, 12) / palm_size
    d_thumb_rng = dist2d(lm, 4, 16) / palm_size
    d_thumb_pnk = dist2d(lm, 4, 20) / palm_size
    d_idx_mid   = dist2d(lm, 8, 12) / palm_size
    d_idx_tip_pip = dist2d(lm, 8, 6) / palm_size

    idx_horiz = abs(lm[8].y - lm[5].y) < 0.13
    idx_down  = lm[8].y > lm[6].y + 0.03
    thm_horiz = abs(lm[4].x - lm[2].x) > abs(lm[4].y - lm[2].y) * 1.2

    # ==================== C - Strongly Improved ====================
    if thm_ext and not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
        tips_y_spread = max(lm[8].y, lm[12].y, lm[16].y, lm[20].y) - min(lm[8].y, lm[12].y, lm[16].y, lm[20].y)
        idx_curved = dist2d(lm, 8, 7) / palm_size < 0.28
        mid_curved = dist2d(lm, 12, 11) / palm_size < 0.28

        if (0.20 < d_thumb_idx < 0.65 and 
            0.20 < d_thumb_mid < 0.68 and
            tips_y_spread / palm_size < 0.60 and
            idx_curved and mid_curved):
            return "C"

    # A
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and thm_ext and d_thumb_idx > 0.25:
        return "A"

    # B
    if idx_ext and mid_ext and rng_ext and pnk_ext and not thm_ext and d_idx_mid < 0.23:
        return "B"

    # D
    if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thm_ext and d_thumb_mid < 0.29:
        return "D"

    # E
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thm_ext:
        if lm[4].y > lm[5].y + 0.02 and d_thumb_idx < 0.37:
            return "E"

    # F
    if not idx_ext and mid_ext and rng_ext and pnk_ext and thm_ext and d_thumb_idx < 0.23:
        return "F"

    # G
    if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thm_ext and idx_horiz and thm_horiz:
        return "G"

    # H
    if idx_ext and mid_ext and not rng_ext and not pnk_ext and not thm_ext and idx_horiz and abs(lm[8].y - lm[12].y) / palm_size < 0.19:
        return "H"

    # I
    if not idx_ext and not mid_ext and not rng_ext and pnk_ext and not thm_ext:
        return "I"

    # K
    if idx_ext and mid_ext and not rng_ext and not pnk_ext and thm_ext and not idx_horiz and d_thumb_idx < 0.33 and d_thumb_mid < 0.33:
        return "K"

    # L
    if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thm_ext and not idx_horiz and thm_horiz:
        return "L"

    # M
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thm_ext:
        if d_thumb_idx < 0.33 and d_thumb_mid < 0.33 and d_thumb_rng < 0.36:
            return "M"

    # N
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thm_ext:
        if d_thumb_idx < 0.31 and d_thumb_mid < 0.31 and d_thumb_rng > 0.24:
            return "N"

    # O
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and thm_ext:
        tips = [d_thumb_idx, d_thumb_mid, d_thumb_rng, d_thumb_pnk]
        if all(0.06 < t < 0.33 for t in tips):
            return "O"

    # P
    if idx_ext and mid_ext and not rng_ext and not pnk_ext and thm_ext and idx_down:
        return "P"

    # Q
    if idx_ext and not mid_ext and not rng_ext and not pnk_ext and thm_ext and idx_down and thm_horiz:
        return "Q"

    # R
    if idx_ext and mid_ext and not rng_ext and not pnk_ext and not thm_ext and d_idx_mid < 0.14 and not idx_horiz:
        return "R"

    # S
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thm_ext:
        if lm[4].y < lm[8].y and d_thumb_idx < 0.33:
            return "S"

    # T
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thm_ext:
        if dist2d(lm, 4, 6) / palm_size < 0.23:
            return "T"

    # U
    if idx_ext and mid_ext and not rng_ext and not pnk_ext and not thm_ext and d_idx_mid < 0.15 and not idx_horiz:
        return "U"

    # V
    if idx_ext and mid_ext and not rng_ext and not pnk_ext and not thm_ext and d_idx_mid >= 0.15 and not idx_horiz:
        return "V"

    # W
    if idx_ext and mid_ext and rng_ext and not pnk_ext and not thm_ext:
        return "W"

    # X
    if not idx_ext and not mid_ext and not rng_ext and not pnk_ext and not thm_ext and d_idx_tip_pip < 0.23:
        return "X"

    # Y
    if not idx_ext and not mid_ext and not rng_ext and pnk_ext and thm_ext:
        return "Y"

    return None


# =========================
# App
# =========================

app = ctk.CTk()
app.title("ASL for Nerds")
app.geometry("1200x720")
app.configure(fg_color=BG)

# =========================
# Webcam State
# =========================

cap                  = None
webcam_running       = False
feedback_text        = ""
feedback_hold_frames = 0
CORRECT_HOLD         = 25

# =========================
# Frames
# =========================

home_frame  = ctk.CTkFrame(app, fg_color=BG)
home_frame.pack(fill="both", expand=True)

learn_frame = ctk.CTkFrame(app, fg_color=BG)

# =========================
# Webcam Functions
# =========================

def start_webcam():
    global cap, webcam_running, prev_index_tip, j_motion_frames
    cap = cv2.VideoCapture(0)
    webcam_running = True
    prev_index_tip = None
    j_motion_frames = 0
    update_webcam()

def stop_webcam():
    global cap, webcam_running
    webcam_running = False
    if cap:
        cap.release()
        cap = None

def update_webcam():
    global feedback_text, feedback_hold_frames, prev_index_tip, j_motion_frames

    if not webcam_running or cap is None:
        return

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)

        predicted = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                predicted = classify_asl(lm)

                # J motion detection
                if letter_keys[current_letter] == 'J' and predicted is None:
                    idx_tip = lm[8]
                    if prev_index_tip is not None:
                        dy = idx_tip.y - prev_index_tip.y
                        if dy > 0.015:
                            j_motion_frames += 1
                        else:
                            j_motion_frames = max(0, j_motion_frames - 2)
                    prev_index_tip = idx_tip

                    if j_motion_frames >= J_MOTION_THRESHOLD:
                        predicted = "J"
                        j_motion_frames = 0

        current_key = letter_keys[current_letter]

        if predicted == current_key:
            feedback_hold_frames = CORRECT_HOLD

        if feedback_hold_frames > 0:
            feedback_hold_frames -= 1
            feedback_text = "Correct! Great job!"
            overlay_color = (34, 197, 94)
        elif predicted:
            feedback_text = f"Detected: {predicted} — try {current_key}"
            overlay_color = (100, 130, 180)
        else:
            feedback_text = f"Show the sign for {current_key}"
            overlay_color = (80, 100, 140)

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), (20, 30, 50), -1)
        cv2.putText(frame, feedback_text, (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, overlay_color, 2, cv2.LINE_AA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(420, 315))
        webcam_label.configure(image=ctk_img)
        webcam_label.image = ctk_img

    app.after(30, update_webcam)

# =========================
# Navigation Functions
# =========================

def show_learn_page():
    home_frame.pack_forget()
    learn_frame.pack(fill="both", expand=True)
    update_letter()
    start_webcam()

def show_home_page():
    stop_webcam()
    learn_frame.pack_forget()
    home_frame.pack(fill="both", expand=True)

def next_letter():
    global current_letter
    current_letter = (current_letter + 1) % len(letter_keys)
    update_letter()

def previous_letter():
    global current_letter
    current_letter = (current_letter - 1) % len(letter_keys)
    update_letter()

def update_letter():
    current_key = letter_keys[current_letter]
    letter_label.configure(text=f"Letter  {current_key}")
    description_label.configure(text=letters[current_key]["description"])
    try:
        img = Image.open(letters[current_key]["image"])
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(260, 260))
        image_label.configure(image=ctk_img, text="")
        image_label.image = ctk_img
    except Exception:
        image_label.configure(image=None, text="[no image]", text_color="#64748B")

# =========================
# Home Page UI
# =========================

ctk.CTkLabel(
    home_frame,
    text="🤟 ASL for Nerds",
    font=("Arial", 48, "bold"),
    text_color=TEXT
).pack(pady=(120, 20))

ctk.CTkLabel(
    home_frame,
    text="Learn ASL the fun way",
    font=("Arial", 20),
    text_color="#CBD5E1"
).pack(pady=(0, 50))

ctk.CTkButton(
    home_frame,
    text="START LEARNING",
    width=260,
    height=70,
    corner_radius=20,
    font=("Arial", 24, "bold"),
    fg_color=PRIMARY,
    hover_color=PRIMARY_HOVER,
    command=show_learn_page
).pack()

# =========================
# Learn Page UI
# =========================

ctk.CTkButton(
    learn_frame,
    text="← Back",
    command=show_home_page
).pack(anchor="w", padx=20, pady=20)

content_frame = ctk.CTkFrame(learn_frame, fg_color="transparent")
content_frame.pack(fill="both", expand=True, padx=30, pady=10)

# Left column - Reference
left_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
left_frame.pack(side="left", padx=10, fill="y")

ctk.CTkLabel(left_frame, text="REFERENCE", font=("Arial", 12, "bold"), text_color="#64748B").pack(pady=(10, 4))

image_label = ctk.CTkLabel(left_frame, text="")
image_label.pack(pady=(0, 10))

nav_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
nav_frame.pack()

ctk.CTkButton(nav_frame, text="◀ Prev", width=110, fg_color=CARD, hover_color="#2D3F55", command=previous_letter).pack(side="left", padx=6)
ctk.CTkButton(nav_frame, text="Next ▶", width=110, fg_color=CARD, hover_color="#2D3F55", command=next_letter).pack(side="left", padx=6)

# Info column
info_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
info_frame.pack(side="left", fill="y", padx=10, pady=10)

letter_label = ctk.CTkLabel(info_frame, text="", font=("Arial", 54, "bold"), text_color=PRIMARY)
letter_label.pack(anchor="w", pady=(30, 6))

description_label = ctk.CTkLabel(info_frame, text="", wraplength=220, justify="left", font=("Arial", 16), text_color="#CBD5E1")
description_label.pack(anchor="w")

# Tip box
tip_frame = ctk.CTkFrame(info_frame, fg_color=CARD, corner_radius=12)
tip_frame.pack(anchor="w", pady=(30, 0), fill="x")

ctk.CTkLabel(tip_frame, text="💡 Tip", font=("Arial", 13, "bold"), text_color=PRIMARY).pack(anchor="w", padx=14, pady=(10, 2))
ctk.CTkLabel(tip_frame, text="Hold your hand clearly in\nfront of the camera.\nGood lighting helps!", 
             font=("Arial", 13), text_color="#94A3B8", justify="left").pack(anchor="w", padx=14, pady=(0, 12))

# Webcam column
cam_frame = ctk.CTkFrame(content_frame, fg_color=CARD, corner_radius=16)
cam_frame.pack(side="left", padx=20, pady=10, fill="both", expand=True)

ctk.CTkLabel(cam_frame, text="📷 YOUR CAMERA", font=("Arial", 12, "bold"), text_color="#64748B").pack(pady=(14, 6))

webcam_label = ctk.CTkLabel(cam_frame, text="Starting camera…", text_color="#64748B")
webcam_label.pack(padx=16, pady=(0, 16))

# =========================
# Cleanup
# =========================

def on_close():
    stop_webcam()
    hands_detector.close()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)

# Start
update_letter()
app.mainloop()
