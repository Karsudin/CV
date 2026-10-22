"""
main.py

Gesture Particle Universe -- aplikasi partikel 3D yang dikendalikan gesture tangan.

Alur singkat:
  - THREAD KAMERA : baca webcam -> MediaPipe -> kenali gesture -> simpan ke shared_state
  - MAIN THREAD   : baca gesture -> pilih effect -> morphing partikel -> render OpenGL

Tombol:  S = screenshot,  F = fullscreen,  ESC = keluar
Pilih kamera:  python main.py 1
"""

import math
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pygame
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_ARRAY, GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_DYNAMIC_DRAW, GL_FLOAT, GL_LINEAR,
    GL_MODELVIEW, GL_ONE_MINUS_SRC_ALPHA, GL_PACK_ALIGNMENT, GL_POINTS,
    GL_PROJECTION, GL_QUADS, GL_RGB, GL_RGBA, GL_SRC_ALPHA, GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_UNSIGNED_BYTE,
    GL_VERTEX_ARRAY,
    glBegin, glBindBuffer, glBindTexture, glBlendFunc, glBufferData,
    glBufferSubData, glClear, glClearColor, glColor4f, glColorPointer,
    glDisable, glDisableClientState, glDrawArrays, glEnable,
    glEnableClientState, glEnd, glGenBuffers, glGenTextures, glLoadIdentity,
    glMatrixMode, glOrtho, glPixelStorei, glPointSize, glPopMatrix,
    glPushMatrix, glReadPixels, glRotatef, glTexCoord2f, glTexImage2D,
    glTexParameteri, glTranslatef, glVertex2f, glVertexPointer, glViewport,
)
from OpenGL.GLU import gluPerspective
from pygame.locals import DOUBLEBUF, FULLSCREEN, K_ESCAPE, K_f, K_s, KEYDOWN, OPENGL, QUIT

from config.settings import (
    HEIGHT, IDLE_AMPLITUDE, IDLE_SPEED, LERP_SPEED, MODEL_PATH, NUM_PARTICLES,
    ROTATION_SPEED, SCREENSHOT_DIR, TARGET_FPS, VIEW_DISTANCE, WIDTH,
    WINDOW_TITLE,
)
from detectors.gesture_detector import recognize_gesture
from detectors.hand_detector import HandDetector
from effects.firework_effect import FireworkEffect
from effects.galaxy_effect import GalaxyEffect
from effects.heart_effect import HeartEffect
from effects.lightning_effect import LightningEffect
from effects.nebula_effect import NebulaEffect
from effects.saturn_effect import SaturnEffect
from effects.text_effect import TextEffect
from utils.camera_utils import list_available_cameras

# Gesture yang memicu Fireworks (effect ini punya animasi internal lewat update())
FIREWORK_KEY = "THREE"


# ==========================================
# SHARED STATE ANTAR-THREAD
# ==========================================
data_lock = threading.Lock()
shared_state = {
    "running": True,
    "camera_frame": None,   # frame webcam terbaru (untuk window monitor)
    "gesture": "NO_HAND",   # gesture terbaru
}


def camera_worker(camera_index=0):
    """
    Berjalan di THREAD TERPISAH: baca webcam, deteksi tangan, kenali gesture,
    simpan hasilnya ke shared_state (dilindungi Lock).
    """
    cap = cv2.VideoCapture(camera_index)
    detector = HandDetector(max_hands=1)

    while shared_state["running"]:
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # efek cermin
        frame, results = detector.find_hands(frame, draw=True)

        gesture_name = "NO_HAND"
        if results.hand_landmarks:
            gesture_name = recognize_gesture(results.hand_landmarks[0])

        cv2.putText(
            frame, f"Gesture: {gesture_name}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        with data_lock:
            shared_state["camera_frame"] = frame
            shared_state["gesture"] = gesture_name

    cap.release()


# ==========================================
# EFFECT REGISTRY: gesture -> effect
# Menambah effect baru = tambah 1 baris di sini (+ 1 file di folder effects/)
# ==========================================
def build_effects():
    return {
        "POINTING": SaturnEffect(),
        "OPEN_PALM": TextEffect("Karsudin"),
        "PEACE": HeartEffect(),
        "FIST": GalaxyEffect(),
        "THREE": FireworkEffect(),
        "THUMBS_UP": LightningEffect(),
        "DEFAULT": NebulaEffect(),  # dipakai bila gesture belum punya effect
    }


# ==========================================
# OPENGL: KAMERA, VBO, PARTIKEL
# ==========================================
def setup_opengl(width, height):
    """Menyiapkan viewport, 'lensa kamera' (projection), depth test, warna latar."""
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)


def create_vbo(data):
    """Kirim NumPy array (float32) ke GPU, return ID buffer."""
    vbo_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
    return vbo_id


def create_color_vbo(num_particles, default_color=(1.0, 1.0, 1.0)):
    """VBO khusus warna per-partikel."""
    colors = np.tile(np.array(default_color, dtype=np.float32), (num_particles, 1))
    return create_vbo(colors), colors


def update_vbo(vbo_id, data):
    """Timpa isi VBO yang sudah ada (dipanggil tiap frame)."""
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)


def draw_particles(vbo_id, color_vbo_id, count):
    """Gambar SEMUA partikel sekaligus, warna diambil per-partikel dari color VBO."""
    glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, color_vbo_id)
    glEnableClientState(GL_COLOR_ARRAY)
    glColorPointer(3, GL_FLOAT, 0, None)

    glPointSize(3.0)
    glDrawArrays(GL_POINTS, 0, count)

    glDisableClientState(GL_COLOR_ARRAY)


# ==========================================
# OPENGL: TEKS OVERLAY (FPS COUNTER)
# ==========================================
def upload_text_texture(texture_id, text, font, color=(255, 255, 255)):
    """
    Render teks dengan font Pygame lalu isi ke texture OpenGL yang SUDAH ada
    (texture dipakai ulang tiap frame, tidak membuat texture baru terus-menerus).
    Return (width, height) teks.
    """
    surface = font.render(text, True, color)
    data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return width, height


def draw_text_overlay(texture_id, width, height, x, y, screen_width, screen_height):
    """Gambar texture teks sebagai overlay 2D di depan scene 3D."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, screen_width, screen_height, 0, -1, 1)  # (0,0) di kiri-atas

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glBindTexture(GL_TEXTURE_2D, texture_id)
    glEnable(GL_TEXTURE_2D)
    glColor4f(1.0, 1.0, 1.0, 1.0)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x, y)
    glTexCoord2f(1, 1); glVertex2f(x + width, y)
    glTexCoord2f(1, 0); glVertex2f(x + width, y + height)
    glTexCoord2f(0, 0); glVertex2f(x, y + height)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


# ==========================================
# FITUR: SCREENSHOT & FULLSCREEN
# ==========================================
def save_screenshot(width, height, folder=SCREENSHOT_DIR):
    """Simpan isi window OpenGL saat ini sebagai PNG dengan nama berdasarkan waktu."""
    os.makedirs(folder, exist_ok=True)

    glPixelStorei(GL_PACK_ALIGNMENT, 1)  # lebar tidak selalu kelipatan 4 (mis. fullscreen)
    raw_data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    image = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
    image = np.flipud(image)                          # OpenGL membaca dari kiri-BAWAH
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV memakai BGR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(folder, f"screenshot_{timestamp}.png")
    cv2.imwrite(filepath, image_bgr)
    print(f"[INFO] Screenshot disimpan: {filepath}")


def get_desktop_size():
    """Resolusi layar asli (dipanggil SEBELUM window dibuat)."""
    try:
        return pygame.display.get_desktop_sizes()[0]
    except (AttributeError, pygame.error):
        info = pygame.display.Info()
        return info.current_w, info.current_h


def toggle_fullscreen(is_fullscreen, desktop_size):
    """
    Pindah windowed <-> fullscreen.
    Return (width, height, status_fullscreen_baru).
    setup_opengl() HARUS dipanggil ulang setelah ini (aspect ratio berubah).
    """
    new_state = not is_fullscreen
    if new_state:
        width, height = desktop_size
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | FULLSCREEN)
    else:
        width, height = WIDTH, HEIGHT
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    return width, height, new_state


# ==========================================
# MAIN
# ==========================================
def main():
    # --- Pilih kamera dari argumen command line (default 0) ---
    try:
        camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    except ValueError:
        print("[ERROR] Index kamera harus angka, contoh: python main.py 1")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] File model tidak ditemukan: {MODEL_PATH}")
        print("        Jalankan dulu:  python download_model.py")
        return

    print(f"[INFO] Kamera yang tersedia: {list_available_cameras()}")
    print(f"[INFO] Menggunakan kamera index: {camera_index}")

    camera_thread = threading.Thread(
        target=camera_worker, args=(camera_index,), daemon=True
    )
    camera_thread.start()

    pygame.init()
    desktop_size = get_desktop_size()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(WINDOW_TITLE)

    current_width, current_height = WIDTH, HEIGHT
    is_fullscreen = False
    setup_opengl(current_width, current_height)

    pygame.font.init()
    fps_font = pygame.font.SysFont("Consolas", 24)
    fps_texture_id = glGenTextures(1)

    # --- Siapkan effect & posisi target tiap effect (dihitung SEKALI) ---
    effect_map = build_effects()
    effect_positions = {
        key: effect.get_positions(NUM_PARTICLES) for key, effect in effect_map.items()
    }

    current_positions = effect_positions["DEFAULT"].copy()
    vbo_id = create_vbo(current_positions)
    color_vbo_id, current_colors = create_color_vbo(NUM_PARTICLES)

    clock = pygame.time.Clock()
    running = True
    screenshot_requested = False

    rotation_angle = 0.0
    idle_time = 0.0

    while running:
        # ---------- 1. Input ----------
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_s:
                    screenshot_requested = True  # diambil nanti, setelah frame selesai digambar
                elif event.key == K_f:
                    current_width, current_height, is_fullscreen = toggle_fullscreen(
                        is_fullscreen, desktop_size
                    )
                    setup_opengl(current_width, current_height)
                    fps_texture_id = glGenTextures(1)

        # ---------- 2. Pilih effect aktif berdasarkan gesture ----------
        with data_lock:
            current_gesture = shared_state["gesture"]

        active_key = current_gesture if current_gesture in effect_map else "DEFAULT"
        active_effect = effect_map[active_key]

        # ---------- 3. Update posisi partikel ----------
        dt = min(clock.get_time() / 1000.0, 0.1)  # detik sejak frame lalu (dibatasi)

        if active_key == FIREWORK_KEY:
            # Fireworks punya fisika sendiri, bukan lerp ke target diam
            active_effect.update(dt)
            current_positions = active_effect.positions.copy()
        else:
            # Lerp: tiap frame bergerak sebagian kecil jarak ke target
            target_positions = effect_positions[active_key]
            current_positions += (target_positions - current_positions) * LERP_SPEED

        update_vbo(vbo_id, current_positions)

        # ---------- 4. Warna: per-partikel (jika ada) atau satu warna rata ----------
        colors_array = active_effect.get_colors_array(NUM_PARTICLES)
        if colors_array is not None:
            current_colors = colors_array
        else:
            flat_color = np.array(active_effect.get_color(), dtype=np.float32)
            current_colors = np.tile(flat_color, (NUM_PARTICLES, 1))
        update_vbo(color_vbo_id, current_colors)

        # ---------- 5. Render scene 3D ----------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -VIEW_DISTANCE)
        glRotatef(active_effect.get_tilt(), 1, 0, 0)

        motion = active_effect.get_motion_type()
        if motion == "full":
            glRotatef(rotation_angle, 0, 1, 0)
        elif motion == "idle":
            glRotatef(math.sin(idle_time) * IDLE_AMPLITUDE, 0, 1, 0)
        # motion == "none" -> tidak ada rotasi tambahan

        draw_particles(vbo_id, color_vbo_id, NUM_PARTICLES)

        rotation_angle = (rotation_angle + ROTATION_SPEED) % 360.0
        idle_time += IDLE_SPEED

        # ---------- 6. Overlay FPS ----------
        fps_text = f"FPS: {clock.get_fps():.1f}"
        text_w, text_h = upload_text_texture(fps_texture_id, fps_text, fps_font, (0, 255, 0))
        draw_text_overlay(
            fps_texture_id, text_w, text_h, 10, 10, current_width, current_height
        )

        # ---------- 7. Screenshot (sebelum flip, saat frame masih ada di buffer) ----------
        if screenshot_requested:
            save_screenshot(current_width, current_height)
            screenshot_requested = False

        # ---------- 8. Window monitor kamera ----------
        with data_lock:
            cam_frame = shared_state["camera_frame"]
        if cam_frame is not None:
            cv2.imshow("Camera Monitor", cam_frame)
            cv2.waitKey(1)

        pygame.display.flip()
        clock.tick(TARGET_FPS)

    # --- Bersihkan ---
    shared_state["running"] = False
    camera_thread.join(timeout=1.0)
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
