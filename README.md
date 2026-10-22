# Gesture Particle Universe

Aplikasi partikel 3D real-time yang dikendalikan gesture tangan lewat webcam.
Stack: **Python 3.11**, **Pygame + PyOpenGL** (render 3D), **MediaPipe Tasks API** (hand tracking), **NumPy**, **OpenCV**.

## Setup (Windows, PowerShell)

> Pakai Python **3.11** (bukan 3.13/3.14) — MediaPipe belum mendukung versi terbaru.

```powershell
py -3.11 -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser   # sekali saja
.\venv\Scripts\Activate
pip install -r requirements.txt
python download_model.py        # unduh model hand tracking ke assets/ (sekali saja)
python main.py
```

Pilih kamera lain: `python main.py 1`

## Gesture & Effect

| Gesture | Bentuk tangan | Effect |
|---|---|---|
| (default / tidak ada tangan) | — | Nebula (oranye, acak) |
| `POINTING` | hanya telunjuk | Saturnus (inti kuning + cincin biru) |
| `OPEN_PALM` | 4 jari terbuka | Teks "Karsudin" |
| `PEACE` | telunjuk + tengah (V) | Heart |
| `FIST` | kepalan | Galaxy spiral |
| `THREE` | telunjuk + tengah + manis | Fireworks multi-letusan |
| `THUMBS_UP` | kepalan, jempol tegak ke atas | Lightning |

## Tombol

| Tombol | Fungsi |
|---|---|
| `S` | Screenshot → `assets/screenshots/` |
| `F` | Fullscreen on/off |
| `ESC` | Keluar |

## Struktur

```
config/settings.py      semua konstanta (ukuran window, jumlah partikel, kecepatan, dst)
detectors/              hand_detector.py (MediaPipe) + gesture_detector.py (aturan gesture)
effects/                base_effect.py (kontrak) + satu file per effect
utils/camera_utils.py   deteksi kamera yang tersedia
main.py                 loop utama: thread kamera + render OpenGL
tests/test_core.py      tes tanpa webcam/GPU:  python -m unittest discover tests -v
```

## Menambah effect baru

1. Buat `effects/nama_effect.py`, turunkan dari `BaseEffect`, isi `get_positions()` dan `get_color()`
   (opsional: `get_colors_array()`, `get_tilt()`, `get_motion_type()`, `update(dt)`).
2. Daftarkan di `build_effects()` pada `main.py`: `"NAMA_GESTURE": NamaEffect(),`
3. Belum punya gesture-nya? Tambah satu kondisi di `detectors/gesture_detector.py`.

## Tips

- FPS kurang lancar → turunkan `NUM_PARTICLES` di `config/settings.py`.
- `THUMBS_UP` sering salah terbaca `FIST` (atau sebaliknya) → atur `THUMB_UP_MARGIN` di `gesture_detector.py`.
- Webcam tidak terbuka → tutup aplikasi lain yang memakai kamera (Zoom, Snipping Tool, dll).
