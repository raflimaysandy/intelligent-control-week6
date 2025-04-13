<<<<<<< HEAD
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Pastikan ultralytics sudah terinstal

# Path ke model dan gambar
MODEL_PATH = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\best.pt"
IMAGE_PATH = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\relhd.jpg"

# Output file
OUTPUT_PATH_YOLO = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\output_yolo.jpg"
OUTPUT_PATH_CANNY = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\output_canny.jpg"
OUTPUT_PATH_COMBINED = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\output_combined.jpg"

# Memuat model YOLO
try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Fungsi untuk mendeteksi tepi menggunakan Canny dan model YOLO
def detect_canny_edges_with_model(image_path, low_threshold=50, high_threshold=150):
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Error: Gambar tidak ditemukan atau gagal dibaca!")
        return

    # Prediksi menggunakan model YOLO
    results = model(image_path, save=True)  # Simpan hasil YOLO ke folder default `runs/detect`
    results[0].save(OUTPUT_PATH_YOLO)  # Simpan hasil YOLO ke path yang ditentukan

    # Konversi ke grayscale untuk Canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Simpan hasil Canny ke file
    cv2.imwrite(OUTPUT_PATH_CANNY, edges)

    # Gabungkan gambar asli dan hasil deteksi tepi
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, edges_bgr))

    # Simpan hasil gabungan ke file
    cv2.imwrite(OUTPUT_PATH_COMBINED, combined)

    print(f"Hasil YOLO disimpan di: {OUTPUT_PATH_YOLO}")
    print(f"Hasil Canny disimpan di: {OUTPUT_PATH_CANNY}")
    print(f"Hasil gabungan disimpan di: {OUTPUT_PATH_COMBINED}")

    # Tampilkan hasil
    cv2.imshow("YOLO & Canny Edge Detection", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Jalankan program
detect_canny_edges_with_model(IMAGE_PATH)

=======
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # Pastikan ultralytics sudah terinstal

# Path ke model dan gambar
MODEL_PATH = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\best.pt"
IMAGE_PATH = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\relhd.jpg"

# Output file
OUTPUT_PATH_YOLO = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\output_yolo.jpg"
OUTPUT_PATH_CANNY = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\output_canny.jpg"
OUTPUT_PATH_COMBINED = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\\output_combined.jpg"

# Memuat model YOLO
try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Fungsi untuk mendeteksi tepi menggunakan Canny dan model YOLO
def detect_canny_edges_with_model(image_path, low_threshold=50, high_threshold=150):
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Error: Gambar tidak ditemukan atau gagal dibaca!")
        return

    # Prediksi menggunakan model YOLO
    results = model(image_path, save=True)  # Simpan hasil YOLO ke folder default `runs/detect`
    results[0].save(OUTPUT_PATH_YOLO)  # Simpan hasil YOLO ke path yang ditentukan

    # Konversi ke grayscale untuk Canny
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Simpan hasil Canny ke file
    cv2.imwrite(OUTPUT_PATH_CANNY, edges)

    # Gabungkan gambar asli dan hasil deteksi tepi
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, edges_bgr))

    # Simpan hasil gabungan ke file
    cv2.imwrite(OUTPUT_PATH_COMBINED, combined)

    print(f"Hasil YOLO disimpan di: {OUTPUT_PATH_YOLO}")
    print(f"Hasil Canny disimpan di: {OUTPUT_PATH_CANNY}")
    print(f"Hasil gabungan disimpan di: {OUTPUT_PATH_COMBINED}")

    # Tampilkan hasil
    cv2.imshow("YOLO & Canny Edge Detection", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Jalankan program
detect_canny_edges_with_model(IMAGE_PATH)

>>>>>>> 7798759 (Week 6: Canny Edge + Instance Segmentation for Lane Detection)
