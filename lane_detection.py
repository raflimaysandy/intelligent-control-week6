<<<<<<< HEAD
import cv2
import torch
from ultralytics import YOLO

# Load model YOLOv8 Instance Segmentation
model = YOLO(r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path)  # Jalankan deteksi
    
    # Ambil hasil deteksi pertama dan plot hasilnya
    annotated_image = results[0].plot()
    
    # Simpan hasil deteksi
    output_path = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\lane_detection_result.jpg"
    cv2.imwrite(output_path, annotated_image)
    
    # Tampilkan gambar hasil deteksi
    cv2.imshow("Deteksi Jalur Rel", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contoh penggunaan
detect_rail_lane(r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\relhd.jpg")
=======
import cv2
import torch
from ultralytics import YOLO

# Load model YOLOv8 Instance Segmentation
model = YOLO(r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path)  # Jalankan deteksi
    
    # Ambil hasil deteksi pertama dan plot hasilnya
    annotated_image = results[0].plot()
    
    # Simpan hasil deteksi
    output_path = r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\lane_detection_result.jpg"
    cv2.imwrite(output_path, annotated_image)
    
    # Tampilkan gambar hasil deteksi
    cv2.imshow("Deteksi Jalur Rel", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Contoh penggunaan
detect_rail_lane(r"C:\Rafli\Semester 6\Prak.Kontrol Cerdas\week6\intelligent-control-week6\relhd.jpg")
>>>>>>> 7798759 (Week 6: Canny Edge + Instance Segmentation for Lane Detection)
