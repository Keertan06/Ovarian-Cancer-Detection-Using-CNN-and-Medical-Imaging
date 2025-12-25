import os
import pydicom
import cv2
import numpy as np
from tqdm import tqdm
print("Start executing")

SOURCE_DIR = "/Users/keertankumar/Documents/TCGA-OV"
DEST_DIR = "dataset/cancerous"

os.makedirs(DEST_DIR, exist_ok=True)

print(f"🔍 Scanning DICOM files in: {SOURCE_DIR}")

dcm_files = []
for root, _, files in os.walk(SOURCE_DIR):
    for f in files:
        if f.lower().endswith(".dcm"):
            dcm_files.append(os.path.join(root, f))

print(f"📁 Found {len(dcm_files)} DICOM files")

if not dcm_files:
    print("⚠️ No DICOM files found! Check your SOURCE_DIR path.")
    exit()

count_converted = 0
for dcm_path in tqdm(dcm_files[:2000], desc="Converting", unit="file"):
    try:
        dicom_data = pydicom.dcmread(dcm_path)
        image = dicom_data.pixel_array.astype(float)
        image = (np.maximum(image, 0) / image.max()) * 255.0
        image = np.uint8(image)
        image = cv2.resize(image, (256, 256))
        filename = os.path.splitext(os.path.basename(dcm_path))[0] + ".png"
        cv2.imwrite(os.path.join(DEST_DIR, filename), image)
        count_converted += 1
    except Exception as e:
        print(f"❌ Error on {dcm_path}: {str(e)}")

print(f"✅ Converted {count_converted} DICOMs → PNGs saved in {DEST_DIR}")
