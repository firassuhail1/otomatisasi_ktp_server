# import sys
# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image
# import json
# import os

# # Path ke Tesseract OCR (Sesuaikan dengan lokasi di sistem Anda)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# def preprocess_image(image_path):
#     """Preprocessing gambar KTP untuk meningkatkan akurasi OCR."""
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Ubah ke grayscale
#     img = cv2.GaussianBlur(img, (5, 5), 0)  # Kurangi noise
#     _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarisasi
    
#     return img

# def extract_text(image_path):
#     """Lakukan OCR pada gambar setelah preprocessing."""
#     processed_img = preprocess_image(image_path)
#     processed_pil = Image.fromarray(processed_img)

#     # Gunakan Tesseract untuk ekstrak teks
#     custom_config = r'--oem 3 --psm 6'
#     text = pytesseract.image_to_string(processed_pil, config=custom_config, lang="ind")

#     return text

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print(json.dumps({"error": "No image path provided"}))
#         sys.exit(1)

#     image_path = sys.argv[1]

#     if not os.path.exists(image_path):
#         print(json.dumps({"error": "Image file not found"}))
#         sys.exit(1)

#     # Ekstrak teks dari gambar
#     extracted_text = extract_text(image_path)

#     # Format hasil sebagai JSON
#     result = {"text": extracted_text}
#     print(json.dumps(result))

#     # Hapus gambar setelah diproses
#     os.remove(image_path)



# import json
# import os
# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image
# import matplotlib.pyplot as plt

# # Path ke Tesseract OCR (Sesuaikan dengan lokasi di sistem Anda)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# script_dir = os.path.dirname(os.path.realpath(__file__))

# # Tentukan nama gambar secara langsung
# image_filename = 'foto_ktp.jpg' 
# image_path = os.path.join(script_dir, image_filename)

# def preprocess_image(img):
#     """Preprocessing gambar KTP untuk meningkatkan akurasi OCR."""
#     results = {}

#     # 1. Konversi ke grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     results["Grayscale"] = gray

#     # 2. Gaussian Blur untuk mengurangi noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     results["Gaussian Blur"] = blurred

#     # 3. Adaptive Thresholding untuk memperjelas teks
#     threshold = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
#     results["Thresholding"] = threshold

#     # # 4. Morphological Closing untuk memperjelas teks putus-putus
#     # kernel = np.ones((2, 2), np.uint8)
#     # closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
#     # results["Morphological Closing"] = closing

#     # # 5. Sharpening Filter untuk mempertajam teks
#     # sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     # sharpened = cv2.filter2D(closing, -1, sharpen_kernel)
#     # results["Sharpened Image"] = sharpened

#     # # 6. Non-Local Means Denoising
#     # denoised = cv2.fastNlMeansDenoising(sharpened, None, 30, 7, 21)
#     # results["Denoised Image"] = denoised

#     # # 7. Canny Edge Detection
#     # edges = cv2.Canny(denoised, 50, 150)
#     # results["Edge Detection"] = edges

#     # # 8. Bitwise AND untuk mempertajam hasil akhir
#     # final = cv2.bitwise_and(denoised, denoised, mask=edges)
#     # results["Final Processed Image"] = final

#     return threshold, results

# def extract_text(processed_img):
#     """Lakukan OCR setelah preprocessing."""
#     processed_pil = Image.fromarray(processed_img)

#     # Konfigurasi Tesseract OCR
#     custom_config = r'--oem 3 --psm 11'
#     text = pytesseract.image_to_string(processed_pil, config=custom_config, lang="ind")

#     return text

# def visualize_preprocessing(results):
#     """Menampilkan hasil tiap tahap preprocessing."""
#     fig, axes = plt.subplots(3, 3, figsize=(15, 10))
#     fig.suptitle("Preprocessing Steps", fontsize=16)

#     for ax, (title, img) in zip(axes.ravel(), results.items()):
#         ax.imshow(img, cmap='gray')
#         ax.set_title(title)
#         ax.axis("off")

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     img = cv2.imread(image_path)
#     if img is None:
#         print(json.dumps({"error": "Gambar tidak ditemukan atau tidak bisa dibaca."}))
#         exit(1)
    
#     processed_img, results = preprocess_image(img)
    
#     # Ekstrak teks setelah preprocessing
#     extracted_text = extract_text(processed_img)
    
#     # Tampilkan visualisasi tiap langkah preprocessing
#     visualize_preprocessing(results)
    
#     # Format hasil sebagai JSON
#     result = {"text": extracted_text}
#     print(json.dumps(result))

    # Hapus gambar setelah diproses
    # os.remove(image_path)


# import json
# import os
# import cv2
# import numpy as np
# import pytesseract
# import base64
# from PIL import Image
# from io import BytesIO

# # Path ke Tesseract OCR (Sesuaikan dengan lokasi di sistem Anda)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# def preprocess_image(img):
#     """Threshold tanpa grayscale, menggunakan channel hijau"""
#     green_channel = img[:, :, 1]  # Ambil channel hijau

#     threshold = cv2.adaptiveThreshold(
#         green_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )

#     return img

# def extract_text(processed_img):
#     """Lakukan OCR setelah preprocessing."""
#     processed_pil = Image.fromarray(processed_img)
#     custom_config = r'--oem 3 --psm 11'
#     text = pytesseract.image_to_string(processed_pil, config=custom_config, lang="ind")
#     return text

# def image_to_base64(img):
#     """Konversi gambar OpenCV ke format Base64."""
#     img_pil = Image.fromarray(img)
#     buffered = BytesIO()
#     img_pil.save(buffered, format="JPEG")  # Simpan sebagai JPEG
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

# if __name__ == "__main__":
#     import sys
#     image_path = sys.argv[1]

#     img = cv2.imread(image_path)
#     if img is None:
#         print(json.dumps({"error": "Gambar tidak ditemukan atau tidak bisa dibaca."}))
#         exit(1)

#     processed_img = preprocess_image(img)
#     extracted_text = extract_text(processed_img)
#     processed_img_base64 = image_to_base64(processed_img)

#     result = {
#         "text": extracted_text,
#         "processed_image": processed_img_base64  # Kirim gambar dalam format Base64
#     }
#     print(json.dumps(result))


# import os
# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image
# import re
# import json

# # Konfigurasi path Tesseract jika diperlukan (uncomment dan sesuaikan jika perlu)
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # contoh Windows

# # Hardcode nama file gambar KTP - UBAH INI sesuai nama file Anda
# IMAGE_FILENAME = "foto_ktp.jpg"  # Pastikan file ini ada di folder yang sama dengan script
# script_dir = os.path.dirname(os.path.realpath(__file__))
# IMAGE_PATH = os.path.join(script_dir, IMAGE_FILENAME)

# print(IMAGE_PATH)

# def preprocess_image(image):
#     """
#     Preprocess gambar untuk meningkatkan akurasi OCR
#     """
#     # Konversi ke grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Menerapkan threshold untuk memperjelas teks
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Menerapkan denoising
#     denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
#     # Menerapkan adaptive threshold untuk membantu dengan area yang memiliki pencahayaan berbeda
#     adaptive_thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                            cv2.THRESH_BINARY, 11, 2)
    
#     # Operasi morfologi untuk membersihkan noise
#     kernel = np.ones((1, 1), np.uint8)
#     opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
#     # Menerapkan dilatasi untuk membuat teks lebih jelas
#     kernel = np.ones((1, 1), np.uint8)
#     dilated = cv2.dilate(opening, kernel, iterations=1)
    
#     return dilated

# def extract_text_from_image(preprocessed_img):
#     """
#     Ekstrak semua teks dari gambar yang telah dipreprocessing
#     """
#     # Konversi gambar OpenCV ke format PIL untuk pytesseract
#     pil_img = Image.fromarray(preprocessed_img)
    
#     # Konfigurasi OCR untuk meningkatkan akurasi
#     custom_config = r'--oem 3 --psm 6 -l ind'
    
#     # Ekstrak teks menggunakan pytesseract dengan bahasa Indonesia
#     text = pytesseract.image_to_string(pil_img, config=custom_config)
    
#     return text

# def extract_structured_data(text):
#     """
#     Ekstrak data terstruktur dari teks OCR
#     """
#     data = {}
    
#     # Pola regex untuk field KTP Indonesia
#     patterns = {
#         'nik': r'NIK\s*:?\s*(\d[\d\s]+)',
#         'nama': r'Nama\s*:?\s*([A-Z\s]+)',
#         'tempat_tgl_lahir': r'Tempat[/\-\s]*Tgl\s*Lahir\s*:?\s*([A-Z\s]+,\s*[\d\-]+)',
#         'jenis_kelamin': r'Jenis[\s\-]*[Kk]elamin\s*:?\s*([A-Z/-]+)',
#         'alamat': r'Alamat\s*:?\s*([A-Z\s]+)',
#         'rt_rw': r'RT[/\-\s]*RW\s*:?\s*(\d+[/\-\s]*\d+)',
#         'kel_desa': r'Kel[/\-\s]*Desa\s*:?\s*([A-Z\s]+)',
#         'kecamatan': r'Kecamatan\s*:?\s*([A-Z\s]+)',
#         'agama': r'Agama\s*:?\s*([A-Z\s]+)',
#         'status_perkawinan': r'Status\s*Perkawinan\s*:?\s*([A-Z\s]+)',
#         'pekerjaan': r'Pekerjaan\s*:?\s*([A-Z/\s]+)',
#         'kewarganegaraan': r'Kewarganegaraan\s*:?\s*([A-Z\s]+)',
#         'berlaku_hingga': r'Berlaku\s*Hingga\s*:?\s*([A-Z\s]+)',
#         'gol_darah': r'Gol[.\s]*Darah\s*:?\s*([A-ZO\-]+)'
#     }
    
#     # Ekstrak data menggunakan pola regex
#     for key, pattern in patterns.items():
#         match = re.search(pattern, text)
#         if match:
#             # Bersihkan hasil (hapus spasi berlebih, karakter non-alfanumerik di awal/akhir)
#             data[key] = re.sub(r'^\W+|\W+$', '', match.group(1).strip())
#             # Untuk NIK, hapus semua spasi
#             if key == 'nik':
#                 data[key] = re.sub(r'\s', '', data[key])
#         else:
#             data[key] = "Tidak ditemukan"
    
#     return data

# def clean_extracted_data(data):
#     """
#     Membersihkan dan memvalidasi data yang diekstrak
#     """
#     # Membersihkan NIK - pastikan hanya berisi digit
#     if 'nik' in data and data['nik'] != "Tidak ditemukan":
#         data['nik'] = re.sub(r'\D', '', data['nik'])
    
#     # Membersihkan nama - hapus karakter khusus
#     if 'nama' in data and data['nama'] != "Tidak ditemukan":
#         data['nama'] = re.sub(r'[^A-Z\s]', '', data['nama']).strip()
    
#     return data

# def main():
#     try:
#         # Baca gambar dari file hardcode
#         image_path = IMAGE_PATH
#         img = cv2.imread(image_path)
        
#         if img is None:
#             print(json.dumps({"error": f"Gagal membaca gambar {image_path}. Pastikan file ada dan path benar."}, 
#                           indent=2, ensure_ascii=False))
#             return
        
#         # Preprocessing gambar
#         print("Melakukan preprocessing gambar...")
#         preprocessed_img = preprocess_image(img)
        
#         # Simpan hasil preprocessing untuk debugging
#         cv2.imwrite("preprocessed_debug.jpg", preprocessed_img)
#         print("Gambar hasil preprocessing disimpan ke preprocessed_debug.jpg")
        
#         # Ekstrak teks
#         print("Mengekstrak teks dari gambar...")
#         extracted_text = extract_text_from_image(preprocessed_img)
        
#         # Ekstrak data terstruktur
#         print("Mengekstrak data terstruktur...")
#         data = extract_structured_data(extracted_text)
        
#         # Bersihkan data
#         clean_data = clean_extracted_data(data)
        
#         # Tambahkan teks mentah untuk debugging
#         clean_data['raw_text'] = extracted_text
        
#         # Tampilkan hasil sebagai JSON
#         result_json = json.dumps(clean_data, indent=2, ensure_ascii=False)
#         print("\nHASIL EKSTRAKSI:")
#         print(result_json)
        
#         # Simpan hasil ke file JSON
#         with open("hasil_ktp.json", "w", encoding="utf-8") as f:
#             f.write(result_json)
#         print("\nHasil disimpan ke file hasil_ktp.json")
        
#     except Exception as e:
#         error_json = json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)
#         print(error_json)

# if __name__ == "__main__":
#     main()


# import json
# import os
# import sys
# import cv2
# import numpy as np
# import base64
# from PIL import Image
# from io import BytesIO
# from google.cloud import vision
# import re
# import spacy
# import requests

# # Pastikan kredensial Google Vision diatur
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\pdam\pdam-server\scripts\ocr-dg-google-vision.json"
# API_KEY = "AIzaSyC3mxtm_A1kyv89RFjGTZt50CrhockO5QY"

# # Load model NLP spaCy
# nlp = spacy.load("en_core_web_sm")

# def preprocess_image(img):
#     """Threshold tanpa grayscale, menggunakan channel hijau"""

#     return img

# def preprocess_image_scam(image):
#     """
#     Preprocess gambar untuk meningkatkan akurasi OCR
#     """
#     # Konversi ke grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Menerapkan threshold untuk memperjelas teks
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Menerapkan denoising
#     denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
#     return thresh

# # def detect_text(image):
# #     """Deteksi teks menggunakan Google Cloud Vision OCR"""
# #     client = vision.ImageAnnotatorClient()

# #     # Encode image as bytes
# #     success, encoded_image = cv2.imencode('.jpg', image)
# #     if not success:
# #         return {"error": "Failed to encode image"}

# #     # Create vision image from encoded bytes
# #     vision_image = vision.Image(content=encoded_image.tobytes())
    
# #     # Perform text detection
# #     response = client.text_detection(image=vision_image)

# #     if response.error.message:
# #         return {"error": response.error.message}

# #     texts = response.text_annotations
# #     detected_text = texts[0].description if texts else ""

# #     return detected_text

# def detect_text(image):
#     # Encode image to base64
#     success, encoded_image = cv2.imencode('.jpg', image)
#     if not success:
#         return {"error": "Failed to encode image"}

#     img_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

#     # Prepare request payload
#     url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
#     headers = {"Content-Type": "application/json"}
#     data = {
#         "requests": [
#             {
#                 "image": {"content": img_base64},
#                 "features": [{"type": "TEXT_DETECTION"}]
#             }
#         ]
#     }

#     # Send request
#     response = requests.post(url, headers=headers, json=data)
#     if response.status_code != 200:
#         return {"error": f"Request failed: {response.text}"}

#     result = response.json()
#     try:
#         text = result['responses'][0]['textAnnotations'][0]['description']
#     except (KeyError, IndexError):
#         text = ""

#     return text

# def image_to_base64(img):
#     """Konversi gambar OpenCV ke format Base64"""
#     img_pil = Image.fromarray(img)
#     buffered = BytesIO()
#     img_pil.save(buffered, format="JPEG")  # Simpan sebagai JPEG
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

# def extract_information(text):
#     """Ekstrak informasi dari teks OCR KTP"""
    
#     # # Preprocessing: Hapus newline & bersihkan teks
#     # text = text.replace("\n", " ")
#     # text = re.sub(r'\s+', ' ', text).strip()

#     # Cari NIK (16 digit angka)
#     nik_match = re.search(r'\b\d{16}\b', text)
#     nik = nik_match.group(0) if nik_match else None

#     # Membuat regex untuk mengekstrak semua kata/kalimat yang bukan bagian dari label
#     exclude_keywords_for_nama = r"(NIK|\s*Nama\s*|\s*Tempat\s*|\s*Lahir|Jenis kelamin|\s*Alamat\s*|PROVINSI|KABUPATEN|RT/RW|Kel/Desa|Kecamatan|Agama|Status Perkawinan|Pekerjaan|Gol Darah|Kewarganegaraan|perempuan|laki-laki|Berlaku Hingga|\d{16})"
#     exclude_keywords_for_alamat = r"(NIK|\s*Nama\s*|\s*Tempat\s*|\s*Lahir|Jenis kelamin|PROVINSI|KABUPATEN|RT/RW|Kel/Desa|Kecamatan|Agama|Status Perkawinan|Pekerjaan|Kewarganegaraan|Berlaku Hingga|gol|LAKI-LAKI|PEREMPUAN|ISLAM|KRISTEN|KATOLIK|HINDU|BUDHA|KONGHUCU|\d{16})"
    
#     # Memisahkan teks per baris
#     lines = text.split("\n")

#     # Menyaring baris yang tidak mengandung kata kunci yang dikecualikan
#     nama = ambil_nama_dari_lines(lines, exclude_keywords_for_nama)

#     alamat = [
#         re.sub(r"[^\w\s]", "", line.strip())  # Menghapus karakter spesial seperti : , . dll
#         for line in lines
#         if line.strip() and not re.search(exclude_keywords_for_alamat, line, re.IGNORECASE)
#     ]

#     # Cari jenis kelamin
#     jenis_kelamin_match = re.search(r'\b(LAKI-LAKI|PEREMPUAN)\b', text)
#     jenis_kelamin = jenis_kelamin_match.group(0) if jenis_kelamin_match else None

#     # Cari TTL (Tanggal Lahir)
#     ttl_match = re.search(r'(\d{2}-\d{2}-\d{4})', text)
#     ttl = ttl_match.group(0) if ttl_match else None

#     # Cari agama (hanya kata "ISLAM", "KRISTEN", dll)
#     agama_match = re.search(r'\b(ISLAM|KRISTEN|KATOLIK|HINDU|BUDHA|KONGHUCU)\b', text)
#     agama = agama_match.group(0) if agama_match else None

#     result = {
#         "nik": nik,
#         "nama": nama,
#         "alamat": alamat[3],
#         "ttl": ttl,
#         "jenis_kelamin": jenis_kelamin,
#         "agama": agama,
#     }

#     return result

# def ambil_nama_dari_lines(lines, exclude_words):
#     nama = None
#     for i, line in enumerate(lines):
#         # Cek apakah ini baris yang mengandung NIK (16 digit angka)
#         if re.search(r"\b\d{16}\b", line):
#             # Cek baris berikutnya (kalau ada)
#             if i + 1 < len(lines):
#                 kandidat = lines[i + 1].strip()
                
#                 # Filter sederhana: jangan ambil jika mengandung kata-kata exclude
#                 if not re.search(r"(Tempat|Lahir|Jenis kelamin|Alamat|KABUPATEN|PROVINSI)", kandidat, re.IGNORECASE):
#                     # Bersihkan karakter selain huruf dan spasi
#                     nama = re.sub(r"[^\w\s]", "", kandidat)
#                     break

#     return nama

# def process_image(image_path):
#     """Memproses gambar untuk mendeteksi teks dan mengekstrak informasi"""
#     try:
#         # Baca gambar
#         img = cv2.imread(image_path)
#         if img is None:
#             return {"error": f"Failed to load image: {image_path}"}

#         # Preprocessing image
#         processed_img = preprocess_image(img)

#         processed_img_scam = preprocess_image_scam(img)

#         # Deteksi teks dengan Google Cloud Vision
#         detected_text = detect_text(processed_img)

#         # Ekstrak informasi dari teks
#         extracted_info = extract_information(detected_text)

#         # Konversi gambar ke Base64
#         processed_img_base64 = image_to_base64(processed_img)
#         processed_img_base64_2 = image_to_base64(processed_img_scam)

#         # Hasilkan JSON
#         result = {
#             "text": detected_text,
#             "extracted_info": extracted_info,
#             "processed_image": processed_img_base64,
#             "processed_image_scam": processed_img_base64_2
#         }

#         return result

#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     # if len(sys.argv) < 2:
#     #     print(json.dumps({"error": "No image path provided"}))
#         # sys.exit(1)

#     # image_path = sys.argv[1]
#     image_path = os.path.join(os.path.dirname(__file__), "foto_ktp.JPG")
#     result = process_image(image_path)
#     print(json.dumps(result))  # Output dalam format JSON agar bisa dibaca Laravel


import json
import os
import sys
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import re
import spacy
import requests
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\joki\otomatisasi_ktp_server\scripts\ocr-dg-google-vision.json"

# Load model NLP spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess_image(img):
    return img

def preprocess_image_scam(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    return thresh

def detect_text(image):
    # --- Perubahan di sini: Menggunakan Google Cloud Vision Client Library ---
    client = vision.ImageAnnotatorClient()

    # Konversi gambar OpenCV (numpy array) ke bytes
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return {"error": "Failed to encode image to bytes"}

    image_content = encoded_image.tobytes()
    image = vision.Image(content=image_content)

    try:
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            # Teks penuh yang dideteksi akan ada di elemen pertama (indeks 0) dari text_annotations
            detected_full_text = texts[0].description
        else:
            detected_full_text = ""

        if response.error.message:
            return {"error": f"Google Vision API error: {response.error.message}"}

    except Exception as e:
        return {"error": f"Error during Google Vision API call: {str(e)}"}

    return detected_full_text
    # --- Akhir Perubahan ---

def image_to_base64(img):
    img_pil = Image.fromarray(img)
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_information(text):
    nik_match = re.search(r'\b\d{16}\b', text)
    nik = nik_match.group(0) if nik_match else None

    exclude_keywords_for_nama = r"(NIK|\s*Nama\s*|\s*Tempat\s*|\s*Lahir|Jenis kelamin|\s*Alamat\s*|PROVINSI|KABUPATEN|RT/RW|Kel/Desa|Kecamatan|Agama|Status Perkawinan|Pekerjaan|Gol Darah|Kewarganegaraan|perempuan|laki-laki|Berlaku Hingga|\d{16})"
    exclude_keywords_for_alamat = r"(NIK|\s*Nama\s*|\s*Tempat\s*|\s*Lahir|Jenis kelamin|PROVINSI|KABUPATEN|RT/RW|Kel/Desa|Kecamatan|Agama|Status Perkawinan|Pekerjaan|Kewarganegaraan|Berlaku Hingga|gol|LAKI-LAKI|PEREMPUAN|ISLAM|KRISTEN|KATOLIK|HINDU|BUDHA|KONGHUCU|\d{16})"

    lines = text.split("\n")
    nama = ambil_nama_dari_lines(lines, exclude_keywords_for_nama)

    alamat = [
        re.sub(r"[^\w\s]", "", line.strip())
        for line in lines
        if line.strip() and not re.search(exclude_keywords_for_alamat, line, re.IGNORECASE)
    ]

    jenis_kelamin_match = re.search(r'\b(LAKI-LAKI|PEREMPUAN)\b', text)
    jenis_kelamin = jenis_kelamin_match.group(0) if jenis_kelamin_match else None

    ttl_match = re.search(r'(\d{2}-\d{2}-\d{4})', text)
    ttl = ttl_match.group(0) if ttl_match else None

    agama_match = re.search(r'\b(ISLAM|KRISTEN|KATOLIK|HINDU|BUDHA|KONGHUCU)\b', text)
    agama = agama_match.group(0) if agama_match else None

    result = {
        "nik": nik,
        "nama": nama,
        "alamat": alamat[3] if len(alamat) > 3 else None,
        "ttl": ttl,
        "jenis_kelamin": jenis_kelamin,
        "agama": agama,
    }

    return result

def ambil_nama_dari_lines(lines, exclude_words):
    nama = None
    for i, line in enumerate(lines):
        if re.search(r"\b\d{16}\b", line):
            if i + 1 < len(lines):
                kandidat = lines[i + 1].strip()
                if not re.search(r"(Tempat|Lahir|Jenis kelamin|Alamat|KABUPATEN|PROVINSI)", kandidat, re.IGNORECASE):
                    nama = re.sub(r"[^\w\s]", "", kandidat)
                    break
    return nama

def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Failed to load image: {image_path}"}

        processed_img = preprocess_image(img)
        processed_img_scam = preprocess_image_scam(img)

        detected_text = detect_text(processed_img)

        if isinstance(detected_text, dict) and "error" in detected_text:
            return {"error": detected_text["error"]}

        extracted_info = extract_information(detected_text)

        processed_img_base64 = image_to_base64(processed_img)
        processed_img_base64_2 = image_to_base64(processed_img_scam)

        result = {
            "text": detected_text,
            "extracted_info": extracted_info,
            "processed_image": processed_img_base64,
            "processed_image_scam": processed_img_base64_2
        }

        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # image_path = os.path.join(os.path.dirname(__file__), "foto_ktp.JPG")
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        # sys.exit(1)

    image_path = sys.argv[1]
    result = process_image(image_path)
    print(json.dumps(result))
