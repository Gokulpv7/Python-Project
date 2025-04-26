import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from deepface import DeepFace
from pdf2image import convert_from_path
import pytesseract

# Set Tesseract Path 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

# ✅ Extract Face from Aadhaar Document
def extract_face_from_aadhaar(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    if len(faces) == 0:
        return None  # No face detected

    x, y, w, h = faces[0]  # Assume first detected face is Aadhaar photo
    face = image[y:y+h, x:x+w]
    
    aadhaar_face_path = "aadhaar_face.jpg"
    cv2.imwrite(aadhaar_face_path, face)
    
    return aadhaar_face_path

# ✅ Capture Live Face from Webcam
def capture_live_face():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        cv2.imshow("Live Face Capture - Press 'C' to Capture", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('c'):  # Press 'c' to capture
            user_face_path = "live_face.jpg"
            cv2.imwrite(user_face_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return user_face_path

# ✅ Match Aadhaar Face with Live Face
def match_faces(aadhaar_face, live_face):
    result = DeepFace.verify(aadhaar_face, live_face, model_name="VGG-Face")

    if result["verified"]:
        return "✅ Face Match Successful! e-KYC Approved!"
    else:
        return "❌ Face Match Failed! e-KYC Rejected!"

# ✅ Extract Text from Image Using OCR
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# ✅ Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    full_text = ""
    for img in images:
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"
    return full_text.strip()

# ✅ Process Aadhaar Document (Image/PDF)
def process_document(file_path):
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_text_from_image(file_path)
    elif file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        return "Unsupported file type!"

# ✅ Flask API for e-KYC
@app.route("/ekyc", methods=["POST"])
def ekyc():
    aadhaar_file = request.files["aadhaar"]
    aadhaar_path = "uploaded_aadhaar.jpg"
    aadhaar_file.save(aadhaar_path)

    extracted_text = process_document(aadhaar_path)

    # Step 1: Extract Face from Aadhaar
    aadhaar_face = extract_face_from_aadhaar(aadhaar_path)
    if not aadhaar_face:
        return jsonify({"status": "failed", "message": "❌ No face detected in Aadhaar!"})

    # Step 2: Capture Live Face
    live_face = capture_live_face()
    
    # Step 3: Match Faces
    match_result = match_faces(aadhaar_face, live_face)

    return jsonify({
        "status": "success",
        "match_result": match_result,
        "extracted_text": extracted_text
    })

if __name__ == "__main__":
    app.run(debug=True)
