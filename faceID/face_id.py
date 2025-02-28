import cv2
import face_recognition
import numpy as np
import os
import pyttsx3  # 🔊 Sesli uyarı için
import re  # Dosya adından gereksiz kısımları çıkarmak için

# Ses motorunu başlat
engine = pyttsx3.init()

# Kayıtlı yüzlerin bulunduğu klasör
KNOWN_FACES_DIR = "known_faces"

# Bilinen yüzleri ve isimlerini saklayan listeler
known_face_encodings = []
known_face_names = []

# Dosya adındaki "_front", "_right" gibi kısımları temizleyen fonksiyon
def clean_name(filename):
    return re.sub(r'(_front|_right|_left|_back|\d+)', '', os.path.splitext(filename)[0])

# Kayıtlı resimleri yükleyelim
for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:
        known_face_encodings.append(encoding[0])
        name = clean_name(filename)  # Gereksiz kısımları kaldır
        known_face_names.append(name)

# Kamerayı aç
cap = cv2.VideoCapture(0)
last_spoken_name = None  # Aynı ismi tekrar tekrar söylememesi için

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü küçült, hız kazandır
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Yüzleri tespit et
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    anyone_recognized = False  # Algılanan bir yüz var mı?

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Bilinmeyen"

        # En iyi eşleşmeyi bul
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            anyone_recognized = True

        # Sesli uyarıyı yalnızca yeni bir isim algıladığında söyle
        if name != last_spoken_name:
            engine.say(f"{name} algılandı")
            engine.runAndWait()
            last_spoken_name = name  # Aynı ismi tekrar söylememesi için

        # Dikdörtgen çiz
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # İsmi ekrana yazdır
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Eğer hiç tanınan kişi yoksa "Yüz Tanımlanamadı" sesli uyarısı ver
    if not anyone_recognized and last_spoken_name != "Bilinmeyen":
        engine.say("Yüz Tanımlanamadı")
        engine.runAndWait()
        last_spoken_name = "Bilinmeyen"

    # Görüntüyü göster
    cv2.imshow("Yüz Tanıma", frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
