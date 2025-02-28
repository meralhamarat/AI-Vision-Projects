import cv2
import numpy as np

# Kamerayı aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü HSV formatına çevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Kırmızı renk aralığını belirle
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])

    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2  # İki maske birleştirildi

    # Maskeyi uygula
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Konturları bul ve çiz
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Küçük gürültüleri engelle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Görüntüleri göster
    cv2.imshow("Original", frame)
    cv2.imshow("Red Detection", result)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()
