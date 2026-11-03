import cv2
import torch
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_ADDRESS = 'bilalyousafyousaf33@gmail.com'
EMAIL_PASSWORD = ''
ADMIN_EMAIL = 'bilalyousaf950@gmail.com'

def send_email_alert(subject, message):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"Email sent to {ADMIN_EMAIL}: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

SET_CHAIR_COUNT = 15
ALERT_THRESHOLD_MOTION = 5000
MOVEMENT_FRAMES = 10
HEATMAP_DECAY = 0.9

def send_alert(message):
    print(f"[ALERT] {datetime.now()}: {message}")

def calculate_engagement_score(human_count, total_movements, movement_threshold=3000):
    base_score = 50 if human_count > 0 else 0
    activity_score = min(total_movements / movement_threshold, 50)
    return int(base_score + activity_score)

cap = cv2.VideoCapture(0)
chair_count_alerted = False
frame_counter = 0
prev_frame = None
total_movements = 0
heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    height, width = frame.shape[:2]

    results = model(frame)
    detections = results.pandas().xyxy[0]

    chair_count = 0
    human_count = 0
    movement_detected = 0
    mobile_detected = False

    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        color = (0, 255, 0) if label == 'person' else (0, 255, 255) if label == 'chair' else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if label == "chair":
            chair_count += 1
        elif label == "person":
            human_count += 1
        elif label == "mobile":
            mobile_detected = True
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_email_alert(
                "URGENT: Mobile Detected in Class",
                f"A mobile device was detected by the surveillance system at {current_time}.\n\n"
                "Action Required:\nPlease address this issue immediately.\n\n"
                "This message was sent automatically by the classroom surveillance system."
            )

    if mobile_detected:
        send_alert("Mobile detected!")

    total_movements += movement_detected
    heatmap *= HEATMAP_DECAY
    if movement_detected > 0:
        heatmap[y1:y2, x1:x2] += movement_detected

    engagement_score = calculate_engagement_score(human_count, total_movements)

    font_color = (0, 0, 0)
    cv2.putText(frame, f"Engagement Score: {engagement_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color, 2)
    cv2.putText(frame, f"Humans: {human_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, font_color, 2)

    heatmap_display = cv2.applyColorMap(cv2.convertScaleAbs(heatmap, alpha=0.03), cv2.COLORMAP_BONE)
    frame = cv2.addWeighted(frame, 0.9, heatmap_display, 0.1, 0)

    if chair_count != SET_CHAIR_COUNT and not chair_count_alerted:
        send_alert(f"Chair count mismatch! Expected {SET_CHAIR_COUNT}, but found {chair_count}")
        chair_count_alerted = True

    if frame_counter % MOVEMENT_FRAMES == 0:
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_intensity = np.sum(thresh) / 255
            if motion_intensity > ALERT_THRESHOLD_MOTION:
                send_alert("Potential fight detected due to high motion")

        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Classroom Surveillance', cv2.WINDOW_NORMAL)
    cv2.imshow('Classroom Surveillance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
