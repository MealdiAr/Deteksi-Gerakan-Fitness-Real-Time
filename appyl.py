from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pymysql
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Inisialisasi MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi model YOLO dengan raw string
yolo_model = YOLO(r'D:\project3\runs\detect\train\weights\best.pt')


# Konfigurasi Database
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='gym_pose_detection',
        cursorclass=pymysql.cursors.DictCursor
    )

def save_pose_to_db(pose_name, status):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            query = "INSERT INTO detected_poses (pose_name, status) VALUES (%s, %s)"
            cursor.execute(query, (pose_name, status))
            connection.commit()
        connection.close()
    except Exception as e:
        print(f"Error menyimpan ke database: {e}")

def fetch_pose_history():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            query = "SELECT * FROM detected_poses ORDER BY id DESC"
            cursor.execute(query)
            poses = cursor.fetchall()
        connection.close()
        return poses
    except Exception as e:
        print(f"Error mengambil data dari database: {e}")
        return []

def generate_frames(selected_pose=None):
    cap = cv2.VideoCapture("squad.mp4")  # Gunakan kamera webcam (0) atau file video
    confidence_threshold = 0.5  # Ambang batas kepercayaan untuk deteksi YOLO

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Langkah 1: Deteksi pose menggunakan YOLO
        yolo_results = yolo_model(frame, conf=confidence_threshold)
        
        # Periksa apakah ada pose terdeteksi
        pose_detected = False
        pose_bbox = None
        yolo_confidence = 0.0
        detected_class_name = ""
        
        # Ambil hasil deteksi YOLO
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if box.conf[0].item() > confidence_threshold:
                        pose_detected = True
                        # Ambil koordinat bounding box (x1, y1, x2, y2)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        pose_bbox = [int(coord) for coord in [x1, y1, x2, y2]]
                        yolo_confidence = box.conf[0].item()
                        
                        # Ambil nama class
                        class_id = int(box.cls[0].item())
                        detected_class_name = yolo_model.names[class_id]
                        break
            
            if pose_detected:
                break

        # Jika pose terdeteksi oleh YOLO, lanjutkan dengan MediaPipe
        if pose_detected:
            # Konversi frame ke RGB untuk deteksi pose MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # VISUALISASI DENGAN POSISI TEKS YANG TERPISAH JAUH
                h, w, c = frame.shape
                
                # Membuat latar belakang semi-transparan untuk teks
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)  # Latar belakang hitam untuk header
                alpha = 0.7  # Transparansi
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                # 1. Menampilkan jenis pose di KIRI ATAS dengan jarak yang cukup
                pose_text = f"Target: {selected_pose}"
                cv2.putText(frame, pose_text, (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 2. Menampilkan pose terdeteksi YOLO di KANAN ATAS dengan jarak yang cukup
                detected_text = f"{detected_class_name} ({yolo_confidence:.2f})"
                text_size = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                # Hitung posisi x agar text ada di sebelah kanan dengan jarak aman
                detected_x = w - text_size[0] - 30
                cv2.putText(frame, detected_text, (detected_x, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 3. Menggambar bounding box hijau dari YOLO dengan label
                box_color = (0, 255, 0)  # Hijau
                cv2.rectangle(frame, (pose_bbox[0], pose_bbox[1]), 
                             (pose_bbox[2], pose_bbox[3]), box_color, 2)
                
                # Menampilkan label di atas bounding box
                label_text = f"{detected_class_name}: {yolo_confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (pose_bbox[0], pose_bbox[1] - label_size[1] - 10), 
                             (pose_bbox[0] + label_size[0], pose_bbox[1]), box_color, -1)
                cv2.putText(frame, label_text, (pose_bbox[0], pose_bbox[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            else:
                # MediaPipe tidak bisa mendeteksi pose meskipun YOLO menemukan pose
                cv2.putText(frame, "MediaPipe pose tidak terdeteksi", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Tetap tampilkan bounding box YOLO
                cv2.rectangle(frame, (pose_bbox[0], pose_bbox[1]), (pose_bbox[2], pose_bbox[3]), (255, 0, 0), 2)
                
                # Label untuk YOLO detection
                label_text = f"{detected_class_name}: {yolo_confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (pose_bbox[0], pose_bbox[1] - label_size[1] - 10), 
                             (pose_bbox[0] + label_size[0], pose_bbox[1]), (255, 0, 0), -1)
                cv2.putText(frame, label_text, (pose_bbox[0], pose_bbox[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # YOLO tidak mendeteksi pose
            cv2.putText(frame, "Tidak ada pose terdeteksi", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame ke format JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    categories = {
        "Otot Tangan": ["Arm Press", "Push up", "plank", "warrior-pose", "crunch", "bicep curl", "pilates", "align"],
        "Otot Punggung": ["cabble frontraise", "cabble row", "deltoid press", "dumble row", "lets pulldown", "t-bar row"],
        "Otot Kaki": ["hai squad", "langus", "leg extention", "leg press", "squad", "standing caflraise", "sumo squad"]
    }
    return render_template('index.html', categories=categories)

@app.route('/category/<string:category>')
def category_page(category):
    poses = {
        "Otot Tangan": ["Arm Press", "Push up", "plank", "warrior-pose", "crunch", "bicep curl", "pilates", "align"],
        "Otot Punggung": ["cabble frontraise", "cabble row", "deltoid press", "dumble row", "lets pulldown", "t-bar row"],
        "Otot Kaki": ["hai squad", "langus", "leg extention", "leg press", "squad", "standing caflraise", "sumo squad"]
    }.get(category, [])
    return render_template('category.html', category=category, poses=poses)

@app.route('/pose/<string:pose>')
def pose_page(pose):
    return render_template('video_feed.html', pose_name=pose)

@app.route('/video_feed/<string:pose>')
def video_feed(pose):
    # Pastikan nama pose diteruskan ke generate_frames
    return Response(generate_frames(selected_pose=pose), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history():
    poses = fetch_pose_history()
    return render_template('history.html', poses=poses)

if __name__ == '__main__':
    app.run(debug=True)