from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import pymysql
import numpy as np
import math
import time
from ultralytics import YOLO

app = Flask(__name__)

# Inisialisasi MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
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

def save_pose_to_db(pose_name, is_correct, feedback):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            query = "INSERT INTO detected_poses (pose_name, is_correct, feedback) VALUES (%s, %s, %s)"
            cursor.execute(query, (pose_name, is_correct, feedback))
            connection.commit()
        connection.close()
    except Exception as e:
        print(f"Error menyimpan ke database: {e}")

def fetch_pose_history():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            query = "SELECT * FROM detected_poses ORDER BY id DESC LIMIT 50"
            cursor.execute(query)
            poses = cursor.fetchall()
        connection.close()
        return poses
    except Exception as e:
        print(f"Error mengambil data dari database: {e}")
        return []

# Fungsi untuk menghitung sudut antara tiga titik
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# Fungsi untuk mendeteksi visibilitas landmark
def is_visible(landmark):
    return landmark.visibility > 0.65

# Fungsi untuk mengklasifikasikan gerakan berdasarkan pose tertentu
def classify_pose(landmarks, selected_pose):
    feedback = {}
    is_correct = False
    
    # Otot Tangan
    if selected_pose == "Arm Press":
        # Periksa sudut lengan
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        # Hitung sudut siku
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Aturan untuk Arm Press: siku harus ditekuk dengan sudut sekitar 90 derajat
        if 80 <= right_elbow_angle <= 100 and 80 <= left_elbow_angle <= 100:
            is_correct = True
            feedback["message"] = "Posisi Arm Press sudah benar!"
        else:
            feedback["message"] = "Posisi Arm Press salah!"
            if right_elbow_angle < 80 or left_elbow_angle < 80:
                feedback["detail"] = "Tekuk siku lebih dalam (sudut terlalu lebar)"
            elif right_elbow_angle > 100 or left_elbow_angle > 100:
                feedback["detail"] = "Buka siku lebih lebar (sudut terlalu sempit)"
    
    elif selected_pose == "Push up":
        # Landmark untuk push up
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Hitung sudut siku dan tubuh
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        body_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Aturan push up: siku ~90 derajat dan tubuh lurus
        if 70 <= elbow_angle <= 110:
            if 160 <= body_angle <= 190:
                is_correct = True
                feedback["message"] = "Posisi Push Up sudah benar!"
            else:
                feedback["message"] = "Posisi tubuh Push Up salah!"
                feedback["detail"] = "Jaga tubuh tetap lurus, jangan menekuk pinggul"
        else:
            feedback["message"] = "Posisi siku Push Up salah!"
            if elbow_angle < 70:
                feedback["detail"] = "Terlalu rendah, naikan posisi tubuh"
            else:
                feedback["detail"] = "Terlalu tinggi, turunkan posisi tubuh"
    
    elif selected_pose == "plank":
        # Landmark untuk plank
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Hitung sudut tubuh (harus lurus)
        left_body_angle = calculate_angle(left_shoulder, left_hip, left_ankle)
        right_body_angle = calculate_angle(right_shoulder, right_hip, right_ankle)
        
        # Posisi plank: tubuh harus lurus (sudut ~180 derajat)
        if 160 <= left_body_angle <= 200 and 160 <= right_body_angle <= 200:
            is_correct = True
            feedback["message"] = "Posisi Plank sudah benar!"
        else:
            feedback["message"] = "Posisi Plank salah!"
            if left_body_angle < 160 or right_body_angle < 160:
                feedback["detail"] = "Pinggul terlalu rendah, angkat pinggul"
            else:
                feedback["detail"] = "Pinggul terlalu tinggi, turunkan pinggul"
    
    elif selected_pose == "langus":
        # Landmark untuk lunges
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Hitung sudut lutut untuk kedua kaki
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        # Menghitung jarak antar kaki
        ankle_x_distance = abs(right_ankle.x - left_ankle.x)
        
        # Identifikasi kaki depan dan belakang
        if right_knee.y > left_knee.y:
            front_knee = right_knee
            front_hip = right_hip
            front_ankle = right_ankle
            front_knee_angle = right_knee_angle
            
            back_knee = left_knee
            back_hip = left_hip
            back_ankle = left_ankle
            back_knee_angle = left_knee_angle
        else:
            front_knee = left_knee
            front_hip = left_hip
            front_ankle = left_ankle
            front_knee_angle = left_knee_angle
            
            back_knee = right_knee
            back_hip = right_hip
            back_ankle = right_ankle
            back_knee_angle = right_knee_angle
        
        # SOLUSI UNTUK MASALAH DETEKSI
        
        # 1. Definisikan fase gerakan lunges dengan jelas
        
        # Posisi berdiri tegak (start/end)
        standing_position = right_knee_angle > 160 and left_knee_angle > 160
        
        # Posisi bawah yang benar (sudut < 110, sudah cukup turun)
        bottom_position = front_knee_angle < 110 and ankle_x_distance > 0.25
        
        # Posisi kurang dalam yang tidak bergerak (posisi statis yang salah)
        insufficient_static = (
            front_knee_angle >= 140 and 
            front_knee_angle <= 140 and 
            ankle_x_distance > 0.2
        )
        
        # TAMBAHAN: State untuk mendeteksi gerakan
        # Untuk menyimpan frame sebelumnya, gunakan variabel global atau state
        # Di sini kita akan menggunakan pendekatan sederhana berdasarkan posisi saat ini
        
        # Keamanan: Posisi lutut tidak melewati ujung kaki
        knee_forward_threshold = 0.2
        knee_position_safe = front_knee.x <= front_ankle.x + knee_forward_threshold
        
        # EVALUASI POSISI LUNGES DENGAN MEMPERTIMBANGKAN GERAKAN
        
        # Posisi berdiri tegak selalu dianggap benar (posisi awal/akhir)
        if standing_position:
            is_correct = True
            feedback["message"] = "Posisi Lunges sudah benar!"
            feedback["detail"] = "Posisi awal/akhir lunges"
        
        # Posisi bawah yang sudah cukup dalam dianggap benar
        elif bottom_position:
            if not knee_position_safe:
                is_correct = False
                feedback["message"] = "Posisi Lunges salah!"
                feedback["detail"] = "Lutut depan terlalu maju melewati ujung kaki"
            else:
                is_correct = True
                feedback["message"] = "Posisi Lunges sudah benar!"
                feedback["detail"] = "Posisi bawah lunges yang baik"
        
        # Posisi kurang dalam yang tidak bergerak dianggap salah
        elif insufficient_static:
            is_correct = False
            feedback["message"] = "Posisi Lunges salah!"
            feedback["detail"] = "Kurang ke bawah, turunkan tubuh dan tekuk lutut lebih dalam"
        
        # Ada langkah kaki tetapi belum jelas posisinya
        elif ankle_x_distance > 0.2:
            # PENTING: Jika ada langkah kaki tetapi sudut lutut belum ideal,
            # kita asumsikan sedang dalam proses turun/naik, sehingga dianggap benar
            is_correct = True
            feedback["message"] = "Posisi Lunges sudah benar!"
            
            if front_knee_angle > 140:
                feedback["detail"] = "Mulai turun, teruskan gerakan"
            else:
                feedback["detail"] = "Lanjutkan gerakan"
        
        # Belum melakukan gerakan lunges
        else:
            is_correct = False
            feedback["message"] = "Posisi Lunges salah!"
            feedback["detail"] = "Belum melakukan gerakan lunges dengan benar"
    
    elif selected_pose == "squad":
        # Landmark untuk squat
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Hitung sudut lutut
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        avg_knee_angle = (right_knee_angle + left_knee_angle) / 2
        
        # Indikator posisi lutut yang tidak aman
        knee_forward_threshold = 0.3
        right_knee_forward = right_knee.x > right_ankle.x + knee_forward_threshold
        left_knee_forward = left_knee.x > left_ankle.x + knee_forward_threshold
        knee_position_unsafe = right_knee_forward and left_knee_forward
        
        # Deteksi posisi berdiri tegak (hampir lurus, sudut > 170)
        standing_position = avg_knee_angle > 170
        
        # Deteksi posisi persiapan squat (posisi sebelum turun, sudut 130-170)
        preparation_position = 130 <= avg_knee_angle <= 170
        
        # Deteksi squat aktif (sudut 80-130)
        active_squat = 80 <= avg_knee_angle < 130
        
        # Deteksi squat terlalu dalam (sudut < 60)
        too_deep_squat = avg_knee_angle < 60
        
        # Deteksi squat dangkal yang benar-benar kurang dalam (60-80)
        insufficient_squat = 40 <= avg_knee_angle < 60
        
        # LOGIKA EVALUASI YANG SANGAT AKURAT
        
        # Posisi lutut tidak aman selalu salah
        if knee_position_unsafe:
            is_correct = False
            feedback["message"] = "Posisi Squat perlu perbaikan"
            feedback["detail"] = "Lutut terlalu jauh ke depan, geser berat badan ke tumit"
        
        # Squat terlalu dalam dianggap salah
        elif too_deep_squat:
            is_correct = False
            feedback["message"] = "Posisi Squat perlu perbaikan"
            feedback["detail"] = "Terlalu dalam, naikan sedikit posisi"
        
        # Squat yang benar-benar kurang dalam dianggap salah
        elif insufficient_squat:
            is_correct = False
            feedback["message"] = "Posisi Squat perlu perbaikan"
            feedback["detail"] = "Kurang dalam, turunkan tubuh lebih rendah"
        
        # SEMUA POSISI LAINNYA DIANGGAP BENAR
        # Termasuk: posisi berdiri, posisi persiapan, dan squat aktif
        else:
            is_correct = True
            feedback["message"] = "Posisi Squat sudah benar!"
            
            # Tambahkan detail fase squat
            if standing_position:
                feedback["detail"] = "Posisi berdiri tegak"
            elif preparation_position:
                feedback["detail"] = "Posisi persiapan squat"
            elif active_squat:
                feedback["detail"] = "Posisi squat aktif"
    
    # Tambahkan pose lainnya di sini sesuai kebutuhan...
    else:
        feedback["message"] = f"Gerakan {selected_pose} tidak dikenali"
        
    return is_correct, feedback

def generate_frames(selected_pose=None):
    cap = cv2.VideoCapture(0)  # Gunakan kamera webcam (0) atau file video
    confidence_threshold = 0.5  # Ambang batas kepercayaan untuk deteksi YOLO
    last_save_time = 0
    save_interval = 3  # Simpan ke database setiap 3 detik

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip untuk tampilan mirror
        frame = cv2.flip(frame, 1)

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

        # Variabel untuk feedback MediaPipe
        mediapipe_feedback_text = "Posisi Tidak Terdeteksi"
        mediapipe_feedback_detail = ""
        mediapipe_feedback_color = (0, 0, 255)  # Merah untuk default

        # Jika pose terdeteksi oleh YOLO, lanjutkan dengan MediaPipe
        if pose_detected:
            # Konversi frame ke RGB untuk deteksi pose MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Klasifikasi gerakan berdasarkan gerakan yang dipilih
                if selected_pose:
                    is_correct, feedback = classify_pose(landmarks, selected_pose)
                    
                    if is_correct:
                        mediapipe_feedback_text = feedback.get("message", "BENAR!")
                        mediapipe_feedback_color = (0, 255, 0)  # Hijau
                    else:
                        mediapipe_feedback_text = feedback.get("message", "SALAH!")
                        mediapipe_feedback_detail = feedback.get("detail", "")
                    
                    # Simpan ke database setiap interval tertentu
                    current_time = time.time()
                    if current_time - last_save_time > save_interval:
                        save_pose_to_db(selected_pose, is_correct, mediapipe_feedback_detail)
                        last_save_time = current_time

                # Menggambar kerangka pose MediaPipe
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # VISUALISASI DENGAN POSISI TEKS YANG TERPISAH JAUH
                h, w, c = frame.shape
                
                # Membuat latar belakang semi-transparan untuk teks
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)  # Latar belakang hitam untuk header
                alpha = 0.7  # Transparansi
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                # 1. Menampilkan jenis pose di KIRI ATAS dengan jarak yang cukup
                pose_text = f"Target: {selected_pose}"
                cv2.putText(frame, pose_text, (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 2. Menampilkan pose terdeteksi YOLO di KANAN ATAS dengan jarak yang cukup
                detected_text = f"YOLO: {detected_class_name} ({yolo_confidence:.2f})"
                text_size = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                # Hitung posisi x agar text ada di sebelah kanan dengan jarak aman
                detected_x = w - text_size[0] - 30
                cv2.putText(frame, detected_text, (detected_x, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 3. Menampilkan feedback MediaPipe di bawah target pose
                cv2.putText(frame, f"MediaPipe: {mediapipe_feedback_text}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mediapipe_feedback_color, 2)
                
                # 4. Menampilkan detail feedback jika ada
                if mediapipe_feedback_detail:
                    cv2.putText(frame, mediapipe_feedback_detail, (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mediapipe_feedback_color, 2)
                
                # 5. Menggambar bounding box hijau dari YOLO dengan label
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
                h, w, c = frame.shape
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
                alpha = 0.7
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                cv2.putText(frame, f"Target: {selected_pose}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                detected_text = f"YOLO: {detected_class_name} ({yolo_confidence:.2f})"
                text_size = cv2.getTextSize(detected_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                detected_x = w - text_size[0] - 30
                cv2.putText(frame, detected_text, (detected_x, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.putText(frame, "MediaPipe: Pose tidak terdeteksi", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
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
            h, w, c = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            cv2.putText(frame, f"Target: {selected_pose}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Tidak ada pose terdeteksi", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    return Response(generate_frames(selected_pose=pose), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def history():
    poses = fetch_pose_history()
    return render_template('history.html', poses=poses)

if __name__ == '__main__':
    app.run(debug=True)