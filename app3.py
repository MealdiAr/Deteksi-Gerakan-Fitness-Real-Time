from flask import Flask, render_template, Response, request, jsonify
import cv2
import deteksi_pose as mp
import pymysql
import numpy as np
import math
import time
import mediapipe as mp

app = Flask(__name__)

# Inisialisasi MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Global variables untuk tracking akurasi
accuracy_metrics = {
    'total_frames': 0,
    'detected_frames': 0,
    'correct_poses': 0,
    'incorrect_poses': 0,
    'confidence_scores': [],
    'visibility_scores': [],
    'session_start_time': time.time()
}

# Konfigurasi Database
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='gym_pose_detection',
        cursorclass=pymysql.cursors.DictCursor
    )

def save_pose_to_db(pose_name, is_correct, feedback, accuracy_data=None):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            query = """INSERT INTO detected_poses 
                      (pose_name, is_correct, feedback, detection_confidence, 
                       avg_visibility, frame_accuracy, timestamp) 
                      VALUES (%s, %s, %s, %s, %s, %s, NOW())"""
            
            if accuracy_data:
                cursor.execute(query, (
                    pose_name, 
                    is_correct, 
                    feedback,
                    accuracy_data.get('detection_confidence', 0.0),
                    accuracy_data.get('avg_visibility', 0.0),
                    accuracy_data.get('frame_accuracy', 0.0)
                ))
            else:
                cursor.execute(query, (pose_name, is_correct, feedback, 0.0, 0.0, 0.0))
            
            connection.commit()
        connection.close()
    except Exception as e:
        print(f"Error menyimpan ke database: {e}")

def fetch_pose_history():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            query = """SELECT *, 
                      ROUND(detection_confidence * 100, 2) as confidence_percent,
                      ROUND(avg_visibility * 100, 2) as visibility_percent,
                      ROUND(frame_accuracy * 100, 2) as accuracy_percent
                      FROM detected_poses 
                      ORDER BY id DESC LIMIT 50"""
            cursor.execute(query)
            poses = cursor.fetchall()
        connection.close()
        return poses
    except Exception as e:
        print(f"Error mengambil data dari database: {e}")
        return []

def calculate_pose_accuracy(landmarks, results):
    """
    Menghitung akurasi deteksi pose berdasarkan berbagai metrik
    """
    accuracy_data = {
        'detection_confidence': 0.0,
        'avg_visibility': 0.0,
        'landmark_quality': 0.0,
        'frame_accuracy': 0.0
    }
    
    if not landmarks or not results.pose_landmarks:
        return accuracy_data
    
    # 1. Detection Confidence (dari MediaPipe internal)
    # Estimasi berdasarkan kualitas landmark detection
    detection_scores = []
    visibility_scores = []
    
    # Key landmarks untuk evaluasi kualitas deteksi
    key_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE
    ]
    
    for landmark_idx in key_landmarks:
        landmark = landmarks[landmark_idx.value]
        visibility_scores.append(landmark.visibility)
        
        # Hitung kualitas berdasarkan visibility dan posisi
        if hasattr(landmark, 'presence'):
            detection_scores.append(landmark.presence)
        else:
            # Estimasi presence berdasarkan visibility
            detection_scores.append(min(landmark.visibility * 1.2, 1.0))
    
    # 2. Average Visibility Score
    avg_visibility = np.mean(visibility_scores) if visibility_scores else 0.0
    
    # 3. Detection Confidence Score
    avg_detection = np.mean(detection_scores) if detection_scores else 0.0
    
    # 4. Landmark Quality Assessment
    # Periksa konsistensi posisi landmark
    quality_score = 0.0
    if len(visibility_scores) > 0:
        # Hitung berapa banyak landmark yang terdeteksi dengan baik
        good_landmarks = sum(1 for v in visibility_scores if v > 0.5)
        quality_score = good_landmarks / len(visibility_scores)
    
    # 5. Frame Accuracy (kombinasi semua metrik)
    frame_accuracy = (avg_visibility * 0.4 + avg_detection * 0.4 + quality_score * 0.2)
    
    accuracy_data.update({
        'detection_confidence': avg_detection,
        'avg_visibility': avg_visibility,
        'landmark_quality': quality_score,
        'frame_accuracy': frame_accuracy
    })
    
    return accuracy_data

def update_global_accuracy(is_detected, is_correct, accuracy_data):
    """
    Update global accuracy metrics
    """
    global accuracy_metrics
    
    accuracy_metrics['total_frames'] += 1
    
    if is_detected:
        accuracy_metrics['detected_frames'] += 1
        
        if is_correct:
            accuracy_metrics['correct_poses'] += 1
        else:
            accuracy_metrics['incorrect_poses'] += 1
        
        # Store accuracy scores for analysis
        if accuracy_data:
            accuracy_metrics['confidence_scores'].append(accuracy_data['detection_confidence'])
            accuracy_metrics['visibility_scores'].append(accuracy_data['avg_visibility'])
            
            # Keep only last 100 scores to prevent memory issues
            if len(accuracy_metrics['confidence_scores']) > 100:
                accuracy_metrics['confidence_scores'] = accuracy_metrics['confidence_scores'][-100:]
            if len(accuracy_metrics['visibility_scores']) > 100:
                accuracy_metrics['visibility_scores'] = accuracy_metrics['visibility_scores'][-100:]

def get_accuracy_stats():
    """
    Menghitung statistik akurasi real-time
    """
    global accuracy_metrics
    
    total_frames = accuracy_metrics['total_frames']
    detected_frames = accuracy_metrics['detected_frames']
    correct_poses = accuracy_metrics['correct_poses']
    
    stats = {
        'detection_rate': (detected_frames / total_frames * 100) if total_frames > 0 else 0,
        'accuracy_rate': (correct_poses / detected_frames * 100) if detected_frames > 0 else 0,
        'avg_confidence': np.mean(accuracy_metrics['confidence_scores']) * 100 if accuracy_metrics['confidence_scores'] else 0,
        'avg_visibility': np.mean(accuracy_metrics['visibility_scores']) * 100 if accuracy_metrics['visibility_scores'] else 0,
        'total_frames': total_frames,
        'detected_frames': detected_frames,
        'correct_poses': correct_poses,
        'session_duration': time.time() - accuracy_metrics['session_start_time']
    }
    
    return stats

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
        
        # Hitung sudut pinggul
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        
        # Hitung sudut punggung terhadap vertikal
        back_angle = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        
        # PENDEKATAN FINAL YANG SEDERHANA DAN AKURAT
        
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
        insufficient_squat = 60 <= avg_knee_angle < 80
        
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
        
    return is_correct, feedback

def generate_frames(selected_pose=None):
    cap = cv2.VideoCapture("lexxexsis.mp4")  # Gunakan webcam, ganti dengan file video jika perlu
    last_save_time = 0
    save_interval = 3  # Simpan ke database setiap 3 detik

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip untuk tampilan mirror
        frame = cv2.flip(frame, 1)
        
        # Konversi frame ke RGB untuk deteksi pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Variabel untuk feedback dan bounding box
        feedback_text = "Posisi Tidak Terdeteksi"
        feedback_detail = ""
        feedback_color = (0, 0, 255)  # Merah untuk default
        is_detected = False
        is_correct = False
        accuracy_data = None

        if results.pose_landmarks:
            is_detected = True
            landmarks = results.pose_landmarks.landmark
            
            # Hitung akurasi deteksi
            accuracy_data = calculate_pose_accuracy(landmarks, results)
            
            # Klasifikasi gerakan berdasarkan gerakan yang dipilih
            if selected_pose:
                is_correct, feedback = classify_pose(landmarks, selected_pose)
                
                if is_correct:
                    feedback_text = feedback.get("message", "BENAR!")
                    feedback_color = (0, 255, 0)  # Hijau
                else:
                    feedback_text = feedback.get("message", "SALAH!")
                    feedback_detail = feedback.get("detail", "")
                
                # Simpan ke database setiap interval tertentu
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    save_pose_to_db(selected_pose, is_correct, feedback_detail, accuracy_data)
                    last_save_time = current_time

            # Menggambar kerangka pose
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Bounding box sekitar tubuh
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in landmarks]) * w)
            y_min = int(min([lm.y for lm in landmarks]) * h)
            x_max = int(max([lm.x for lm in landmarks]) * w)
            y_max = int(max([lm.y for lm in landmarks]) * h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), feedback_color, 2)
            
            # Tambahkan feedback text di atas frame
            cv2.putText(frame, feedback_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
            
            # Tambahkan detail feedback jika ada
            if feedback_detail:
                cv2.putText(frame, feedback_detail, (x_min, y_min - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)

        # Update global accuracy metrics
        update_global_accuracy(is_detected, is_correct, accuracy_data)
        
        # Tambahkan informasi akurasi real-time di frame
        stats = get_accuracy_stats()
        
        # Tampilkan statistik akurasi di sudut kiri atas
        accuracy_text = [
            f"Detection: {stats['detection_rate']:.1f}%",
            f"Accuracy: {stats['accuracy_rate']:.1f}%",
            f"Confidence: {stats['avg_confidence']:.1f}%",
            f"Visibility: {stats['avg_visibility']:.1f}%"
        ]
        
        for i, text in enumerate(accuracy_text):
            cv2.putText(frame, text, (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Tambahkan informasi gerakan yang sedang dilakukan
        if selected_pose:
            cv2.putText(frame, f"Gerakan: {selected_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

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
    return Response(generate_frames(selected_pose=pose), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/accuracy_stats')
def api_accuracy_stats():
    """
    API endpoint untuk mendapatkan statistik akurasi real-time
    """
    stats = get_accuracy_stats()
    return jsonify(stats)

@app.route('/api/reset_accuracy')
def api_reset_accuracy():
    """
    API endpoint untuk reset statistik akurasi
    """
    global accuracy_metrics
    accuracy_metrics = {
        'total_frames': 0,
        'detected_frames': 0,
        'correct_poses': 0,
        'incorrect_poses': 0,
        'confidence_scores': [],
        'visibility_scores': [],
        'session_start_time': time.time()
    }
    return jsonify({"status": "reset", "message": "Accuracy metrics reset successfully"})

@app.route('/history')
def history():
    poses = fetch_pose_history()
    return render_template('history.html', poses=poses)

@app.route('/accuracy_dashboard')
def accuracy_dashboard():
    """
    Dashboard untuk menampilkan detail statistik akurasi
    """
    stats = get_accuracy_stats()
    recent_poses = fetch_pose_history()
    return render_template('accuracy_dashboard.html', stats=stats, recent_poses=recent_poses)

if __name__ == '__main__':
    app.run(debug=True)