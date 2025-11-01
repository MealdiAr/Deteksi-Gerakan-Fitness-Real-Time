from flask import Flask, render_template, Response, request
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
    
    elif selected_pose == "warrior-pose":
        # Landmark untuk warrior pose
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Hitung sudut lutut (satu kaki harus ditekuk ~90 derajat, yang lain lurus)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Warrior pose: satu lutut tekuk, satu lutut lurus
        if (80 <= left_knee_angle <= 110 and right_knee_angle >= 160) or \
           (80 <= right_knee_angle <= 110 and left_knee_angle >= 160):
            is_correct = True
            feedback["message"] = "Posisi Warrior Pose sudah benar!"
        else:
            feedback["message"] = "Posisi Warrior Pose salah!"
            feedback["detail"] = "Pastikan satu kaki ditekuk (~90°) dan kaki lain lurus"
    
    elif selected_pose == "crunch":
        # Landmark untuk crunch
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        
        # Hitung sudut tubuh bagian atas dan lutut
        upper_body_angle = calculate_angle(nose, left_shoulder, left_hip)
        knee_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        
        # Crunch: tubuh atas terangkat, lutut ditekuk
        if upper_body_angle < 160 and knee_angle < 130:
            is_correct = True
            feedback["message"] = "Posisi Crunch sudah benar!"
        else:
            feedback["message"] = "Posisi Crunch salah!"
            if upper_body_angle >= 160:
                feedback["detail"] = "Angkat tubuh atas lebih tinggi"
            if knee_angle >= 130:
                feedback["detail"] = "Tekuk lutut lebih dalam"
    
    elif selected_pose == "bicep curl":
        # Landmark untuk bicep curl
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        # Hitung sudut siku
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Bicep curl: siku ditekuk (<90 derajat)
        if right_elbow_angle < 90 or left_elbow_angle < 90:
            is_correct = True
            feedback["message"] = "Posisi Bicep Curl sudah benar!"
        else:
            feedback["message"] = "Posisi Bicep Curl salah!"
            feedback["detail"] = "Tekuk siku lebih dalam saat mengangkat beban"
    
    elif selected_pose == "pilates":
        # Landmark untuk pilates
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Hitung sudut tubuh dan kaki
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        # Pilates: tubuh tegak, kaki lurus
        if 160 <= hip_angle <= 200 and 160 <= knee_angle <= 200:
            is_correct = True
            feedback["message"] = "Posisi Pilates sudah benar!"
        else:
            feedback["message"] = "Posisi Pilates salah!"
            if hip_angle < 160:
                feedback["detail"] = "Tegakkan punggung, jangan menekuk"
            if knee_angle < 160:
                feedback["detail"] = "Luruskan kaki lebih baik"
    
    elif selected_pose == "align":
        # Landmark untuk align
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Hitung alignment tubuh
        shoulder_hip_x = abs(left_shoulder.x - left_hip.x)
        hip_ankle_x = abs(left_hip.x - left_ankle.x)
        
        # Alignment: tubuh lurus (koordinat x harus hampir sama)
        if shoulder_hip_x < 0.1 and hip_ankle_x < 0.1:
            is_correct = True
            feedback["message"] = "Posisi Align sudah benar!"
        else:
            feedback["message"] = "Posisi Align salah!"
            feedback["detail"] = "Sejajarkan bahu, pinggul dan pergelangan kaki"
    
    # Otot Punggung
    elif selected_pose == "cabble frontraise":
        # Landmark untuk cabble frontraise
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Hitung sudut lengan
        arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Posisi wrist harus lebih tinggi dari shoulder
        if right_wrist.y < right_shoulder.y and 160 <= arm_angle <= 200:
            is_correct = True
            feedback["message"] = "Posisi Cabble Frontraise sudah benar!"
        else:
            feedback["message"] = "Posisi Cabble Frontraise salah!"
            if right_wrist.y >= right_shoulder.y:
                feedback["detail"] = "Angkat lengan lebih tinggi dari bahu"
            if arm_angle < 160 or arm_angle > 200:
                feedback["detail"] = "Pertahankan lengan tetap lurus"
    
    elif selected_pose == "cabble row":
        # Landmark untuk cabble row
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Hitung sudut siku dan posisi
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        shoulder_hip_elbow_angle = calculate_angle(right_shoulder, right_hip, right_elbow)
        
        # Cabble row: siku ditekuk, torso maju
        if elbow_angle < 110 and shoulder_hip_elbow_angle < 120:
            is_correct = True
            feedback["message"] = "Posisi Cabble Row sudah benar!"
        else:
            feedback["message"] = "Posisi Cabble Row salah!"
            if elbow_angle >= 110:
                feedback["detail"] = "Tarik siku lebih ke belakang"
            if shoulder_hip_elbow_angle >= 120:
                feedback["detail"] = "Condongkan tubuh sedikit ke depan"
    
    elif selected_pose == "deltoid press":
        # Landmark untuk deltoid press
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Hitung sudut siku
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Deltoid press: lengan lurus ke atas
        if 160 <= elbow_angle <= 200 and right_wrist.y < right_elbow.y:
            is_correct = True
            feedback["message"] = "Posisi Deltoid Press sudah benar!"
        else:
            feedback["message"] = "Posisi Deltoid Press salah!"
            if elbow_angle < 160:
                feedback["detail"] = "Luruskan lengan lebih baik"
            if right_wrist.y >= right_elbow.y:
                feedback["detail"] = "Angkat lengan lebih tinggi"
    
    elif selected_pose == "dumble row":
        # Landmark untuk dumble row
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        
        # Hitung sudut siku dan posisi tubuh
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        body_angle = calculate_angle(left_shoulder, left_hip, left_elbow)
        
        # Dumble row: siku ditekuk, tubuh maju
        if elbow_angle < 100 and body_angle < 120:
            is_correct = True
            feedback["message"] = "Posisi Dumble Row sudah benar!"
        else:
            feedback["message"] = "Posisi Dumble Row salah!"
            if elbow_angle >= 100:
                feedback["detail"] = "Tekuk siku lebih dalam saat menarik"
            if body_angle >= 120:
                feedback["detail"] = "Condongkan tubuh lebih maju"
    
    elif selected_pose == "lets pulldown":
        # Landmark untuk lat pulldown
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        # Hitung sudut siku
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # Lat pulldown: siku ditekuk, tangan di sekitar bahu
        if right_elbow_angle < 120 and left_elbow_angle < 120:
            is_correct = True
            feedback["message"] = "Posisi Lat Pulldown sudah benar!"
        else:
            feedback["message"] = "Posisi Lat Pulldown salah!"
            feedback["detail"] = "Tarik bar ke bawah hingga siku menekuk lebih dalam"
    
    elif selected_pose == "t-bar row":
        # Landmark untuk t-bar row
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Hitung sudut siku dan posisi tubuh
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        body_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        
        # T-bar row: siku ditekuk, tubuh condong ke depan
        if elbow_angle < 110 and body_angle < 150:
            is_correct = True
            feedback["message"] = "Posisi T-Bar Row sudah benar!"
        else:
            feedback["message"] = "Posisi T-Bar Row salah!"
            if elbow_angle >= 110:
                feedback["detail"] = "Tarik beban lebih ke atas"
            if body_angle >= 150:
                feedback["detail"] = "Condongkan tubuh lebih ke depan"
    
    # Otot Kaki
    elif selected_pose == "hai squad":
        # Landmark untuk high squat
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Hitung sudut lutut
        knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # High squat: lutut ditekuk tapi tidak terlalu dalam (120-150 derajat)
        if 120 <= knee_angle <= 150:
            is_correct = True
            feedback["message"] = "Posisi High Squat sudah benar!"
        else:
            feedback["message"] = "Posisi High Squat salah!"
            if knee_angle < 120:
                feedback["detail"] = "Terlalu dalam, angkat sedikit tubuh"
            else:
                feedback["detail"] = "Terlalu tinggi, turunkan tubuh lebih dalam"
    
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
    
    elif selected_pose == "leg press":
        # Landmark untuk leg press
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Hitung sudut lutut
        knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        # Leg press: lutut ditekuk (90-120 derajat)
        if 90 <= knee_angle <= 120:
            is_correct = True
            feedback["message"] = "Posisi Leg Press sudah benar!"
        else:
            feedback["message"] = "Posisi Leg Press salah!"
            if knee_angle < 90:
                feedback["detail"] = "Terlalu dalam, bisa melukai lutut"
            else:
                feedback["detail"] = "Tekuk lutut lebih dalam"
    
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
        
    elif selected_pose == "sumo squad":
        # Landmark untuk sumo squat
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        # Hitung sudut lutut dan jarak antara kaki
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        
        ankle_distance = math.sqrt((right_ankle.x - left_ankle.x)**2 + (right_ankle.y - left_ankle.y)**2)
        
        # Sumo squat: lutut ditekuk dan kaki terbuka lebar
        if 90 <= right_knee_angle <= 110 and 90 <= left_knee_angle <= 110 and ankle_distance > 0.3:
            is_correct = True
            feedback["message"] = "Posisi Sumo Squat sudah benar!"
        else:
            feedback["message"] = "Posisi Sumo Squat salah!"
            if right_knee_angle < 90 or left_knee_angle < 90:
                feedback["detail"] = "Terlalu dalam, naikan posisi tubuh"
            elif right_knee_angle > 110 or left_knee_angle > 110:
                feedback["detail"] = "Tekuk lutut lebih dalam"
            elif ankle_distance <= 0.3:
                feedback["detail"] = "Buka kaki lebih lebar untuk posisi sumo yang benar"
    else:
        feedback["message"] = f"Gerakan {selected_pose} tidak dikenali"
        
    return is_correct, feedback

def generate_frames(selected_pose=None):
    cap = cv2.VideoCapture(0)
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

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
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
                    save_pose_to_db(selected_pose, is_correct, feedback_detail)
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

@app.route('/history')
def history():
    poses = fetch_pose_history()
    return render_template('history.html', poses=poses)

if __name__ == '__main__':
    app.run(debug=True)