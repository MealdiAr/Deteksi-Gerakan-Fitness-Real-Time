from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pymysql

app = Flask(__name__)

# Inisialisasi MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
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

def save_pose_to_db(pose_name):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            query = "INSERT INTO detected_poses (pose_name) VALUES (%s)"
            cursor.execute(query, (pose_name,))
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
    cap = cv2.VideoCapture("langus.mp4")

    pose_saved = False  # Supaya tidak terus-terusan simpan ke database

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            if not pose_saved and selected_pose is not None:
                save_pose_to_db(selected_pose)
                pose_saved = True  # Sudah simpan, supaya tidak duplikat terus menerus

            # Menggambar kerangka tubuh
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        else:
            pose_saved = False  # Reset kalau tidak terdeteksi orang

        # Encode frame ke JPEG
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