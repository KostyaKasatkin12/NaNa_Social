import eventlet

eventlet.monkey_patch()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import re
from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify, Response, \
    send_from_directory
from flask_socketio import SocketIO, emit, join_room
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
from flask_wtf.csrf import CSRFProtect
import sqlite3
import cv2
from fer import FER
import mediapipe as mp
import numpy as np
import base64
import time
import json
import logging
import pymorphy3 as pymorphy
import google.generativeai as genai
from langdetect import detect, DetectorFactory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import speech_recognition as sr
from io import BytesIO
import wave
from collections import defaultdict
from threading import Lock

# Initialize Flask and SocketIO
app = Flask(__name__)
app.secret_key = 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5'  # Replace with a secure key
app.config['UPLOAD_FOLDER'] = 'static/avatars'
app.config['STORIES_FOLDER'] = 'static/stories'
app.config['AUDIO_FOLDER'] = 'static/audio'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
csrf = CSRFProtect(app)

# Configure directories
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
STORIES_FOLDER = app.config['STORIES_FOLDER']
AUDIO_FOLDER = app.config['AUDIO_FOLDER']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi', 'wav'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STORIES_FOLDER):
    os.makedirs(STORIES_FOLDER)
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Initialize emotion and hand detection
emotion_detector = FER(mtcnn=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Initialize speech recognition
recognizer = sr.Recognizer()

# Глобальные переменные для отслеживания онлайн-статуса
online_users = set()
online_lock = Lock()

# Define gestures based on finger states (0 = down, 1 = up)
GESTURES = {
    (0, 0, 0, 0, 0): "FIST",
    (1, 1, 1, 1, 1): "OPEN_HAND",
    (0, 1, 1, 0, 0): "VICTORY",
    (0, 1, 0, 0, 1): "ROCK",
    (1, 0, 0, 0, 0): "THUMBS_UP",
    (0, 0, 0, 0, 1): "THUMBS_DOWN",
    (0, 1, 0, 0, 0): "POINTING",
    (0, 0, 0, 1, 0): "POINTING",
    (0, 0, 1, 0, 0): "MIDDLE_FINGER",
    (1, 1, 0, 0, 1): "SPIDERMAN"
}

# Define finger colors for drawing
finger_colors = {
    "thumb": (255, 0, 0),
    "index": (0, 255, 0),
    "middle": (0, 0, 255),
    "ring": (0, 255, 255),
    "pinky": (255, 255, 0)
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure consistent language detection results
DetectorFactory.seed = 0

# Initialize language tools
morph = pymorphy.MorphAnalyzer()

# Configure Gemini AI
genai.configure(api_key="AIzaSyBNR9ULDDEAJ2iW_0b6GgT9lfSOqs-dwMw")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired()])
    description = StringField('Description')
    city = StringField('City')
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], default='Male')
    interests = StringField('Interests')
    submit = SubmitField('Register')


class StoryForm(FlaskForm):
    content = StringField('Content', validators=[DataRequired()])
    image = FileField('Image', validators=[DataRequired()])
    submit = SubmitField('Add Story')


class AddFriendForm(FlaskForm):
    submit = SubmitField('Добавить в друзья')


# Database initialization
def init_db():
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    # Create tables
    tables = [
        '''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            description TEXT,
            relationship_status TEXT DEFAULT 'не интересуюсь',
            avatar TEXT,
            city TEXT,
            gender TEXT,
            interests TEXT
        )''',
        '''CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image TEXT,
            emotion TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS friends (
            user_id INTEGER NOT NULL,
            friend_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, friend_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (friend_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user1_id INTEGER NOT NULL,
            user2_id INTEGER NOT NULL,
            FOREIGN KEY (user1_id) REFERENCES users(id),
            FOREIGN KEY (user2_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            sender_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_read INTEGER DEFAULT 0,
            FOREIGN KEY (chat_id) REFERENCES chats(id),
            FOREIGN KEY (sender_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''',
        '''CREATE TABLE IF NOT EXISTS post_reactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            reaction TEXT NOT NULL,
            UNIQUE(post_id, user_id),
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS post_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT,
            image TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            views INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS speech_recognition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            recognized_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )'''
    ]

    for table in tables:
        cursor.execute(table)

    conn.commit()
    conn.close()


def process_audio(audio_data, sample_rate=16000):
    """
    Обработка аудио данных и распознавание речи
    """
    try:
        # Создаем временный WAV файл в памяти
        audio_buffer = BytesIO()

        # Создаем WAV файл
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # моно
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        # Перемещаем указатель в начало буфера
        audio_buffer.seek(0)

        # Используем speech_recognition для распознавания
        with sr.AudioFile(audio_buffer) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

            # Пытаемся распознать речь на русском языке
            text = recognizer.recognize_google(audio, language='ru-RU')
            return text, True

    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return "Речь не распознана", False
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service: {e}")
        return f"Ошибка сервиса распознавания: {e}", False
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return f"Ошибка обработки аудио: {e}", False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_gesture(landmarks):
    fingers = []
    is_right_hand = landmarks[17].x < landmarks[5].x
    fingers.append(1 if (landmarks[4].x < landmarks[3].x if is_right_hand else landmarks[4].x > landmarks[3].x) else 0)
    for tip_id in [8, 12, 16, 20]:
        fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)
    fingers_tuple = tuple(fingers)
    if sum(fingers) >= 4:
        return "OPEN_HAND"
    return GESTURES.get(fingers_tuple, f"UNKNOWN ({sum(fingers)} fingers)")


def process_emotions(frame):
    try:
        if frame is None or frame.size == 0:
            logger.error("[Emotion] Input frame is invalid")
            return None, 0.0, (0, 0, 0, 0)
        logger.info(f"[Emotion] Processing frame of shape: {frame.shape}")
        resized_frame = cv2.resize(frame, (64, 64))
        emotions = emotion_detector.detect_emotions(resized_frame)
        if emotions:
            (x, y, w, h) = emotions[0]["box"]
            emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
            scale_x = frame.shape[1] / 64
            scale_y = frame.shape[0] / 64
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            logger.info(f"[Emotion] Detected: {emotion} ({score:.2f}) at box ({x}, {y}, {w}, {h})")
            return emotion, score, (x, y, w, h)
        else:
            logger.info("[Emotion] No faces detected")
    except Exception as e:
        logger.error(f"[Emotion] Error: {e}")
    return None, 0.0, (0, 0, 0, 0)


def process_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_finger_tips(frame, hand_landmarks.landmark, frame.shape[1], frame.shape[0])
            gesture = detect_gesture(hand_landmarks.landmark)
    return gesture


def draw_finger_tips(frame, landmarks, image_width, image_height):
    for i, finger in enumerate([4, 8, 12, 16, 20]):
        x = int(landmarks[finger].x * image_width)
        y = int(landmarks[finger].y * image_height)
        finger_name = ["thumb", "index", "middle", "ring", "pinky"][i]
        color = finger_colors[finger_name]
        cv2.circle(frame, (x, y), 10, color, -1)


def send_notifications_real_time(user_id, notification_content=None):
    """
    Улучшенная функция отправки уведомлений в реальном времени
    """
    try:
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()

        # Если передано конкретное уведомление, сохраняем его
        if notification_content:
            cursor.execute(
                "INSERT INTO notifications (user_id, content) VALUES (?, ?)",
                (user_id, notification_content)
            )
            conn.commit()

        # Получаем все актуальные уведомления
        cursor.execute("""
            SELECT 
                n.content, 
                n.created_at,
                CASE 
                    WHEN n.content LIKE '%sent you%message%' THEN 'message'
                    WHEN n.content LIKE '%friend request%' THEN 'friend_request'
                    WHEN n.content LIKE '%liked your post%' THEN 'like'
                    WHEN n.content LIKE '%commented on your post%' THEN 'comment'
                    WHEN n.content LIKE '%accepted your friend request%' THEN 'friend_accept'
                    ELSE 'system'
                END as type
            FROM notifications n
            WHERE n.user_id = ? 
            ORDER BY n.created_at DESC 
            LIMIT 20
        """, (user_id,))

        notifications = cursor.fetchall()

        # Получаем количество непрочитанных сообщений
        cursor.execute("""
            SELECT COUNT(*) as unread_messages
            FROM chat_messages cm
            JOIN chats c ON cm.chat_id = c.id
            WHERE (c.user1_id = ? OR c.user2_id = ?)
            AND cm.sender_id != ?
            AND cm.is_read = 0
        """, (user_id, user_id, user_id))

        unread_messages = cursor.fetchone()[0] or 0

        # Получаем количество заявок в друзья
        cursor.execute("""
            SELECT COUNT(*) as pending_requests
            FROM friends f
            WHERE f.friend_id = ? 
            AND f.status = 'pending'
        """, (user_id,))

        pending_requests = cursor.fetchone()[0] or 0

        conn.close()

        # Подготавливаем данные для отправки
        notification_data = {
            'user_id': user_id,
            'notifications': [
                {
                    'content': n[0],
                    'created_at': n[1],
                    'type': n[2]
                } for n in notifications
            ],
            'unread_count': len(notifications),
            'unread_messages': unread_messages,
            'pending_requests': pending_requests,
            'timestamp': datetime.now().isoformat()
        }

        # Проверяем, онлайн ли пользователь
        with online_lock:
            is_online = user_id in online_users

        # Отправляем только если пользователь онлайн
        if is_online:
            logger.info(f"Sending real-time notifications to user_id {user_id}")
            socketio.emit('update_notifications', notification_data, room=str(user_id))

        return notification_data

    except Exception as e:
        logger.error(f"Error in send_notifications_real_time: {e}")
        return None


# Routes
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')


@app.route('/speech_history')
def speech_history():
    """
    Страница с историей распознанной речи
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT recognized_text, created_at 
        FROM speech_recognition 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 50
    ''', (user_id,))

    speech_history = cursor.fetchall()
    conn.close()

    return render_template('speech_history.html', speech_history=speech_history)


@app.route('/', methods=['GET'])
def home():
    if 'user_id' not in session:
        logger.warning("No user_id in session, redirecting to login")
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    if not user:
        logger.error(f"No user found for user_id: {user_id}, clearing session")
        session.pop('user_id', None)
        return redirect(url_for('login'))
    cursor.execute("""
        SELECT posts.id, posts.content, posts.created_at, users.username, posts.image,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = posts.id AND reaction = 'like') AS likes,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = posts.id AND reaction = 'dislike') AS dislikes,
               (SELECT reaction FROM post_reactions WHERE post_id = posts.id AND user_id = ?) AS user_reaction,
               (SELECT json_group_array(json_array(c.content, c.created_at, u2.username))
                FROM post_comments c
                JOIN users u2 ON c.user_id = u2.id
                WHERE c.post_id = posts.id
                ORDER BY c.created_at DESC
                LIMIT 1) AS latest_comment,
               posts.emotion
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        ORDER BY posts.created_at DESC
    """, (user_id,))
    posts_raw = cursor.fetchall()
    posts = []
    for post in posts_raw:
        latest_comment = json.loads(post[8])[0] if post[8] and json.loads(post[8]) else None
        posts.append(post[:8] + (latest_comment, post[9]))
    cursor.execute("""
        SELECT users.username, users.id FROM friends 
        JOIN users ON friends.friend_id = users.id
        WHERE friends.user_id = ? AND friends.status = 'accepted'
    """, (user_id,))
    friends = cursor.fetchall()
    cursor.execute("""
        SELECT users.id, users.username FROM friends 
        JOIN users ON friends.user_id = users.id
        WHERE friends.friend_id = ? AND friends.status = 'pending'
    """, (user_id,))
    friend_requests = cursor.fetchall()
    cursor.execute("""
        SELECT chats.id, users.username,
               (SELECT COUNT(*) FROM chat_messages 
                WHERE chat_messages.chat_id = chats.id 
                AND chat_messages.sender_id != ? 
                AND chat_messages.is_read = 0) AS unread_count
        FROM chats 
        JOIN users ON (chats.user1_id = users.id OR chats.user2_id = users.id)
        WHERE (chats.user1_id = ? OR chats.user2_id = ?) AND users.id != ?
    """, (user_id, user_id, user_id, user_id))
    chats = cursor.fetchall()
    cursor.execute("""
        SELECT users.username, 
               (SELECT COUNT(*) FROM chat_messages 
                WHERE chat_messages.chat_id = chats.id 
                AND chat_messages.sender_id != ? 
                AND chat_messages.is_read = 0) AS unread_count
        FROM chats 
        JOIN users ON (chats.user1_id = users.id OR chats.user2_id = users.id)
        WHERE (chats.user1_id = ? OR chats.user2_id = ?) 
        AND users.id != ?
        AND (SELECT COUNT(*) FROM chat_messages 
             WHERE chat_messages.chat_id = chats.id 
             AND chat_messages.sender_id != ? 
             AND chat_messages.is_read = 0) > 0
    """, (user_id, user_id, user_id, user_id, user_id))
    unread_notifications = cursor.fetchall()
    notifications = [(f"{username} sent you {unread_count} message(s)", None) for username, unread_count in
                     unread_notifications]
    cursor.execute("SELECT content, created_at FROM notifications WHERE user_id = ? ORDER BY created_at DESC",
                   (user_id,))
    notifications.extend(cursor.fetchall())

    # Fetch active stories only from friends (valid for 24 hours)
    cursor.execute("""
        SELECT s.id, s.user_id, s.content, s.image, s.created_at, s.expires_at, s.views 
        FROM stories s
        JOIN friends f ON s.user_id = f.friend_id
        WHERE f.user_id = ? AND f.status = 'accepted' AND s.expires_at > ?
    """, (user_id, datetime.now()))
    stories = cursor.fetchall()
    conn.close()
    send_notifications_real_time(user_id)
    search_form = AddFriendForm()
    form = AddFriendForm()
    return render_template('home.html',
                           username=user[0],
                           posts=posts,
                           friends=friends,
                           friend_requests=friend_requests,
                           notifications=notifications,
                           chats=chats,
                           search_form=search_form,
                           form=form,
                           stories=stories)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        logger.info(f"Login attempt for username: {username}, user: {user}")
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            logger.info(f"User logged in: {username}, user_id: {user[0]}")
            return redirect(url_for('home'))
        logger.warning(f"Invalid login attempt for username: {username}, user: {user}")
        return render_template('login.html', form=form, error="Invalid username or password")
    return render_template('login.html', form=form)


@app.route('/get_more_posts', methods=['GET'])
def get_more_posts():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    offset = request.args.get('offset', type=int, default=5)
    limit = 5
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT posts.id, posts.content, posts.created_at, users.username, posts.image,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = posts.id AND reaction = 'like') AS likes,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = posts.id AND reaction = 'dislike') AS dislikes,
               (SELECT reaction FROM post_reactions WHERE post_id = posts.id AND user_id = ?) AS user_reaction,
               (SELECT json_group_array(json_array(c.content, c.created_at, u2.username))
                FROM post_comments c
                JOIN users u2 ON c.user_id = u2.id
                WHERE c.post_id = posts.id
                ORDER BY c.created_at DESC
                LIMIT 1) AS latest_comment,
               posts.emotion
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        ORDER BY posts.created_at DESC
        LIMIT ? OFFSET ?
    """, (user_id, limit, offset))
    posts_raw = cursor.fetchall()
    posts = []
    for post in posts_raw:
        latest_comment = json.loads(post[8])[0] if post[8] and json.loads(post[8]) else None
        posts.append({
            'id': post[0],
            'content': post[1],
            'created_at': post[2],
            'username': post[3],
            'image': post[4],
            'likes': post[5],
            'dislikes': post[6],
            'user_reaction': post[7],
            'latest_comment': latest_comment,
            'emotion': post[9]
        })
    conn.close()
    return jsonify({'status': 'success', 'posts': posts})


@app.route('/face_detector')
def face_detector():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('NaNa_Face.html')


@app.route('/face_chat')
def face_chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT users.id, users.username 
        FROM friends 
        JOIN users ON friends.friend_id = users.id 
        WHERE friends.user_id = ? AND friends.status = 'accepted'
    """, (session['user_id'],))
    friends = [{'id': row[0], 'username': row[1]} for row in cursor.fetchall()]
    conn.close()
    return render_template('Face_Chat.html', friends=friends)


@app.route('/friends', methods=['POST'])
def get_friends():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    user_id = request.json.get('user_id', session['user_id'])
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT users.id, users.username 
        FROM friends 
        JOIN users ON friends.friend_id = users.id 
        WHERE friends.user_id = ? AND friends.status = 'accepted'
    """, (user_id,))
    friends = [{'id': row[0], 'username': row[1]} for row in cursor.fetchall()]
    conn.close()
    return jsonify(friends)


@app.route('/search_user', methods=['GET', 'POST'])
def search_user():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = request.form.get('username') if request.method == 'POST' else request.args.get('username')
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    if username:
        cursor.execute("SELECT id, username, relationship_status, avatar FROM users WHERE username LIKE ? AND id != ?",
                       (f'%{username}%', user_id))
        users = cursor.fetchall()
    else:
        users = []
    cursor.execute("""
        SELECT users.id, users.username FROM friends 
        JOIN users ON friends.user_id = users.id
        WHERE friends.friend_id = ? AND friends.status = 'pending'
    """, (user_id,))
    friend_requests = [{'id': row[0], 'username': row[1]} for row in cursor.fetchall()]
    conn.close()
    form = AddFriendForm()
    return render_template('search_results.html', users=users, form=form, friend_requests=friend_requests)


@app.route('/add_friend/<int:friend_id>', methods=['POST'])
def add_friend(friend_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    form = AddFriendForm()
    if form.validate_on_submit():
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM friends WHERE user_id = ? AND friend_id = ?", (user_id, friend_id))
            if not cursor.fetchone():
                cursor.execute("INSERT INTO friends (user_id, friend_id, status) VALUES (?, ?, 'pending')",
                               (user_id, friend_id))
                cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
                sender_username = cursor.fetchone()[0]

                # Отправляем уведомление в реальном времени
                send_notifications_real_time(friend_id, f"{sender_username} sent you a friend request")

                conn.commit()
                logger.info(f"Friend request sent to user {friend_id}")

                socketio.emit('new_friend_request', {
                    'sender_id': user_id,
                    'sender_username': sender_username
                }, room=str(friend_id))

            else:
                logger.info(f"Friend request to user {friend_id} already exists")
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
        finally:
            conn.close()
    return redirect(request.referrer or url_for('home'))


@app.route('/accept_friend/<int:friend_id>', methods=['POST'])
def accept_friend(friend_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    form = AddFriendForm()
    if form.validate_on_submit():
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE friends SET status = 'accepted' WHERE user_id = ? AND friend_id = ?",
                           (friend_id, user_id))
            cursor.execute("INSERT OR IGNORE INTO friends (user_id, friend_id, status) VALUES (?, ?, 'accepted')",
                           (user_id, friend_id))
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            acceptor_username = cursor.fetchone()[0]
            cursor.execute("SELECT username FROM users WHERE id = ?", (friend_id,))
            friend_username = cursor.fetchone()[0]

            # Отправляем уведомление в реальном времени
            send_notifications_real_time(friend_id, f"{acceptor_username} accepted your friend request")

            conn.commit()
            logger.info(f"Friend request from {friend_id} accepted by {user_id}")

            socketio.emit('friend_request_accepted', {
                'friend_id': user_id,
                'friend_username': acceptor_username
            }, room=str(friend_id))

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
        finally:
            conn.close()
        return redirect(url_for('home'))
    logger.error("CSRF validation failed for accept_friend")
    return "Bad Request: CSRF token missing", 400


@app.route('/reject_friend/<int:friend_id>', methods=['POST'])
def reject_friend(friend_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    form = AddFriendForm()
    if form.validate_on_submit():
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM friends WHERE user_id = ? AND friend_id = ?", (friend_id, user_id))
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            rejector_username = cursor.fetchone()[0]

            # Отправляем уведомление в реальном времени
            send_notifications_real_time(friend_id, f"{rejector_username} rejected your friend request")

            conn.commit()
            logger.info(f"Friend request from {friend_id} rejected by {user_id}")

            socketio.emit('friend_request_rejected', {'friend_id': user_id}, room=str(friend_id))

        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
        finally:
            conn.close()
        return redirect(url_for('home'))
    logger.error("CSRF validation failed for reject_friend")
    return "Bad Request: CSRF token missing", 400


@app.route('/create_chat/<int:friend_id>', methods=['GET'])
def create_chat(friend_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM chats 
        WHERE (user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?)
    """, (user_id, friend_id, friend_id, user_id))
    chat = cursor.fetchone()
    if not chat:
        cursor.execute("INSERT INTO chats (user1_id, user2_id) VALUES (?, ?)", (user_id, friend_id))
        chat_id = cursor.lastrowid
    else:
        chat_id = chat[0]
    conn.commit()
    conn.close()
    return redirect(url_for('chat', chat_id=chat_id))


@app.route('/chat/<int:chat_id>', methods=['GET'])
def chat(chat_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT chats.id, users.username, chats.user1_id, chats.user2_id
        FROM chats 
        JOIN users ON (chats.user1_id = users.id OR chats.user2_id = users.id)
        WHERE chats.id = ? AND (chats.user1_id = ? OR chats.user2_id = ?) AND users.id != ?
    """, (chat_id, user_id, user_id, user_id))
    chat = cursor.fetchone()
    if not chat:
        conn.close()
        return redirect(url_for('home'))
    cursor.execute("""
        SELECT chat_messages.id, chat_messages.content, chat_messages.created_at,
               users.username, chat_messages.sender_id
        FROM chat_messages
        JOIN users ON chat_messages.sender_id = users.id
        WHERE chat_messages.chat_id = ?
        ORDER BY chat_messages.created_at ASC
    """, (chat_id,))
    messages = cursor.fetchall()
    logger.info(f"Loaded messages for chat {chat_id}: {len(messages)} messages")
    cursor.execute("""
        UPDATE chat_messages
        SET is_read = 1
        WHERE chat_id = ? AND sender_id != ? AND is_read = 0
    """, (chat_id, user_id))
    conn.commit()
    other_user_id = chat[2] if chat[3] == user_id else chat[3]
    send_notifications_real_time(other_user_id)
    send_notifications_real_time(user_id)
    conn.close()
    return render_template('chat.html', chat=chat, messages=messages, user_id=user_id, chat_id=chat_id)


@app.route('/send_message', methods=['POST'])
def send_message():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    data = request.get_json()
    chat_id = data.get('chat_id')
    content = data.get('content')
    if not chat_id or not content:
        logger.error(f"Missing chat_id or content: {data}")
        return jsonify({'status': 'error', 'message': 'Missing chat_id or content'}), 400
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO chat_messages (chat_id, sender_id, content) VALUES (?, ?, ?)",
                       (chat_id, user_id, content))
        conn.commit()
        logger.info(f"Message saved: chat_id={chat_id}, sender_id={user_id}, content={content}")
        cursor.execute("SELECT created_at FROM chat_messages WHERE id = LAST_INSERT_ROWID()")
        created_at = cursor.fetchone()[0]
        cursor.execute("SELECT user1_id, user2_id FROM chats WHERE id = ?", (chat_id,))
        chat = cursor.fetchone()
        if not chat:
            logger.error(f"Chat {chat_id} not found")
            return jsonify({'status': 'error', 'message': 'Chat not found'}), 404
        other_user_id = chat[1] if chat[0] == user_id else chat[0]
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        username = cursor.fetchone()[0]
        conn.close()

        message_data = {
            'chat_id': chat_id,
            'sender_id': user_id,
            'username': username,
            'content': content,
            'created_at': created_at
        }

        # Отправляем уведомление о новом сообщении
        send_notifications_real_time(other_user_id, f"{username}: {content[:50]}{'...' if len(content) > 50 else ''}")

        socketio.emit('new_message', message_data, room=str(other_user_id))
        socketio.emit('new_message', message_data, room=str(user_id))
        socketio.emit('message_sent', message_data, room=str(user_id))

        return jsonify({
            'status': 'success',
            'chat_id': chat_id,
            'sender_id': user_id,
            'content': content,
            'created_at': created_at
        })
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/clear_notifications', methods=['POST'])
def clear_notifications():
    if 'user_id' not in session:
        logger.warning("Attempt to clear notifications without login")
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    try:
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notifications WHERE user_id = ?", (user_id,))
        conn.commit()
        logger.info(f"Notifications cleared for user_id: {user_id}")

        # Отправляем пустой список уведомлений в реальном времени
        socketio.emit('update_notifications', {
            'user_id': user_id,
            'notifications': [],
            'unread_count': 0,
            'unread_messages': 0,
            'pending_requests': 0,
            'timestamp': datetime.now().isoformat()
        }, room=str(user_id))

        return jsonify({'status': 'success', 'message': 'Notifications cleared'})
    except sqlite3.Error as e:
        logger.error(f"Database error while clearing notifications for user_id {user_id}: {e}")
        conn.rollback()
        return jsonify({'status': 'error', 'message': 'Database error'}), 500
    finally:
        conn.close()


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        confirm_password = form.confirm_password.data
        description = form.description.data
        city = form.city.data
        gender = form.gender.data
        interests = form.interests.data
        if password != confirm_password:
            return render_template('register.html', form=form, error="Passwords do not match")
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return render_template('register.html', form=form, error="Username already exists")
        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (username, password, description, city, gender, interests) VALUES (?, ?, ?, ?, ?, ?)",
            (username, hashed_password, description, city, gender, interests))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html', form=form, cities=['Moscow', 'Saint Petersburg', 'Novosibirsk'])


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT username, description, relationship_status, avatar, city, gender, interests FROM users WHERE id = ?",
        (user_id,))
    user = cursor.fetchone()
    cursor.execute("""
        SELECT posts.id, posts.content, posts.created_at, users.username, posts.image,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = posts.id AND reaction = 'like') AS likes,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = posts.id AND reaction = 'dislike') AS dislikes,
               (SELECT reaction FROM post_reactions WHERE post_id = posts.id AND user_id = ?) AS user_reaction,
               (SELECT json_group_array(json_array(c.content, c.created_at, u2.username))
                FROM post_comments c
                JOIN users u2 ON c.user_id = u2.id
                WHERE c.post_id = posts.id
                ORDER BY c.created_at DESC
                LIMIT 1) AS latest_comment,
               posts.emotion
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        WHERE posts.user_id = ?
        ORDER BY posts.created_at DESC
    """, (user_id, user_id))
    posts_raw = cursor.fetchall()
    posts = []
    for post in posts_raw:
        latest_comment = json.loads(post[8])[0] if post[8] and json.loads(post[8]) else None
        posts.append({
            'id': post[0],
            'content': post[1],
            'created_at': post[2],
            'username': post[3],
            'image': post[4],
            'likes': post[5],
            'dislikes': post[6],
            'user_reaction': post[7],
            'latest_comment': latest_comment,
            'emotion': post[9]
        })
    cursor.execute("""
        SELECT users.username, users.id FROM friends 
        JOIN users ON friends.friend_id = users.id
        WHERE friends.user_id = ? AND friends.status = 'accepted'
    """, (user_id,))
    friends = cursor.fetchall()
    cities = ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань", "Нижний Новгород", "Челябинск",
              "Самара", "Омск", "Ростов-на-Дону"]
    if request.method == 'POST':
        csrf_token = request.form.get('csrf_token')
        if not csrf_token:
            logger.error("CSRF token missing in request")
            return jsonify({'status': 'error', 'message': 'CSRF token missing'}), 400
        description = request.form.get('description')
        relationship_status = request.form.get('relationship_status')
        city = request.form.get('city')
        gender = request.form.get('gender')
        interests = request.form.get('interests')
        avatar = request.files.get('avatar')
        avatar_filename = user[3]
        if avatar and allowed_file(avatar.filename):
            filename = secure_filename(avatar.filename)
            avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            avatar.save(avatar_path)
            avatar_filename = filename
        cursor.execute(
            "UPDATE users SET description = ?, relationship_status = ?, avatar = ?, city = ?, gender = ?, interests = ? WHERE id = ?",
            (description, relationship_status, avatar_filename, city, gender, interests, user_id))
        conn.commit()
        flash('Profile updated successfully', 'success')
        return jsonify({'status': 'success', 'avatar': f'/static/avatars/{avatar_filename}'})
    conn.close()
    return render_template('profile.html', user=user, posts=posts, friends=friends, cities=cities)


@app.route('/like_post/<int:post_id>', methods=['POST'])
def like_post(post_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("SELECT reaction FROM post_reactions WHERE post_id = ? AND user_id = ?", (post_id, user_id))
    existing_reaction = cursor.fetchone()
    if existing_reaction:
        if existing_reaction[0] == 'like':
            cursor.execute("DELETE FROM post_reactions WHERE post_id = ? AND user_id = ?", (post_id, user_id))
        else:
            cursor.execute("UPDATE post_reactions SET reaction = 'like' WHERE post_id = ? AND user_id = ?",
                           (post_id, user_id))
    else:
        cursor.execute("INSERT INTO post_reactions (post_id, user_id, reaction) VALUES (?, ?, 'like')",
                       (post_id, user_id))
        cursor.execute("SELECT user_id, username FROM posts JOIN users ON posts.user_id = users.id WHERE posts.id = ?",
                       (post_id,))
        post_owner = cursor.fetchone()
        if post_owner and post_owner[0] != user_id:
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            liker_username = cursor.fetchone()[0]

            # Отправляем уведомление в реальном времени
            send_notifications_real_time(post_owner[0], f"{liker_username} liked your post")

    conn.commit()
    cursor.execute("""
        SELECT (SELECT COUNT(*) FROM post_reactions WHERE post_id = ? AND reaction = 'like') AS likes,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = ? AND reaction = 'dislike') AS dislikes,
               (SELECT reaction FROM post_reactions WHERE post_id = ? AND user_id = ?) AS user_reaction
    """, (post_id, post_id, post_id, user_id))
    reaction_data = cursor.fetchone()
    likes, dislikes, user_reaction = reaction_data if reaction_data else (0, 0, None)
    conn.close()
    socketio.emit('post_reaction_updated', {
        'post_id': post_id,
        'likes': likes,
        'dislikes': dislikes,
        'user_id': user_id,
        'user_reaction': user_reaction
    })
    return jsonify({
        'status': 'success',
        'post_id': post_id,
        'likes': likes,
        'dislikes': dislikes,
        'user_reaction': user_reaction
    })


@app.route('/dislike_post/<int:post_id>', methods=['POST'])
def dislike_post(post_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("SELECT reaction FROM post_reactions WHERE post_id = ? AND user_id = ?", (post_id, user_id))
    existing_reaction = cursor.fetchone()
    if existing_reaction:
        if existing_reaction[0] == 'dislike':
            cursor.execute("DELETE FROM post_reactions WHERE post_id = ? AND user_id = ?", (post_id, user_id))
        else:
            cursor.execute("UPDATE post_reactions SET reaction = 'dislike' WHERE post_id = ? AND user_id = ?",
                           (post_id, user_id))
    else:
        cursor.execute("INSERT INTO post_reactions (post_id, user_id, reaction) VALUES (?, ?, 'dislike')",
                       (post_id, user_id))
        cursor.execute("SELECT user_id, username FROM posts JOIN users ON posts.user_id = users.id WHERE posts.id = ?",
                       (post_id,))
        post_owner = cursor.fetchone()
        if post_owner and post_owner[0] != user_id:
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            disliker_username = cursor.fetchone()[0]

            # Отправляем уведомление в реальном времени
            send_notifications_real_time(post_owner[0], f"{disliker_username} disliked your post")

    conn.commit()
    cursor.execute("""
        SELECT (SELECT COUNT(*) FROM post_reactions WHERE post_id = ? AND reaction = 'like') AS likes,
               (SELECT COUNT(*) FROM post_reactions WHERE post_id = ? AND reaction = 'dislike') AS dislikes,
               (SELECT reaction FROM post_reactions WHERE post_id = ? AND user_id = ?) AS user_reaction
    """, (post_id, post_id, post_id, user_id))
    reaction_data = cursor.fetchone()
    likes, dislikes, user_reaction = reaction_data if reaction_data else (0, 0, None)
    conn.close()
    socketio.emit('post_reaction_updated', {
        'post_id': post_id,
        'likes': likes,
        'dislikes': dislikes,
        'user_id': user_id,
        'user_reaction': user_reaction
    })
    return jsonify({
        'status': 'success',
        'post_id': post_id,
        'likes': likes,
        'dislikes': dislikes,
        'user_reaction': user_reaction
    })


@app.route('/create_post', methods=['POST'])
def create_post():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    content = request.form['content']
    image = request.files.get('image')
    photo_path = request.form.get('photo_path')
    emotion = request.form.get('emotion', None)
    speech_text = request.form.get('speech_text', '')

    # Если есть распознанная речь, используем её как содержание поста
    if speech_text and speech_text.strip() and (not content or content.strip() == ''):
        content = speech_text.strip()

    image_filename = None

    logger.info(
        f"[CreatePost] Received: content={content}, speech_text={speech_text}, photo_path={photo_path}, emotion={emotion}, image={image.filename if image else None}")

    if photo_path and os.path.exists(photo_path):
        logger.info(f"[CreatePost] Processing photo_path: {photo_path}")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = os.path.basename(photo_path)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            import shutil
            shutil.move(photo_path, target_path)
            logger.info(f"[CreatePost] Moved photo from {photo_path} to {target_path}")
            image_filename = filename
        except Exception as e:
            logger.error(f"[CreatePost] Failed to move photo from {photo_path} to {target_path}: {e}")
    elif image and allowed_file(image.filename):
        logger.info(f"[CreatePost] Processing uploaded image: {image.filename}")
        filename = secure_filename(image.filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(target_path)
        logger.info(f"[CreatePost] Saved image to {target_path}")
        image_filename = filename

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO posts (user_id, content, image, emotion) VALUES (?, ?, ?, ?)",
                   (user_id, content, image_filename, emotion))
    conn.commit()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    username = cursor.fetchone()[0]
    cursor.execute("SELECT id, created_at FROM posts WHERE id = LAST_INSERT_ROWID()")
    post_id, created_at = cursor.fetchone()
    conn.close()

    socketio.emit('new_post', {
        'id': post_id,
        'username': username,
        'content': content,
        'image': image_filename,
        'created_at': created_at,
        'likes': 0,
        'dislikes': 0,
        'user_reaction': None,
        'emotion': emotion,
        'has_speech': bool(speech_text and speech_text.strip())
    })
    return redirect(url_for('home'))


@app.route('/enhance_post', methods=['POST'])
def enhance_post():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    data = request.get_json()
    content = data.get('content')
    if not content:
        return jsonify({'status': 'error', 'message': 'No content provided'}), 400
    try:
        lang = detect(content)
        if lang != 'ru':
            enhanced_content = content + "."
        else:
            prompt = f"Улучшите следующий текст на русском языке, сохранив его основной смысл, но сделав стиль более живым, эмоциональным и естественным. Обязательно добавьте знаки препинания (точки, запятые, восклицательные знаки, если уместно), используйте разговорный тон и исправьте орфографические или стилистические ошибки. Верните только улучшенный текст без дополнительных комментариев. Текст: {content}"
            response = gemini_model.generate_content(prompt)
            enhanced_content = response.text.strip()
        # Ensure at least a period at the end if missing
        if not enhanced_content.endswith(('.', '!', '?')):
            enhanced_content += '.'
        word_count = len(re.findall(r'\b\w+\b', enhanced_content))
        if word_count > 100:
            return jsonify({
                'status': 'error',
                'message': 'Enhanced content exceeds 100 words'
            }), 400
        return jsonify({
            'status': 'success',
            'original_content': content,
            'enhanced_content': enhanced_content
        })
    except Exception as e:
        logger.error(f"Error enhancing post: {e}")
        # Fallback with basic punctuation
        enhanced_content = content.strip() + '.'
        return jsonify({
            'status': 'success',
            'original_content': content,
            'enhanced_content': enhanced_content
        })


@app.route('/add_comment', methods=['POST'])
def add_comment():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    data = request.get_json()
    post_id = data.get('post_id')
    content = data.get('content')
    if not post_id or not content:
        return jsonify({'status': 'error', 'message': 'Missing post_id or content'}), 400
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO post_comments (post_id, user_id, content) VALUES (?, ?, ?)",
                   (post_id, user_id, content))
    conn.commit()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    username = cursor.fetchone()[0]
    cursor.execute("SELECT created_at FROM post_comments WHERE id = LAST_INSERT_ROWID()")
    created_at = cursor.fetchone()[0]
    cursor.execute("SELECT user_id, username FROM posts JOIN users ON posts.user_id = users.id WHERE posts.id = ?",
                   (post_id,))
    post_owner = cursor.fetchone()
    if post_owner and post_owner[0] != user_id:
        # Отправляем уведомление в реальном времени
        send_notifications_real_time(post_owner[0], f"{username} commented on your post")

        conn.commit()

    conn.close()
    return jsonify({
        'status': 'success',
        'username': username,
        'created_at': created_at
    })


@app.route('/get_comments/<int:post_id>', methods=['GET'])
def get_comments(post_id):
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT post_comments.content, post_comments.created_at, users.username
        FROM post_comments
        JOIN users ON post_comments.user_id = users.id
        WHERE post_comments.post_id = ?
        ORDER BY post_comments.created_at ASC
    """, (post_id,))
    comments = cursor.fetchall()
    conn.close()
    return jsonify({
        'status': 'success',
        'comments': [{'content': c[0], 'created_at': c[1], 'username': c[2]} for c in comments]
    })


@app.route('/create_story', methods=['POST'])
def create_story():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    form = StoryForm()
    if form.validate_on_submit():
        content = request.form['content']
        image = request.files.get('image')
        image_filename = None

        if image and allowed_file(image.filename):
            filename = secure_filename(f"{user_id}_{int(datetime.now().timestamp())}_{image.filename}")
            target_path = os.path.join(app.config['STORIES_FOLDER'], filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            image.save(target_path)
            image_filename = filename

        expires_at = datetime.now() + timedelta(hours=24)
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO stories (user_id, content, image, expires_at) VALUES (?, ?, ?, ?)",
                       (user_id, content, image_filename, expires_at))
        conn.commit()
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        username = cursor.fetchone()[0]
        cursor.execute("SELECT id, created_at FROM stories WHERE id = LAST_INSERT_ROWID()")
        story_id, created_at = cursor.fetchone()
        conn.close()

        socketio.emit('new_story', {
            'story_id': story_id,
            'user_id': user_id,
            'username': username,
            'content': content,
            'image': image_filename,
            'created_at': created_at,
            'expires_at': expires_at
        })
        return redirect(url_for('home'))
    return redirect(url_for('home'))


@app.route('/view_story/<int:story_id>', methods=['POST'])
def view_story(story_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE stories SET views = views + 1 WHERE id = ?", (story_id,))
        conn.commit()
        cursor.execute("SELECT views FROM stories WHERE id = ?", (story_id,))
        views = cursor.fetchone()[0]
        return jsonify({'status': 'success', 'views': views})
    except sqlite3.Error as e:
        logger.error(f"Database error updating story views: {e}")
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/get_story/<int:story_id>', methods=['GET'])
def get_story(story_id):
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT content, image FROM stories 
        WHERE id = ? AND user_id IN (
            SELECT friend_id FROM friends 
            WHERE user_id = ? AND status = 'accepted'
        )
    """, (story_id, user_id))
    story = cursor.fetchone()
    conn.close()
    if story:
        return jsonify({
            'status': 'success',
            'content': story[0],
            'image': story[1]
        })
    return jsonify({'status': 'error', 'message': 'Story not found or not accessible'}), 404


@app.route('/static/stories/<path:filename>')
def serve_story_file(filename):
    return send_from_directory(app.config['STORIES_FOLDER'], filename)


# Новые endpoints для уведомлений
@app.route('/api/notifications/latest')
def get_latest_notifications():
    """Получить последние уведомления"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT content, created_at 
        FROM notifications 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    """, (user_id,))

    notifications = cursor.fetchall()

    # Считаем количество непрочитанных
    cursor.execute("SELECT COUNT(*) FROM notifications WHERE user_id = ?", (user_id,))
    unread_count = cursor.fetchone()[0]

    conn.close()

    return jsonify({
        'notifications': notifications,
        'unread_count': unread_count,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/notifications/check')
def check_new_notifications():
    """Проверить наличие новых уведомлений"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = session['user_id']
    last_check = request.args.get('last_check', datetime.now().isoformat())

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) 
        FROM notifications 
        WHERE user_id = ? AND created_at > ?
    """, (user_id, last_check))

    new_count = cursor.fetchone()[0]
    conn.close()

    return jsonify({
        'has_new': new_count > 0,
        'new_count': new_count,
        'timestamp': datetime.now().isoformat()
    })


# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    logger.info('Клиент подключился')
    if 'user_id' in session:
        user_id = session['user_id']
        join_room(str(user_id))

        # Добавляем пользователя в онлайн
        with online_lock:
            online_users.add(user_id)

        # Отправляем текущий онлайн статус
        emit('online_status', {'status': 'online', 'user_id': user_id})

        # Немедленно отправляем уведомления при подключении
        send_notifications_real_time(user_id)

        logger.info(f'User {user_id} connected and subscribed to notifications')


@socketio.on('user_online')
def handle_user_online():
    """Пользователь в сети"""
    user_id = session.get('user_id')
    if user_id:
        with online_lock:
            online_users.add(user_id)
        logger.info(f"User {user_id} is now online")

        # Немедленно отправляем уведомления при подключении
        send_notifications_real_time(user_id)
        emit('online_status', {'status': 'online', 'user_id': user_id})


@socketio.on('user_offline')
def handle_user_offline():
    """Пользователь вышел из сети"""
    user_id = session.get('user_id')
    if user_id:
        with online_lock:
            online_users.discard(user_id)
        logger.info(f"User {user_id} is now offline")
        emit('online_status', {'status': 'offline', 'user_id': user_id})


@socketio.on('disconnect')
def handle_disconnect():
    """Обработка отключения"""
    user_id = session.get('user_id')
    if user_id:
        with online_lock:
            online_users.discard(user_id)
        logger.info(f"User {user_id} disconnected")


@socketio.on('audio_data')
def handle_audio_data(data):
    """
    Обработка аудио данных от клиента
    """
    try:
        user_id = session.get('user_id')
        if not user_id:
            emit('speech_result', {'error': 'Not logged in'})
            return

        # Декодируем base64 аудио данные
        if 'audio' in data:
            audio_base64 = data['audio'].split(',')[1]  # Убираем префикс data:audio/wav;base64,
            audio_data = base64.b64decode(audio_base64)

            # Обрабатываем аудио
            recognized_text, success = process_audio(audio_data)

            # Сохраняем в базу данных
            if success and recognized_text and recognized_text != "Речь не распознана":
                conn = sqlite3.connect('nana.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO speech_recognition (user_id, recognized_text) VALUES (?, ?)",
                               (user_id, recognized_text))
                conn.commit()
                conn.close()

            # Отправляем результат обратно клиенту
            emit('speech_result', {
                'text': recognized_text,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }, room=str(user_id))

        else:
            emit('speech_result', {'error': 'No audio data received'})

    except Exception as e:
        logger.error(f"Error handling audio data: {e}")
        emit('speech_result', {'error': str(e)})


@socketio.on('join_speech_room')
def handle_join_speech_room(data):
    """
    Присоединение к комнате для обновлений речи
    """
    user_id = session.get('user_id')
    if user_id:
        room_name = f'speech_room_{user_id}'
        join_room(room_name)
        emit('speech_room_joined', {'room': room_name})


@socketio.on('join_room')
def on_join(data):
    try:
        if isinstance(data, dict):
            room = data.get('room')
        else:
            room = str(data)
        if room:
            join_room(room)
            logger.info(f'Клиент присоединился к комнате: {room}')
        else:
            logger.warning('No room specified in join_room event')
    except Exception as e:
        logger.error(f'Error in on_join: {e}')


@socketio.on('frame')
def handle_frame(data):
    try:
        user_id = session.get('user_id')
        if not user_id:
            emit('error', {'error': 'Not logged in'})
            return

        previous_gesture = None
        middle_finger_sent = False

        img_data = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            logger.error("[Frame] Invalid or empty frame")
            emit('error', {'error': 'Empty frame'})
            return

        emotion, score, bbox = process_emotions(frame)
        gesture = process_hands(frame)

        response = {
            'emotion': {'name': emotion, 'score': float(score), 'bbox': list(bbox)} if emotion else None,
            'gesture': gesture,
            'photo': None
        }

        if gesture == "FIST":
            time.sleep(5)
            filename = f"user_photo/{user_id}_{int(time.time())}.jpg"
            success = cv2.imwrite(filename, frame)
            if success:
                logger.info(f"[Saved] {filename}")
                response['photo'] = filename
            else:
                logger.error(f"[Failed] Could not save {filename}")
                emit('error', {'error': f'Failed to save photo: {filename}'})
                return

        if gesture == "VICTORY":
            emit('spiderman_gesture', {'message': 'Вы хотите перейти на страницу своего профиля?'})
            return

        if gesture == "ROCK":
            logger.info("[Gesture] Detected: ROCK")
            emit('rock_gesture')
            return

        if gesture == "MIDDLE_FINGER" and not middle_finger_sent:
            filename = f"user_photo/{user_id}_{int(time.time())}.jpg"
            success = cv2.imwrite(filename, frame)
            if success:
                logger.info(f"[Captured] {filename}")
                os.remove(filename)
                middle_finger_sent = True
            else:
                logger.error(f"[Failed] Could not save {filename}")

        if gesture != "MIDDLE_FINGER":
            middle_finger_sent = False

        previous_gesture = gesture
        emit('response', response, room=str(user_id))

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        emit('error', {'error': str(e)}, room=str(user_id))


import os
from pyngrok import ngrok

if __name__ == '__main__':
    try:
        # Если запускаемся в Colab — пробрасываем ngrok вручную
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            from pyngrok import ngrok

            public_url = ngrok.connect(5000)
            print(f"✅ Открой сайт по ссылке: {public_url}")

        init_db()
        logger.info("Starting Flask-SocketIO server with speech recognition...")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False)

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
