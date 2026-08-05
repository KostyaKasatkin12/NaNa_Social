import eventlet

eventlet.monkey_patch()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import re
import json
import time
import base64
import logging
import random
import sqlite3
from datetime import datetime, timedelta
from threading import Lock
from io import BytesIO
import wave
from collections import defaultdict
from collections import Counter

from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, SubmitField, FileField, SelectField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, Optional, Length, EqualTo

from forms import LoginForm, RegisterForm, StoryForm, AddFriendForm, SearchForm

import cv2
import numpy as np
from fer import FER
import mediapipe as mp
import pymorphy3 as pymorphy
from textblob import TextBlob
import google.generativeai as genai
from langdetect import detect, DetectorFactory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import speech_recognition as sr

# ==================== ИНИЦИАЛИЗАЦИЯ ====================

app = Flask(__name__)
app.secret_key = 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5'

app.config['UPLOAD_FOLDER'] = 'static/avatars'
app.config['STORIES_FOLDER'] = 'static/stories'
app.config['AUDIO_FOLDER'] = 'static/audio'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_REFRESH_EACH_REQUEST'] = True

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
csrf = CSRFProtect(app)

for folder in [app.config['UPLOAD_FOLDER'], app.config['STORIES_FOLDER'], app.config['AUDIO_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi', 'wav'}

# ==================== AI ИНИЦИАЛИЗАЦИЯ ====================

emotion_detector = FER(mtcnn=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

recognizer = sr.Recognizer()
DetectorFactory.seed = 0
morph = pymorphy.MorphAnalyzer()

genai.configure(api_key="AIzaSyBNR9ULDDEAJ2iW_0b6GgT9lfSOqs-dwMw")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==================== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ====================

online_users = set()
online_lock = Lock()

user_locations = {}
user_mood_history = {}
mood_cache = {}
city_mood_data = {}
user_city = {}

# ==================== АНОНИМНЫЙ ЧАТ ПО НАСТРОЕНИЯМ ====================

MOOD_ROOMS = {
    'happy': {'name': '😊 Счастливые', 'emoji': '😊', 'color': '#34d399', 'desc': 'Делимся радостью и позитивом'},
    'sad': {'name': '😢 Грустные', 'emoji': '😢', 'color': '#60a5fa', 'desc': 'Поддержим друг друга в трудный момент'},
    'angry': {'name': '😡 Злые', 'emoji': '😡', 'color': '#f87171', 'desc': 'Выплесните гнев в безопасной обстановке'},
    'anxious': {'name': '😰 Тревожные', 'emoji': '😰', 'color': '#fbbf24', 'desc': 'Поделитесь тревогами и страхами'},
    'lonely': {'name': '🥺 Одинокие', 'emoji': '🥺', 'color': '#a78bfa', 'desc': 'Вы не одни, мы с вами'},
    'hopeful': {'name': '🌟 Надеющиеся', 'emoji': '🌟', 'color': '#34d399', 'desc': 'Вместе верим в лучшее'},
    'tired': {'name': '😴 Уставшие', 'emoji': '😴', 'color': '#9ca3af', 'desc': 'Отдыхаем и восстанавливаемся'},
    'love': {'name': '❤️ Влюблённые', 'emoji': '❤️', 'color': '#f472b6', 'desc': 'Делимся теплотой и нежностью'},
    'grateful': {'name': '🙏 Благодарные', 'emoji': '🙏', 'color': '#fbbf24', 'desc': 'Ценим моменты и друг друга'},
    'confused': {'name': '🤔 Растерянные', 'emoji': '🤔', 'color': '#6b7280', 'desc': 'Ищем ответы вместе'},
    'support': {'name': '🤗 Поддержка', 'emoji': '🤗', 'color': '#10b981', 'desc': 'Пришли поддержать других'},
    'help': {'name': '🆘 Нужна помощь', 'emoji': '🆘', 'color': '#ef4444', 'desc': 'Срочная эмоциональная поддержка'}
}

mood_chat_rooms = {}
mood_chat_users = {}
user_current_mood = {}
user_mood_timestamp = {}

GESTURES = {
    (0, 0, 0, 0, 0): "FIST",
    (1, 1, 1, 1, 1): "OPEN_HAND",
    (0, 1, 1, 0, 0): "VICTORY",
    (0, 1, 0, 0, 1): "ROCK",
    (1, 0, 0, 0, 0): "THUMBS_UP",
    (0, 0, 0, 0, 1): "THUMBS_DOWN",
    (0, 0, 1, 0, 0): "MIDDLE_FINGER",
    (1, 1, 0, 0, 1): "SPIDERMAN"
}

finger_colors = {
    "thumb": (255, 0, 0),
    "index": (0, 255, 0),
    "middle": (0, 0, 255),
    "ring": (0, 255, 255),
    "pinky": (255, 255, 0)
}

MOOD_EMOJIS = {
    'very_positive': '🥳', 'positive': '😊', 'neutral': '😐',
    'negative': '😔', 'very_negative': '😢', 'angry': '😡',
    'surprised': '😮', 'fearful': '😨', 'joyful': '😄', 'sad': '😭'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== ФУНКЦИИ ДЛЯ РАБОТЫ С НАСТРОЕНИЯМИ ====================

def set_user_mood(user_id, mood_key):
    """Установить настроение пользователя"""
    user_current_mood[user_id] = mood_key
    user_mood_timestamp[user_id] = datetime.now().isoformat()

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO mood_history (user_id, mood, created_at) 
        VALUES (?, ?, ?)
    """, (user_id, mood_key, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return user_current_mood[user_id]


def get_user_mood(user_id):
    """Получить текущее настроение пользователя"""
    return user_current_mood.get(user_id)


def get_user_mood_with_emoji(user_id):
    """Получить настроение пользователя с эмодзи"""
    mood = user_current_mood.get(user_id)
    if mood and mood in MOOD_ROOMS:
        return {
            'key': mood,
            'name': MOOD_ROOMS[mood]['name'],
            'emoji': MOOD_ROOMS[mood]['emoji'],
            'color': MOOD_ROOMS[mood]['color']
        }
    return None


# ==================== ФУНКЦИИ БАЗЫ ДАННЫХ ====================

def init_db():
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    tables = [
        '''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            description TEXT,
            relationship_status TEXT DEFAULT 'не интересуюсь',
            avatar TEXT,
            city TEXT,
            gender TEXT,
            interests TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''',
        '''CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image TEXT,
            emotion TEXT,
            mood TEXT,
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        )''',
        '''CREATE TABLE IF NOT EXISTS anonymous_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''',
        '''CREATE TABLE IF NOT EXISTS mood_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            score REAL,
            text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''',
        '''CREATE TABLE IF NOT EXISTS city_moods (
            city_name TEXT PRIMARY KEY,
            mood TEXT NOT NULL,
            score REAL DEFAULT 0,
            count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )'''
    ]

    for table in tables:
        cursor.execute(table)

    conn.commit()
    conn.close()


def migrate_database():
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(posts)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'mood' not in columns:
        try:
            cursor.execute("ALTER TABLE posts ADD COLUMN mood TEXT")
            print("[OK] Добавлена колонка 'mood' в таблицу posts")
        except sqlite3.OperationalError as e:
            print(f"[WARN] Ошибка добавления колонки mood: {e}")

    conn.commit()
    conn.close()
    print("[OK] Миграция базы данных завершена")


def create_test_user():
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed_password = generate_password_hash('admin123')
        cursor.execute(
            "INSERT INTO users (username, password, description, city) VALUES (?, ?, ?, ?)",
            ('admin', hashed_password, 'Администратор', 'Москва')
        )
        conn.commit()
        print("[OK] Создан администратор: admin / admin123")

    conn.close()


# ==================== ФУНКЦИИ ДЛЯ НАСТРОЕНИЯ ====================

def analyze_mood_text(text):
    if not text or not isinstance(text, str):
        return {'mood': 'neutral', 'score': 0, 'emoji': '😐', 'label': 'Нейтральное'}

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.5:
            mood, label, emoji = 'very_positive', 'Очень позитивное', '🥳'
        elif polarity > 0.1:
            mood, label, emoji = 'positive', 'Позитивное', '😊'
        elif polarity < -0.5:
            mood, label, emoji = 'very_negative', 'Очень негативное', '😢'
        elif polarity < -0.1:
            mood, label, emoji = 'negative', 'Негативное', '😔'
        else:
            mood, label, emoji = 'neutral', 'Нейтральное', '😐'

        return {'mood': mood, 'score': polarity, 'emoji': emoji, 'label': label}
    except:
        return {'mood': 'neutral', 'score': 0, 'emoji': '😐', 'label': 'Нейтральное'}


# ==================== ФУНКЦИИ ДЛЯ ЧАТА ПО НАСТРОЕНИЯМ (ПОЛНАЯ АНОНИМНОСТЬ) ====================

def get_mood_room(mood):
    if mood not in mood_chat_rooms:
        mood_chat_rooms[mood] = {
            'users': set(),
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
    return mood_chat_rooms[mood]


def add_user_to_mood_room(user_id, mood):
    """Добавить пользователя в комнату (анонимно, без имени)"""
    if mood not in mood_chat_rooms:
        mood_chat_rooms[mood] = {
            'users': set(),
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
    mood_chat_rooms[mood]['users'].add(user_id)
    mood_chat_users[user_id] = mood
    return mood_chat_rooms[mood]


def remove_user_from_mood_room(user_id):
    if user_id in mood_chat_users:
        mood = mood_chat_users[user_id]
        if mood in mood_chat_rooms:
            mood_chat_rooms[mood]['users'].discard(user_id)
        del mood_chat_users[user_id]


def save_mood_message(mood, message, user_id):
    """Сохранить сообщение в комнате (полностью анонимно)"""
    room = get_mood_room(mood)
    room['messages'].append({
        'user_id': user_id,
        'message': message,
        'created_at': datetime.now().isoformat()
    })
    if len(room['messages']) > 100:
        room['messages'] = room['messages'][-100:]
    return room['messages'][-1]


# ==================== ОБЫЧНЫЕ ФУНКЦИИ ====================

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
            return None, 0.0, (0, 0, 0, 0)
        resized_frame = cv2.resize(frame, (64, 64))
        emotions = emotion_detector.detect_emotions(resized_frame)
        if emotions:
            (x, y, w, h) = emotions[0]["box"]
            emotion, score = max(emotions[0]["emotions"].items(), key=lambda x: x[1])
            scale_x = frame.shape[1] / 64
            scale_y = frame.shape[0] / 64
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            return emotion, score, (x, y, w, h)
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


def process_voice_command(text):
    if not text or not isinstance(text, str):
        return None
    text = text.lower().strip()

    commands = {
        'открыть профиль': 'open_profile', 'профиль': 'open_profile',
        'открыть друзей': 'open_friends', 'друзья': 'open_friends',
        'создать пост': 'create_post', 'выйти': 'logout',
        'домой': 'go_home', 'помощь': 'help',
        'сделать фото': 'take_photo', 'анонимный чат': 'anonymous_chat',
        'карта': 'mood_map', 'настроение': 'mood_chat'
    }

    for cmd, action in commands.items():
        if cmd in text:
            return action
    return None


def process_audio(audio_data, sample_rate=16000):
    try:
        audio_buffer = BytesIO()
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        audio_buffer.seek(0)

        with sr.AudioFile(audio_buffer) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language='ru-RU')
            command = process_voice_command(text)
            return text, True, command
    except sr.UnknownValueError:
        return "Речь не распознана", False, None
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return f"Ошибка: {e}", False, None


def send_notifications_real_time(user_id, notification_content=None):
    try:
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        if notification_content:
            cursor.execute(
                "INSERT INTO notifications (user_id, content) VALUES (?, ?)",
                (user_id, notification_content)
            )
            conn.commit()

        cursor.execute("""
            SELECT content, created_at FROM notifications 
            WHERE user_id = ? ORDER BY created_at DESC LIMIT 10
        """, (user_id,))
        notifications = cursor.fetchall()

        with online_lock:
            is_online = user_id in online_users

        if is_online:
            socketio.emit('update_notifications', {
                'user_id': user_id,
                'notifications': [{'content': n[0], 'created_at': n[1]} for n in notifications],
                'unread_count': len(notifications),
                'timestamp': datetime.now().isoformat()
            }, room=str(user_id))

        conn.close()
        return notifications
    except Exception as e:
        logger.error(f"Error in send_notifications_real_time: {e}")
        return None


# ==================== МАРШРУТЫ ====================

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/x-icon')


@app.route('/check_session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'user_id': session['user_id']})
    return jsonify({'logged_in': False, 'message': 'No active session'})


@app.route('/call/<int:friend_id>')
def call_page(friend_id):
    """Страница для звонка"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    # Проверяем, что пользователь является другом
    cursor.execute("""
        SELECT COUNT(*) FROM friends 
        WHERE ((user_id = ? AND friend_id = ?) OR (user_id = ? AND friend_id = ?))
        AND status = 'accepted'
    """, (user_id, friend_id, friend_id, user_id))

    if cursor.fetchone()[0] == 0:
        conn.close()
        flash('Этот пользователь не в вашем списке друзей', 'error')
        return redirect(url_for('home'))

    # Получаем информацию о друге
    cursor.execute("SELECT id, username FROM users WHERE id = ?", (friend_id,))
    friend = cursor.fetchone()
    conn.close()

    if not friend:
        flash('Пользователь не найден', 'error')
        return redirect(url_for('home'))

    return render_template('call.html',
                           friend_id=friend_id,
                           friend_username=friend[1],
                           user_id=user_id)

@app.route('/mood_chat')
def mood_chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    current_mood = get_user_mood(user_id)
    current_mood_data = get_user_mood_with_emoji(user_id) if current_mood else None

    room_stats = {}
    for mood, room in mood_chat_rooms.items():
        room_stats[mood] = len(room['users'])

    return render_template('mood_chat.html',
                           moods=MOOD_ROOMS,
                           current_mood=current_mood,
                           current_mood_data=current_mood_data,
                           room_stats=room_stats)


@app.route('/mood_map')
def mood_map():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('mood_map.html')


@app.route('/', methods=['GET'])
def home():
    if 'user_id' not in session:
        logger.warning("No user_id in session, redirecting to login")
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, city FROM users WHERE id = ?", (user_id,))
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
               posts.emotion, posts.mood
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        ORDER BY posts.created_at DESC
    """, (user_id,))
    posts_raw = cursor.fetchall()
    posts = []
    for post in posts_raw:
        posts.append({
            'id': post[0],
            'content': post[1],
            'created_at': post[2],
            'username': post[3],
            'image': post[4],
            'likes': post[5],
            'dislikes': post[6],
            'user_reaction': post[7],
            'emotion': post[8],
            'mood': post[9]
        })

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

    cursor.execute("SELECT content, created_at FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
                   (user_id,))
    notifications = cursor.fetchall()

    cursor.execute("""
        SELECT s.id, s.user_id, s.content, s.image, s.created_at, s.expires_at, s.views 
        FROM stories s
        JOIN friends f ON s.user_id = f.friend_id
        WHERE f.user_id = ? AND f.status = 'accepted' AND s.expires_at > ?
    """, (user_id, datetime.now()))
    stories = cursor.fetchall()

    conn.close()

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
                           stories=stories,
                           user_city=user[1] if len(user) > 1 else None)


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

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', form=form, error="Неверное имя пользователя или пароль")

    return render_template('login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))

    form = RegisterForm()

    if request.method == 'POST':
        if form.validate_on_submit():
            username = form.username.data.strip()
            password = form.password.data
            confirm_password = form.confirm_password.data
            description = form.description.data.strip() if form.description.data else ''
            city = form.city.data.strip() if form.city.data else ''
            gender = form.gender.data if form.gender.data else ''
            interests = form.interests.data.strip() if form.interests.data else ''

            if password != confirm_password:
                return render_template('register.html', form=form, error="Пароли не совпадают")

            conn = sqlite3.connect('nana.db')
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                conn.close()
                return render_template('register.html', form=form, error="Пользователь с таким именем уже существует")

            hashed_password = generate_password_hash(password)
            cursor.execute(
                """INSERT INTO users (username, password, description, city, gender, interests) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (username, hashed_password, description, city, gender, interests)
            )
            conn.commit()
            conn.close()

            flash('Регистрация успешна! Теперь вы можете войти.', 'success')
            return redirect(url_for('login'))
        else:
            error_messages = []
            for field, errors in form.errors.items():
                for error in errors:
                    error_messages.append(f"{field}: {error}")
            return render_template('register.html', form=form, error="; ".join(error_messages))

    return render_template('register.html', form=form)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/speech_history')
def speech_history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT recognized_text, created_at 
        FROM speech_recognition 
        WHERE user_id = ? ORDER BY created_at DESC LIMIT 50
    ''', (user_id,))
    speech_history = cursor.fetchall()
    conn.close()
    return render_template('speech_history.html', speech_history=speech_history)


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
               posts.emotion, posts.mood
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        ORDER BY posts.created_at DESC
        LIMIT ? OFFSET ?
    """, (user_id, limit, offset))
    posts_raw = cursor.fetchall()
    posts = []
    for post in posts_raw:
        posts.append({
            'id': post[0],
            'content': post[1],
            'created_at': post[2],
            'username': post[3],
            'image': post[4],
            'likes': post[5],
            'dislikes': post[6],
            'user_reaction': post[7],
            'emotion': post[8],
            'mood': post[9]
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
    return render_template('Face_Chat.html')


@app.route('/friends', methods=['POST'])
def get_friends():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user_id = request.json.get('user_id', session['user_id'])
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
                send_notifications_real_time(friend_id, f"{sender_username} sent you a friend request")
                conn.commit()

                socketio.emit('new_friend_request', {
                    'sender_id': user_id,
                    'sender_username': sender_username
                }, room=str(friend_id))
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

            send_notifications_real_time(friend_id, f"{acceptor_username} accepted your friend request")
            conn.commit()

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

            send_notifications_real_time(friend_id, f"{rejector_username} rejected your friend request")
            conn.commit()

            socketio.emit('friend_request_rejected', {'friend_id': user_id}, room=str(friend_id))
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
        finally:
            conn.close()
        return redirect(url_for('home'))
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

    cursor.execute("""
        UPDATE chat_messages
        SET is_read = 1
        WHERE chat_id = ? AND sender_id != ? AND is_read = 0
    """, (chat_id, user_id))
    conn.commit()

    # ВАЖНО: вычисляем other_user_id
    other_user_id = chat[2] if chat[3] == user_id else chat[3]

    send_notifications_real_time(other_user_id)
    send_notifications_real_time(user_id)
    conn.close()

    # Передаём other_user_id в шаблон
    return render_template('chat.html',
                           chat=chat,
                           messages=messages,
                           user_id=user_id,
                           chat_id=chat_id,
                           other_user_id=other_user_id)  # <-- ДОБАВЛЯЕМ ЭТУ ПЕРЕМЕННУЮ


@app.route('/send_message', methods=['POST'])
@csrf.exempt  # Временно отключаем CSRF для этого маршрута, если не работает
def send_message():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    user_id = session['user_id']
    data = request.get_json()
    chat_id = data.get('chat_id')
    content = data.get('content')

    if not chat_id or not content:
        return jsonify({'status': 'error', 'message': 'Missing chat_id or content'}), 400

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO chat_messages (chat_id, sender_id, content) VALUES (?, ?, ?)",
                       (chat_id, user_id, content))
        conn.commit()
        cursor.execute("SELECT created_at FROM chat_messages WHERE id = LAST_INSERT_ROWID()")
        created_at = cursor.fetchone()[0]
        cursor.execute("SELECT user1_id, user2_id FROM chats WHERE id = ?", (chat_id,))
        chat = cursor.fetchone()
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

        send_notifications_real_time(other_user_id, f"{username}: {content[:50]}{'...' if len(content) > 50 else ''}")
        socketio.emit('new_message', message_data, room=str(other_user_id))
        socketio.emit('new_message', message_data, room=str(user_id))

        return jsonify({'status': 'success', **message_data})
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/clear_notifications', methods=['POST'])
def clear_notifications():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401

    user_id = session['user_id']
    try:
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM notifications WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Notifications cleared'})
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return jsonify({'status': 'error', 'message': 'Database error'}), 500


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
               posts.emotion, posts.mood
        FROM posts 
        JOIN users ON posts.user_id = users.id 
        WHERE posts.user_id = ?
        ORDER BY posts.created_at DESC
    """, (user_id, user_id))
    posts_raw = cursor.fetchall()
    posts = []
    for post in posts_raw:
        posts.append({
            'id': post[0],
            'content': post[1],
            'created_at': post[2],
            'username': post[3],
            'image': post[4],
            'likes': post[5],
            'dislikes': post[6],
            'user_reaction': post[7],
            'emotion': post[8],
            'mood': post[9]
        })

    cursor.execute("""
        SELECT users.username, users.id FROM friends 
        JOIN users ON friends.friend_id = users.id
        WHERE friends.user_id = ? AND friends.status = 'accepted'
    """, (user_id,))
    friends = cursor.fetchall()

    cities = [('', 'Выберите город'), ('Москва', 'Москва'), ('Санкт-Петербург', 'Санкт-Петербург'),
              ('Новосибирск', 'Новосибирск'), ('Екатеринбург', 'Екатеринбург'), ('Казань', 'Казань'),
              ('Нижний Новгород', 'Нижний Новгород'), ('Челябинск', 'Челябинск'), ('Самара', 'Самара'),
              ('Омск', 'Омск'), ('Ростов-на-Дону', 'Ростов-на-Дону'), ('Уфа', 'Уфа'), ('Красноярск', 'Красноярск'),
              ('Пермь', 'Пермь'), ('Воронеж', 'Воронеж'), ('Волгоград', 'Волгоград'), ('Краснодар', 'Краснодар'),
              ('Саратов', 'Саратов'), ('Тюмень', 'Тюмень'), ('Тольятти', 'Тольятти'), ('Ижевск', 'Ижевск'),
              ('Барнаул', 'Барнаул'), ('Ульяновск', 'Ульяновск'), ('Иркутск', 'Иркутск'), ('Хабаровск', 'Хабаровск'),
              ('Ярославль', 'Ярославль'), ('Владивосток', 'Владивосток'), ('Махачкала', 'Махачкала'),
              ('Томск', 'Томск'), ('Оренбург', 'Оренбург'), ('Кемерово', 'Кемерово'), ('Новокузнецк', 'Новокузнецк'),
              ('Рязань', 'Рязань'), ('Астрахань', 'Астрахань'), ('Набережные Челны', 'Набережные Челны'),
              ('Пенза', 'Пенза'), ('Киров', 'Киров'), ('Липецк', 'Липецк'), ('Балашиха', 'Балашиха'),
              ('Чебоксары', 'Чебоксары'), ('Калининград', 'Калининград'), ('Тула', 'Тула'), ('Курск', 'Курск'),
              ('Ставрополь', 'Ставрополь'), ('Улан-Удэ', 'Улан-Удэ'), ('Тверь', 'Тверь'),
              ('Магнитогорск', 'Магнитогорск'), ('Севастополь', 'Севастополь'), ('Сочи', 'Сочи'),
              ('Белгород', 'Белгород')]

    form = AddFriendForm()

    if request.method == 'POST':
        if not form.validate_on_submit():
            return jsonify({'status': 'error', 'message': 'CSRF validation failed'}), 400

        description = request.form.get('description')
        relationship_status = request.form.get('relationship_status')
        city = request.form.get('city')
        gender = request.form.get('gender')
        interests = request.form.get('interests')
        avatar = request.files.get('avatar')
        avatar_filename = user[3] if user else None

        if avatar and allowed_file(avatar.filename):
            filename = secure_filename(avatar.filename)
            avatar_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            avatar.save(avatar_path)
            avatar_filename = filename

        cursor.execute(
            "UPDATE users SET description = ?, relationship_status = ?, avatar = ?, city = ?, gender = ?, interests = ? WHERE id = ?",
            (description, relationship_status, avatar_filename, city, gender, interests, user_id))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'avatar': f'/static/avatars/{avatar_filename}'})

    conn.close()
    return render_template('profile.html', user=user, posts=posts, friends=friends, cities=cities, form=form)


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
        cursor.execute("SELECT user_id FROM posts WHERE id = ?", (post_id,))
        post_owner = cursor.fetchone()
        if post_owner and post_owner[0] != user_id:
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            liker_username = cursor.fetchone()[0]
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

    return jsonify({'status': 'success', 'likes': likes, 'dislikes': dislikes, 'user_reaction': user_reaction})


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
        cursor.execute("SELECT user_id FROM posts WHERE id = ?", (post_id,))
        post_owner = cursor.fetchone()
        if post_owner and post_owner[0] != user_id:
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            disliker_username = cursor.fetchone()[0]
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

    return jsonify({'status': 'success', 'likes': likes, 'dislikes': dislikes, 'user_reaction': user_reaction})


@app.route('/create_post', methods=['POST'])
def create_post():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    content = request.form.get('content', '')
    image = request.files.get('image')
    photo_path = request.form.get('photo_path')
    emotion = request.form.get('emotion', None)
    speech_text = request.form.get('speech_text', '')

    if speech_text and speech_text.strip() and (not content or content.strip() == ''):
        content = speech_text.strip()

    image_filename = None

    if photo_path and os.path.exists(photo_path):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = os.path.basename(photo_path)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            import shutil
            shutil.move(photo_path, target_path)
            image_filename = filename
        except Exception as e:
            logger.error(f"Failed to move photo: {e}")
    elif image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(target_path)
        image_filename = filename

    mood_result = analyze_mood_text(content) if content else {'mood': 'neutral', 'score': 0, 'emoji': '😐',
                                                              'label': 'Нейтральное'}

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO posts (user_id, content, image, emotion, mood) VALUES (?, ?, ?, ?, ?)",
                   (user_id, content, image_filename, emotion, mood_result['mood']))
    conn.commit()

    cursor.execute("SELECT username, city FROM users WHERE id = ?", (user_id,))
    user_info = cursor.fetchone()
    username = user_info[0]
    city = user_info[1] if user_info and len(user_info) > 1 else None

    cursor.execute("SELECT id, created_at FROM posts WHERE id = LAST_INSERT_ROWID()")
    post_id, created_at = cursor.fetchone()
    conn.close()

    if city:
        conn = sqlite3.connect('nana.db')
        cursor = conn.cursor()
        cursor.execute("SELECT mood, score, count FROM city_moods WHERE city_name = ?", (city,))
        existing = cursor.fetchone()
        if existing:
            current_mood, current_score, current_count = existing
            new_count = current_count + 1
            new_score = (current_score * current_count + mood_result['score']) / new_count
            cursor.execute("""
                UPDATE city_moods SET mood = ?, score = ?, count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE city_name = ?
            """, (mood_result['mood'], new_score, new_count, city))
        else:
            cursor.execute("""
                INSERT INTO city_moods (city_name, mood, score, count)
                VALUES (?, ?, ?, ?)
            """, (city, mood_result['mood'], mood_result['score'], 1))
        conn.commit()
        conn.close()

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO mood_history (user_id, mood, score, text) 
        VALUES (?, ?, ?, ?)
    """, (user_id, mood_result['mood'], mood_result['score'], content[:200]))
    conn.commit()
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
        'mood': mood_result['mood'],
        'mood_emoji': mood_result['emoji']
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
            prompt = f"Улучшите следующий текст на русском языке, сохранив его основной смысл, но сделав стиль более живым, эмоциональным и естественным. Верните только улучшенный текст: {content}"
            response = gemini_model.generate_content(prompt)
            enhanced_content = response.text.strip()

        if not enhanced_content.endswith(('.', '!', '?')):
            enhanced_content += '.'

        return jsonify({
            'status': 'success',
            'original_content': content,
            'enhanced_content': enhanced_content
        })
    except Exception as e:
        logger.error(f"Error enhancing post: {e}")
        return jsonify({
            'status': 'success',
            'original_content': content,
            'enhanced_content': content.strip() + '.'
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

    cursor.execute("SELECT user_id FROM posts WHERE id = ?", (post_id,))
    post_owner = cursor.fetchone()
    if post_owner and post_owner[0] != user_id:
        send_notifications_real_time(post_owner[0], f"{username} commented on your post")
        conn.commit()

    conn.close()
    return jsonify({'status': 'success', 'username': username, 'created_at': created_at})


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
        content = request.form.get('content', '')
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

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE stories SET views = views + 1 WHERE id = ?", (story_id,))
        conn.commit()
        cursor.execute("SELECT views FROM stories WHERE id = ?", (story_id,))
        views = cursor.fetchone()[0]
        return jsonify({'status': 'success', 'views': views})
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
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
        return jsonify({'status': 'success', 'content': story[0], 'image': story[1]})
    return jsonify({'status': 'error', 'message': 'Story not found'}), 404


@app.route('/static/stories/<path:filename>')
def serve_story_file(filename):
    return send_from_directory(app.config['STORIES_FOLDER'], filename)


# ==================== API ДЛЯ КАРТЫ МАРОДЁРОВ ====================

@app.route('/api/mood_rooms/stats')
def api_mood_rooms_stats():
    """Получить статистику по комнатам настроений"""
    stats = {}
    total = 0
    for mood, room in mood_chat_rooms.items():
        count = len(room['users'])
        stats[mood] = count
        total += count

    return jsonify({
        'status': 'success',
        'data': stats,
        'total': total
    })


@app.route('/api/cities/stats')
def api_cities_stats():
    """Получить статистику по городам с количеством пользователей и настроением (АНОНИМНО)"""
    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, city 
        FROM users 
        WHERE city IS NOT NULL AND city != ''
    """)
    users = cursor.fetchall()

    city_data = {}

    for user in users:
        user_id, username, city = user
        if not city or city == '':
            continue

        mood = user_current_mood.get(user_id)

        if not mood:
            cursor.execute("""
                SELECT mood FROM mood_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC LIMIT 1
            """, (user_id,))
            mood_result = cursor.fetchone()
            mood = mood_result[0] if mood_result else 'neutral'

        if city not in city_data:
            city_data[city] = {
                'count': 0,
                'moods': []
            }

        city_data[city]['count'] += 1
        city_data[city]['moods'].append(mood)

    conn.close()

    result = []
    for city, data in city_data.items():
        mood_counts = Counter(data['moods'])
        dominant_mood = mood_counts.most_common(1)[0][0] if mood_counts else 'neutral'

        mood_data = MOOD_ROOMS.get(dominant_mood, {})

        mood_percentages = {}
        total_moods = len(data['moods'])
        for mood_type, count in mood_counts.items():
            mood_percentages[mood_type] = round(count / total_moods * 100, 1) if total_moods > 0 else 0

        result.append({
            'city': city,
            'count': data['count'],
            'dominant_mood': dominant_mood,
            'emoji': mood_data.get('emoji', '😐'),
            'label': mood_data.get('name', 'Нейтральное'),
            'mood_distribution': mood_percentages
        })

    result.sort(key=lambda x: x['count'], reverse=True)

    return jsonify({
        'status': 'success',
        'data': result,
        'total_cities': len(result),
        'total_users': sum(c['count'] for c in result)
    })


@app.route('/api/notifications/latest')
def get_latest_notifications():
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


# ==================== WEBSOCKET ОБРАБОТЧИКИ ====================

@socketio.on('connect')
def handle_connect():
    if 'user_id' in session:
        user_id = session['user_id']
        join_room(str(user_id))
        with online_lock:
            online_users.add(user_id)
        emit('online_status', {'status': 'online', 'user_id': user_id})
        send_notifications_real_time(user_id)
        logger.info(f'User {user_id} connected')


@socketio.on('disconnect')
def handle_disconnect():
    user_id = session.get('user_id')
    if user_id:
        with online_lock:
            online_users.discard(user_id)
        logger.info(f'User {user_id} disconnected')


# ==================== WEBSOCKET ДЛЯ ЗВОНКОВ ====================

@socketio.on('call_user')
def handle_call_user(data):
    """Инициация звонка"""
    user_id = session.get('user_id')
    if not user_id:
        return

    friend_id = data.get('friend_id')
    call_type = data.get('type', 'video')

    if not friend_id:
        return

    call_id = f"{user_id}_{friend_id}_{int(time.time())}"

    conn = sqlite3.connect('nana.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()

    username = user[0] if user else 'Пользователь'

    emit('incoming_call', {
        'call_id': call_id,
        'caller_id': user_id,
        'caller_username': username,
        'type': call_type
    }, room=str(friend_id))

    emit('call_initiated', {
        'call_id': call_id,
        'status': 'ringing'
    }, room=str(user_id))


@socketio.on('accept_call')
def handle_accept_call(data):
    """Принятие звонка"""
    user_id = session.get('user_id')
    if not user_id:
        return

    call_id = data.get('call_id')
    caller_id = data.get('caller_id')

    if not call_id or not caller_id:
        return

    emit('call_accepted', {
        'call_id': call_id,
        'status': 'connected'
    }, room=str(user_id))

    emit('call_accepted', {
        'call_id': call_id,
        'status': 'connected',
        'friend_id': user_id
    }, room=str(caller_id))


@socketio.on('reject_call')
def handle_reject_call(data):
    """Отклонение звонка"""
    user_id = session.get('user_id')
    if not user_id:
        return

    call_id = data.get('call_id')
    caller_id = data.get('caller_id')

    if not call_id or not caller_id:
        return

    emit('call_rejected', {
        'call_id': call_id,
        'status': 'rejected'
    }, room=str(caller_id))


@socketio.on('end_call')
def handle_end_call(data):
    """Завершение звонка"""
    user_id = session.get('user_id')
    if not user_id:
        return

    call_id = data.get('call_id')
    friend_id = data.get('friend_id')

    if not call_id or not friend_id:
        return

    emit('call_ended', {
        'call_id': call_id,
        'status': 'ended'
    }, room=str(friend_id))

    emit('call_ended', {
        'call_id': call_id,
        'status': 'ended'
    }, room=str(user_id))


@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    """Передача SDP offer"""
    user_id = session.get('user_id')
    if not user_id:
        return

    target_user_id = data.get('target_user_id')
    offer = data.get('offer')
    call_id = data.get('call_id')

    if not target_user_id or not offer:
        return

    emit('webrtc_offer_received', {
        'offer': offer,
        'call_id': call_id,
        'from_user_id': user_id
    }, room=str(target_user_id))


@socketio.on('webrtc_answer')
def handle_webrtc_answer(data):
    """Передача SDP answer"""
    user_id = session.get('user_id')
    if not user_id:
        return

    target_user_id = data.get('target_user_id')
    answer = data.get('answer')
    call_id = data.get('call_id')

    if not target_user_id or not answer:
        return

    emit('webrtc_answer_received', {
        'answer': answer,
        'call_id': call_id,
        'from_user_id': user_id
    }, room=str(target_user_id))


@socketio.on('webrtc_ice_candidate')
def handle_webrtc_ice_candidate(data):
    """Передача ICE кандидатов"""
    user_id = session.get('user_id')
    if not user_id:
        return

    target_user_id = data.get('target_user_id')
    candidate = data.get('candidate')
    call_id = data.get('call_id')

    if not target_user_id or not candidate:
        return

    emit('webrtc_ice_candidate_received', {
        'candidate': candidate,
        'call_id': call_id,
        'from_user_id': user_id
    }, room=str(target_user_id))

@socketio.on('join_room')
def on_join(data):
    try:
        room = data.get('room') if isinstance(data, dict) else str(data)
        if room:
            join_room(room)
            logger.info(f'Client joined room: {room}')
    except Exception as e:
        logger.error(f'Error in on_join: {e}')


# ==================== WEBSOCKET ДЛЯ ЧАТА ПО НАСТРОЕНИЯМ (ПОЛНАЯ АНОНИМНОСТЬ) ====================

@socketio.on('select_mood')
def handle_select_mood(data):
    user_id = session.get('user_id')
    if not user_id:
        return

    mood = data.get('mood')
    if not mood or mood not in MOOD_ROOMS:
        return

    set_user_mood(user_id, mood)

    emit('mood_selected', {
        'mood': mood,
        'mood_name': MOOD_ROOMS[mood]['name'],
        'emoji': MOOD_ROOMS[mood]['emoji'],
        'color': MOOD_ROOMS[mood]['color']
    }, room=str(user_id))

    update_mood_room_stats()


@socketio.on('join_mood_room')
def handle_join_mood_room(data):
    user_id = session.get('user_id')
    if not user_id:
        return

    mood = data.get('mood')
    if not mood or mood not in MOOD_ROOMS:
        return

    set_user_mood(user_id, mood)

    if user_id in mood_chat_users:
        old_mood = mood_chat_users[user_id]
        if old_mood != mood:
            if old_mood in mood_chat_rooms:
                mood_chat_rooms[old_mood]['users'].discard(user_id)
            del mood_chat_users[user_id]

    # Полностью анонимно - не передаём имя пользователя
    room = add_user_to_mood_room(user_id, mood)
    join_room(f"mood_{mood}")

    # Отправляем историю (без имён)
    emit('mood_history', {
        'messages': room['messages'][-50:],
        'users_count': len(room['users']),
        'current_mood': mood,
        'mood_name': MOOD_ROOMS[mood]['name'],
        'mood_emoji': MOOD_ROOMS[mood]['emoji']
    }, room=f"mood_{mood}")

    # Уведомляем всех о новом пользователе (анонимно)
    emit('mood_user_joined', {
        'users_count': len(room['users'])
    }, room=f"mood_{mood}")

    # Отправляем подтверждение пользователю
    emit('mood_joined', {
        'mood': mood,
        'mood_name': MOOD_ROOMS[mood]['name'],
        'mood_emoji': MOOD_ROOMS[mood]['emoji'],
        'users_count': len(room['users']),
        'messages': room['messages'][-50:]
    }, room=str(user_id))

    update_mood_room_stats()


@socketio.on('leave_mood_room')
def handle_leave_mood_room(data):
    user_id = session.get('user_id')
    if not user_id:
        return

    if user_id in mood_chat_users:
        mood = mood_chat_users[user_id]
        remove_user_from_mood_room(user_id)

        room = get_mood_room(mood)
        emit('mood_user_left', {
            'users_count': len(room['users'])
        }, room=f"mood_{mood}")

        emit('mood_left', {
            'mood': mood,
            'success': True
        }, room=str(user_id))

        update_mood_room_stats()


@socketio.on('mood_message')
def handle_mood_message(data):
    user_id = session.get('user_id')
    if not user_id:
        return

    mood = data.get('mood')
    message = data.get('message', '').strip()

    if not mood or not message or mood not in MOOD_ROOMS:
        return

    if user_id not in mood_chat_users or mood_chat_users[user_id] != mood:
        return

    # Сохраняем сообщение (полностью анонимно, без имени)
    saved_message = save_mood_message(mood, message, user_id)
    mood_analysis = analyze_mood_text(message)

    # Отправляем всем в комнате (анонимно)
    emit('mood_new_message', {
        'message': message,
        'created_at': saved_message['created_at'],
        'mood_emoji': MOOD_ROOMS[mood]['emoji'],
        'message_mood': mood_analysis['emoji']
    }, room=f"mood_{mood}")


def update_mood_room_stats():
    stats = {}
    total = 0
    for mood, room in mood_chat_rooms.items():
        count = len(room['users'])
        stats[mood] = count
        total += count

    socketio.emit('mood_users_update', stats)


# ==================== ОСТАЛЬНЫЕ WEBSOCKET ОБРАБОТЧИКИ ====================

@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        user_id = session.get('user_id')
        if not user_id:
            emit('speech_result', {'error': 'Not logged in'})
            return

        if 'audio' in data:
            audio_base64 = data['audio'].split(',')[1]
            audio_data = base64.b64decode(audio_base64)
            recognized_text, success, command = process_audio(audio_data)

            if success and recognized_text and recognized_text != "Речь не распознана":
                conn = sqlite3.connect('nana.db')
                cursor = conn.cursor()
                cursor.execute("INSERT INTO speech_recognition (user_id, recognized_text) VALUES (?, ?)",
                               (user_id, recognized_text))
                conn.commit()
                conn.close()

            if command:
                emit('voice_command', {
                    'type': command,
                    'message': f'Выполняется команда: {command}',
                    'text': recognized_text
                }, room=str(user_id))

            emit('speech_result', {
                'text': recognized_text,
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'is_command': bool(command)
            }, room=str(user_id))
    except Exception as e:
        logger.error(f"Error handling audio: {e}")
        emit('speech_result', {'error': str(e)})


@socketio.on('frame')
def handle_frame(data):
    try:
        user_id = session.get('user_id')
        if not user_id:
            emit('error', {'error': 'Not logged in'})
            return

        img_data = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
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
                response['photo'] = filename

        emit('response', response, room=str(user_id))
    except Exception as e:
        logger.error(f"Frame error: {e}")
        emit('error', {'error': str(e)})


# ==================== ЗАПУСК ====================

if __name__ == '__main__':
    try:
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            from pyngrok import ngrok

            public_url = ngrok.connect(5000)
            print(f"✅ Открой сайт по ссылке: {public_url}")

        init_db()
        migrate_database()
        create_test_user()

        logger.info("Starting NaNa Social Network with Mood Map and Anonymous Chat...")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
