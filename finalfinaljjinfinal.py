from flask import Flask, render_template, Response, send_file, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
import io
from datetime import datetime
import pygame
import simpleaudio as sa
import os
import threading
from flask_sqlalchemy import SQLAlchemy
import tkinter as tk

def play_sound(sound):
    pygame.mixer.Sound(sound).play()
    
# pygame 초기화
pygame.init()
pygame.mixer.init()

# 사운드 로드
sound_file = 'plank_complete.wav'  # 재생할 파일 이름
sound = pygame.mixer.Sound(sound_file)

font_path = "C:/Users/chaee/anaconda3/pkgs/matplotlib-base-3.4.3-py39h49ac443_0/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/NanumGothicBold.ttf"
fontprop = fm.FontProperties(fname=font_path, size=10)
plt.rc('font', family=fontprop.get_name())

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///squat_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  
db = SQLAlchemy(app)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pygame.mixer.init()

class SquatRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), unique=True, nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __init__(self, date, count):
        self.date = date
        self.count = count

class LungeRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), unique=True, nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __init__(self, date, count):
        self.date = date
        self.count = count
        
class PlankRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), unique=True, nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __init__(self, date, count):
        self.date = date
        self.count = count
        
class DolphinRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(10), unique=True, nullable=False)
    count = db.Column(db.Integer, nullable=False)
    def __init__(self, date, count):
        self.date = date
        self.count = count
        
with app.app_context():
    db.create_all()

# feedback_played = {}

# feedback_sounds = {
#     "무릎을 안으로 넣어주세요. 엉덩이를 조금 더 내려주세요." : "무릎을-안으로-넣어주세요.-엉덩이를-조금-더-올려주세요.wav",
#     "무릎을 안으로 넣어주세요. 엉덩이를 조금 더 내려주세요." : "무릎을-안으로-넣어주세요.-엉덩이를-조금-더-내려주세요.wav",
#     "어깨너비로 발을 벌리고 스쿼트를 시작해주세요!" : "어깨너비로-발을-벌리고-스쿼트를-시작해주세요_.wav",
#     "무릎을 안으로 넣어주세요.": "무릎을-안으로-넣어주세요.wav",
#     "무릎이 튀어나왔습니다. 엉덩이를 조금 더 내려주세요.": "무릎이-튀어나왔습니다.-엉덩이를-조금-더-내려주세요.wav",
#     "무릎이 튀어나왔습니다. 엉덩이를 조금 더 올려주세요.": "무릎이-튀어나왔습니다.-엉덩이를-조금-더-올려주세요.wav",
#     "좋은 스쿼트 자세입니다!": "좋은-스쿼트-자세입니다_.wav",
#     "엉덩이를 조금 더 내려주세요.": "엉덩이를-조금-더-내려주세요.wav",
#     "엉덩이를 조금 더 올려주세요.": "엉덩이를-조금-더-올려주세요.wav",
#     "이제 허리를 세우고 천천히 올라와주세요!": "이제-허리를-세우고-천천히-올라와주세요_.wav"
# }

# def play_feedback(feedback):
#     # 피드백이 이미 재생된 경우 재생하지 않음
#     if feedback_played.get(feedback, False):
#         return

#     sound_file = feedback_sounds.get(feedback)
#     if sound_file:
#         sound_path = f'D:/4-1/jongsul/sqart_tts/{sound_file}'
#         if os.path.exists(sound_path):
#             sound = pygame.mixer.Sound(sound_path)
#             threading.Thread(target=sound.play).start()
#             feedback_played[feedback] = True  
            
def reset_playback():
    global feedback_played
    feedback_played = {}
      
class CameraManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CameraManager, cls).__new__(cls)
            cls._instance.cap = cv2.VideoCapture(0)
        return cls._instance

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
                    
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def draw_text(img, text, position, font_size, font_color=(255, 255, 255), background_color=(0, 0, 0), padding=5):
    font_path = "C:/Users/chaee/anaconda3/pkgs/matplotlib-base-3.4.3-py39h49ac443_0/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/NanumGothicBold.ttf"  # 적절한 폰트 경로 설정
    font = ImageFont.truetype(font_path, font_size)

    # 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # 텍스트 크기 계산
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # 텍스트 배경 그리기
    text_position = position
    draw.rectangle((text_position[0] - padding, text_position[1] - padding,
                    text_position[0] + text_width + padding, text_position[1] + text_height + padding), fill=background_color)
    # 텍스트 그리기
    draw.text(text_position, text, font=font, fill=font_color)

    return np.array(img_pil)

def flip_image(image):
    return cv2.flip(image, 1)

feedback_counts = {}
exclude_feedbacks = ['어깨너비로 발을 벌리고 스쿼트를 시작해주세요!', '']  # 카운트에서 제외할 메시지 목록
body_part_counts = {"무릎": 0, "엉덩이": 0}  # 무릎과 엉덩이 카운트를 위한 딕셔너리
squat_counts = {}

lunge_feedback_counts = {}
exclude_feedbacks_lunge = ['발을 골반 너비로 벌리고 런지를 시작해주세요!', '']  # 카운트에서 제외할 메시지 목록
body_part_counts_lunge = {"몸통": 0, "오른쪽 무릎": 0, "왼쪽 무릎": 0}   # 무릎과 엉덩이 카운트를 위한 딕셔너리
lunge_counts = {}
lunge_detected = False
lunge_start_time = None

plank_feedback_counts = {}
exclude_feedbacks_plank = ['플랭크를 시작해주세요!', '']  # 카운트에서 제외할 메시지 목록
body_part_counts_plank = {"어깨": 0, "엉덩이": 0, "무릎": 0}   # 무릎과 엉덩이 카운트를 위한 딕셔너리
plank_counts = {}
plank_detected = False
plank_start_time = None

dolphin_feedback_counts = {}
exclude_feedbacks_dolphin = ['돌고래자세를 시작해주세요!', '']  # 카운트에서 제외할 메시지 목록
body_part_counts_dolphin = {"엉덩이 고관절": 0, "무릎": 0, "어깨 팔꿈치": 0}  
dolphin_counts = {}
dolphin_detected = False
dolphin_start_time = None

camera_manager = CameraManager()

def generate_frames_squat():
    global body_part_counts, squat_counts, app  # 전역 변수 사용 선언
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # cap = cv2.VideoCapture(0)
        
        initial_head_height = None  # 초기 머리 높이 저장 변수
        feedback_flag = True  # 초기 피드백 표시 플래그
        squat_start = False  # 스쿼트 내려갔는지 상태
        squat_completed = False
        squat_up = False
        squat_down = False
        good_squat_feedback_counts = {}  # 좋은 스쿼트 피드백 카운트 딕셔너리 초기화
        
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue

            frame = flip_image(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)            
            
            if feedback_flag:
                feedback = "어깨너비로 발을 벌리고 스쿼트를 시작해주세요!"
                font_color = (255, 255, 255)
            else:
                feedback = ""
                font_color = (255, 255, 255)
            
            if results.pose_landmarks:
                num_people = len(results.pose_landmarks.landmark)
                if num_people > 0:
                    head = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]
                    left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                    hip_width = abs(left_hip[0] - right_hip[0])
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    
                    if (left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1]) and \
                            (left_shoulder[1] < 0.5 and right_shoulder[1] < 0.5):
                        if shoulder_width < 0.1 and hip_width < 0.1:
                            feedback_flag = False
                            if (not squat_start) and (not squat_completed):
                                initial_head_height = head[1]  # 스쿼트 시작 시 머리 높이 측정
                                squat_start = True  # 스쿼트 시작 상태 활성화
                                squat_down = True
                                
                            if (squat_start) and (head[1] > initial_head_height):                                                       
                                if (right_knee[0] - right_ankle[0] > 0.05) and (75 < right_knee_angle < 105):
                                    feedback = "무릎을 안으로 넣어주세요."
                                    font_color = (255, 0, 0)
                                elif (right_knee[0] - right_ankle[0] > 0.05) and (105 < right_knee_angle < 135): 
                                    feedback = "무릎이 튀어나왔습니다. 엉덩이를 조금 더 내려주세요."
                                    font_color = (255, 0, 0)
                                elif (right_knee[0] - right_ankle[0] > 0.05) and (0 < right_knee_angle < 75): 
                                    feedback = "무릎이 튀어나왔습니다. 엉덩이를 조금 더 올려주세요."
                                    font_color = (255, 0, 0)
                                elif (right_knee[0] - right_ankle[0] < 0.05) and (75 < right_knee_angle < 105):
                                    feedback = "좋은 스쿼트 자세입니다!"            
                                    font_color = (0, 255, 0)
                                elif (right_knee[0] - right_ankle[0] < 0.05) and (105 < right_knee_angle < 135):
                                    feedback = "엉덩이를 조금 더 내려주세요."
                                    font_color = (255, 0, 0)
                                elif (right_knee[0] - right_ankle[0] < 0.05) and (0 < right_knee_angle < 75):
                                    feedback = "엉덩이를 조금 더 올려주세요."
                                    font_color = (255, 0, 0)
                                elif (left_knee[0] - left_ankle[0] < 0.05) and (75 < left_knee_angle < 105):
                                    feedback = "무릎을 안으로 넣어주세요."
                                    font_color = (255, 0, 0)
                                elif (left_knee[0] - left_ankle[0] < 0.05) and (105 < left_knee_angle < 135):
                                    feedback = "무릎을 안으로 넣어주세요. 엉덩이를 조금 더 내려주세요."
                                    font_color = (255, 0, 0)
                                elif (left_knee[0] - left_ankle[0] < 0.05) and (0 < left_knee_angle < 75):
                                    feedback = "무릎을 안으로 넣어주세요. 엉덩이를 조금 더 올려주세요."
                                    font_color = (255, 0, 0)
                                elif (left_knee[0] - left_ankle[0] > 0.05) and (75 < left_knee_angle < 105):
                                    feedback = "좋은 스쿼트 자세입니다!"
                                    font_color = (0, 255, 0)
                                elif (left_knee[0] - left_ankle[0] > 0.05) and (105 < left_knee_angle < 135):
                                    feedback = "엉덩이를 조금 더 내려주세요."
                                    font_color = (255, 0, 0)
                                elif (left_knee[0] - left_ankle[0] > 0.05) and (0 < left_knee_angle < 75):
                                    feedback = "엉덩이를 조금 더 올려주세요."
                                    font_color = (255, 0, 0)
                                else:
                                    feedback = ""
                                
                                if squat_completed:
                                    if (right_knee_angle <= 165) and (left_knee_angle <= 165):
                                        feedback = "이제 허리를 세우고 천천히 올라와주세요!"
                                        font_color = (255, 255, 0)
                                    else:
                                        feedback = ""  # 무릎 각도가 105도를 넘으면 피드백을 비워 표시 중지
                                        font_color = (255, 255, 255)  # 피드백이 없으므로 글씨색은 관계 없음
                                        # 피드백 카운트 초기화 및 새 스쿼트 준비
                                        good_squat_feedback_counts.clear()  # 모든 피드백 카운트 초기화
                                        date_key = datetime.now().strftime('%Y-%m-%d')
                                        if date_key not in squat_counts:
                                            squat_counts[date_key] = 0
                                        squat_counts[date_key] += 1
                                        # 초기화 횟수 증가
                                        squat_start = False
                                        squat_completed = False
                                        squat_up = False
                                        squat_down = False

                            # 스쿼트 동작 감지 및 피드백 로직
                                if (squat_down) and (not squat_up) and not squat_completed:
                                    if feedback not in exclude_feedbacks:  # 제외 대상이 아닐 때만 카운트
                                        if "무릎" in feedback:
                                            body_part_counts["무릎"] += 1
                                        if "엉덩이" in feedback:
                                            body_part_counts["엉덩이"] += 1
                                            
                                        if feedback not in feedback_counts:
                                            feedback_counts[feedback] = 1
                                        else:
                                            feedback_counts[feedback] += 1
                                            
                                        if feedback == "좋은 스쿼트 자세입니다!":
                                            if feedback not in good_squat_feedback_counts:
                                                good_squat_feedback_counts[feedback] = 1
                                            else:
                                                good_squat_feedback_counts[feedback] += 1
                                                
                                            if good_squat_feedback_counts[feedback] >= 15:
                                                with app.app_context():
                                                    save_squat_count()
                                                squat_completed = True
                                                squat_up = True # 스쿼트 상승 시작을 표시
                                                feedback_flag = False
                                                
            
            image = draw_text(image, f"스쿼트 횟수: {squat_counts}", (10, 420), 30, font_color=(17, 255, 127))                                    
            image = draw_text(image, feedback, (10, 20), 30, font_color=font_color)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.putText(frame, f'Squats: {squat_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def save_squat_count():
    with app.app_context():
        today = datetime.now().strftime('%Y-%m-%d')
        record = SquatRecord.query.filter_by(date=today).first()
        if not record:
            record = SquatRecord(date=today, count=0)
            db.session.add(record)
        record.count += 1
        
        db.session.commit()
        
def plot_graph_squat():
    global squat_counts  # 전역 변수 squat_counts 사용 선언
    
    # 틀린 자세 통계 그래프를 그리기 위해
    labels = body_part_counts.keys()
    values = body_part_counts.values()
    total = sum(values) if values else 1  # 분모가 0인 경우 방지

    percentages = [100 * (v / total) for v in values]
    
    plt.figure(figsize=(14, 7))  # 그래프의 전체 크기를 늘림
    
    # 첫 번째 그래프: 틀린 자세 기록
    plt.subplot(1, 2, 1)
    plt.bar(labels, percentages, color='red')
    plt.xlabel('신체부위', fontsize=18)  # 텍스트 크기 조정
    plt.ylabel('빈도 (%)', fontsize=18)  # 텍스트 크기 조정
    plt.title('틀린 자세 기록', fontsize=20)  # 텍스트 크기 조정
    plt.xticks(fontsize=16)  # x축 틱 레이블의 폰트 크기
    plt.yticks(fontsize=16)  # y축 틱 레이블의 폰트 크기
    plt.ylim(0, 100)

    # 두 번째 그래프: 스쿼트 총 횟수
    plt.subplot(1, 2, 2)
    dates = list(squat_counts.keys())
    counts = list(squat_counts.values())
    plt.bar(dates, counts, color='blue', width=0.1)
    plt.title("스쿼트 총 횟수", fontsize=20)  # 텍스트 크기 조정
    plt.ylabel("총 횟수 (개)", fontsize=18)  # 텍스트 크기 조정
    plt.ylim(0, max(counts + [3]))  # 최소 값은 3 이상을 유지하도록 설정
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=16)  # x축 틱 레이블의 폰트 크기
    plt.yticks(fontsize=16)  # y축 틱 레이블의 폰트 크기

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', transparent=True)
    img_buf.seek(0)
    plt.close()
    return img_buf

def generate_frames_lunge():
    global body_part_counts_lunge, lunge_counts, app 
    date_key = datetime.now().strftime('%Y-%m-%d')# 전역 변수 사용 선언
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        initial_head_height_lunge = None  # 초기 머리 높이 저장 변수
        feedback_flag_lunge = True  # 초기 피드백 표시 플래그
        lunge_start = False  # 스쿼트 내려갔는지 상태
        lunge_completed = False
        lunge_up = False
        lunge_down = False
        good_lunge_feedback_counts = {}  # 좋은 스쿼트 피드백 카운트 딕셔너리 초기화
        lunge_detected = False
        lunge_start_time = None
        
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue

            frame = flip_image(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 초기 피드백 설정
            if feedback_flag_lunge:
                feedback = "발을 골반 너비로 벌리고 런지를 시작해주세요!"
                font_color = (255, 255, 255)
            else:
                feedback = ""
                font_color = (255, 255, 255)
            

            if results.pose_landmarks:
                num_people = len(results.pose_landmarks.landmark)
                if num_people > 0:
                    head = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]
                    left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                    hip_width = abs(left_hip[0] - right_hip[0])
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    waist = (left_hip[0] + right_hip[0]) / 2
                    average_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
                    average_hip_x = (left_hip[0] + right_hip[0]) / 2
                    
                    # 런지 시작 조건 -> 옆모습 감지
                    if (left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1]) and \
                            (left_shoulder[1] < 0.5 and right_shoulder[1] < 0.5):
                        if shoulder_width < 0.1 and hip_width < 0.1:
                            feedback_flag_lunge = False  # 초기 피드백 제거
                            if (not lunge_start) and (not lunge_completed):
                                initial_head_height_lunge = head[1]  # 런지 시작 시 머리 높이 측정
                                lunge_start = True  # 런지 시작 상태 활성화
                                lunge_down = True
                            # if average_shoulder_x > average_hip_x:
                            #     # 어깨가 엉덩이보다 카메라 쪽에 가깝게 위치하면 사람이 오른쪽을 향하고 있음
                            #     feedback = "left"
                            # else:
                            #     # 그 반대의 경우, 사람이 왼쪽을 향하고 있음
                            #     feedback = "right"
                            # print("left_shoulder: ", left_shoulder[0], "left_hip: ", left_hip[0])    
                            
                            if (lunge_start) and (head[1] > initial_head_height_lunge):
                                # print("right_shoulder: ", right_shoulder[0], "right_hip: ", right_hip[0])           
                                if (left_shoulder[0] - left_hip[0] < -0.05) and (0 <= right_knee_angle < 70) and (0 <= left_knee_angle < 70) and (average_shoulder_x < average_hip_x):
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 작아요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (0 <= right_knee_angle < 70) and (70 <= left_knee_angle <= 110) and (average_shoulder_x < average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (0 <= right_knee_angle < 70) and (110 < left_knee_angle <= 140) and (average_shoulder_x < average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 작아요. 오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)  
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (70 <= right_knee_angle <= 110) and (0 <= left_knee_angle < 70) and (average_shoulder_x < average_hip_x):
                                    feedback = "어깨를 뒤로 당겨주세요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (70 <= right_knee_angle <= 110) and (70 <= left_knee_angle <= 110) and (average_shoulder_x < average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (70 <= right_knee_angle <= 110) and (110 < left_knee_angle <= 140) and (average_shoulder_x < average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (110 < right_knee_angle <= 140) and (0 <= left_knee_angle < 70) and (average_shoulder_x < average_hip_x):
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (110 < right_knee_angle <= 140) and (70 <= left_knee_angle <= 110) and (average_shoulder_x < average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] < -0.05) and (110 < right_knee_angle <= 140) and (110 < left_knee_angle <= 140) and (average_shoulder_x < average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 커요"
                                    font_color = (255, 0, 0)
                                    
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (0 <= right_knee_angle < 70) and (0 <= left_knee_angle < 70) and (average_shoulder_x < average_hip_x):
                                    feedback = "왼쪽 무릎이 90도 보다 작아요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 255)
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (0 <= right_knee_angle < 70) and (70 <= left_knee_angle <= 110) and (average_shoulder_x < average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 작아요. "
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (0 <= right_knee_angle < 70) and (110 < left_knee_angle <= 140) and (average_shoulder_x < average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 작아요.  오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)  
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (70 <= right_knee_angle <= 110) and (0 <= left_knee_angle < 70) and (average_shoulder_x < average_hip_x):
                                    feedback = "오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (70 <= right_knee_angle <= 110) and (70 <= left_knee_angle <= 110) and (average_shoulder_x < average_hip_x): 
                                    feedback = "좋은 런지 자세입니다!"
                                    font_color = (0, 255, 0)
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (70 <= right_knee_angle <= 110) and (110 < left_knee_angle <= 140) and (average_shoulder_x < average_hip_x): 
                                    feedback = "오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (110 < right_knee_angle <= 140) and (0 <= left_knee_angle < 70) and (average_shoulder_x < average_hip_x):
                                    feedback = "왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (110 < right_knee_angle <= 140) and (70 <= left_knee_angle <= 110) and (average_shoulder_x < average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_hip[0] - left_shoulder[0]) and (110 < right_knee_angle <= 140) and (110 < left_knee_angle <= 140) and (average_shoulder_x < average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)    
                                
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (0 <= right_knee_angle < 70) and (0 <= left_knee_angle < 70) and (average_shoulder_x > average_hip_x):
                                    feedback = "왼쪽 무릎이 90도 보다 작아요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (0 <= right_knee_angle < 70) and (70 <= left_knee_angle <= 110) and (average_shoulder_x > average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 작아요. "
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (0 <= right_knee_angle < 70) and (110 < left_knee_angle <= 140) and (average_shoulder_x > average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 작아요.  오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)  
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (70 <= right_knee_angle <= 110) and (0 <= left_knee_angle < 70) and (average_shoulder_x > average_hip_x):
                                    feedback = "오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (70 <= right_knee_angle <= 110) and (70 <= left_knee_angle <= 110) and (average_shoulder_x > average_hip_x): 
                                    feedback = "좋은 런지 자세입니다!"
                                    font_color = (0, 255, 0)
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (70 <= right_knee_angle <= 110) and (110 < left_knee_angle <= 140) and (average_shoulder_x > average_hip_x): 
                                    feedback = "오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (110 < right_knee_angle <= 140) and (0 <= left_knee_angle < 70) and (average_shoulder_x > average_hip_x):
                                    feedback = "왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (110 < right_knee_angle <= 140) and (70 <= left_knee_angle <= 110) and (average_shoulder_x > average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                elif (-0.05 <= left_shoulder[0] - left_hip[0] <= 0.05) and (110 < right_knee_angle <= 140) and (110 < left_knee_angle <= 140) and (average_shoulder_x > average_hip_x): 
                                    feedback = "왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                    
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (0 <= right_knee_angle < 70) and (0 <= left_knee_angle < 70) and (average_shoulder_x > average_hip_x):
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 작아요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (0 <= right_knee_angle < 70) and (70 <= left_knee_angle <= 110) and (average_shoulder_x > average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (0 <= right_knee_angle < 70) and (110 < left_knee_angle <= 140) and (average_shoulder_x > average_hip_x):
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 작아요. 오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)  
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (70 <= right_knee_angle <= 110) and (0 <= left_knee_angle < 70) and (average_shoulder_x > average_hip_x):
                                    feedback = "어깨를 뒤로 당겨주세요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (70 <= right_knee_angle <= 110) and (70 <= left_knee_angle <= 110) and (average_shoulder_x > average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. "
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (70 <= right_knee_angle <= 110) and (110 < left_knee_angle <= 140) and (average_shoulder_x > average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (110 < right_knee_angle <= 140) and (0 <= left_knee_angle < 70) and (average_shoulder_x > average_hip_x):
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 작아요."
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (110 < right_knee_angle <= 140) and (70 <= left_knee_angle <= 110) and (average_shoulder_x > average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 커요. "
                                    font_color = (255, 0, 0)
                                elif (left_shoulder[0] - left_hip[0] > 0.05) and (110 < right_knee_angle <= 140) and (110 < left_knee_angle <= 140) and (average_shoulder_x > average_hip_x): 
                                    feedback = "어깨를 뒤로 당겨주세요. 왼쪽 무릎이 90도 보다 커요. 오른쪽 무릎이 90도 보다 커요."
                                    font_color = (255, 0, 0)
                                else:
                                    feedback = ""
                                    font_color = (255, 255, 255)
                                
                                if lunge_completed:
                                    if (right_knee_angle <= 165) and (left_knee_angle <= 165):
                                        feedback = "앞다리에 힘을 주며 천천히 올라와주세요!"
                                        font_color = (255, 255, 0)  # 피드백 색상을 노란색으로 유지
                                    else:
                                        feedback = ""  # 무릎 각도가 165도를 넘으면 피드백을 비워 표시 중지
                                        font_color = (255, 255, 255)  # 피드백이 없으므로 글씨색은 관계 없음
                                        # 피드백 카운트 초기화 및 새 스쿼트 준비
                                        good_lunge_feedback_counts.clear()  # 모든 피드백 카운트 초기화
                                        date_key = datetime.now().strftime('%Y-%m-%d')
                                        if date_key not in lunge_counts:
                                            lunge_counts[date_key] = 0
                                        lunge_counts[date_key] += 1  # 초기화 횟수 증가
                                        lunge_start = False
                                        lunge_completed = False
                                        lunge_up = False
                                        lunge_down = False
                                
                                if (lunge_down) and (not lunge_up) and not lunge_completed:
                                    if feedback not in exclude_feedbacks_lunge:  # 제외 대상이 아닐 때만 카운트
                                        if "어깨" in feedback:
                                            body_part_counts_lunge["몸통"] += 1
                                        if "오른쪽 무릎" in feedback:
                                            body_part_counts_lunge["오른쪽 무릎"] += 1
                                        if "왼쪽 무릎" in feedback:
                                            body_part_counts_lunge["왼쪽 무릎"] += 1
                                            
                                        if feedback not in lunge_feedback_counts:
                                            lunge_feedback_counts[feedback] = 1
                                        else:
                                            lunge_feedback_counts[feedback] += 1 
                                                        
                                        if feedback == "좋은 런지 자세입니다!":
                                            if feedback not in good_lunge_feedback_counts:
                                                good_lunge_feedback_counts[feedback] = 1
                                            else:
                                                good_lunge_feedback_counts[feedback] += 1
                                                
                                            if good_lunge_feedback_counts[feedback] >= 40:
                                                with app.app_context():
                                                    save_lunge_count()
                                                lunge_completed = True
                                                lunge_up = True  # 스쿼트 상승 시작을 표시
                                                feedback_flag_lunge = False  # 추가 피드백 표시 안 함
                    
            image = draw_text(image, f"런지 횟수: {lunge_counts}", (10, 420), 30, font_color=(17, 255, 127))                                    
            image = draw_text(image, feedback, (10, 20), 30, font_color=font_color)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.putText(frame, f'Squats: {squat_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  

def save_lunge_count():
    with app.app_context():
        today = datetime.now().strftime('%Y-%m-%d')
        record = LungeRecord.query.filter_by(date=today).first()
        if not record:
            record = LungeRecord(date=today, count=0)
            db.session.add(record)
        record.count += 1
        db.session.commit()
        
def plot_graph_lunge():
    global lunge_counts  # 전역 변수 lunge_counts 사용 선언

    # 틀린 자세 통계 그래프를 그리기 위해
    labels_lunge = body_part_counts_lunge.keys()
    values_lunge = body_part_counts_lunge.values()
    total = sum(values_lunge) if values_lunge else 1  # 분모가 0인 경우 방지

    percentages = [100 * (v / total) for v in values_lunge]

    plt.figure(figsize=(16, 8))

    # 첫 번째 그래프: 틀린 자세 기록
    plt.subplot(1, 2, 1)
    plt.bar(labels_lunge, percentages, color='red')
    plt.xlabel('신체부위', fontsize=18)
    plt.ylabel('빈도 (%)', fontsize=18)
    plt.title('틀린 자세 기록', fontsize=20)
    plt.xticks( fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 100)

    # 두 번째 그래프: 런지 총 횟수
    plt.subplot(1, 2, 2)
    dates = list(lunge_counts.keys())
    counts = list(lunge_counts.values())
    plt.bar(dates, counts, color='blue', width=0.1)
    plt.title("런지 총 횟수", fontsize=20)
    plt.ylabel('개수 (개)', fontsize=18)
    plt.ylim(0, max(counts + [3]))  # 최소 값은 3 이상을 유지하도록 설정
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))  # y축의 값을 정수로 설정
    plt.xticks( fontsize=16)
    plt.yticks(fontsize=16)

    img_buf_lunge = io.BytesIO()
    plt.savefig(img_buf_lunge, format='png', transparent=True)
    img_buf_lunge.seek(0)
    plt.close()
    return img_buf_lunge

def generate_frames_plank(target_plank):
    global body_part_counts_plank, plank_counts, app  # 전역 변수 사용 선언
    date_key = datetime.now().strftime('%Y-%m-%d')
    if date_key not in plank_counts:
        plank_counts[date_key] = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        initial_head_height_plank = None  # 초기 머리 높이 저장 변수
        feedback_flag_plank = True  # 초기 피드백 표시 플래그
        plank_start = False  # 플랭크 내려갔는지 상태
        plank_completed = False
        plank_up = False
        plank_down = False
        good_plank_feedback_counts = {}  # 좋은 플랭크 피드백 카운트 딕셔너리 초기화
        plank_detected = False
        plank_start_time = None
        
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                continue

            frame = flip_image(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            # 초기 피드백 설정
            if feedback_flag_plank:
                feedback = "플랭크를 시작해주세요!"
                font_color = (255, 255, 255)
            else:
                feedback = ""
                font_color = (255, 255, 255)
            

            if results.pose_landmarks:
                num_people = len(results.pose_landmarks.landmark)
                if num_people > 0:
                    head = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]
                    shoulders = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]]
                    hips = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]]
                    ankles = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]]
                    left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    knees = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]]
                    left_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    elbows = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]]
                    
                    hip_width = abs(left_hip[0] - right_hip[0])
                    hip_ankle_linearity = calculate_angle([hips[0].x, hips[0].y], [ankles[0].x, ankles[0].y], [ankles[1].x, ankles[1].y])
                    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                    shoulder_hip_angle = calculate_angle([shoulders[0].x, shoulders[0].y], [hips[0].x, hips[0].y], [ankles[0].x, ankles[0].y])
                    shoulder_elbow_angle = calculate_angle([shoulders[0].x, shoulders[0].y], [elbows[0].x, elbows[0].y], [hips[0].x, hips[0].y])
                    knee_angle = calculate_angle([hips[0].x, hips[0].y], [knees[0].x, knees[0].y], [ankles[0].x, ankles[0].y])
                    right_shoulder_elbow_wrist = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    
                    # 플랭크 시작 조건
                    if (left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1]) and \
                            (left_shoulder[1] < 0.5 and right_shoulder[1] < 0.5):
                        if shoulder_width < 0.1 and hip_width < 0.1:
                            feedback_flag_plank = False  # 초기 피드백 제거
                            
                    if (not plank_start) and (not plank_completed):
                        initial_head_height_plank = head[1]  # 플랭크 시작 시 머리 높이 측정
                        plank_start = True  # 플랭크 시작 상태 활성화
                        plank_down = True
                        
                    if (plank_start) and (head[1] - initial_head_height_plank > 0.2) and (right_elbow[1] > 0.7):
                        # print("팔꿈치: ", right_shoulder_elbow_wrist)
                        if (shoulder_hip_angle < 160) and (right_shoulder[1] > right_hip[1]) and (knee_angle < 160) and (right_shoulder_elbow_wrist >= 70):
                            feedback = "엉덩이를 내려주세요. 무릎을 펴주세요."                                       
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 160) and (right_shoulder[1] > right_hip[1]) and (160 <= knee_angle <= 200) and (right_shoulder_elbow_wrist >= 70):
                            feedback = "엉덩이를 내려주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 160) and (right_hip[1] > right_shoulder[1]) and (knee_angle < 160) and (right_shoulder_elbow_wrist >= 70):
                            feedback = "엉덩이를 올려주세요. 무릎을 펴주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 160) and (right_hip[1] > right_shoulder[1]) and (160 <= knee_angle <= 200) and (right_shoulder_elbow_wrist >= 70):
                            feedback = "엉덩이를 올려주세요."
                            font_color = (0, 0, 255)
                        elif (160 <= shoulder_hip_angle <= 200) and (knee_angle < 160) and (right_shoulder_elbow_wrist >= 70):
                            feedback = "무릎을 펴주세요."
                            font_color = (0, 0, 255)
                        elif (160 <= shoulder_hip_angle <= 200) and (160 <= knee_angle <= 200) and (70 <= right_shoulder_elbow_wrist <= 150):
                            feedback = "좋은 플랭크 자세입니다!"
                            font_color = (0, 255, )
                            
                        elif (shoulder_hip_angle < 160) and (right_shoulder[1] > right_hip[1]) and (knee_angle < 160) and (right_shoulder_elbow_wrist < 70):
                            feedback = "엉덩이를 내려주세요. 무릎을 펴주세요. 어깨를 올려주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 160) and (right_shoulder[1] > right_hip[1]) and (160 <= knee_angle <= 200) and (right_shoulder_elbow_wrist < 70):
                            feedback = "엉덩이를 내려주세요. 어깨를 올려주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 160) and (right_hip[1] > right_shoulder[1]) and (knee_angle < 160) and (right_shoulder_elbow_wrist < 70):
                            feedback = "엉덩이를 올려주세요. 무릎을 펴주세요. 어깨를 올려주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 160) and (right_hip[1] > right_shoulder[1]) and (160 <= knee_angle <= 200) and (right_shoulder_elbow_wrist < 70):
                            feedback = "엉덩이를 올려주세요. 어깨를 올려주세요."
                            font_color = (0, 0, 255)
                        elif (160 <= shoulder_hip_angle <= 200) and (knee_angle < 160) and (right_shoulder_elbow_wrist < 70):
                            feedback = "무릎을 펴주세요. 어깨를 올려주세요."
                            font_color = (0, 0, 255)
                        elif (160 <= shoulder_hip_angle <= 200) and (160 <= knee_angle <= 200) and (right_shoulder_elbow_wrist < 70):
                            feedback = " 어깨를 올려주세요."
                            font_color = (0, 0, 255)
                        else:
                            feedback = ""
                        
                        if (160 <= shoulder_hip_angle <= 200) and (160 <= knee_angle <= 200) and (70 <= right_shoulder_elbow_wrist <= 150):
                            if plank_start_time is None:
                                plank_start_time = time.time()
                            current_time = time.time() - plank_start_time
                            feedback = f"플랭크 유지 시간: {int(current_time+1)}초"
                            font_color = (0, 255, 0)
                            if current_time >= 5:
                                plank_completed = True
                                
                        if feedback not in exclude_feedbacks_plank:  # 제외 대상이 아닐 때만 카운트
                            if "어깨" in feedback:
                                body_part_counts_plank["어깨"] += 1
                            if "엉덩이" in feedback:
                                body_part_counts_plank["엉덩이"] += 1
                            if "무릎" in feedback:
                                body_part_counts_plank["무릎"] += 1
                                
                            if feedback not in plank_feedback_counts:
                                plank_feedback_counts[feedback] = 1
                            else:
                                plank_feedback_counts[feedback] += 1       
                                 
                        if plank_completed:
                            feedback = ""
                            good_plank_feedback_counts.clear()  # 모든 피드백 카운트 초기화
                            date_key = datetime.now().strftime('%Y-%m-%d')
                            plank_counts[date_key] += 1
                            save_plank_count()  # 데이터베이스에 플랭크 횟수 저장
                            if plank_counts[date_key] >= target_plank:
                                threading.Thread(target=play_sound, args=(sound_file,)).start()
                                print("목표 플랭크 횟수와 시간에 도달했습니다.")
                                break
                            
                            plank_completed = False
                            plank_start_time = None

            # 플랭크 횟수 비교
            # if plank_counts[date_key] == target_plank:
            #     # 사운드 재생
            #     threading.Thread(target=play_sound, args=(sound_file,)).start()
            #     print("목표 플랭크 횟수와 시간에 도달했습니다.")
            #     break
            
            # 텍스트 그리기                        
            image = draw_text(image, feedback, (10, 20), 30, font_color=font_color)
            image = draw_text(image, f"플랭크 횟수: {plank_counts}", (10, 420), 30, font_color=(17, 255, 127))                                    
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.putText(frame, f'Squats: {squat_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def save_plank_count():
    with app.app_context():
        today = datetime.now().strftime('%Y-%m-%d')
        record = PlankRecord.query.filter_by(date=today).first()
        if not record:
            record = PlankRecord(date=today, count=0)
            db.session.add(record)
        record.count += 1
        db.session.commit()
        
def plot_graph_plank():
    global plank_counts  # 전역 변수 plank_counts 사용 선언

    # 틀린 자세 통계 그래프를 그리기 위해
    labels_plank = body_part_counts_plank.keys()
    values_plank = body_part_counts_plank.values()
    total = sum(values_plank) if values_plank else 1  # 분모가 0인 경우 방지

    percentages = [100 * (v / total) for v in values_plank]

    plt.figure(figsize=(18, 8))  # 그래프 크기 조정

    # 첫 번째 그래프: 틀린 자세 기록
    plt.subplot(1, 3, 1)
    plt.bar(labels_plank, percentages, color='red')
    plt.xlabel('신체부위', fontsize=16)
    plt.ylabel('빈도 (%)', rotation=90, fontsize=16)
    plt.title('틀린 자세 기록', fontsize=18)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 100)

    # 두 번째 그래프: 플랭크 총 횟수
    plt.subplot(1, 3, 2)
    dates = list(plank_counts.keys())
    counts = list(plank_counts.values())
    plt.bar(dates, counts, color='blue', width=0.1)
    plt.title("플랭크 총 횟수", fontsize=18)
    plt.ylabel('개수 (개)', fontsize=16)
    plt.ylim(0, max(counts + [3]))  # 최소 값은 3 이상을 유지하도록 설정
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)

    # 세 번째 그래프: 플랭크 유지시간
    plt.subplot(1, 3, 3)
    dates = list(plank_counts.keys())
    counts = [count * 5 for count in plank_counts.values()]  # 각 횟수에 5를 곱하여 시간을 계산
    plt.bar(dates, counts, color='green', width=0.1)
    plt.title("플랭크 유지시간", fontsize=18)
    plt.ylabel('시간 (초)', fontsize=16)
    plt.ylim(0, max(counts + [15]))  # 최소 값은 15 이상을 유지하도록 설정
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)

    img_buf_plank = io.BytesIO()
    plt.savefig(img_buf_plank, format='png', transparent=True)
    img_buf_plank.seek(0)
    plt.close()
    return img_buf_plank


def generate_frames_dolphin(target_dolphin):
    global body_part_counts_dolphin, dolphin_counts, app  # 전역 변수 사용 선언
    date_key = datetime.now().strftime('%Y-%m-%d')
    if date_key not in dolphin_counts:
        dolphin_counts[date_key] = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        
        initial_head_height_dolphin = None  # 초기 머리 높이 저장 변수
        feedback_flag_dolphin = True  # 초기 피드백 표시 플래그
        dolphin_start = False  # 돌고래 시작 상태
        dolphin_completed = False
        dolphin_up = False
        dolphin_down = False
        good_dolphin_feedback_counts = {}  # 좋은 돌고래 자세 피드백 카운트 딕셔너리 초기화
        dolphin_detected = False
        dolphin_start_time = None
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                time.sleep(0.1)  # 웹캠에서 프레임을 가져오지 못할 때 잠시 대기
                continue
            
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)            
            
            # 초기 피드백 설정
            if feedback_flag_dolphin:
                feedback = "어깨를 귀 옆에 붙이지 않은 상태로 돌고래 자세를 시작해주세요!"
                font_color = (255, 255, 255)
            else:
                feedback = ""
                font_color = (255, 255, 255)
            

            if results.pose_landmarks:
                num_people = len(results.pose_landmarks.landmark)
                if num_people > 0:
                    head = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]
                    shoulders = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]]
                    hips = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]]
                    ankles = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]]
                    left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    left_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    right_hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    left_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    right_elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    knees = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]]
                    left_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    right_knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    left_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    right_ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    elbows = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value]]
                    
                    hip_width = abs(left_hip[0] - right_hip[0])
                    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                    shoulder_hip_angle = calculate_angle([shoulders[0].x, shoulders[0].y], [hips[0].x, hips[0].y], [ankles[0].x, ankles[0].y])
                    hip_shoulder_elbow = calculate_angle([hips[0].x, hips[0].y], [shoulders[0].x, shoulders[0].y], [elbows[0].x, elbows[0].y])
                    knee_angle = calculate_angle([hips[0].x, hips[0].y], [knees[0].x, knees[0].y], [ankles[0].x, ankles[0].y])
                    
                    # 돌고래 시작 조건
                    if (left_shoulder[1] < left_hip[1] and right_shoulder[1] < right_hip[1]) and \
                            (left_shoulder[1] < 0.5 and right_shoulder[1] < 0.5):
                        if shoulder_width < 0.1 and hip_width < 0.1:
                            feedback_flag = False  # 초기 피드백 제거
                            
                    if (not dolphin_start) and (not dolphin_completed):
                        initial_head_height = head[1]  # 돌고래 시작 시 머리 높이 측정
                        dolphin_start = True  # 돌고래 시작 상태 활성화
                        dolphin_down = True
                        
                    if (dolphin_start) and (head[1] - initial_head_height > 0.2) and (right_wrist[1] > 0.5):
                        print("엉덩이 고관절: ", shoulder_hip_angle)
                        print("무릎: ", knee_angle)
                        print("어깨 팔꿈치: ", hip_shoulder_elbow)
                        if (70 <= shoulder_hip_angle <= 90) and (knee_angle < 160) and (120 <= hip_shoulder_elbow <= 180):
                            feedback = "무릎을 펴주세요."
                            font_color = (0, 0, 255)
                        elif (70 <= shoulder_hip_angle <= 90) and (160 <= knee_angle <= 200) and (120 <= hip_shoulder_elbow <= 180):
                            feedback = "좋은 돌고래 자세 자세입니다!"
                            font_color = (0, 255, 0)
                        elif (shoulder_hip_angle > 90) and (knee_angle < 160) and (120 <= hip_shoulder_elbow <= 180):
                            feedback = "엉덩이 고관절 각도를 좁혀주세요. 무릎을 펴주세요"
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle > 90) and (160 <= knee_angle <= 200) and (120 <= hip_shoulder_elbow <= 180):
                            feedback = "엉덩이 고관절 각도를 줄여주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 70) and (knee_angle < 160) and (120 <= hip_shoulder_elbow <= 180):
                            feedback = "엉덩이 고관절 각도를 넓혀주세요. 무릎을 펴주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 70) and (160 <= knee_angle <= 200) and (120 <= hip_shoulder_elbow <= 180):
                            feedback = "엉덩이 고관절 각도를 넓혀주세요."
                            font_color = (0, 0, 255)
                            
                        elif (70 <= shoulder_hip_angle <= 90) and (knee_angle < 160) and (hip_shoulder_elbow < 120):
                            feedback = "무릎과 팔꿈치를 펴주세요."
                            font_color = (0, 0, 255)
                        elif (70 <= shoulder_hip_angle <= 90) and (160 <= knee_angle <= 200) and (hip_shoulder_elbow < 120):
                            feedback = " 팔꿈치를 펴주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle > 90) and (knee_angle < 160) and (hip_shoulder_elbow < 120):
                            feedback = "엉덩이 고관절 각도를 좁혀주세요. 무릎과 팔꿈치를 펴주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle > 90) and (160 <= knee_angle <= 200) and (hip_shoulder_elbow < 120):
                            feedback = "엉덩이 고관절 각도를 줄여주세요. 팔꿈치를 펴주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 70) and (knee_angle < 160) and (hip_shoulder_elbow < 120):
                            feedback = "엉덩이 고관절 각도를 넓혀주세요. 무릎과 팔꿈치를 펴주세요."
                            font_color = (0, 0, 255)
                        elif (shoulder_hip_angle < 70) and (160 <= knee_angle <= 200) and (hip_shoulder_elbow < 120):
                            feedback = "엉덩이 고관절 각도를 넓혀주세요. 팔꿈치를 펴주세요."
                            font_color = (0, 0, 255)
                        else:
                            feedback = ""
                        
                        if (70 <= shoulder_hip_angle <= 90) and (160 <= knee_angle <= 200) and (120 <= hip_shoulder_elbow <= 180):
                            if dolphin_start_time is None:
                                dolphin_start_time = time.time()
                            current_time = time.time() - dolphin_start_time
                            feedback = f"돌고래 자세 유지 시간: {int(current_time)}초"
                            font_color = (0, 255, 0)
                            if current_time >= 5:
                                dolphin_completed = True
                        
                        if feedback not in exclude_feedbacks_dolphin:  # 제외 대상이 아닐 때만 카운트
                            if "엉덩이 고관절" in feedback:
                                body_part_counts_dolphin["엉덩이 고관절"] += 1
                            if "무릎" in feedback:
                                body_part_counts_dolphin["무릎"] += 1
                            if "필꿈치" in feedback:
                                body_part_counts_dolphin["팔꿈치"] += 1
                                
                            if feedback not in dolphin_feedback_counts:
                                dolphin_feedback_counts[feedback] = 1
                            else:
                                dolphin_feedback_counts[feedback] += 1
                                            
                    if dolphin_completed:
                        feedback = ""
                        good_dolphin_feedback_counts.clear()  # 모든 피드백 카운트 초기화
                        date_key = datetime.now().strftime('%Y-%m-%d')
                        dolphin_counts[date_key] += 1
                        save_dolphin_count()  # 데이터베이스에 플랭크 횟수 저장
                        if dolphin_counts[date_key] >= target_dolphin:
                            threading.Thread(target=play_sound, args=(sound_file,)).start()
                            print("목표 돌고래자세 횟수와 시간에 도달했습니다.")
                            break
                            
                        dolphin_completed = False
                        dolphin_start_time = None
                                            
            
            # 텍스트 그리기                        
            image = draw_text(image, feedback, (10, 20), 30, font_color=font_color)
            image = draw_text(image, f"돌고래 자세 횟수: {dolphin_counts}", (10, 420), 30, font_color=(17, 255, 127))                                    
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.putText(frame, f'Squats: {squat_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_dolphin_count():
    with app.app_context():
        today = datetime.now().strftime('%Y-%m-%d')
        record = DolphinRecord.query.filter_by(date=today).first()
        if not record:
            record = DolphinRecord(date=today, count=0)
            db.session.add(record)
        record.count += 1
        db.session.commit()
                
            
def plot_graph_dolphin():
    global dolphin_counts  # 전역 변수 dolphin_count 사용 선언
    
    # 틀린 자세 통계 그래프를 그리기 위해
    labels_dolphin = body_part_counts_dolphin.keys()
    values_dolphin = body_part_counts_dolphin.values()
    total = sum(values_dolphin) if values_dolphin else 1  # 분모가 0인 경우 방지

    percentages = [100 * (v / total) for v in values_dolphin]
    
    plt.figure(figsize=(12, 6))
    
    # 첫 번째 그래프: 틀린 자세 기록
    plt.subplot(1, 3, 1)
    plt.bar(labels_dolphin, percentages, color='red')
    plt.xlabel('신체부위', fontsize=16)
    plt.ylabel('빈도 (%)', rotation=90, fontsize=16)
    plt.title('틀린 자세 기록', fontsize=18)
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 100)
    
    plt.subplot(1, 3, 2)
    dates = list(dolphin_counts.keys())
    counts = list(dolphin_counts.values())
    plt.bar(dates, counts, color='blue', width=0.1)
    plt.title("돌고래자세 총 횟수", fontsize=18)
    plt.ylabel('개수 (개)', fontsize=16)
    plt.ylim(0, max(counts + [3]))  # 최소 값은 3 이상을 유지하도록 설정
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(1, 3, 3)
    dates = list(dolphin_counts.keys())
    counts = [count * 5 for count in dolphin_counts.values()]  # 각 횟수에 5를 곱하여 시간을 계산
    plt.bar(dates, counts, color='green', width=0.1)
    plt.title("돌고래자세 유지시간", fontsize=18)
    plt.ylabel('시간 (초)', fontsize=16)
    plt.ylim(0, max(counts + [15]))  # 최소 값은 15 이상을 유지하도록 설정
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)

    img_buf_dolphin = io.BytesIO()
    plt.savefig(img_buf_dolphin, format='png', transparent=True)
    img_buf_dolphin.seek(0)
    plt.close()
    return img_buf_dolphin  
            
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames_squat(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_lunge')
def video_feed_lunge():
    """Video streaming route for lunge. Put this in the src attribute of an img tag."""
    return Response(generate_frames_lunge(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_plank_10sec')
def video_feed_plank_10sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_plank_20sec')
def video_feed_plank_20sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(4),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_plank_30sec')
def video_feed_plank_30sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(6),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_plank_40sec')
def video_feed_plank_40sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(8),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_plank_50sec')
def video_feed_plank_50sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(10),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_plank_60sec')
def video_feed_plank_60sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(12),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/video_feed_plank_90sec')
def video_feed_plank_90sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(18),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_plank_180sec')
def video_feed_plank_180sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(36),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_plank_300sec')
def video_feed_plank_300sec():
    """Video streaming route for plank. Put this in the src attribute of an img tag."""
    return Response(generate_frames_plank(60),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_dolphin_10sec')
def video_feed_dolphin_10sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_dolphin_20sec')
def video_feed_dolphin_20sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(4),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_dolphin_30sec')
def video_feed_dolphin_30sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(6),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_dolphin_40sec')
def video_feed_dolphin_40sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(8),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_dolphin_50sec')
def video_feed_dolphin_50sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(10),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_dolphin_60sec')
def video_feed_dolphin_60sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(12),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/video_feed_dolphin_90sec')
def video_feed_dolphin_90sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(18),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_dolphin_180sec')
def video_feed_dolphin_180sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(36),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_dolphin_300sec')
def video_feed_dolphin_300sec():
    """Video streaming route for dolphin. Put this in the src attribute of an img tag."""
    return Response(generate_frames_dolphin(60),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
           
@app.route('/calendar')
def calendar_page():
    return render_template('calendar.html')

@app.route('/get_all_squat_data', methods=['GET'])
def get_all_squat_data():
    records = SquatRecord.query.all()
    data = {record.date: record.count for record in records}
    return jsonify(data)

@app.route('/get_squat_data', methods=['GET'])
def get_squat_data():
    date = request.args.get('date', '')
    squat_count = squat_counts.get(date, 0)
    return jsonify({'date': date, 'squat_count': squat_count})

    
@app.route('/get_all_lunge_data', methods=['GET'])
def get_all_lunge_data():
    records = LungeRecord.query.all()
    data = {record.date: record.count for record in records}
    return jsonify(data)

@app.route('/get_lunge_data', methods=['GET'])
def get_lunge_data():
    date = request.args.get('date', '')
    lunge_count = lunge_counts.get(date, 0)
    return jsonify({'date': date, 'lunge_count': lunge_count})

@app.route('/get_all_plank_data', methods=['GET'])
def get_all_plank_data():
    records = PlankRecord.query.all()
    data = {record.date: record.count for record in records}
    return jsonify(data)

@app.route('/get_plank_data', methods=['GET'])
def get_plank_data():
    date = request.args.get('date', '')
    plank_count = plank_counts.get(date, 0)
    return jsonify({'date': date, 'plank_count': plank_count})

@app.route('/get_all_dolphin_data', methods=['GET'])
def get_all_dolphin_data():
    records = DolphinRecord.query.all()
    data = {record.date: record.count for record in records}
    return jsonify(data)

@app.route('/get_dolphin_data', methods=['GET'])
def get_dolphin_data():
    date = request.args.get('date', '')
    dolphin_count = dolphin_counts.get(date, 0)
    return jsonify({'date': date, 'dolphin_count': dolphin_count})

@app.route('/get_daily_stats', methods=['GET'])
def get_daily_stats():
    result = {}
    squat_records = SquatRecord.query.all()
    lunge_records = LungeRecord.query.all()
    plank_records = PlankRecord.query.all()
    dolphin_records = DolphinRecord.query.all()
    
    for record in squat_records:
        if record.date not in result:
            result[record.date] = {'squat': 0, 'lunge': 0, 'plank': 0, 'dolphin': 0}
        result[record.date]['squat'] += record.count

    for record in lunge_records:
        if record.date not in result:
            result[record.date] = {'squat': 0, 'lunge': 0, 'plank': 0, 'dolphin': 0}
        result[record.date]['lunge'] += record.count

    for record in plank_records:
        if record.date not in result:
            result[record.date] = {'squat': 0, 'lunge': 0, 'plank': 0, 'dolphin': 0}
        result[record.date]['plank'] += record.count
        
    for record in dolphin_records:
        if record.date not in result:
            result[record.date] = {'squat': 0, 'lunge': 0, 'plank': 0, 'dolphin': 0}
        result[record.date]['dolphin'] += record.count

    return jsonify(result)


@app.route('/squat')
def squat_video():
    return send_file("D:/4-1/jongsul/squat.mp4", mimetype='video/mp4')

@app.route('/plank')
def plank_video():
    return send_file("D:/4-1/jongsul/plank.mp4", mimetype='video/mp4')

@app.route('/lunge')
def lunge_video():
    return send_file("D:/4-1/jongsul/lunge.mp4", mimetype='video/mp4')

@app.route('/dolphin')
def dolphin_video():
    return send_file("D:/4-1/jongsul/dolphin.mp4", mimetype='video/mp4')

@app.route('/records')
def records():
    return render_template('records.html')

@app.route('/records_lunge')
def records_lunge():
    return render_template('records_lunge.html')

@app.route('/records_plank')
def records_plank():
    return render_template('records_plank.html')

@app.route('/records_plank_10sec')
def records_plank_10sec():
    return render_template('records_plank_10sec.html')

@app.route('/records_plank_20sec')
def records_plank_20sec():
    return render_template('records_plank_20sec.html')

@app.route('/records_plank_30sec')
def records_plank_30sec():
    return render_template('records_plank_30sec.html')

@app.route('/records_plank_40sec')
def records_plank_40sec():
    return render_template('records_plank_40sec.html')

@app.route('/records_plank_50sec')
def records_plank_50sec():
    return render_template('records_plank_50sec.html')

@app.route('/records_plank_60sec')
def records_plank_60sec():
    return render_template('records_plank_60sec.html')

@app.route('/records_plank_90sec')
def records_plank_90sec():
    return render_template('records_plank_90sec.html')

@app.route('/records_plank_180sec')
def records_plank_180sec():
    return render_template('records_plank_180sec.html')

@app.route('/records_plank_300sec')
def records_plank_300sec():
    return render_template('records_plank_300sec.html')

@app.route('/records_dolphin')
def records_dolphin():
    return render_template('records_dolphin.html')

@app.route('/records_dolphin_10sec')
def records_dolphin_10sec():
    return render_template('records_dolphin_10sec.html')

@app.route('/records_dolphin_20sec')
def records_dolphin_20sec():
    return render_template('records_dolphin_20sec.html')

@app.route('/records_dolphin_30sec')
def records_dolphin_30sec():
    return render_template('records_dolphin_30sec.html')

@app.route('/records_dolphin_40sec')
def records_dolphin_40sec():
    return render_template('records_dolphin_40sec.html')

@app.route('/records_dolphin_50sec')
def records_dolphin_50sec():
    return render_template('records_dolphin_50sec.html')

@app.route('/records_dolphin_60sec')
def records_dolphin_60sec():
    return render_template('records_dolphin_60sec.html')

@app.route('/records_dolphin_90sec')
def records_dolphin_90sec():
    return render_template('records_dolphin_90sec.html')

@app.route('/records_dolphin180sec')
def records_dolphin_180sec():
    return render_template('records_dolphin_180sec.html')

@app.route('/records_dolphin_300sec')
def records_dolphin_300sec():
    return render_template('records_dolphin_300sec.html')
        
@app.route('/graph_image')
def graph_image():
    img_buf = plot_graph_squat()
    return send_file(img_buf, mimetype='image/png')

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/graph_image_lunge')
def graph_image_lunge():
    img_buf_lunge = plot_graph_lunge()
    return send_file(img_buf_lunge, mimetype='image/png')

@app.route('/graph_lunge')
def graph_lunge():
    return render_template('graph_lunge.html')

@app.route('/graph_image_plank')
def graph_image_plank():
    img_buf_plank = plot_graph_plank()
    return send_file(img_buf_plank, mimetype='image/png')

@app.route('/graph_plank')
def graph_plank():
    return render_template('graph_plank.html')

@app.route('/howmany_plank')
def howMany_plank():
    return render_template('howmany_plank.html')

@app.route('/graph_image_dolphin')
def graph_image_dolphin():
    img_buf_dolphin = plot_graph_dolphin()
    return send_file(img_buf_dolphin, mimetype='image/png')

@app.route('/graph_dolphin')
def graph_dolphin():
    return render_template('graph_dolphin.html')

@app.route('/howmany_dolphin')
def howMany_dolphin():
    return render_template('howmany_dolphin.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='127.0.0.1', threaded=True)

# @app.route('/squat_data')
# def squat_data():
#     return jsonify({'count': squat_count})  # `squat_count`는 전역 변수로 관리