import cv2
import mediapipe as mp

# Mediapipe의 포즈 모델을 초기화합니다.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 웹캠으로부터 비디오 캡처를 시작합니다.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("비디오 읽기를 실패했습니다. 종료합니다.")
        break

    # 이미지의 색상을 BGR에서 RGB로 변환합니다.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # 이미지에서 포즈를 검출합니다.
    results = pose.process(image)

    # 포즈 키포인트를 이미지에 그립니다.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('BlazePose', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료합니다.
        break

cap.release()
cv2.destroyAllWindows()