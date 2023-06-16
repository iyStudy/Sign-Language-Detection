import cv2
import mediapipe as mp
import numpy as np

try:

    # MediaPipeの手検出モデルを初期化
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()

    # Webカメラからの映像を取得
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 状態の初期化
    state = 0

    # 映像のフレームごとに処理
    while cap.isOpened():
        ret, frame = cap.read()
        

        # フレームをMediaPipeで処理
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # 検出された手のランドマークを描画
        if results.multi_hand_landmarks:
            # 手のランドマーク取得
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 手の位置取得
            if len(results.multi_hand_landmarks) == 2:
                right_hand = results.multi_hand_landmarks[0]
                left_hand = results.multi_hand_landmarks[1]
                
                # 両手の人差し指の位置を取得
                right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # 状態0: 両手が交差しているか確認
                if state == 0 and abs(right_index.x - left_index.x) < 0.1:
                    state = 1
                    
                # 状態1: 両手が開かれているか確認
                elif state == 1 and abs(right_index.x - left_index.x) > 0.3:
                    state = 2
                    
                # 状態2: 人差し指が折り曲げられているか確認
                elif state == 2:
                    right_pip = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    left_pip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

                    if abs(right_index.y - right_pip.y) < 0.05 and abs(left_index.y - left_pip.y) < 0.05:
                        print("こんにちは")
                        state = 0  # 状態をリセット

        # フレームを表示
        cv2.imshow('MediaPipe Hands', frame)
        
        # 'q'キーで終了
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"エラーが発生しました: {e}")
finally:
    # カメラとウィンドウを解放
    cap.release()
    cv2.destroyAllWindows()
