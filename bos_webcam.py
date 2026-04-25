import cv2
import numpy as np
import os

# Camera and visualization settings
CAMERA_INDEX = 1
SENSITIVITY = 25.0
WINDOW_NAME = "BOS Visualizer (Original | BOS)"

# Farneback optical flow parameters (balanced for real-time webcam use)
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def get_bos_tensor(reference_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    """AI 모델 학습/추론을 위한 원본 2채널(dx, dy) float32 텐서(flow)를 계산합니다."""
    flow = cv2.calcOpticalFlowFarneback(
        reference_gray,
        current_gray,
        None,
        FARNEBACK_PARAMS["pyr_scale"],
        FARNEBACK_PARAMS["levels"],
        FARNEBACK_PARAMS["winsize"],
        FARNEBACK_PARAMS["iterations"],
        FARNEBACK_PARAMS["poly_n"],
        FARNEBACK_PARAMS["poly_sigma"],
        FARNEBACK_PARAMS["flags"],
    )
    return flow


def render_bos_frame(flow: np.ndarray) -> np.ndarray:
    """계산된 flow 텐서를 받아 사람이 보기 좋은 HSV 컬러 영상으로 변환합니다."""
    # X, Y 이동량을 크기(magnitude)와 각도(angle)로 변환
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8((angle * 180.0 / np.pi) / 2.0)  # 방향 -> 색상(Hue)
    hsv[..., 1] = 255  # 채도 최대
    hsv[..., 2] = np.clip(magnitude * SENSITIVITY, 0, 255).astype(np.uint8)  # 크기 -> 밝기(Value)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main() -> int:
    # 데이터 저장을 위한 폴더 자동 생성 및 카운터 초기화
    os.makedirs("dataset", exist_ok=True)
    frame_count = 0

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Webcam open failed. Check camera connection/index.")
        return 1

    ok, first_frame = cap.read()
    if not ok:
        print("[ERROR] Failed to read first frame from webcam.")
        cap.release()
        return 1

    reference_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    print("[INFO] BOS started. Press 'r' to reset reference, 's' to save data, 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame read failed. Exiting loop.")
                break

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. 광학 흐름(텐서) 연산은 딱 한 번만 수행! (속도 저하 방지)
            bos_tensor = get_bos_tensor(reference_gray, current_gray)
            
            # 2. 계산된 텐서를 넘겨주어 화면에 띄울 시각화 이미지 생성
            bos_frame = render_bos_frame(bos_tensor)

            combined = np.hstack((frame, bos_frame))
            cv2.imshow(WINDOW_NAME, combined)

            # 3. 키 입력 이벤트 (루프 당 1번만 처리)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            elif key == ord("r"):
                reference_gray = current_gray.copy()
                print("[INFO] Reference frame reset.")
            elif key == ord("s"):
                # 4. 's'를 누르면 현재 텐서를 .npy 파일로 저장
                save_path = f"dataset/bos_flow_{frame_count:04d}.npy"
                np.save(save_path, bos_tensor)
                print(f"[SAVE] 데이터 저장 완료: {save_path}")
                frame_count += 1

            # 5. 배경을 미세하게 지속 업데이트 (조명 변화 및 노이즈 적응)
            reference_gray = cv2.addWeighted(reference_gray, 0.99, current_gray, 0.01, 0)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())