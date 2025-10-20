%%writefile /content/Practical-RIFE/train_log/inference_img.py
import os
import cv2
import torch
import numpy as np
import argparse
import sys

# =========================================================
# ⚙️ 경로 설정
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       # /content/Practical-RIFE/train_log
PROJECT_DIR = os.path.dirname(CURRENT_DIR)                     # /content/Practical-RIFE

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)  # train_log 안에서 import 가능하게

from RIFE_HDv3 import Model   # ✅ 같은 폴더 안에서 가져오기

INPUT_DIR = '/content/Practical-RIFE/input_frames'   # 입력 프레임 폴더
OUTPUT_DIR = '/content/Practical-RIFE/output'        # 결과 저장 폴더
MODEL_PATH = os.path.join(CURRENT_DIR, 'flownet.pkl')

# =========================================================
# 🧠 모델 초기화
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_model(MODEL_PATH, -1)
model.eval()
model.device()

# =========================================================
# 📥 입력 인자 파싱
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--img', nargs=2, required=True, help='두 개의 입력 프레임 경로')
args = parser.parse_args()

# =========================================================
# 🖼️ 입력 이미지 로드
# =========================================================
img1_path, img2_path = args.img
I0 = cv2.imread(img1_path)
I1 = cv2.imread(img2_path)

if I0 is None or I1 is None:
    raise FileNotFoundError(f"❌ 입력 이미지 로드 실패:\n - {img1_path}\n - {img2_path}")

h, w, _ = I0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32

I0 = cv2.resize(I0, (pw, ph))
I1 = cv2.resize(I1, (pw, ph))

I0 = torch.from_numpy(np.transpose(I0, (2, 0, 1))).unsqueeze(0).float() / 255.
I1 = torch.from_numpy(np.transpose(I1, (2, 0, 1))).unsqueeze(0).float() / 255.
I0 = I0.to(device)
I1 = I1.to(device)

# =========================================================
# 🚀 중간 프레임 예측
# =========================================================
with torch.no_grad():
    mid = model.inference(I0, I1, 0.5)
    mid = (mid[0] * 255.0).byte().cpu().numpy().transpose(1, 2, 0)
    mid = mid[:h, :w, ::-1]  # RGB → BGR

# =========================================================
# 💾 결과 저장
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, 'img1.png')
cv2.imwrite(out_path, mid)
print(f"✅ RIFE 보간 완료 → {out_path}")
