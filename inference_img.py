import os
import cv2
import torch
import argparse
import numpy as np
from train_log.RIFE_HDv3 import Model  # ✅ 정확한 경로로 수정

# =====================================
# ✅ 명령행 인자 처리
# =====================================
parser = argparse.ArgumentParser(description="RIFE Frame Interpolation")
parser.add_argument("--img", nargs=2, required=True, help="입력 이미지 2장 (절대경로)")
args = parser.parse_args()

# =====================================
# ✅ 경로 및 모델 초기화
# =====================================
MODEL_PATH = "/content/Practical-RIFE/train_log/flownet.pkl"
OUTPUT_DIR = "/content/Practical-RIFE/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# ✅ 입력 이미지 로드
# =====================================
I0 = cv2.imread(args.img[0])
I1 = cv2.imread(args.img[1])

if I0 is None or I1 is None:
    raise FileNotFoundError(f"❌ 입력 이미지 로드 실패:\n{args.img[0]}\n{args.img[1]}")

# RGB 변환
I0 = cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)

# float32 변환 및 정규화
I0 = torch.from_numpy(I0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
I1 = torch.from_numpy(I1).permute(2, 0, 1).unsqueeze(0).float() / 255.0

I0 = I0.to(device)
I1 = I1.to(device)

# =====================================
# ✅ 모델 로드 및 추론
# =====================================
model = Model()
model.load_model(MODEL_PATH)
model.eval()
model.device()

with torch.no_grad():
    pred = model.inference(I0, I1)

# =====================================
# ✅ 결과 저장
# =====================================
out_img = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
out_path = os.path.join(OUTPUT_DIR, "img1.png")
cv2.imwrite(out_path, out_img)

torch.cuda.empty_cache()

print(f"✅ 중간 프레임 저장 완료 → {out_path}")
