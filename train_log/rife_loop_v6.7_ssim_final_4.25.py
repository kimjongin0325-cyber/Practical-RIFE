# =====================================================
# ✅ [RIFE v4.25 - SSIM Corrected Patch64 Final v6.7]
# - 입력: /content/Practical-RIFE/output    (v3.8 결과)
# - 출력: /content/Practical-RIFE/output2   (최종 결과)
# - 4.25 가중치 전용 / SSIM 로깅 / NaN 감시 / 패딩 안정화
# =====================================================

import os, glob, torch, shutil, re
import numpy as np, cv2

# -------------------- [1] 경로 설정 --------------------
BASE_DIR = "/content/Practical-RIFE"
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)

opt = {
    "threads": min(os.cpu_count(), 8),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_dir":  f"{BASE_DIR}/output",   # ✅ v3.1 FastSafe가 만든 결과
    "output_dir": f"{BASE_DIR}/output2",  # ✅ 최종 출력
    "model_path": f"{BASE_DIR}/train_log/flownet_v425.pkl",  # ← 4.25 가중치여야 함
}

os.makedirs(opt["input_dir"], exist_ok=True)
os.makedirs(opt["output_dir"], exist_ok=True)

# -------------------- [2] 모델 로드 --------------------
import sys, os
BASE_DIR = "/content/Practical-RIFE"  # ✅ 실제 프로젝트 루트
os.chdir(BASE_DIR)                    # ✅ 작업 디렉토리 이동
sys.path.append(BASE_DIR)             # ✅ train_log 모듈 인식 보장

from train_log.rf425 import Model  # v4.25도 동일 클래스명 사용
device = torch.device(opt["device"])

model = Model()
model.load_model(os.path.dirname(opt["model_path"]), -1)
model.eval()
model.device()

# SSIM (선택적): model/__init__.py에 SSIM 있으면 사용
try:
    from model import SSIM
    ssim_loss = SSIM(window_size=11, size_average=True).to(device)
    print("✅ SSIM 준비 완료 (DSSIM→SSIM 변환)")
except Exception as e:
    ssim_loss = None
    print(f"⚠️ SSIM 비활성화: {e}")

print(f"✅ 모델 로드 완료: {opt['model_path']}")
print(f"📂 입력 폴더: {opt['input_dir']}")
print(f"📂 출력 폴더: {opt['output_dir']}")

# -------------------- [3] 유틸 --------------------
def to_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0) / 255.0
    return t.to(device)

def pad64(img):
    h, w, _ = img.shape
    ph = ((h // 64) + 1) * 64 if h % 64 != 0 else h
    pw = ((w // 64) + 1) * 64 if w % 64 != 0 else w
    if ph != h or pw != w:
        img = cv2.copyMakeBorder(img, 0, ph - h, 0, pw - w, cv2.BORDER_REFLECT_101)
    return img, (h, w)

def unpad(img, size):
    h, w = size
    return img[:h, :w]

def contains_nan(t: torch.Tensor) -> bool:
    return torch.isnan(t).any().item() or not torch.isfinite(t).all().item()

# -------------------- [4] 메인 루프 --------------------
frame_files = sorted(glob.glob(os.path.join(opt["input_dir"], "*.png")))
try:
    frame_files.sort(key=lambda f: int(re.search(r"\d+", os.path.basename(f)).group()))
except:
    pass

n = len(frame_files)
if n < 2:
    raise SystemExit("❌ 최소 2개 이상의 프레임이 필요합니다.")

# 첫 프레임 복사
shutil.copy(frame_files[0], os.path.join(opt["output_dir"], "img00000.png"))
print(f"🧠 총 {n}개 프레임 감지됨 — v4.25 보간 시작")

success = 0
ssim_log = []

for i in range(n - 1):
    f0, f1 = frame_files[i], frame_files[i+1]
    img0 = cv2.imread(f0); img1 = cv2.imread(f1)
    if img0 is None or img1 is None:
        print(f"[{i+1}/{n-1}] ⚠️ 프레임 로드 실패 → 건너뜀: {f0} or {f1}")
        continue

    # 64배수 패딩
    img0p, orig_size = pad64(img0)
    img1p, _        = pad64(img1)

    I0 = to_tensor(img0p)
    I1 = to_tensor(img1p)

    try:
        mid = model.inference(I0, I1, 0.5)
        if contains_nan(mid):
            print(f"[{i+1}/{n-1}] ⚠️ NaN 발생 → 건너뜀")
            torch.cuda.empty_cache()
            continue

        out = (mid[0].detach().cpu().numpy().transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = unpad(out, orig_size)

        # 저장: 보간 프레임 + 다음 원본
        cv2.imwrite(os.path.join(opt["output_dir"], f"img{2*i+1:05d}.png"), out)
        shutil.copy(f1, os.path.join(opt["output_dir"], f"img{2*i+2:05d}.png"))

        # SSIM (선택)
        if ssim_loss is not None:
            # ssim_loss는 DSSIM을 반환하니 SSIM으로 변환
            try:
                dssim = ssim_loss(to_tensor(out), to_tensor(img1)).item()
                ssim_val = 1.0 - (2.0 * dssim)
                ssim_log.append(ssim_val)
                print(f"[{i+1}/{n-1}] ✅ {os.path.basename(f0)} ↔ {os.path.basename(f1)} | SSIM={ssim_val:.4f}")
            except Exception as e:
                print(f"[{i+1}/{n-1}] ✅ 저장 완료 (SSIM 계산 실패: {e})")
        else:
            print(f"[{i+1}/{n-1}] ✅ 저장 완료")

        success += 1

    except Exception as e:
        print(f"[{i+1}/{n-1}] ❌ 오류: {e}")
    finally:
        torch.cuda.synchronize() if device.type == "cuda" else None
        torch.cuda.empty_cache() if device.type == "cuda" else None

print(f"\n🎉 RIFE v4.25 보간 완료!")
print(f"✅ 성공: {success}/{n - 1}")
print(f"📁 출력: {opt['output_dir']}")
if ssim_log:
    print(f"📈 평균 SSIM: {np.mean(ssim_log):.4f}")
