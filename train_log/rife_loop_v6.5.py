# =====================================================
# ✅ [실전용 RIFE Interpolation Loop v6.5-Remodel]
# - Colab/Kaggle 완전 호환
# - 1000장 이상 대량 프레임 안정 지원
# - SSIM 품질 평가 추가
# =====================================================
import os, sys, glob, torch, shutil, re, time
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------- [1] 기본 경로 설정 --------------------
BASE_DIR = "/content/Practical-RIFE"
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)

# 필수 경로 추가
sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "train_log"),
    os.path.join(BASE_DIR, "model")
])

# 디버깅
print(f"✅ 경로 준비 완료: {BASE_DIR}")
print(f"sys.path 추가 완료: {sys.path[-3:]}")
print(f"train_log/__init__.py 존재: {os.path.exists(os.path.join(BASE_DIR, 'train_log', '__init__.py'))}")
print(f"pytorch_msssim/__init__.py 존재: {os.path.exists(os.path.join(BASE_DIR, 'model', 'pytorch_msssim', '__init__.py'))}")

# -------------------- [2] 사용자 옵션 --------------------
opt = {
    "scale": 2,             # 2=2x FPS, 4=4x FPS 등
    "threads": min(os.cpu_count(), 4),  # T4 환경 최적화
    "fps_limit": 60,        # FFmpeg 병합 시 FPS
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_dir": os.path.join(BASE_DIR, "input_frames"),
    "output_dir": os.path.join(BASE_DIR, "output"),
    "model_path": os.path.join(BASE_DIR, "train_log/flownet.pkl"),
}

# -------------------- [3] RIFE 모델 로드 --------------------
try:
    from train_log import Model
    print("✅ train_log.Model import 성공!")
except ImportError as e:
    raise ImportError(f"🚨 train_log.Model import 실패: {e}. train_log/__init__.py 확인하세요.")

# SSIM 손실 함수 로드
try:
    from model import SSIM
    ssim_loss = SSIM(window_size=11, size_average=True).to(opt["device"])
    print("✅ SSIM 손실 함수 로드 성공!")
except ImportError as e:
    print(f"⚠️ SSIM import 실패: {e}. pytorch_msssim/__init__.py 확인하세요.")

device = torch.device(opt["device"])
rife_model = Model()
try:
    rife_model.load_model(os.path.dirname(opt["model_path"]))  # 디렉토리 전달
    rife_model.eval()
    rife_model.device()
    print("✅ 모델 로딩 완료:", opt["model_path"])
except Exception as e:
    raise RuntimeError(f"🚨 모델 로드 실패: {e}")

# -------------------- [4] 보간 함수 --------------------
@torch.inference_mode()
def run_rife_inference(img0_bgr, img1_bgr, timestep=0.5):
    if img0_bgr is None or img1_bgr is None:
        return None
    # T4 메모리 최적화: 4K 이미지 다운스케일
    if img0_bgr.shape[0] > 1080:
        img0_bgr = cv2.resize(img0_bgr, (1920, 1080))
        img1_bgr = cv2.resize(img1_bgr, (1920, 1080))
    img0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    I0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I0, I1 = I0.to(device), I1.to(device)
    out_img = rife_model.inference(I0, I1, scale=max(1.0, 1.0 / timestep))
    out = (out_img[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # SSIM 계산
    if 'ssim_loss' in globals():
        out_tensor = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        ssim_value = 1 - 2 * ssim_loss(out_tensor, img1_tensor)  # SSIM (0~1)
        return out_bgr, ssim_value
    return out_bgr, None

# -------------------- [5] 프레임 초기화 --------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

INPUT_DIR = opt["input_dir"]
OUTPUT_DIR = opt["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

frame_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
try:
    frame_files.sort(key=lambda f: int(re.search(r"\d+", os.path.basename(f)).group()))
except:
    print("⚠️ 숫자 정렬 실패 → 사전순 정렬.")

num_frames = len(frame_files)
if num_frames < 2:
    raise SystemExit("❌ 최소 2개 프레임 필요")
if num_frames < 1000:
    print(f"⚠️ {num_frames}장 감지됨. 최소 1000장 권장.")

print(f"총 {num_frames}개 프레임 감지됨 — 보간 시작")

# 첫 프레임 복사
shutil.copy(frame_files[0], os.path.join(OUTPUT_DIR, f"img00000.png"))

# -------------------- [6] 프리로딩 --------------------
def preload_pair(i):
    try:
        f1, f2 = frame_files[i], frame_files[i+1]
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        if img1 is None or img2 is None:
            raise ValueError(f"프레임 로드 실패: {f1} 또는 {f2}")
        return (i, img1, img2)
    except Exception as e:
        print(f"⚠️ 프레임 {i} 로드 실패: {e}")
        return (i, None, None)

preloaded = {}
print("🧠 CPU 프리로딩 중...")
with ThreadPoolExecutor(max_workers=opt["threads"]) as ex:
    futures = {ex.submit(preload_pair, i): i for i in range(num_frames - 1)}
    for fut in as_completed(futures):
        i, a, b = fut.result()
        if a is not None and b is not None:
            preloaded[i] = (a, b)
print(f"✅ 프리로딩 완료: {len(preloaded)}/{num_frames-1} 쌍 성공")

# -------------------- [7] 메인 루프 --------------------
success, idx = 0, 0
for i in range(num_frames - 1):
    if i not in preloaded:
        print(f"⚠️ {i}번째 쌍 누락 — 원본 복사")
        idx += 1
        shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f"img{idx:05d}.png"))
        idx += 1
        continue

    img0, img1 = preloaded[i]
    print(f"\n[{i+1}/{num_frames-1}] 보간 중: {os.path.basename(frame_files[i])} ↔ {os.path.basename(frame_files[i+1])}")

    for j in range(1, opt["scale"]):
        t = j / opt["scale"]
        out, ssim_value = run_rife_inference(img0, img1, timestep=t)
        if out is None or np.isnan(out).any() or np.max(out) == 0:
            print(f"❌ NaN 감지 (t={t}) → 스킵")
            continue
        dst = os.path.join(OUTPUT_DIR, f"img{idx + j:05d}.png")
        cv2.imwrite(dst, out)
        print(f"✅ 보간 프레임 저장: {os.path.basename(dst)} (SSIM: {ssim_value:.4f} if available)")
        success += 1
        torch.cuda.empty_cache()  # T4 메모리 관리

    idx += opt["scale"]
    shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f"img{idx:05d}.png"))

print(f"\n🎉 RIFE 보간 완료! 성공: {success} 프레임")
print(f"📁 출력: {OUTPUT_DIR}")
print(f"🔗 FFmpeg 병합: cd {OUTPUT_DIR} && ffmpeg -r {opt['fps_limit']} -i img%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
