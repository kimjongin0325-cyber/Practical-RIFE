# =====================================================
# ✅ [RIFE Interpolation Loop v6.6-SSIM-Corrected-Final]
# - 실전용 / Colab, Kaggle 완전 호환
# - SSIM 정확 계산 (DSSIM → SSIM 변환)
# - 1000장 이상 대량 프레임 안정 지원
# =====================================================
import os, sys, glob, torch, shutil, re, time
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------- [1] 기본 경로 설정 --------------------
BASE_DIR = "/content/Practical-RIFE"
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)

sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "train_log"),
    os.path.join(BASE_DIR, "model")
])

# -------------------- [2] 사용자 옵션 --------------------
opt = {
    "scale": 2,
    "threads": min(os.cpu_count(), 4),
    "fps_limit": 60,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_dir": os.path.join(BASE_DIR, "input_frames"),
    "output_dir": os.path.join(BASE_DIR, "output"),
    "model_path": os.path.join(BASE_DIR, "train_log/flownet.pkl"),
}

# -------------------- [3] 모델 및 SSIM 로드 --------------------
try:
    from train_log import Model
    print("✅ train_log.Model import 성공!")
except ImportError as e:
    raise ImportError(f"🚨 train_log.Model import 실패: {e}")

try:
    from model import SSIM
    ssim_loss = SSIM(window_size=11, size_average=True).to(opt["device"])
    print("✅ SSIM 손실 함수 로드 성공! (DSSIM 기반)")
except Exception as e:
    ssim_loss = None
    print(f"⚠️ SSIM import 실패: {e}")

device = torch.device(opt["device"])
rife_model = Model()
try:
    rife_model.load_model(os.path.dirname(opt["model_path"]))
    rife_model.eval()
    rife_model.device()
    print("✅ 모델 로딩 완료:", opt["model_path"])
except Exception as e:
    raise RuntimeError(f"🚨 모델 로드 실패: {e}")

# -------------------- [4] 보간 함수 --------------------
@torch.inference_mode()
def run_rife_inference(img0_bgr, img1_bgr, timestep=0.5):
    if img0_bgr is None or img1_bgr is None:
        return None, None

    original_h, original_w, _ = img0_bgr.shape
    downscaled = False
    if original_h > 1080:
        downscaled = True
        img0_bgr = cv2.resize(img0_bgr, (1920, 1080))
        img1_bgr = cv2.resize(img1_bgr, (1920, 1080))

    img0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    I0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I0, I1 = I0.to(device), I1.to(device)

    out_tensor = rife_model.inference(I0, I1, timestep)
    out = (out_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    if downscaled:
        out_bgr = cv2.resize(out_bgr, (original_w, original_h), interpolation=cv2.INTER_CUBIC)

    ssim_value = None
    if ssim_loss is not None:
        try:
            dssim_val = ssim_loss(out_tensor, I1).item()
            ssim_value = 1 - (2 * dssim_val)  # DSSIM → SSIM 변환
        except Exception as e:
            print(f"⚠️ SSIM 계산 실패: {e}")
    return out_bgr, ssim_value

# -------------------- [5] 프레임 초기화 --------------------
torch.cuda.empty_cache()
INPUT_DIR = opt["input_dir"]
OUTPUT_DIR = opt["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

frame_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
try:
    frame_files.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
except:
    print('⚠️ 숫자 정렬 실패 → 사전순 정렬.')

num_frames = len(frame_files)
if num_frames < 2:
    raise SystemExit('❌ 최소 2개 프레임 필요')
if num_frames < 1000:
    print(f'⚠️ {num_frames}장 감지됨. 최소 1000장 권장.')

print(f'총 {num_frames}개 프레임 감지됨 — 보간 시작')

# 첫 프레임 복사
shutil.copy(frame_files[0], os.path.join(OUTPUT_DIR, 'img00000.png'))

# -------------------- [6] 프리로딩 --------------------
def preload_pair(i):
    try:
        f1, f2 = frame_files[i], frame_files[i+1]
        img1, img2 = cv2.imread(f1), cv2.imread(f2)
        if img1 is None or img2 is None:
            raise ValueError(f'프레임 로드 실패: {f1} 또는 {f2}')
        return (i, img1, img2)
    except Exception as e:
        print(f'⚠️ 프레임 {i} 로드 실패: {e}')
        return (i, None, None)

preloaded = {}
print('🧠 CPU 프리로딩 중...')
with ThreadPoolExecutor(max_workers=opt['threads']) as ex:
    futures = {ex.submit(preload_pair, i): i for i in range(num_frames - 1)}
    for fut in as_completed(futures):
        i, a, b = fut.result()
        if a is not None and b is not None:
            preloaded[i] = (a, b)
print(f'✅ 프리로딩 완료: {len(preloaded)}/{num_frames-1} 쌍 성공')

# -------------------- [7] 메인 루프 --------------------
success, idx = 0, 0
for i in range(num_frames - 1):
    if i not in preloaded:
        print(f'⚠️ {i}번째 쌍 누락 — 원본 복사')
        idx += opt['scale']
        shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f'img{idx:05d}.png'))
        continue

    img0, img1 = preloaded[i]
    print(f'\n[{i+1}/{num_frames-1}] 보간 중: {os.path.basename(frame_files[i])} ↔ {os.path.basename(frame_files[i+1])}')

    for j in range(1, opt['scale']):
        t = j / opt['scale']
        out, ssim_value = run_rife_inference(img0, img1, timestep=t)
        if out is None:
            print(f'❌ 보간 실패 (t={t}) → 스킵')
            continue

        dst = os.path.join(OUTPUT_DIR, f'img{idx + j:05d}.png')
        cv2.imwrite(dst, out)
        if ssim_value is not None:
            print(f'✅ 보간 프레임 저장: {os.path.basename(dst)} (SSIM: {ssim_value:.4f})')
        else:
            print(f'✅ 보간 프레임 저장: {os.path.basename(dst)} (SSIM 계산 안 됨)')
        success += 1
        torch.cuda.empty_cache()

    idx += opt['scale']
    shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f'img{idx:05d}.png'))

print(f'\n🎉 보간 완료! 성공: {success} 프레임')
print(f'📁 출력: {OUTPUT_DIR}')
print(f'🔗 FFmpeg 병합 명령: ffmpeg -framerate {opt['fps_limit']} -i {os.path.join(OUTPUT_DIR, 'img%05d.png')} -c:v libx264 -pix_fmt yuv420p -r {opt['fps_limit']} output.mp4')
