# =====================================================
# ✅ [코랩 최적화 RIFE Interpolation Loop v6.2-FINAL]
# - v3.1 호환 opt + v6 구조 반영 + scale/timestep 지원
# - inference_img.py 기능 내장 + 성능/안정성 극대화
# - NaN 감시 + GPU 안정화 + CPU 프리로딩
# =====================================================
import os, glob, torch, shutil, re, time
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# -------------------- 사용자 옵션 (v3.1 호환) --------------------
BASE_DIR = "/content/Practical-RIFE"
opt = {
    "scale": 2,  # 보간 배율 (2=2x FPS, 4=4x FPS 등)
    "device": "cuda",
    "threads": 4,
    "fps_limit": 60,  # FFmpeg 출력용
    "input_dir": os.path.join(BASE_DIR, "input_frames"),
    "output_dir": os.path.join(BASE_DIR, "output"),
    "demo_dir": os.path.join(BASE_DIR, "demo"),  # 사용 안 함 (호환용)
    "script_path": os.path.join(BASE_DIR, "train_log/inference_img.py"),  # 사용 안 함
    "model_path": os.path.join(BASE_DIR, "train_log/flownet.pkl")
}

# -------------------- RIFE 모델 및 추론 함수 --------------------
sys.path.append(os.path.join(BASE_DIR, 'train_log'))
try:
    from RIFE_HDv3 import Model
except ImportError:
    print("="*80)
    print("🚨 오류: RIFE_HDv3.py 파일을 찾을 수 없습니다!")
    print(f"경로 확인: {os.path.join(BASE_DIR, 'train_log/RIFE_HDv3.py')}")
    print("리포지토리에서 다운로드 후 train_log 폴더에 넣어주세요.")
    print("="*80)
    raise

device = torch.device(opt["device"] if torch.cuda.is_available() else "cpu")

print("⚡ RIFE 모델 로딩 중...")
rife_model = Model()
rife_model.load_model(opt["model_path"])
rife_model.eval()
rife_model.device()
print("✅ 모델 로딩 완료.")

def run_rife_inference(img0_bgr, img1_bgr, timestep=0.5):
    """
    두 이미지와 timestep(0~1)을 입력받아 보간 이미지 반환.
    """
    if img0_bgr is None or img1_bgr is None:
        return None

    img0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    I0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I0, I1 = I0.to(device), I1.to(device)

    with torch.no_grad():
        pred = rife_model.inference(I0, I1, timestep=timestep)

    out_img_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    out_img_bgr = cv2.cvtColor(out_img_np, cv2.COLOR_RGB2BGR)

    return out_img_bgr

# -------------------- 초기화 --------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

INPUT_DIR = opt["input_dir"]
OUTPUT_DIR = opt["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(opt["demo_dir"], exist_ok=True)  # 호환용

# -------------------- 프레임 목록 정렬 --------------------
frame_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
try:
    # ✅ 안정 정렬: 첫 숫자만 추출
    frame_files.sort(key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
except Exception:
    print("⚠️ 숫자 정렬 실패 → 일반 사전순 정렬.")

num_frames = len(frame_files)
if num_frames < 2:
    print("오류: 최소 2개 프레임 필요")
    exit(1)

print(f"총 {num_frames}개의 프레임 감지됨. [보간 시작]")

# 첫 프레임 복사
shutil.copy(frame_files[0], os.path.join(OUTPUT_DIR, f'img00000.png'))

# -------------------- 유틸 함수 --------------------
def contains_nan(img_data):
    """NaN 검사 최적화: NaN만 검사, 완전 빈 이미지만 스킵"""
    if img_data is None:
        return True
    return np.isnan(img_data).any() or np.max(img_data) == 0

def preload_pair_data(i):
    f1_path = frame_files[i]
    f2_path = frame_files[i + 1]
    f1_data = cv2.imread(f1_path)
    f2_data = cv2.imread(f2_path)
    return (i, f1_data, f2_data)

# -------------------- CPU 프리로딩 --------------------
print("🧠 CPU 프리로딩 중...")
preloaded_images = {}
with ThreadPoolExecutor(max_workers=opt["threads"]) as executor:
    futures = {executor.submit(preload_pair_data, i): i for i in range(num_frames - 1)}
    for future in as_completed(futures):
        i, f1_data, f2_data = future.result()
        if f1_data is not None and f2_data is not None:
            preloaded_images[i] = (f1_data, f2_data)
print(f"✅ 프리로딩 완료 ({len(preloaded_images)}/{num_frames - 1} 쌍 성공)\n")

# -------------------- 메인 보간 루프 --------------------
success_count = 0
current_idx = 0  # 출력 프레임 인덱스
for i in range(num_frames - 1):
    if i not in preloaded_images:
        print(f"⚠️ 쌍 {i} 로드 실패 — 원본 복사")
        current_idx += 1
        shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f'img{current_idx:05d}.png'))
        current_idx += 1
        continue

    img0_data, img1_data = preloaded_images[i]

    print(f"\n--- [{i + 1}/{num_frames - 1}] 보간: {os.path.basename(frame_files[i])} ↔ {os.path.basename(frame_files[i+1])} ---")

    # scale만큼 보간 프레임 생성
    for j in range(1, opt["scale"]):
        timestep = j / opt["scale"]
        interpolated_img = run_rife_inference(img0_data, img1_data, timestep=timestep)

        if contains_nan(interpolated_img):
            print(f"❌ NaN 감지 (timestep={timestep}) → 스킵")
            continue

        dst_path = os.path.join(OUTPUT_DIR, f'img{current_idx + j:05d}.png')
        cv2.imwrite(dst_path, interpolated_img)
        print(f"  ✅ 보간 프레임 저장: {os.path.basename(dst_path)} (timestep={timestep})")
        success_count += 1

    # 원본 다음 프레임 복사
    shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f'img{current_idx + opt["scale"]:05d}.png'))
    current_idx += opt["scale"]

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    time.sleep(0.1)  # GPU 휴식

print(f"\n🎉 RIFE 보간 완료!")
print(f"✅ 성공: {success_count}/{num_frames - 1}")
print(f"📁 출력: {OUTPUT_DIR}")
print(f"🔗 FFmpeg 병합: cd {OUTPUT_DIR} && ffmpeg -r {opt['fps_limit']} -i img%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
