# =====================================================
# ✅ [코랩 최적화 RIFE Interpolation Loop v6-Final]
# - inference_img.py의 기능을 함수(run_rife_inference)로 내장
# - subprocess/파일 I/O 대신 함수 직접 호출로 성능 및 안정성 극대화
# - NaN 감시 + GPU 안정화 + CPU 프리로딩
# =====================================================
import os, glob, torch, shutil, re, time
import numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# -------------------- 사용자 옵션 --------------------
# 코랩 환경 경로에 맞게 설정
BASE_DIR = "/content/Practical-RIFE"
opt = {
    "threads": 4,  # CPU 코어 수에 맞춰 조절
    "input_dir": os.path.join(BASE_DIR, "input_frames"),
    "output_dir": os.path.join(BASE_DIR, "output"),
    "model_path": os.path.join(BASE_DIR, "train_log/flownet.pkl")
}

# -------------------- RIFE 모델 및 추론 함수 (inference_img.py 내장) --------------------
# RIFE 모델 클래스 코드가 필요합니다. train_log/RIFE_HDv3.py 에서 가져와야 합니다.
# 우선 경로를 추가하여 import 할 수 있도록 설정합니다.
sys.path.append(os.path.join(BASE_DIR, 'train_log'))
try:
    from RIFE_HDv3 import Model
except ImportError:
    print("="*80)
    print("🚨 오류: RIFE_HDv3.py 파일을 찾을 수 없습니다!")
    print(f"'{os.path.join(BASE_DIR, 'train_log')}' 폴더에 RIFE_HDv3.py 파일이 있는지 확인해주세요.")
    print("RIFE 모델 코드가 없으면 보간을 진행할 수 없습니다.")
    print("="*80)
    # RIFE_HDv3.py가 없으면 진행이 불가능하므로 여기서 중단합니다.
    # 만약 RIFE_HDv3.py 파일이 있다면 이 부분은 그냥 지나갑니다.
    raise

# --- 추론 함수 정의 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델을 한 번만 로드하여 재사용
print("⚡ RIFE 모델 로딩 중...")
rife_model = Model()
rife_model.load_model(opt["model_path"])
rife_model.eval()
rife_model.device()
print("✅ 모델 로딩 완료.")

def run_rife_inference(img0_bgr, img1_bgr):
    """
    두 개의 OpenCV 이미지(BGR)를 입력받아 보간된 이미지를 반환합니다.
    """
    if img0_bgr is None or img1_bgr is None:
        return None

    # BGR to RGB, HWC to CHW, Normalize
    img0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    I0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    I0, I1 = I0.to(device), I1.to(device)

    with torch.no_grad():
        pred = rife_model.inference(I0, I1)

    # Denormalize, CHW to HWC, RGB to BGR
    out_img_np = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    out_img_bgr = cv2.cvtColor(out_img_np, cv2.COLOR_RGB2BGR)

    return out_img_bgr

# -------------------- 초기화 --------------------
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

INPUT_DIR  = opt["input_dir"]
OUTPUT_DIR = opt["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 프레임 목록 정렬 --------------------
frame_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')))
try:
    frame_files.sort(key=lambda f: [int(s) for s in re.findall(r'\d+', os.path.basename(f))])
except Exception:
    print("⚠️ 숫자 정렬 실패 → 일반 사전순 정렬로 대체.")

num_frames = len(frame_files)
if num_frames < 2:
    print("오류: 보간을 위해 최소 2개 이상의 프레임이 필요합니다.")
else:
    print(f"총 {num_frames}개의 프레임 감지됨. [코랩 최적화 보간 시작]")

    # 첫 프레임 복사
    shutil.copy(frame_files[0], os.path.join(OUTPUT_DIR, f'img{0:05d}.png'))

    # -------------------- 유틸 함수 --------------------
    def contains_nan(img_data):
        """이미지 데이터(numpy array)에 NaN 또는 손상 여부 검사"""
        if img_data is None: return True
        if np.isnan(img_data).any() or np.all(img_data == 0):
            return True
        return False

    # CPU 프리로딩 함수 (이미지를 파일 경로가 아닌 실제 데이터로 로드)
    def preload_pair_data(i):
        f1_path = frame_files[i]
        f2_path = frame_files[i + 1]
        f1_data = cv2.imread(f1_path)
        f2_data = cv2.imread(f2_path)
        return (i, f1_data, f2_data)

    # -------------------- CPU 프리로딩 --------------------
    print("🧠 CPU 프리로딩 중 (이미지 데이터를 직접 로딩)...")
    preloaded_images = {}
    with ThreadPoolExecutor(max_workers=opt["threads"]) as executor:
        futures = {executor.submit(preload_pair_data, i): i for i in range(num_frames - 1)}
        for future in as_completed(futures):
            i, f1_data, f2_data = future.result()
            if f1_data is not None and f2_data is not None:
                preloaded_images[i] = (f1_data, f2_data)
    print(f"✅ 프리로딩 완료 ({len(preloaded_images)}/{num_frames - 1} 쌍 성공)\n")

    # -------------------- 메인 보간 루프 --------------------
    for i in range(num_frames - 1):
        if i not in preloaded_images:
            print(f"⚠️ 프레임 쌍 {i} 로드 실패 — 건너뜀")
            # 원본 프레임이라도 복사해서 프레임 순서를 유지
            shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f'img{2 * (i + 1):05d}.png'))
            continue

        img0_data, img1_data = preloaded_images[i]

        print(f"\n--- [{i + 1}/{num_frames - 1}] 보간 중: {os.path.basename(frame_files[i])} ↔ {os.path.basename(frame_files[i+1])} ---")

        # ✨ 핵심 변경: 파일 실행 대신 함수 직접 호출 ✨
        interpolated_img = run_rife_inference(img0_data, img1_data)

        # NaN 검사
        if contains_nan(interpolated_img):
            print(f"❌ NaN 또는 손상된 결과 감지됨 → 프레임 스킵")
            torch.cuda.empty_cache()
            # 보간 실패 시 원본 프레임이라도 복사
            shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f'img{2 * (i + 1):05d}.png'))
            continue

        # 정상 파일 저장 및 원본 프레임 복사
        # 보간된 프레임: img00001, img00003, ...
        dst_interpolated_path = os.path.join(OUTPUT_DIR, f'img{2 * i + 1:05d}.png')
        cv2.imwrite(dst_interpolated_path, interpolated_img)

        # 원본 다음 프레임: img00002, img00004, ...
        # 이 부분은 미리 로드한 데이터를 저장해도 되지만, 파일 복사가 더 간단합니다.
        shutil.copy(frame_files[i+1], os.path.join(OUTPUT_DIR, f'img{2 * (i + 1):05d}.png'))

        # GPU 메모리 안정화 (함수 호출 방식에서는 덜 중요하지만, 안정성을 위해 유지)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("\n✅ RIFE 보간 완료 (코랩 최적화 버전)")
    print(f"출력 폴더: {OUTPUT_DIR}")
