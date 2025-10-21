# =====================================================
# ✅ [RIFE v4.25 Fine Detail Edition – FP16 + CUDA Stream]
# - ifnet_hd3_v425.py + flownet_v425.pkl 조합 전용
# - FP16 가속 / SSIM 선택적 / 1080~2K 안정화
# =====================================================

import os, cv2, torch, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# -------------------- 사용자 옵션 --------------------
opt = {
    "input_dir": "/content/Practical-RIFE/input_frames",
    "output_dir": "/content/Practical-RIFE/output",
    "model_py": "/content/Practical-RIFE/train_log/ifnet_hd3_v425.py",
    "model_path": "/content/Practical-RIFE/train_log/flownet_v425.pkl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "threads": 8,
    "use_ssim": False,
    "fp16": True,
}

# -------------------- Torch 초기화 --------------------
torch.backends.cudnn.benchmark = True
device = torch.device(opt["device"])
stream = torch.cuda.Stream(device=device)

# -------------------- 모델 로드 --------------------
import importlib.util
spec = importlib.util.spec_from_file_location("model_v425", opt["model_py"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Model = module.Model

model = Model()
model.load_model(os.path.dirname(opt["model_path"]))
model.eval().to(device)
print(f"✅ RIFE v4.25 모델 로딩 완료: {opt['model_path']}")

# -------------------- SSIM 로드 --------------------
try:
    from model import SSIM
    ssim_loss = SSIM(window_size=11, size_average=True).to(device)
except Exception:
    ssim_loss = None

# -------------------- 보간 함수 --------------------
@torch.inference_mode()
def infer(img0, img1, t=0.5):
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    I0 = torch.from_numpy(img0).permute(2,0,1).unsqueeze(0).float().to(device) / 255.
    I1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).float().to(device) / 255.

    with torch.cuda.stream(stream):
        if opt["fp16"]:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.inference(I0, I1, t)
        else:
            pred = model.inference(I0, I1, t)
        torch.cuda.synchronize()

    out_img = (pred[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    ssim_val = None
    if opt["use_ssim"] and ssim_loss is not None:
        ssim_val = float(ssim_loss(pred, I1).item())
    return out_img, ssim_val

# -------------------- 프레임 보간 루프 --------------------
os.makedirs(opt["output_dir"], exist_ok=True)
frames = sorted([f for f in os.listdir(opt["input_dir"]) if f.endswith(".png")])
start = time.time()

with ThreadPoolExecutor(max_workers=opt["threads"]) as ex:
    for i in range(len(frames) - 1):
        f1, f2 = frames[i], frames[i+1]
        img0 = cv2.imread(os.path.join(opt["input_dir"], f1))
        img1 = cv2.imread(os.path.join(opt["input_dir"], f2))
        out, ssim_val = infer(img0, img1, 0.5)
        save_path = os.path.join(opt["output_dir"], f"img{i:05d}.png")
        cv2.imwrite(save_path, out)
        msg = f"[{i+1}/{len(frames)-1}] 저장: {save_path}"
        if ssim_val is not None: msg += f" | SSIM={ssim_val:.4f}"
        print(msg)

print(f"🎉 완료! 총 {len(frames)-1}프레임 | 처리시간 {time.time()-start:.2f}s")
