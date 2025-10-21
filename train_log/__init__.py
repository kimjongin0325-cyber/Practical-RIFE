# =====================================================
# ✅ train_log/__init__.py (RIFE 자동 감지 통합버전)
# - v3.8 / v4.25 자동 인식
# =====================================================
import os
import importlib.util

BASE_DIR = os.path.dirname(__file__)

def _import_model(pyfile):
    """안전한 동적 import"""
    spec = importlib.util.spec_from_file_location("rife_model", pyfile)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# -----------------------------------------------------
# 모델 경로 감지
# -----------------------------------------------------
v425_py = os.path.join(BASE_DIR, "rife_hd3_v425.py")   # ✅ 수정 완료
v425_pkl = os.path.join(BASE_DIR, "flownet_v425.pkl")

v38_py = os.path.join(BASE_DIR, "rife_hd3.py")
v38_pkl = os.path.join(BASE_DIR, "flownet.pkl")

Model = None

try:
    if os.path.exists(v425_py) and os.path.exists(v425_pkl):
        mod = _import_model(v425_py)
        Model = mod.Model
        print("✅ RIFE v4.25 감지됨 → rife_hd3_v425.py 사용")
    elif os.path.exists(v38_py) and os.path.exists(v38_pkl):
        mod = _import_model(v38_py)
        Model = mod.Model
        print("✅ RIFE v3.8 감지됨 → rife_hd3.py 사용")
    else:
        raise FileNotFoundError("❌ RIFE 모델 파일을 찾을 수 없습니다.")
except Exception as e:
    raise ImportError(f"[FATAL] RIFE 모델 import 실패: {e}")
