import torch
import os
import socket

def detect_platform():
    hostname = socket.gethostname().lower()
    singularity_name = os.environ.get("SINGULARITY_NAME", "").lower()
    rocm_hint = any(k in os.environ for k in ["ROCR_VISIBLE_DEVICES", "HSA_PATH", "ROCM_PATH", "HSA_ENABLE_DEBUG"])

    if "puhti" in hostname or "puhti" in singularity_name:
        return "Puhti"
    if (
        "lumi" in hostname
        or hostname.startswith("uan")
        or hostname.startswith("gpu")
        or singularity_name.startswith("lumi-")
        or rocm_hint
    ):
        return "LUMI"
    return "Unknown"

def check_torch():
    print("=== PyTorch Environment Check ===")
    print(f"PyTorch version          : {torch.__version__}")

    # Detect backend
    has_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)

    print(f"ROCm HIP version          : {hip_version}")
    print(f"torch.version.cuda        : {cuda_version}")

    # Device detection
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()

    print(f"torch.cuda.is_available() : {cuda_available}")
    print(f"torch.cuda.device_count() : {device_count}")

    for i in range(device_count):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    return {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "rocm": has_rocm,
        "hip_version": hip_version,
        "cuda_version": cuda_version,
    }

def validate(platform_name, torch_info):
    print("\n=== Validation ===")
    if platform_name == "LUMI":
        if torch_info["rocm"] and not torch_info["cuda_version"]:
            print("✅ LUMI: ROCm detected, no CUDA version reported — environment is OK.")
        else:
            print("❌ LUMI: Unexpected CUDA version or missing ROCm — check your PyTorch installation.")
    elif platform_name == "Puhti":
        if torch_info["cuda_available"] and not torch_info["rocm"]:
            print("✅ Puhti: CUDA available, ROCm not detected — environment is OK.")
        else:
            print("❌ Puhti: ROCm detected or CUDA missing — check if correct PyTorch is installed.")
    else:
        print("⚠️ Unknown platform — please verify manually. ROCm/CUDA details:")
        if torch_info["rocm"]:
            print("ROCm backend detected.")
        if torch_info["cuda_version"]:
            print(f"CUDA version reported: {torch_info['cuda_version']}")

def main():
    print("=== Torch ROCm/CUDA Diagnostic Script ===\n")
    platform_name = detect_platform()
    print(f"Detected platform: {platform_name}\n")
    torch_info = check_torch()
    validate(platform_name, torch_info)

if __name__ == "__main__":
    main()
