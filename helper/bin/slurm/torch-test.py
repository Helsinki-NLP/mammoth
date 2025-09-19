import os, sys

try:
    import torch
except Exception as e:
    torch = None; print("torch import error:", e)
    
print("==============================================")
print(" PATH                      :", os.environ.get("PATH",""))
print(" LD_LIBRARY_PATH           :", os.environ.get("LD_LIBRARY_PATH"))
print(" FI_PROVIDER (cxi)         :", os.environ.get("FI_PROVIDER"))
print(" FI_HMEM (rocr)            :", os.environ.get("FI_HMEM"))
print(" FI_LOG_LEVEL (warn)       :", os.environ.get("FI_LOG_LEVEL"))
print(" FI_LOG_PROV (cxi)         :", os.environ.get("FI_LOG_PROV"))
print(" PLUGIN_DIR                :", os.environ.get("PLUGIN_DIR"))
print(" RCC_ENABLE_OFI            :", os.environ.get("RCCL_ENABLE_OFI"))
print(" NCCL_SOCKET_IFNAME (hsn0) :", os.environ.get("NCCL_SOCKET_IFNAME"))
print(" NCCL_NET_GDR_LEVEL        :", os.environ.get("NCCL_NET_GDR_LEVEL"))
print(" sys.executable            :", sys.executable)
print(" sys.version               :", sys.version.split()[0])
if torch:
    print(" torch                     :", torch.__version__)
    print(" HIP version               :", getattr(torch.version, "hip", None))
    print(" CUDA version              :", getattr(torch.version, "cuda", None))
    print(" CUDA avail?               :", torch.cuda.is_available())
else: 
    print(" torch                     : not available")
    print("==============================================")
