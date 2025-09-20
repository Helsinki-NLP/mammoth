import os, sys

try:
    import torch
except Exception as e:
    torch = None; print("torch import error:", e)
    
print(" sys.executable            :", sys.executable)
print(" sys.version               :", sys.version.split()[0])
if torch:
    print(" torch                     :", torch.__version__)
    print(" HIP version               :", getattr(torch.version, "hip", None))
    print(" CUDA version              :", getattr(torch.version, "cuda", None))
    print(" CUDA avail?               :", torch.cuda.is_available())
else: 
    print(" torch                     : not available")

