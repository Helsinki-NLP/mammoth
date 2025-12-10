#!/usr/bin/bash

python -c '
import torch

print("=== Fine-Grained PCIe Memory Test ===")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("🚫 CUDA not available. Exiting.")
    exit(1)

device = torch.device("cuda")

# Step 1: Create pinned host memory
pinned = torch.tensor([123], dtype=torch.int32, pin_memory=True)
print(f"[HOST] Initial pinned memory value: {pinned.item()}")

# Step 2: Create device tensor
device_tensor = torch.empty(1, dtype=torch.int32, device=device)
print("[DEVICE] Allocated empty tensor on GPU.")

# Step 3: Copy host → device
torch.cuda.synchronize()
print("[COPY] Copying pinned host → device tensor.")
device_tensor.copy_(pinned, non_blocking=True)
torch.cuda.synchronize()
print(f"[DEVICE] Value after copy: {device_tensor.item()}")

# Step 4: Simulate GPU writing a new value
print("[DEVICE] Writing new value 456 to device tensor.")
device_tensor.fill_(456)
torch.cuda.synchronize()

# Step 5: Copy device → host
print("[COPY] Copying device → pinned host tensor.")
pinned.copy_(device_tensor, non_blocking=True)
torch.cuda.synchronize()

# Final value check
print(f"[HOST] Final pinned memory value (should be 456): {pinned.item()}")

print("=== Done ===")
'

