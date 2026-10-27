"""Print what PyTorch sees on this machine."""
import torch

print(f"torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
else:
    print("No CUDA GPU visible. Check `nvidia-smi` and the CUDA tag in setup_env.sh.")
