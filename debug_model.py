import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.restoration._network import RestoreUNet

device = torch.device("cuda")
model = RestoreUNet().to(device)
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

B, T, H, W = 4, 3, 128, 128
x = torch.randn(B * T, 6, H, W, device=device)
t = torch.randint(0, 1000, (B * T,), device=device)

print("Running forward pass ...")
t0 = time.perf_counter()
with torch.no_grad():
    out = model(x, t)
elapsed = time.perf_counter() - t0
print(f"Forward pass done in {elapsed:.2f}s — output: {out.shape}")
