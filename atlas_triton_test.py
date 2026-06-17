import torch
from atlas_triton import ProductionAtlasEngine

device = "cuda"

B,H,T,D = 1,2,128,64
C = 4
LMAX = 32
VMAX = 4

q = torch.randn(B,H,T,D, device=device, dtype=torch.bfloat16)
k = torch.randn(B,H,T,D, device=device, dtype=torch.bfloat16)
v = torch.randn(B,H,T,D, device=device, dtype=torch.bfloat16)

cluster_positions = torch.randint(
    0, T, (B,C,LMAX),
    device=device,
    dtype=torch.int32
)

cluster_lengths = torch.full(
    (B,C),
    LMAX,
    device=device,
    dtype=torch.int32
)

membership_counts = torch.ones(
    (B,T),
    device=device,
    dtype=torch.int32
)

visible_centroids = torch.randint(
    0, C,
    (B,C,VMAX),
    device=device,
    dtype=torch.int32
)

visible_counts = torch.full(
    (B,C),
    VMAX,
    device=device,
    dtype=torch.int32
)

engine = ProductionAtlasEngine()

y = engine(
    q,
    k,
    v,
    cluster_positions,
    cluster_lengths,
    membership_counts,
    visible_centroids,
    visible_counts,
)

print(y.shape)
print(y.dtype)
print(torch.isnan(y).any())
