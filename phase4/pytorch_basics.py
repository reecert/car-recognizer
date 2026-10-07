"""Phase 4 warm-up: the three ideas every PyTorch training script is built from.

  1. Tensors       - NumPy arrays that can live on the GPU
  2. Autograd      - PyTorch computes gradients for you
  3. The loop      - forward -> loss -> backward -> step -> zero_grad

We fit a straight line y = 3x + 2 to noisy points, first by hand and then the
"real" way. Training a car classifier is exactly this loop, with a bigger model.

Usage:
    python pytorch_basics.py
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


section("1. Tensors")
t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("tensor:", t.tolist(), "| shape", tuple(t.shape), "| dtype", t.dtype)
image_batch = torch.zeros(64, 3, 320, 320)  # (N, C, H, W), the shape from Phase 2
print("a batch of 64 color images:", tuple(image_batch.shape))
from_numpy = torch.from_numpy(np.arange(6).reshape(2, 3))
print("from NumPy:", from_numpy.tolist(), "(shares memory with the array)")
on_gpu = t.to(device)
print(f"moved to '{device}':", on_gpu.device, "| math must use tensors on the SAME device")


section("2. Autograd")
w = torch.tensor(2.0, requires_grad=True)  # "track everything done with w"
x = torch.tensor(5.0)
y = w * x + 1          # y = 11
y.backward()           # compute dy/dw
print(f"y = w*x + 1 = {y.item()}, and dy/dw = x = {w.grad.item()}  <- PyTorch worked that out")


section("3a. The training loop, by hand")
# Fake data: points along y = 3x + 2 with noise
xs = torch.linspace(-1, 1, 100).unsqueeze(1)          # shape (100, 1)
ys = 3 * xs + 2 + 0.1 * torch.randn_like(xs)

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1
for step in range(101):
    pred = w * xs + b                          # 1. forward: make predictions
    loss = ((pred - ys) ** 2).mean()           # 2. loss: how wrong? (mean squared error)
    loss.backward()                            # 3. backward: gradients of loss w.r.t. w and b
    with torch.no_grad():                      # 4. step: move against the gradient
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()                             # 5. zero: gradients ADD UP unless cleared
    b.grad.zero_()
    if step % 25 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")
print("target was w = 3, b = 2")


section("3b. The same loop, the standard way")
model = nn.Linear(1, 1).to(device)                       # w and b live inside the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # does the update step for us
loss_fn = nn.MSELoss()
xs_d, ys_d = xs.to(device), ys.to(device)

for step in range(101):
    pred = model(xs_d)              # 1. forward
    loss = loss_fn(pred, ys_d)      # 2. loss
    optimizer.zero_grad()           # 5. zero (done first, before backward, by convention)
    loss.backward()                 # 3. backward
    optimizer.step()                # 4. step
    if step % 25 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

w_fit, b_fit = model.weight.item(), model.bias.item()
print(f"learned w = {w_fit:.3f}, b = {b_fit:.3f}")
print("\nA car classifier uses this exact loop. Only three things change:")
print("  the model   nn.Linear (2 numbers)  ->  a CNN (millions of numbers)")
print("  the data    100 points             ->  batches of images from a DataLoader")
print("  the loss    MSELoss (a number)     ->  CrossEntropyLoss (a class)")
