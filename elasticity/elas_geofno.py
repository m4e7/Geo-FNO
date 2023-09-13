"""
@author: Zongyi Li, Daniel Zhengyu Huang, Mogab Elleithy
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
import yaml

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utilities3 import count_params, LpLoss
from models import FNO2d
from layers import CoordinateTransform


# device = "cuda"
device = "cpu"

np.random.seed(0)
torch.manual_seed(0)
if device == "cuda":
    torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


################################################################
# configs
################################################################
class Config(NamedTuple):
    data_directory: Path
    filepath_sigma: str
    filepath_xy: str
    filepath_rr: str

    n_total: int
    n_train: int
    n_test: int
    batch_size: int
    learning_rate: float
    epochs: int
    step_size: int
    gamma: float
    modes: int
    width: int

    @staticmethod
    def from_yaml(config_path: str):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)

        return Config(
            data_directory=Path(config.get("data_directory")),
            filepath_sigma=config.get("filepath_sigma"),
            filepath_xy=config.get("filepath_xy"),
            filepath_rr=config.get("filepath_rr"),
            n_total=config.get("n_total"),
            n_train=config.get("n_train"),
            n_test=config.get("n_test"),
            batch_size=config.get("batch_size"),
            learning_rate=config.get("learning_rate"),
            epochs=config.get("epochs"),
            step_size=config.get("step_size"),
            gamma=config.get("gamma"),
            modes=config.get("modes"),
            width=config.get("width"),
        )


################################################################
# load data and data normalization
################################################################
cfg = Config.from_yaml(
    "C:\\Users\\orang\\experimental\\Geo-FNO\\elasticity\\elas_geofno.yaml"
)

input_rr = np.load(cfg.data_directory / cfg.filepath_rr)
input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1, 0)
input_s = np.load(cfg.data_directory / cfg.filepath_sigma)
input_s = torch.tensor(input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
input_xy = np.load(cfg.data_directory / cfg.filepath_xy)
input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)

train_rr = input_rr[: cfg.n_train]
test_rr = input_rr[-cfg.n_test :]
train_s = input_s[: cfg.n_train]
test_s = input_s[-cfg.n_test :]
train_xy = input_xy[: cfg.n_train]
test_xy = input_xy[-cfg.n_test :]

print(train_rr.shape, train_s.shape, train_xy.shape)

train_loader = DataLoader(
    TensorDataset(train_rr, train_s, train_xy),
    batch_size=cfg.batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    TensorDataset(test_rr, test_s, test_xy),
    batch_size=cfg.batch_size,
    shuffle=False,
)


################################################################
# training and evaluation
################################################################
def show_subplots(xy, truth, pred):
    lims = dict(
        cmap="RdBu_r",
        vmin=truth.min(),
        vmax=truth.max(),
        edgecolor="w",
        lw=0.1,
    )

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    ax[0].scatter(xy[:, 0], xy[:, 1], 100, truth, **lims)
    ax[1].scatter(xy[:, 0], xy[:, 1], 100, pred, **lims)
    ax[2].scatter(xy[:, 0], xy[:, 1], 100, truth - pred, **lims)
    fig.show()


model = FNO2d(
    cfg.modes,
    cfg.modes,
    cfg.width,
    in_channels=2,
    out_channels=1,
).to(device)
model_iphi = CoordinateTransform().to(device)
print(
    f"FNO parameter count: {count_params(model)}\n",
    f"Coordinate transformation (inverse Phi) "
    f"parameter count: {count_params(model_iphi)}",
)

params = list(model.parameters()) + list(model_iphi.parameters())
optimizer = Adam(params, lr=cfg.learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.step_size,
    gamma=cfg.gamma,
)

lp_loss = LpLoss(size_average=False)
N_sample = 1000
for ep in range(cfg.epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for rr, sigma, mesh in train_loader:
        rr, sigma, mesh = rr.to(device), sigma.to(device), mesh.to(device)
        samples_x = torch.rand(cfg.batch_size, N_sample, 2).to(device) * 3 - 1

        optimizer.zero_grad()
        out = model(mesh, code=rr, iphi=model_iphi)
        samples_xi = model_iphi(samples_x, code=rr)

        loss_data = lp_loss(
            out.view(cfg.batch_size, -1), sigma.view(cfg.batch_size, -1)
        )
        loss_reg = lp_loss(samples_xi, samples_x)
        loss = loss_data + 0.001 * loss_reg
        loss.backward()

        optimizer.step()
        train_l2 += loss_data.item()
        train_reg += loss_reg.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for rr, sigma, mesh in test_loader:
            rr, sigma, mesh = rr.to(device), sigma.to(device), mesh.to(device)
            # out = model(mesh, iphi=model_iphi)
            out = model(mesh, code=rr, iphi=model_iphi)
            test_l2 += lp_loss(
                out.view(cfg.batch_size, -1), sigma.view(cfg.batch_size, -1)
            ).item()

    train_l2 /= cfg.n_train
    train_reg /= cfg.n_train
    test_l2 /= cfg.n_test

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, train_reg, test_l2)

    if ep % 100 == 0:
        show_subplots(
            mesh[-1].squeeze().detach().cpu().numpy(),
            sigma[-1].squeeze().detach().cpu().numpy(),
            out[-1].squeeze().detach().cpu().numpy(),
        )

show_subplots(
    mesh[-1].squeeze().detach().cpu().numpy(),
    sigma[-1].squeeze().detach().cpu().numpy(),
    out[-1].squeeze().detach().cpu().numpy(),
)
