import numpy as np
import torch
from torch.nn import Conv1d, Conv2d, Linear, Module
import torch.nn.functional as F

from layers import SpectralConv2d


class FNO2d(Module):
    def __init__(
        self,
        modes1,
        modes2,
        width,
        in_channels,
        out_channels,
        is_mesh=True,
        s1=40,
        s2=40,
    ):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = Linear(
            in_channels, self.width
        )  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
            s1,
            s2,
        )
        self.conv1 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )
        self.conv2 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )
        self.conv3 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )
        self.conv4 = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
            s1,
            s2,
        )
        self.w1 = Conv2d(self.width, self.width, 1)
        self.w2 = Conv2d(self.width, self.width, 1)
        self.w3 = Conv2d(self.width, self.width, 1)
        self.b0 = Conv2d(2, self.width, 1)
        self.b1 = Conv2d(2, self.width, 1)
        self.b2 = Conv2d(2, self.width, 1)
        self.b3 = Conv2d(2, self.width, 1)
        self.b4 = Conv1d(2, self.width, 1)

        self.fc1 = Linear(self.width, 128)
        self.fc2 = Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in is None:
            x_in = u
        if self.is_mesh and x_out is None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(
            0, 3, 1, 2
        )

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]

        grid_x = torch.tensor(
            np.linspace(0, 1, size_x),
            dtype=torch.float,
        )
        grid_x = grid_x.reshape(1, size_x, 1, 1).repeat([batch_size, 1, size_y, 1])

        grid_y = torch.tensor(
            np.linspace(0, 1, size_y),
            dtype=torch.float,
        )
        grid_y = grid_y.reshape(1, 1, size_y, 1).repeat([batch_size, size_x, 1, 1])

        return torch.cat((grid_x, grid_y), dim=-1).to(device)
