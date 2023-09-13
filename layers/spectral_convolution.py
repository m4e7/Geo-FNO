import numpy as np

import torch
from torch.nn import Module, Parameter

class SpectralConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
        s1=32,
        s2=32,
    ):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = Parameter(
            self.scale * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = Parameter(
            self.scale * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    @staticmethod
    def compl_mul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batch_size = u.shape[0]

        # Compute Fourier coefficients up to factor of e^(- something constant)
        if x_in is None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = SpectralConv2d.compl_mul2d(
            u_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        factor2 = SpectralConv2d.compl_mul2d(
            u_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        if x_out is None:
            out_ft = torch.zeros(
                batch_size,
                self.out_channels,
                s1,
                s2 // 2 + 1,
                dtype=torch.cfloat,
                device=u.device,
            )
            out_ft[:, :, : self.modes1, : self.modes2] = factor1
            out_ft[:, :, -self.modes1 :, : self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batch_size = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wave-number (m1, m2)
        k_x1 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes1, step=1),
                    torch.arange(start=-self.modes1, end=0, step=1),
                ),
                0,
            )
            .reshape(m1, 1)
            .repeat(1, m2)
            .to(device)
        )
        k_x2 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes2, step=1),
                    torch.arange(start=-(self.modes2 - 1), end=0, step=1),
                ),
                0,
            )
            .reshape(1, m2)
            .repeat(m1, 1)
            .to(device)
        )

        # print(x_in.shape)
        if iphi is None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # k = <y, k_x>,  (batch, N, m1, m2)
        k1 = torch.outer(x[..., 0].view(-1), k_x1.view(-1)).reshape(
            batch_size, N, m1, m2
        )
        k2 = torch.outer(x[..., 1].view(-1), k_x2.view(-1)).reshape(
            batch_size, N, m1, m2
        )
        k = k1 + k2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * k).to(device)

        # y (batch, channels, N)
        u = u + 0j
        y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batch_size = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wave-number (m1, m2)
        k_x1 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes1, step=1),
                    torch.arange(start=-self.modes1, end=0, step=1),
                ),
                0,
            )
            .reshape(m1, 1)
            .repeat(1, m2)
            .to(device)
        )
        k_x2 = (
            torch.cat(
                (
                    torch.arange(start=0, end=self.modes2, step=1),
                    torch.arange(start=-(self.modes2 - 1), end=0, step=1),
                ),
                0,
            )
            .reshape(1, m2)
            .repeat(m1, 1)
            .to(device)
        )

        if iphi is None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # k = <y, k_x>,  (batch, N, m1, m2)
        k1 = torch.outer(x[:, :, 0].view(-1), k_x1.view(-1)).reshape(
            batch_size, N, m1, m2
        )
        k2 = torch.outer(x[:, :, 1].view(-1), k_x2.view(-1)).reshape(
            batch_size, N, m1, m2
        )
        k = k1 + k2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * k).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # y (batch, channels, N)
        y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        return y.real
