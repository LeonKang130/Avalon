import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(60, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )
    def forward(self, latent_code: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        n = latent_code[..., :6].reshape(latent_code.shape[:-1] + (2, 3))
        t = latent_code[..., 6:12].reshape(latent_code.shape[:-1] + (2, 3))
        n = n / (torch.linalg.norm(n, dim=-1).unsqueeze(-1) + 1e-6)
        t = t - n * (n * t).sum(dim=-1, keepdim=True)
        t = t / (torch.linalg.norm(t, dim=-1).unsqueeze(-1) + 1e-6)
        b = torch.linalg.cross(n, t)
        onb = torch.hstack((t, b, n)).reshape(latent_code.shape[:-1] + (6, 3))
        ws_local = torch.einsum("...ij,...kj->...ki", onb, ws).reshape(ws.shape[:-2] + (-1,))
        if ws_local.shape[:-1] == latent_code.shape[:-1]:
            x = torch.hstack((ws_local, latent_code[..., 12:]))
        else:
            x = torch.hstack((ws_local, latent_code[..., 12:].unsqueeze(0).expand(ws_local.shape[:-1] + (-1,))))
        return self.fc(x)


class Avalon(nn.Module):
    def __init__(self):
        super(Avalon, self).__init__()
        self.encoder = torch.jit.script(Encoder())
        self.decoder = torch.jit.script(Decoder())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        absdf, ws = x[..., :-6], x[..., -6:].reshape(x.shape[:-1] + (2, 3))
        lc = self.encoder(absdf)
        return self.decoder(lc, ws)