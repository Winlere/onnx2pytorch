import torch
from torch import nn


class Gather(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim
        self.selection = [slice(None) for _ in range(dim)]

    def forward(self, data: torch.Tensor, indices: torch.Tensor):
        indices = indices.to(torch.int64)

        if indices.numel() == 1 and indices == -1:
            indices = torch.tensor(
                data.shape[self.dim] - 1,
                device=data.device,
                dtype=torch.int64,
            )

        if indices.ndim == 0:
            out = torch.index_select(
                data,
                dim=self.dim,
                index=indices.reshape(1),
            )
            return out.select(self.dim, 0)

        return torch.index_select(data, dim=self.dim, index=indices)
