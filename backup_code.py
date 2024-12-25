class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, wandb: bool = False):
        super().__init__()
        self.sqrt_dim: float = 1 / math.sqrt(dim)
        self.eps: float = eps
        self.wandb: bool = wandb
        if wandb:
            self.scale = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def find_rms_value(self, tensor: torch.Tensor) -> float:
        norm_2 = tensor.norm(2, dim=-1)
        return norm_2 * self.sqrt_dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float() # Using 4 bit float for stability
        rms: float = self.find_rms_value(tensor)
        tensor = tensor/(rms.unsqueeze(-1) + self.eps)

        if self.wandb:
            tensor = tensor * self.scale
            tensor = tensor + self.bias

        return tensor