import torch

class ConvNet(torch.nn.Module):
    def __init__(self, dim_output: int) -> None:
        super().__init__()
        self.dim_output = dim_output
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),

            torch.nn.LazyLinear(out_features=self.dim_output),
            torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.net.forward(input=input)

        return torch.exp(input=out)