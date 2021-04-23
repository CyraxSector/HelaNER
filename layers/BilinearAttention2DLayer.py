import torch
import torch.nn as nn


class BilinearAttention2DLayer(nn.Module):
    def __init__(self, x_hidden_size, y_hidden_size, bias=False):
        super(BilinearAttention2DLayer, self).__init__()

        self.x_hidden_size = x_hidden_size
        self.y_hidden_size = y_hidden_size
        self.linear = torch.nn.Linear(self.x_hidden_size, y_hidden_size, bias=bias)

    def forward(self, x, y):
        xW = self.linear(x)
        xWy = torch.matmul(xW, y.transpose(1, 2))

        return xWy


if __name__ == "__main__":
    torch.cuda.set_device(4)
    device = torch.device("cuda:4")

    a = [[[1.0, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[7, 8, 9], [10, 11, 12], [7, 8, 9]]]
    b = [[[1.0, 2], [3, 4]],
         [[5, 6], [7, 8]]]
    a = torch.tensor(a)
    b = torch.tensor(b)
    layer = BilinearAttention2DLayer(3, 2)
    print(layer(a, b))
