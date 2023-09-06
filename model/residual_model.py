import torch
import torch.nn as nn
import torch.nn.init as init


class Network(nn.Module):
    def __init__(self, input_channels, embedding):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, embedding,
                               kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(embedding, embedding,
                               kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = nn.Conv2d(embedding, input_channels,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out += identity
        return out

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param)
            elif 'bias' in name:
                init.zeros_(param)


if __name__ == '__main__':
    input = torch.randn([16, 3, 500, 500])
    model = Network(input_channels=3, embedding=52)
    out = model.forward(input)
    print(out.shape)
