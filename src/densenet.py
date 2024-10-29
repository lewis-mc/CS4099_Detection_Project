import torch
import torch.nn as nn

# Define the Bottleneck layer for the DenseNet architecture
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inner_channel = 4 * growth_rate
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

# Define the Transition layer for the DenseNet architecture
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

# Define the DenseNet model
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=32, reduction=0.5, num_classes=1000):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_channels = 2 * growth_rate  # initial number of channels

        # Initial convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks and transition layers
        self.features = nn.Sequential()
        for i in range(len(nblocks)):
            self.features.add_module(f"dense_block_{i+1}", self._make_dense_block(block, num_channels, nblocks[i]))
            num_channels += nblocks[i] * growth_rate
            if i != len(nblocks) - 1:  # No transition layer after the last dense block
                out_channels = int(num_channels * reduction)
                self.features.add_module(f"transition_{i+1}", Transition(num_channels, out_channels))
                num_channels = out_channels

        # Final batch normalization and classifier
        self.features.add_module("bn", nn.BatchNorm2d(num_channels))
        self.classifier = nn.Linear(num_channels, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_dense_block(self, block, in_channels, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = torch.relu(out)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        return self.classifier(out)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

# DenseNet-121 configuration
def densenet121(num_classes=21):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, reduction=0.5, num_classes=num_classes)
