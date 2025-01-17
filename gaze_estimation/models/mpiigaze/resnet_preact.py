import torch
import torch.nn as nn
import torch.nn.functional as F
import yacs.config


def initialize_weights(module: torch.nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.zeros_(module.bias)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          padding=0,
                          bias=False))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class Model(nn.Module):
    def __init__(self, config: yacs.config.CfgNode):
        super().__init__()

        depth = 8
        base_channels = 16
        input_shape = (1, 1, 64, 256)

        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv = nn.Conv2d(input_shape[1],
                              n_channels[0],
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1,
                              bias=False)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[0],
                                       n_blocks_per_stage,
                                       BasicBlock,
                                       stride=1)
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks_per_stage,
                                       BasicBlock,
                                       stride=2)
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks_per_stage,
                                       BasicBlock,
                                       stride=2)
        self.bn = nn.BatchNorm2d(n_channels[2])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).size(0)
        print("HERES the feature size", self.feature_size)
        self.fc1 = nn.Linear(6, 3)



        self.fc2 = nn.Linear(self.feature_size + 5, 2)

        self.fc = nn.Linear(5, 2)
        self.apply(initialize_weights)

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, n_blocks: int,
                    block: torch.nn.Module, stride: int) -> torch.nn.Module:
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name, block(in_channels, out_channels,
                                      stride=stride))
            else:
                stage.add_module(block_name,
                                 block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x: torch.tensor) -> torch.tensor:
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x: torch.tensor, y: torch.tensor, z1: torch.tensor, z2: torch.tensor, z3: torch.tensor) -> torch.tensor:
        if(torch.isnan(x).any()):
          print("X has nan")
        
        if(torch.isnan(y).any()):
          print("y has nan")

        if(torch.isnan(z1).any()):
          print("z1 has nan")
        

        if(torch.isnan(z2).any()):
          print("z2 has nan")

        if(torch.isnan(z3).any()):
          print("z3 has nan")
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, y], dim=1)
      
        # print("XXXXXXXXXX")
        # print(x.shape)
        z11 = self.fc1(z3).unsqueeze(2)

        # print("DATA SHAPE", z11.shape, z2.shape, z1.shape)
        z12 = torch.matmul(z2, z11) + z1

        norm_z12 =  torch.norm(z12, dim=1)
        normz = torch.cat((norm_z12, norm_z12, norm_z12), dim=1)
        # print("NORM SHAPE", z12.shape, normz.shape)
        z12 = z12.squeeze(2)
        z12 = torch.div(z12, normz)
        # print("NORM SHAPE", z12.shape)
        
        # print("AFTER DATA TYPE", x.shape, z12.shape)

        x = torch.cat((x, z12), dim = 1)
        # print("X DATA SHAPE", x.shape)


        x = self.fc2(x)
        # print("model(X) DATA SHAPE", x.shape)

        
        return x
