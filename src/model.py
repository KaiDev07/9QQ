# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual Block для сохранения градиентов в глубокой сети."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ToguzZeroResNet(nn.Module):
    """
    Главная нейросеть AlphaZero.
    Input: Состояние доски.
    Output: 
      1. Policy (вероятности ходов)
      2. Value (оценка позиции от -1 до 1)
    """
    def __init__(self, num_res_blocks=5, num_channels=64):
        super().__init__()
        # Вход: 2 канала (доска + туздыки)
        self.start_conv = nn.Conv1d(2, num_channels, kernel_size=3, padding=1)
        self.bn_start = nn.BatchNorm1d(num_channels)
        
        # Магистраль ResNet
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Policy Head (Куда ходить?)
        self.policy_conv = nn.Conv1d(num_channels, 16, kernel_size=1)
        self.policy_bn = nn.BatchNorm1d(16)
        self.policy_fc = nn.Linear(16 * 18, 9)
        
        # Value Head (Кто выигрывает?)
        self.value_conv = nn.Conv1d(num_channels, 4, kernel_size=1)
        self.value_bn = nn.BatchNorm1d(4)
        self.value_fc1 = nn.Linear(4 * 18 + 2, 64) # +2 для текущего счета в казанах
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, kazans):
        out = F.relu(self.bn_start(self.start_conv(x)))
        for block in self.res_blocks:
            out = block(out)
        
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = torch.cat([v, kazans], dim=1) # Добавляем информацию о счете
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v