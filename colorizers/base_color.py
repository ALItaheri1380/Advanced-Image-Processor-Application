import torch
from torch import nn

class ColorNormalizer(nn.Module):
    def __init__(self):
        super(ColorNormalizer, self).__init__()
        self.l_center = 50.
        self.l_range = 100.
        self.ab_range = 110.

    def normalize_l(self, channel_l):
        return (channel_l - self.l_center) / self.l_range

    def denormalize_l(self, channel_l):
        return channel_l * self.l_range + self.l_center

    def normalize_ab(self, channels_ab):
        return channels_ab / self.ab_range

    def denormalize_ab(self, channels_ab):
        return channels_ab * self.ab_range
