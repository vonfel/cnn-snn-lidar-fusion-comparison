import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNEncoder(nn.Module):
    """
    3-layer LIF SNN encoder for 2-channel time surface input.
    Uses surrogate gradient (fast sigmoid) for backprop through spikes.

    Input:  (B, 2, 480, 640) time surface (ON/OFF polarity channels)
    Output: (B, 64, 120, 160) mean firing rate map (values in [0, 1])

    Architecture:
        Conv(2,  16, 3, pad=1)           + LIF -> (B, 16, 480, 640)
        Conv(16, 32, 3, pad=1, stride=2) + LIF -> (B, 32, 240, 320)
        Conv(32, 64, 3, pad=1, stride=2) + LIF -> (B, 64, 120, 160)

    NOTE: No MaxPool — stride-2 convolutions are used for downsampling.
    MaxPool with spiking neurons discards spike information and is
    problematic for surrogate gradient training (Gaurav et al. IJCNN 2022).

    Rate coding: the same time surface is presented for num_steps steps.
    The mean spike count across steps is returned as the attention map.
    """

    def __init__(self, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()

        # Layer 1: (B, 2, 480, 640) -> (B, 16, 480, 640)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               init_hidden=True)

        # Layer 2: (B, 16, 480, 640) -> (B, 32, 240, 320)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               init_hidden=True)

        # Layer 3: (B, 32, 240, 320) -> (B, 64, 120, 160)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               init_hidden=True)

    def forward(self, x, num_steps=4):
        """
        Run num_steps of the SNN with rate coding (same input each step).
        Returns mean firing rate across steps as the attention map.

        Args:
            x:         (B, 2, 480, 640) time surface
            num_steps: number of simulation steps (default 4)

        Returns:
            attention_map: (B, 64, 120, 160), values in [0, 1]
        """
        # Reset membrane potentials for this batch
        self.lif1.init_leaky()
        self.lif2.init_leaky()
        self.lif3.init_leaky()

        spike_accumulator = None

        for _ in range(num_steps):
            # Same input at every step (rate coding)
            cur1 = self.conv1(x)
            spk1 = self.lif1(cur1)

            cur2 = self.conv2(spk1)
            spk2 = self.lif2(cur2)

            cur3 = self.conv3(spk2)
            spk3 = self.lif3(cur3)

            if spike_accumulator is None:
                spike_accumulator = spk3
            else:
                spike_accumulator = spike_accumulator + spk3

        # Mean firing rate: values in {0, 0.25, 0.5, 0.75, 1.0} for num_steps=4
        attention_map = spike_accumulator / num_steps
        return attention_map
