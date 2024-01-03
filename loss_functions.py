import numpy as np
import torch

from torch import nn
from torchaudio import transforms
from torch import nn
from pystoi.utils import thirdoct
from pystoi.stoi import NUMBAND, MINFREQ, N, BETA
from torch.nn.functional import unfold



ToSpectrogram = transforms.Spectrogram(power=1,n_fft=320)
cel = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=False)
l1loss = nn.L1Loss()

@staticmethod
def stft(x, win, fft_size, overlap=4):
    """We can't use torch.stft:
    - It's buggy with center=False as it discards the last frame
    - It pads the frame left and right before taking the fft instead
    of padding right
    Instead we unfold and take rfft. This gives the same result as
    pystoi.utils.stft.
    """
    win_len = win.shape[0]
    hop = int(win_len / overlap)
    frames = unfold(x[:, None, None, :], kernel_size=(1, win_len),
                    stride=(1, hop))
    return torch.fft.rfft(frames*win[:, None], n=fft_size, dim=1)


# (batch x freqbins x time)
def stoi_loss(x_spec, y_spec):
    nfft = 159
    win = torch.from_numpy(np.hanning(80 + 2)[1:-1]).float()
    win = nn.Parameter(win, requires_grad=False)
    obm_mat = thirdoct(16000, nfft, NUMBAND, MINFREQ)[0]
    OBM = nn.Parameter(torch.from_numpy(obm_mat).float(),
                                    requires_grad=False)

    x_tob = torch.clamp(torch.matmul(OBM, x_spec.abs().pow(2) + EPS),min=EPS,max=10**6).sqrt().float()
    y_tob = torch.clamp(torch.matmul(OBM, y_spec.abs().pow(2) + EPS),min=EPS,max=10**6).sqrt().float()

    x_seg = unfold(x_tob.unsqueeze(2),
                    kernel_size=(1, N),
                    stride=(1, 1)).view(1, x_tob.shape[1], N, -1)
    y_seg = unfold(y_tob.unsqueeze(2),
                    kernel_size=(1, N),
                    stride=(1, 1)).view(1, y_tob.shape[1], N, -1)

    norm_const = (
        x_seg.norm(p=2, dim=2, keepdim=True) /
        (y_seg.norm(p=2, dim=2, keepdim=True) + EPS)
    )
    y_seg_normed = y_seg * norm_const
    # Clip as described in [1]
    clip_val = 10 ** (-BETA / 20)
    y_prim = torch.min(y_seg_normed, x_seg * (1 + clip_val))
    # Mean/var normalize vectors
    # No need to pass the mask because zeros do not affect statistics
    y_prim = y_prim - y_prim.mean(2, keepdim=True)
    x_seg = x_seg - x_seg.mean(2, keepdim=True)
    y_prim = y_prim / (y_prim.norm(p=2, dim=2, keepdim=True) + EPS)
    x_seg = x_seg / (x_seg.norm(p=2, dim=2, keepdim=True) + EPS)
    # Matrix with entries summing to sum of correlations of vectors
    corr_comp = y_prim * x_seg
    corr_comp = corr_comp.sum(2)

    output = corr_comp.mean(1)
    output = -output.mean(-1)
    return output


# Spec Entropy
def l2loss(orig_spec, noised_spec):
    entropy = torch.linalg.norm(orig_spec - noised_spec,dim=1).mean()
    return entropy

def crossEntropy(noised_spec,orig_spec):
    print("SHAPES", noised_spec.shape, orig_spec.shape)
    return kl_loss(noised_spec.softmax(dim=1),orig_spec.softmax(dim=1))

def l1Entropy(input,target):
    return l1loss(input,target)