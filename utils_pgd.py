from pathlib import Path
from loss_functions import *

import tqdm
import numpy as np
import torch

from torch import nn

import torchaudio

from tslearn.metrics import dtw_path
from encoder.params_data import *

from encoder import inference as encoder
from encoder.audio import wav_to_mel_spectrogram_torch
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from torch import nn



# Use scipy sparse maybe to save space
# uses path generated from tslearn
def getAlignmentMatrices(path):
    M = len(path)
    l1m = max(path, key=lambda x: x[0])[0]
    l2m = max(path, key=lambda x: x[1])[1]

    A1 = np.zeros((M,l1m+1),dtype=np.float32)
    A2 = np.zeros((M,l2m+1),dtype=np.float32)


    for i in range(len(path)):
        p1,p2 = path[i]
        A1[i,p1] = 1.0
        A2[i,p2] = 1.0

    return torch.from_numpy(A1).detach(),torch.from_numpy(A2).detach()



def rand_assign(delta, eps, clip_min, clip_max):
    """Randomly set the data of parameter delta with uniform sampling"""
    delta.data.uniform_(-1, 1)
    if isinstance(eps, torch.Tensor):
        eps = eps.view(-1, 1)
    delta.data = torch.clamp(l2_clamp_or_normalize(delta.data, eps), clip_min, clip_max)
    return

def l2_clamp_or_normalize(tensor, eps=None):
    """Clamp tensor to eps in L2 norm"""
    xnorm = torch.norm(tensor, dim=list(range(1, tensor.dim())))
    if eps is not None:
        coeff = torch.minimum(eps / xnorm, torch.ones_like(xnorm)).view(-1, 1)
    else:
        coeff = (1.0/xnorm).view(-1, 1)
    return coeff * tensor

def reverse_bound_from_rel_bound(wav, snr, order=2):
    """From a relative eps bound, reconstruct the absolute bound for the given batch"""
    rel_eps = torch.pow(torch.tensor(10.0), float(snr) / 20)
    eps = torch.norm(wav, p=order) / rel_eps
    return eps.clone().detach().to(wav.device)


def getPreProcessedInput(voice_sample_path):
    in_fpath = Path(voice_sample_path.replace("\"", "").replace("\'", ""))
    preprocessed_wav = encoder.preprocess_wav(in_fpath,normalize=False)
    toReturn = torch.tensor(preprocessed_wav,requires_grad=True)
    return toReturn


# Specify only spectogram to skip vocoder step
# add functionality to output spectograms as image too
def FwdPass(sample,text, only_spectrogram=False):

    embed = torch.from_numpy(encoder.embed_utterance(sample,using_partials=True)).detach()
    texts = [text]
    embeds = [embed]
    specs = Synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    
    if (only_spectrogram):
        return spec.T

    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = nn.functional.pad(generated_wav, (0, Synthesizer.sample_rate), mode="constant")
    generated_wav = encoder.preprocess_wav(generated_wav,normalize=False)


    return generated_wav


# Forward Passes

# specs should of shape (t x freqbins) tensor
def compute_objective_spectogram(synth_speech_spec, target_speech_spec, loss_func, threshold=17.5):
    np1 = (synth_speech_spec - synth_speech_spec.mean())/synth_speech_spec.std()
    np2 = (target_speech_spec - target_speech_spec.mean())/target_speech_spec.std()

    pathCalc1 = np1.clone().detach().numpy()
    pathCalc2 = np2.clone().detach().numpy()

    optimal_path, dtw_score = dtw_path(pathCalc1, pathCalc2, global_constraint="sakoe_chiba", sakoe_chiba_radius=5)
    A1,A2 = getAlignmentMatrices(optimal_path)

    stretched1 = torch.matmul(A1,np1)
    stretched2 = torch.matmul(A2,np2)


    return loss_func(stretched1,stretched2)


# Misc

def getLevenshtein(og_text,audio_file_path):
    device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU

    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                          model='silero_stt',
                                          language='en', # also available 'de', 'es'
                                          device=device)

    (read_batch, split_into_batches,
    read_audio, prepare_model_input) = utils  # see function signature for details

    test_files = glob(f'{filename}')
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)

    output = model(input)
    return torchaudio.functional.edit_distance(output[0].split(' '), og_text.split(' '))

# preprocesses spectogram to remove quiet sections
def preprocessSpec(wav_init, clipping_threshold):
    wav_init_spec = wav_to_mel_spectrogram_torch(wav_init,factor=2).detach()
    wav_init_spec = torch.log(torch.clamp(wav_init_spec, min=1e-10, max=1e5))

    np1_scale = wav_init_spec.clone().detach().numpy()
    np1_scale = (np1_scale - np1_scale.min())/np1_scale.std()
    mask = np.linalg.norm(np1_scale, axis=1) <= clipping_threshold
    wav_init_spec = torch.tensor(wav_init_spec[~mask, :],requires_grad=True)

    return wav_init_spec


def pgd(target_text, wav_init, forward, lossfunc, snr = 35, nb_iter = 50, clip_min = -10, clip_max = 10, lr = 0.1, sample_rate=16000, clipping_threshold=17.5):
    """Returns perturbed input"""

    delta = torch.zeros_like(wav_init)
    delta = nn.Parameter(delta)

    eps = reverse_bound_from_rel_bound(wav_init, snr)

    rand_assign(delta, eps, clip_min, clip_max)
    eps_iter = eps * lr

    delta.requires_grad_()
    
    wav_init_spec = preprocessSpec(wav_init, clipping_threshold)

    toReturn = []
    best_noise = None
    max_loss = -99999999999

    for _ in tqdm.tqdm(range(nb_iter)):

        delta = torch.squeeze(delta)

        predictions = forward(wav_init + delta, target_text) # spectrogram
        loss = lossfunc(predictions, wav_init_spec)

        loss.backward(inputs = delta)

        toReturn.append(loss)

        if (loss >= max_loss):
          max_loss = loss
          best_noise = wav_init + delta

        grad = delta.grad.data
        grad = l2_clamp_or_normalize(grad)
        delta.data = delta.data + eps_iter * grad
        delta.data = (
            torch.clamp(wav_init.data + delta.data, clip_min, clip_max)
            - wav_init.data
        )
        delta.data = l2_clamp_or_normalize(delta.data, eps)

    return wav_init + delta, delta, toReturn, best_noise