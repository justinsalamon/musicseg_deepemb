import numpy as np
import librosa
import torch
import torch.cuda
from torch.autograd import Variable
import os
import glob
from .fewshot_model import Conv4
from torchsummary import summary


def get_best_model_file(checkpoint_dir):
    """Return full path to model file given model folder

    Parameters
    ----------
    checkpoint_dir : str
        Path to model folder

    Returns
    -------
    str
        Path to model file
    """
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def load_fewshot_model(checkpoint_dir, is_file=False, gpu=0):
    """Load fewshot model given path to folder containing model file

    Parameters
    ----------
    checkpoint_dir : str
        Path to folder containing model file
    is_file : bool, optional
        Set to True if checkpoint_dir is actually direct path to model file, by default False
    gpu : int, optional
        Legacy, by default 0

    Returns
    -------
    [type]
        Few-shot model
    """
    if is_file:
            modelfile = checkpoint_dir
    else:
        modelfile = get_best_model_file(checkpoint_dir)
    model = Conv4()
    if torch.cuda.is_available():
        model = model.cuda()
        tmp = torch.load(modelfile)
    else:
        # device = torch.device('cpu')
        tmp = torch.load(modelfile, map_location='cpu')
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.", "")
            # an architecture model has attribute 'feature',
            # load architecture feature to backbone by casting name from 
            # 'feature.trunk.xx' to 'trunk.xx'
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()
    return model


def fewshot_inference(model, x):
    """Run fewshot model on input x

    Parameters
    ----------
    model : [type]
        Few shot model
    x : [type]
        Mel spectrogram input

    Returns
    -------
    [type]
        Fewshot embeddings
    """
    if torch.cuda.is_available():
        x = x.cuda()
    x_var = Variable(x)
    feats = model(x_var)
    return feats


# simulate hop size of 512/22050 to match hop size of other features (deepsim and cqt)
def compute_mel_features_customhop(query, 
                                   args, 
                                   custom_hop=512/22050, 
                                   verbose=False):
    """Compute mel spectrogram

    Parameters
    ----------
    query : [type]
        Audio signal sampled at 16 kHz
    args : [type]
        [description]
    custom_hop : [type], optional
        Hop size to use in seconds, by default 512/22050
    verbose : bool, optional
        If True print debug output, by default False

    Returns
    -------
    [type]
        [description]
    """
    # Extract mel.
    fftsize = 1024
    window = 400
    hop = 160
    melBin = 128
    sr = 16000
    y = query
    
    hop_sec = hop/sr
    if verbose:
        print(custom_hop)  # custom hop in seconds
        print(hop_sec)

    # Window is 0.5 second, so we need to pas by 250ms
    # mirror pad signal so that first embedding centered on time 0 
    # and last embedding centered on end of signal
    y = np.insert(y, 0, y[0:int(0.25 * sr)][::-1])
    y = np.insert(y, len(y), y[-int(0.25 * sr):][::-1])

    log_mels = None
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=fftsize,
                                         win_length=window, hop_length=hop,
                                         n_mels=melBin)
    log_mel = librosa.power_to_db(mel, ref=1.0)
    mel_S = log_mel.T

    # Construct frames
    all_frames = mel_S.shape[0]
    num_frames = 51  # mel frames for computing embedding, 50 = 0.5s
    
    # calcaulte how many mel patches we'll have using a custom hop
    num_segment = int(np.floor((all_frames - num_frames) * hop_sec / custom_hop))
    if verbose:
        print("num_segment: {}".format(num_segment))

    mel_feat = np.zeros((num_segment, num_frames, 128))
    frame_idx = 0
    chunk_idx = 0

    while chunk_idx < num_segment:
        mel_feat[chunk_idx] = mel_S[frame_idx:frame_idx + num_frames, :]
        chunk_idx += 1
        frame_idx = int(np.round(chunk_idx * custom_hop / hop_sec))

    if verbose:
        print("chunk_idx: {}".format(chunk_idx))
    
    # Fix dimensions
    mel_feat = np.swapaxes(mel_feat, 1, 2)
    mel_feat = np.expand_dims(mel_feat, 1)
    if verbose:
        print('mel shape of query: ', mel_feat.shape)
    return mel_feat


def run_fewshot_inference(audiofile, fewshot_model, verbose=False):
    """Run fewshot model on audiofile and retirn embeddings as numpy array

    Parameters
    ----------
    audiofile : str
        Path to audio file, must be sampled at 16 kHz
    fewshot_model : [type]
        Few shot model
    verbose : bool, optional
        If True print debug output, by default False

    Returns
    -------
    np.ndarray
        Fewshot model embeddings
    """
    if verbose:
        print("Extracting fewshot embeddings...", audiofile)

    # load the audio file
    sr = 16000
    audio, srload = librosa.load(audiofile, sr=None)
    assert srload == sr

    # compute input features for model (log-mel-spec)
    if verbose:
        print("Computing mel...")
    log_mels = compute_mel_features_customhop(audio, 
                                              None, 
                                              custom_hop=512/22050, 
                                              verbose=False)  
    
    n_mel_frames = 500
    idx = 0
    feats = []
    while idx < log_mels.shape[0]:
        if verbose:
            print("fewshot frames {}:{}".format(idx, idx + n_mel_frames))
        emb = fewshot_inference(fewshot_model, torch.from_numpy(log_mels[idx:idx + n_mel_frames]).float())
        emb = emb.cpu().data.numpy()
        feats.extend(emb.tolist())
        idx += n_mel_frames

    return np.asarray(feats)
