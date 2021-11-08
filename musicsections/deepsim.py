import librosa
import os
import numpy as np
import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.models import load_model
import json

# Reduce gpu memory allocation.
import tensorflow as tf


def configure_gpus():
    """ Set GPU memory to grow
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


# Globally force all GPUS to "grow" their memory
configure_gpus()


def compute_mel_features(query, args, emb_hop=1, verbose=False, magicnorm=True):
    """Compute mel features from input audio signal

    Parameters
    ----------
    query : np.ndarray
        Inpu audio signal sampled at 22kHz
    args : [type]
        Arguments for deepsim model
    emb_hop : int, optional
        Hop size between embeddings, in frames, by default 1
    verbose : bool, optional
        If True print out debug information
    magicnorm : bool, optional
        Apply fixed normalization parameters required for deepsim model, by default True.
    """
    # Extract mel.
    fftsize = 1024
    window = 1024
    hop = 512
    melBin = 128
    sr = 22050
    y = query

    # mirror pad signal so that first embedding centered on time 0 and 
    # last embedding centered on end of signal
    y = np.insert(y, 0, y[0:int(1.5 * sr)][::-1])
    y = np.insert(y, len(y), y[-int(1.5 * sr):][::-1])

    S = librosa.core.stft(y, n_fft=fftsize, hop_length=hop, win_length=window)
    X = np.abs(S)

    mel_basis = librosa.filters.mel(sr, n_fft=fftsize, n_mels=melBin)
    mel_S = np.dot(mel_basis, X)

    # value log compression
    mel_S = np.log10(1 + 10 * mel_S)
    mel_S = mel_S.astype(np.float32)
    mel_S = mel_S.T

    # Construct frames
    all_frames = mel_S.shape[0]
    num_frames = 129  # mel frames for computing embedding, 129 = 3s
    num_segment = int((all_frames - num_frames) / emb_hop)

    mel_feat = np.zeros((num_segment, num_frames, 128))
    frame_idx = 0
    chunk_idx = 0
    while frame_idx < (all_frames - num_frames):
        mel_feat[chunk_idx] = mel_S[frame_idx:frame_idx + num_frames, :]
        frame_idx += emb_hop
        chunk_idx += 1

    mel_feat = np.expand_dims(mel_feat, axis=-1)

    if magicnorm or (getattr(args, "inputnorm", None) is not None and args.inputnorm == 'norm'):
        mel_feat -= 0.20
        mel_feat /= 0.25

    if verbose:
        print('mel shape of query: ', mel_feat.shape)
    return mel_feat


def run_deepsim_inference(query, base_model, args, session, verbose=False, magicnorm=True):
    """Run deepsim model on audio signal

    Parameters
    ----------
    query : np.ndarray
        The audio signal, must be sampled at 22kHz
    base_model : [type]
        Deepsim model object
    args : [type]
        Arguments for deepsim model
    session : [type]
        Tensorflow session
    verbose : bool, optional
        If True print out debug information
    magicnorm : bool, optional
        Apply fixed normalization parameters required for deepsim model, by default True.

    Returns
    -------
    np.ndarray
        Deepsim embeddings
    """
    data_config_conditions = {'track': 0, 'genre': 1, 'mood': 2, 'instr': 3, 'tempo': 4, 'era': 8}  
    num_to_c_map = {v: k for k, v in data_config_conditions.items()}
    conditions_here = [num_to_c_map[int(x)] for x in args.conditions]

    # For condition rearrange.
    new_condition_map = {}
    for i, x in enumerate(conditions_here):
        new_condition_map[x] = i
    new_index_to_condition = {v: k for k, v in new_condition_map.items()}
    new_index_to_condition[len(new_condition_map)] = 'alldim'

    # Embedding index list.
    if args.use_c == 1:
        embedding_choices = list(new_condition_map.values())
        embedding_choices.append(len(embedding_choices))
    elif args.use_c == 0:
        embedding_choices = list(new_condition_map.values())
        embedding_choices = [len(embedding_choices)]

    # Conditions.
    conditions = list(data_config_conditions.keys())

    # BREAK INTO BATCHES OF 5000 MEL FRAMES
    log_mels = compute_mel_features(query, args, emb_hop=1, magicnorm=magicnorm)

    n_mel_frames = 500
    idx = 0
    feats = []

    with session.as_default():
        with session.graph.as_default():
            while idx < log_mels.shape[0]:
                if verbose:
                    print("sim frames {}:{}".format(idx, idx + n_mel_frames))
                emb = np.array(base_model.predict(log_mels[idx:idx + n_mel_frames]))
                feats.extend(emb.tolist())
                idx += n_mel_frames

    feats = np.asarray(feats)
    if verbose:
        print("tmp.shape", feats.shape)
    return feats


class DeepSimModel:
    """
    Simple class that converts an audio file path to an embedding vector
    """

    def __init__(self):
        self.session = None
        self.base_model = None
        self.args = None

    def reset(self):
        # Reset estimators
        pass

    def initialize(self, model_filepath, arg_filepath):
        self.session = tf.compat.v1.keras.backend.get_session()

        # Get base model.
        with self.session.as_default():
            with self.session.graph.as_default():
                model = load_model(model_filepath, compile=False)
                self.base_model = model.get_layer('base_model')

        # Get arguments
        with open(arg_filepath, 'r') as f:
            params = json.load(f)
            from collections import namedtuple
            self.args = namedtuple("my_args", params.keys())(*params.values())

    def run(self, audio_filepath, magicnorm=False):
        y, sr = librosa.load(audio_filepath, sr=22050)
        return run_deepsim_inference(
            y, self.base_model, self.args, self.session, magicnorm=magicnorm).tolist()


def load_deepsim_model(model_folder_path):
    """
    Utility function for loading the deepsim model
    
    Parameters
    ----------
    model_folder_path: str
        Path to folder containing the model files

    Returns
    -------
    model_deepsim : DeepSimModel
        Model object

    """
    # Load the model
    model_deepsim = DeepSimModel()
    model_filepath = os.path.join(model_folder_path, 'best.h5')
    arg_filepath = os.path.join(model_folder_path, 'args.json')
    model_deepsim.initialize(model_filepath, arg_filepath)

    return model_deepsim
