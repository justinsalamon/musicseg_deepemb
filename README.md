# Deep Embeddings and Section Fusion Improve Music Segmentation

<a href="https://replicate.ai/musicseg_deepemb"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=blue"></a>

# Installation

## Step 1: install non-python dependencies
Install non-python dependencies:
* ffmpeg
* sox (with support for mp3)
* libsndfile

On Debian/Ubuntu the command would be:
```bash
apt-get update --fix-missing && apt-get install libsndfile1 ffmpeg libsox-fmt-all sox -y
```

## Step 2: install the `musicsections` package
```bash
cd musicsections
pip install -e .
```

# Example

```python
import musicsections

deepsim_model_folder = "/path/to/deepsim/model/folder"
fewshot_model_folder = "/path/to/fewshot/model/folder"

model_deepsim = musicsections.load_deepsim_model(deepsim_model_folder)
model_fewshot = musicsections.load_fewshot_model(fewshot_model_folder)

audiofile = "/path/to/audiofile.mp3"

segmentations, features = musicsections.segment_file(
    audiofile, 
    deepsim_model=model_deepsim,
    fewshot_model=model_fewshot,
    min_duration=8,
    mu=0.5,
    gamma=0.5,
    beats_alg="madmom",
    beats_file=None)

musicsections.plot_segmentation(segmentations)
```

# Reference
```
@inproceedings{Salamon:Segmentation:ISMIR:2021,
	Author = {J. Salamon and O. Nieto and N.J. Bryan},
	Booktitle = {Proc.~22nd International Conference on Music Information Retrieval (ISMIR)},
	Month = {Nov.},
	Title = {Deep Embeddings and Section Fusion Improve Music Segmentation},
	Year = {2021}}
```