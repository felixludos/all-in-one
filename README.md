# All-In-One Music Structure Analyzer

[![Visual Demo](https://img.shields.io/badge/Visual-Demo-8A2BE2)](https://taejun.kim/music-dissector/)
[![arXiv](https://img.shields.io/badge/arXiv-2307.16425-B31B1B)](http://arxiv.org/abs/2307.16425/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-f9f107)](https://huggingface.co/spaces/taejunkim/all-in-one/)
[![PyPI - Version](https://img.shields.io/pypi/v/allin1.svg)](https://pypi.org/project/allin1)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/allin1.svg)](https://pypi.org/project/allin1)

This package provides models for music structure analysis, predicting:
1. Tempo (BPM)
2. Beats
3. Downbeats
4. Functional segment boundaries
5. Functional segment labels (e.g., intro, verse, chorus, bridge, outro)


-----


**Table of Contents**

- [Installation](#installation)
- [HTTP API and Docker](#http-api-and-docker)
- [Usage for CLI](#usage-for-cli)
- [Usage for Python](#usage-for-python)
- [Visualization & Sonification](#visualization--sonification)
- [Available Models](#available-models)
- [Speed](#speed)
- [Advanced Usage for Research](#advanced-usage-for-research)
- [Concerning MP3 Files](#concerning-mp3-files)
- [Training](TRAINING.md)
- [Citation](#citation)

## Installation

### 1. Install PyTorch

Visit [PyTorch](https://pytorch.org/) and install the appropriate version for your system.

### 2. Install NATTEN (Required for Linux and Windows; macOS will auto-install)
* **Linux**: Download from [NATTEN website](https://www.shi-labs.com/natten/)
* **macOS**: Auto-installs with `allin1`.
* **Windows**: Build from source:
```shell
pip install ninja # Recommended, not required
git clone https://github.com/SHI-Labs/NATTEN
cd NATTEN
make
```

### 3. Install the package

```shell
pip install git+https://github.com/CPJKU/madmom  # install the latest madmom directly from GitHub
pip install allin1  # install this package
```

### 4. (Optional) Install FFmpeg for MP3 support

For ubuntu:

```shell
sudo apt install ffmpeg
```

For macOS:

```shell
brew install ffmpeg
```


## HTTP API and Docker

The repository includes a FastAPI service without changing the existing CLI or
Python API. OpenAPI documentation is available at
[http://localhost:4434/docs](http://localhost:4434/docs), and the readiness
endpoint is `GET /health`.

### GPU build and launch

Docker Engine, Docker Compose v2, the NVIDIA driver, and the NVIDIA Container
Toolkit are required. The default Compose file reserves one GPU and deliberately
runs one Uvicorn worker so model copies cannot contend for GPU memory.

```shell
docker compose up -d --build
docker compose ps
curl --fail http://localhost:4434/health
```

The service is then available at `http://localhost:4434`. Model downloads are
cached in the named `model-cache` volume and survive container recreation.
`restart: unless-stopped` restarts the service after Docker or the host
restarts. The first real analysis can take longer while Demucs and All-In-One
weights are downloaded.

### Analyze with curl

A normal request returns JSON:

```shell
curl --fail-with-body \
  -F 'files=@song.wav' \
  -F 'device=cuda' \
  http://localhost:4434/analyze
```

Repeat the `files` field to analyze more than one track. Requests that ask for
generated files, retained byproducts, activations, or embeddings return a ZIP:

```shell
curl --fail-with-body \
  -F 'files=@song.wav' \
  -F 'visualize=true' \
  -F 'sonify=true' \
  -F 'include_activations=true' \
  -F 'include_embeddings=true' \
  -o allin1-results.zip \
  http://localhost:4434/analyze
unzip allin1-results.zip -d allin1-results
```

The archive contains `results.json`, generated visualizations/sonifications,
and requested data arrays. Activations use compressed NumPy `.npz`; embeddings
use `.npy`. Their paths and formats are referenced from `results.json`.
Retained Demucs audio and spectrograms can be very large.

All `allin1.analyze()` options are multipart form fields:

| Field | API default | Notes |
| --- | --- | --- |
| `out_dir` | unset | Relative archive directory; setting it makes the response a ZIP. |
| `visualize` | `false` | Boolean, or a relative archive directory. |
| `sonify` | `false` | Boolean, or a relative archive directory. |
| `model` | `harmonix-all` | Pretrained model name. |
| `device` | CUDA when available, otherwise CPU | May also be set by `ALLIN1_DEFAULT_DEVICE`. |
| `include_activations` | `false` | Returned as `.npz` in a ZIP. |
| `include_embeddings` | `false` | Returned as `.npy` in a ZIP. |
| `demix_dir` | `./demix` | Relative, request-isolated archive directory. |
| `spec_dir` | `./spec` | Relative, request-isolated archive directory. |
| `keep_byproducts` | `false` | Includes demixed audio and spectrograms in a ZIP. |
| `overwrite` | `false` | Forwarded unchanged; workspaces are new per request. |
| `multiprocess` | `true` | Forwarded to analysis preprocessing and generators. |

For safety, path fields are interpreted inside a temporary per-request
workspace. Absolute paths and `..` traversal are rejected. The workspace,
including uploaded audio, is removed after the response completes.

### Python client

```python
from pathlib import Path

import requests

with Path("song.wav").open("rb") as audio:
    response = requests.post(
        "http://localhost:4434/analyze",
        files={"files": ("song.wav", audio, "audio/wav")},
        data={"device": "cuda", "visualize": "false"},
        timeout=None,
    )
response.raise_for_status()

if response.headers["content-type"].startswith("application/zip"):
    Path("allin1-results.zip").write_bytes(response.content)
else:
    print(response.json())
```

For multiple uploads, pass a list of `("files", (...))` tuples to
`requests.post(files=...)`.

### CPU fallback

The same CUDA-runtime image can execute on CPU without exposing a GPU. Use the
standalone CPU Compose file:

```shell
docker compose -f compose.cpu.yaml up -d --build
curl --fail http://localhost:4434/health
```

CPU analysis, especially Demucs separation, is substantially slower. Send
`device=cpu`, which is already the default in the CPU Compose file.

### LAN access

The published port listens on all host interfaces. From another machine use
`http://HOST_LAN_IP:4434` after allowing TCP port 4434 in the host firewall.
The port number does not enable TLS: this service speaks plain HTTP.

### Configuration and security

`ALLIN1_MAX_UPLOAD_MB` limits each file (default 200 MiB), and
`ALLIN1_MAX_FILES` limits files per request (default 8). The service also
limits total concurrent work and serializes GPU work by default with
`ALLIN1_MAX_CONCURRENT_JOBS=1` and `ALLIN1_MAX_CONCURRENT_GPU_JOBS=1`.
Keep one Uvicorn worker when using a GPU; process-local locks cannot coordinate
multiple workers.

The service has no authentication or TLS. Do not expose it directly to the
public internet. For shared networks, place it behind an authenticated,
rate-limited HTTPS reverse proxy, restrict firewall sources, set conservative
upload limits, and monitor disk/GPU usage. Uploaded filenames are sanitized and
audio extensions are allow-listed, but uploaded media must still be treated as
untrusted data by the surrounding deployment.

### Development tests with uv

Python 3.10 is required for the pinned GPU wheels. Create a local uv-managed
environment with the same locked stack:

```shell
uv venv --python 3.10
uv pip install --python .venv/bin/python Cython==0.29.37 numpy==1.23.5 scipy==1.10.1
uv pip install --python .venv/bin/python --no-build-isolation \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r constraints.docker.txt
uv pip install --python .venv/bin/python --no-deps -e .
uv pip install --python .venv/bin/python pytest==8.3.4
uv run --no-sync pytest tests/test_api.py
```

The local install downloads the large CUDA PyTorch wheel. The Docker image
remains the authoritative environment for the pinned legacy PyTorch, NATTEN,
and madmom stack.

### Troubleshooting and compatibility

- If Compose reports that no matching GPU device exists, verify
  `nvidia-smi` on the host and install/configure NVIDIA Container Toolkit, or
  use `compose.cpu.yaml`.
- If `/health` works but the first analysis is slow or fails offline, allow
  outbound access for the initial model downloads or pre-populate the
  `model-cache` volume.
- Inspect failures with `docker compose logs -f api`; confirm FFmpeg with
  `docker compose exec api ffmpeg -version`.
- A 413 response means the configured upload limit was exceeded. A 415 response
  means the filename extension is not in `ALLIN1_AUDIO_EXTENSIONS`.
- The retired legacy NATTEN host has an expired TLS certificate; the image uses
  the same official 0.14.6 release asset from GitHub instead.
- The image pins PyTorch 2.0.0, torchaudio 2.0.1, CUDA 11.8, and NATTEN 0.14.6.
  All-In-One uses legacy neighborhood-attention functions that are not
  source-compatible with current NATTEN releases.
- NumPy is pinned below 1.24 because madmom 0.16.1 still uses aliases removed in
  NumPy 1.24. A small package-initialization shim restores the madmom
  `collections.MutableSequence` import removed in Python 3.10. These older ML
  dependencies are the main upstream compatibility constraint and should be
  upgraded together, with inference regression tests.

## Usage for CLI

To analyze audio files:
```shell
allin1 your_audio_file1.wav your_audio_file2.mp3
```
Results will be saved in the `./struct` directory by default:
```shell
./struct
└── your_audio_file1.json
└── your_audio_file2.json
```
The analysis results will be saved in JSON format:
```json
{
  "path": "/path/to/your_audio_file.wav",
  "bpm": 100,
  "beats": [ 0.33, 0.75, 1.14, ... ],
  "downbeats": [ 0.33, 1.94, 3.53, ... ],
  "beat_positions": [ 1, 2, 3, 4, 1, 2, 3, 4, 1, ... ],
  "segments": [
    {
      "start": 0.0,
      "end": 0.33,
      "label": "start"
    },
    {
      "start": 0.33,
      "end": 13.13,
      "label": "intro"
    },
    {
      "start": 13.13,
      "end": 37.53,
      "label": "chorus"
    },
    {
      "start": 37.53,
      "end": 51.53,
      "label": "verse"
    },
    ...
  ]
}
```
All available options are as follows:
```shell
$ allin1 -h

usage: allin1 [-h] [-o OUT_DIR] [-v] [--viz-dir VIZ_DIR] [-s] [--sonif-dir SONIF_DIR] [-a] [-e] [-m MODEL] [-d DEVICE] [-k]
              [--demix-dir DEMIX_DIR] [--spec-dir SPEC_DIR]
              paths [paths ...]

positional arguments:
  paths                 Path to tracks

options:
  -h, --help            show this help message and exit
  -o OUT_DIR, --out-dir OUT_DIR
                        Path to a directory to store analysis results (default: ./struct)
  -v, --visualize       Save visualizations (default: False)
  --viz-dir VIZ_DIR     Directory to save visualizations if -v is provided (default: ./viz)
  -s, --sonify          Save sonifications (default: False)
  --sonif-dir SONIF_DIR
                        Directory to save sonifications if -s is provided (default: ./sonif)
  -a, --activ           Save frame-level raw activations from sigmoid and softmax (default: False)
  -e, --embed           Save frame-level embeddings (default: False)
  -m MODEL, --model MODEL
                        Name of the pretrained model to use (default: harmonix-all)
  -d DEVICE, --device DEVICE
                        Device to use (default: cuda if available else cpu)
  -k, --keep-byproducts
                        Keep demixed audio files and spectrograms (default: False)
  --demix-dir DEMIX_DIR
                        Path to a directory to store demixed tracks (default: ./demix)
  --spec-dir SPEC_DIR   Path to a directory to store spectrograms (default: ./spec)
```

## Usage for Python

Available functions:
- [`analyze()`](#analyze)
- [`load_result()`](#load_result)
- [`visualize()`](#visualize)
- [`sonify()`](#sonify)

### `analyze()`
Analyzes the provided audio files and returns the analysis results.

```python
import allin1

# You can analyze a single file:
result = allin1.analyze('your_audio_file.wav')

# Or multiple files:
results = allin1.analyze(['your_audio_file1.wav', 'your_audio_file2.mp3'])
```
A result is a dataclass instance containing:
```python
AnalysisResult(
  path='/path/to/your_audio_file.wav', 
  bpm=100,
  beats=[0.33, 0.75, 1.14, ...],
  beat_positions=[1, 2, 3, 4, 1, 2, 3, 4, 1, ...],
  downbeats=[0.33, 1.94, 3.53, ...], 
  segments=[
    Segment(start=0.0, end=0.33, label='start'), 
    Segment(start=0.33, end=13.13, label='intro'), 
    Segment(start=13.13, end=37.53, label='chorus'), 
    Segment(start=37.53, end=51.53, label='verse'), 
    Segment(start=51.53, end=64.34, label='verse'), 
    Segment(start=64.34, end=89.93, label='chorus'), 
    Segment(start=89.93, end=105.93, label='bridge'), 
    Segment(start=105.93, end=134.74, label='chorus'), 
    Segment(start=134.74, end=153.95, label='chorus'), 
    Segment(start=153.95, end=154.67, label='end'),
  ]),
```
Unlike CLI, it does not save the results to disk by default. You can save them as follows:
```python
result = allin1.analyze(
  'your_audio_file.wav',
  out_dir='./struct',
)
```

#### Parameters:

- `paths` : `Union[PathLike, List[PathLike]]`  
List of paths or a single path to the audio files to be analyzed.
  
- `out_dir` : `PathLike` (optional)  
Path to the directory where the analysis results will be saved. By default, the results will not be saved.
  
- `visualize` : `Union[bool, PathLike]` (optional)  
Whether to visualize the analysis results or not. If a path is provided, the visualizations will be saved in that directory. Default is False. If True, the visualizations will be saved in './viz'.
  
- `sonify` : `Union[bool, PathLike]` (optional)  
Whether to sonify the analysis results or not. If a path is provided, the sonifications will be saved in that directory. Default is False. If True, the sonifications will be saved in './sonif'.
  
- `model` : `str` (optional)  
Name of the pre-trained model to be used for the analysis. Default is 'harmonix-all'. Please refer to the documentation for the available models.
  
- `device` : `str` (optional)  
Device to be used for computation. Default is 'cuda' if available, otherwise 'cpu'.
  
- `include_activations` : `bool` (optional)  
Whether to include activations in the analysis results or not.
  
- `include_embeddings` : `bool` (optional)  
Whether to include embeddings in the analysis results or not.
  
- `demix_dir` : `PathLike` (optional)  
Path to the directory where the source-separated audio will be saved. Default is './demix'.
  
- `spec_dir` : `PathLike` (optional)  
Path to the directory where the spectrograms will be saved. Default is './spec'.
  
- `keep_byproducts` : `bool` (optional)  
Whether to keep the source-separated audio and spectrograms or not. Default is False.
  
- `multiprocess` : `bool` (optional)  
Whether to use multiprocessing for extracting spectrograms. Default is True.

#### Returns:

- `Union[AnalysisResult, List[AnalysisResult]]`  
Analysis results for the provided audio files.


### `load_result()`

Loads the analysis results from the disk.

```python
result = allin1.load_result('./struct/24k_Magic.json')
```


### `visualize()`

Visualizes the analysis results.

```python
fig = allin1.visualize(result)
fig.show()
```

#### Parameters:

- `result` : `Union[AnalysisResult, List[AnalysisResult]]`  
List of analysis results or a single analysis result to be visualized.

- `out_dir` : `PathLike` (optional)  
Path to the directory where the visualizations will be saved. By default, the visualizations will not be saved.

#### Returns:

- `Union[Figure, List[Figure]]`
List of figures or a single figure containing the visualizations. `Figure` is a class from `matplotlib.pyplot`.


### `sonify()`

Sonifies the analysis results.
It will mix metronome clicks for beats and downbeats, and event sounds for segment boundaries
to the original audio file.

```python
y, sr = allin1.sonify(result)
# y: sonified audio with shape (channels=2, samples)
# sr: sampling rate (=44100)
```

#### Parameters:

- `result` : `Union[AnalysisResult, List[AnalysisResult]]`  
List of analysis results or a single analysis result to be sonified.
- `out_dir` : `PathLike` (optional)  
Path to the directory where the sonifications will be saved. By default, the sonifications will not be saved.

#### Returns:

- `Union[Tuple[NDArray, float], List[Tuple[NDArray, float]]]`  
List of tuples or a single tuple containing the sonified audio and the sampling rate.


## Visualization & Sonification
This package provides a simple visualization (`-v` or `--visualize`) and sonification (`-s` or `--sonify`) function for the analysis results.
```shell
allin1 -v -s your_audio_file.wav
```
The visualizations will be saved in the `./viz` directory by default:
```shell
./viz
└── your_audio_file.pdf
```
The sonifications will be saved in the `./sonif` directory by default:
```shell
./sonif
└── your_audio_file.sonif.wav
```
For example, a visualization looks like this:
![Visualization](./assets/viz.png)

You can try it at [Hugging Face Space](https://huggingface.co/spaces/taejunkim/all-in-one).


## Available Models
The models are trained on the [Harmonix Set](https://github.com/urinieto/harmonixset) with 8-fold cross-validation.
For more details, please refer to the [paper](http://arxiv.org/abs/2307.16425).
* `harmonix-all`: (Default) An ensemble model averaging the predictions of 8 models trained on each fold.
* `harmonix-foldN`: A model trained on fold N (0~7). For example, `harmonix-fold0` is trained on fold 0.

By default, the `harmonix-all` model is used. To use a different model, use the `--model` option:
```shell
allin1 --model harmonix-fold0 your_audio_file.wav
```


## Speed
With an RTX 4090 GPU and Intel i9-10940X CPU (14 cores, 28 threads, 3.30 GHz),
the `harmonix-all` model processed 10 songs (33 minutes) in 73 seconds.


## Advanced Usage for Research
This package provides researchers with advanced options to extract **frame-level raw activations and embeddings** 
without post-processing. These have a resolution of 100 FPS, equivalent to 0.01 seconds per frame.

### CLI

#### Activations
The `--activ` option also saves frame-level raw activations from sigmoid and softmax:
```shell
$ allin1 --activ your_audio_file.wav
```
You can find the activations in the `.npz` file:
```shell
./struct
└── your_audio_file1.json
└── your_audio_file1.activ.npz
```
To load the activations in Python:
```python
>>> import numpy as np
>>> activ = np.load('./struct/your_audio_file1.activ.npz')
>>> activ.files
['beat', 'downbeat', 'segment', 'label']
>>> beat_activations = activ['beat']
>>> downbeat_activations = activ['downbeat']
>>> segment_boundary_activations = activ['segment']
>>> segment_label_activations = activ['label']
```
Details of the activations are as follows:
* `beat`: Raw activations from the **sigmoid** layer for **beat tracking** (shape: `[time_steps]`)
* `downbeat`: Raw activations from the **sigmoid** layer for **downbeat tracking** (shape: `[time_steps]`)
* `segment`: Raw activations from the **sigmoid** layer for **segment boundary detection** (shape: `[time_steps]`)
* `label`: Raw activations from the **softmax** layer for **segment labeling** (shape: `[label_class=10, time_steps]`)

You can access the label names as follows:
```python
>>> allin1.HARMONIX_LABELS
['start',
 'end',
 'intro',
 'outro',
 'break',
 'bridge',
 'inst',
 'solo',
 'verse',
 'chorus']
```


#### Embeddings
This package also provides an option to extract raw embeddings from the model.
```shell
$ allin1 --embed your_audio_file.wav
```
You can find the embeddings in the `.npy` file:
```shell
./struct
└── your_audio_file1.json
└── your_audio_file1.embed.npy
```
To load the embeddings in Python:
```python
>>> import numpy as np
>>> embed = np.load('your_audio_file1.embed.npy')
```
Each model embeds for every source-separated stem per time step, 
resulting in embeddings shaped as `[stems=4, time_steps, embedding_size=24]`:
1. The number of source-separated stems (the order is bass, drums, other, vocals).
2. The number of time steps (frames). The time step is 0.01 seconds (100 FPS).
3. The embedding size of 24.

Using the `--embed` option with the `harmonix-all` ensemble model will stack the embeddings, 
saving them with the shape `[stems=4, time_steps, embedding_size=24, models=8]`.

### Python
The Python API `allin1.analyze()` offers the same options as the CLI:
```python
>>> allin1.analyze(
      paths='your_audio_file.wav',
      include_activations=True,
      include_embeddings=True,
    )

AnalysisResult(
  path='/path/to/your_audio_file.wav', 
  bpm=100, 
  beats=[...],
  downbeats=[...],
  segments=[...],
  activations={
    'beat': array(...), 
    'downbeat': array(...), 
    'segment': array(...), 
    'label': array(...)
  }, 
  embeddings=array(...),
)
```

## Concerning MP3 Files
Due to variations in decoders, MP3 files can have slight offset differences.
I recommend you to first convert your audio files to WAV format using FFmpeg (as shown below), 
and use the WAV files for all your data processing pipelines.
```shell
ffmpeg -i your_audio_file.mp3 your_audio_file.wav
```
In this package, audio files are read using [Demucs](https://github.com/facebookresearch/demucs).
To my understanding, Demucs converts MP3 files to WAV using FFmpeg before reading them.
However, using a different MP3 decoder can yield different offsets. 
I've observed variations of about 20~40ms, which is problematic for tasks requiring precise timing like beat tracking, 
where the conventional tolerance is just 70ms. 
Hence, I advise standardizing inputs to the WAV format for all data processing, 
ensuring straightforward decoding.


## Training
Please refer to [TRAINING.md](TRAINING.md).

## Citation
If you use this package for your research, please cite the following paper:
```bibtex
@inproceedings{taejun2023allinone,
  title={All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio},
  author={Kim, Taejun and Nam, Juhan},
  booktitle={IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2023}
}
```