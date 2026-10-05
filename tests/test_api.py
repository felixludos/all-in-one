import io
import json
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from allin1 import api
from allin1.typings import AnalysisResult, Segment


client = TestClient(api.app)


def result_for(path, arrays=False):
  return AnalysisResult(
    path=Path(path),
    bpm=120,
    beats=[0.5, 1.0],
    downbeats=[0.5],
    beat_positions=[1, 2],
    segments=[Segment(start=0.0, end=1.5, label='intro')],
    activations={'beat': np.array([0.1, 0.9])} if arrays else None,
    embeddings=np.array([[1.0, 2.0]]) if arrays else None,
  )


def test_health():
  response = client.get('/health')
  assert response.status_code == 200
  assert response.json()['status'] == 'ok'
  assert 'cuda_available' in response.json()


def test_validation_rejects_missing_file_and_unsafe_path(monkeypatch):
  assert client.post('/analyze').status_code == 422
  called = False

  def fake_analyze(**kwargs):
    nonlocal called
    called = True
    return []

  monkeypatch.setattr(api, 'analyze', fake_analyze)
  response = client.post(
    '/analyze',
    files={'files': ('track.wav', b'audio', 'audio/wav')},
    data={'device': 'cpu', 'demix_dir': '../outside'},
  )
  assert response.status_code == 422
  assert not called


def test_parameter_forwarding_and_filename_sanitizing(monkeypatch):
  captured = {}

  def fake_analyze(**kwargs):
    captured.update(kwargs)
    return [result_for(kwargs['paths'][0])]

  monkeypatch.setattr(api, 'analyze', fake_analyze)
  response = client.post(
    '/analyze',
    files={'files': ('../../evil song.wav', b'audio', 'audio/wav')},
    data={
      'out_dir': 'struct',
      'visualize': 'plots',
      'sonify': 'false',
      'model': 'harmonix-fold0',
      'device': 'cpu',
      'include_activations': 'true',
      'include_embeddings': 'true',
      'demix_dir': 'work/demix',
      'spec_dir': 'work/spec',
      'keep_byproducts': 'true',
      'overwrite': 'true',
      'multiprocess': 'false',
    },
  )
  assert response.status_code == 200
  assert response.headers['content-type'] == 'application/zip'
  assert captured['paths'][0].name == 'evil_song.wav'
  assert captured['model'] == 'harmonix-fold0'
  assert captured['device'] == 'cpu'
  assert captured['include_activations'] is True
  assert captured['include_embeddings'] is True
  assert captured['keep_byproducts'] is True
  assert captured['overwrite'] is True
  assert captured['multiprocess'] is False
  assert captured['sonify'] is False
  assert captured['out_dir'].name == 'struct'
  assert captured['visualize'].name == 'plots'
  assert captured['demix_dir'].parts[-2:] == ('work', 'demix')
  assert not captured['paths'][0].exists()


def test_mocked_analysis_returns_json(monkeypatch):
  monkeypatch.setattr(api, 'analyze', lambda **kwargs: [result_for(kwargs['paths'][0])])
  response = client.post(
    '/analyze',
    files={'files': ('track.wav', b'audio', 'audio/wav')},
    data={'device': 'cpu'},
  )
  assert response.status_code == 200
  assert response.headers['content-type'].startswith('application/json')
  assert response.json() == {
    'path': 'track.wav',
    'bpm': 120,
    'beats': [0.5, 1.0],
    'downbeats': [0.5],
    'beat_positions': [1, 2],
    'segments': [{'start': 0.0, 'end': 1.5, 'label': 'intro'}],
  }


def test_large_arrays_are_zip_artifacts(monkeypatch):
  monkeypatch.setattr(
    api,
    'analyze',
    lambda **kwargs: [result_for(kwargs['paths'][0], arrays=True)],
  )
  response = client.post(
    '/analyze',
    files={'files': ('track.wav', b'audio', 'audio/wav')},
    data={'device': 'cpu', 'include_activations': 'true', 'include_embeddings': 'true'},
  )
  assert response.status_code == 200
  with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    names = set(archive.namelist())
    assert {'results.json', 'arrays/track.activ.npz', 'arrays/track.embed.npy'} <= names
    payload = json.loads(archive.read('results.json'))[0]
    assert payload['activations']['format'] == 'npz'
    assert payload['embeddings']['format'] == 'npy'
