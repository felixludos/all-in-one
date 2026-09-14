"""Production HTTP API for All-In-One."""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from .__about__ import __version__
from .analyze import analyze

LOGGER = logging.getLogger(__name__)


def _positive_int_env(name: str, default: int) -> int:
  try:
    value = int(os.getenv(name, str(default)))
  except ValueError as exc:
    raise RuntimeError(f'{name} must be an integer') from exc
  if value < 1:
    raise RuntimeError(f'{name} must be at least 1')
  return value


MAX_UPLOAD_BYTES = _positive_int_env('ALLIN1_MAX_UPLOAD_MB', 200) * 1024 * 1024
MAX_FILES = _positive_int_env('ALLIN1_MAX_FILES', 8)
MAX_CONCURRENT_JOBS = _positive_int_env('ALLIN1_MAX_CONCURRENT_JOBS', 1)
MAX_CONCURRENT_GPU_JOBS = _positive_int_env('ALLIN1_MAX_CONCURRENT_GPU_JOBS', 1)
DEFAULT_DEVICE = os.getenv(
  'ALLIN1_DEFAULT_DEVICE',
  'cuda' if torch.cuda.is_available() else 'cpu',
)
ALLOWED_EXTENSIONS = {
  extension.strip().lower()
  for extension in os.getenv(
    'ALLIN1_AUDIO_EXTENSIONS',
    '.aac,.aif,.aiff,.flac,.m4a,.mp3,.ogg,.opus,.wav,.wma',
  ).split(',')
  if extension.strip()
}

_analysis_slots: Optional[asyncio.Semaphore] = None
_gpu_slots: Optional[asyncio.Semaphore] = None
_SAFE_FILENAME = re.compile(r'[^A-Za-z0-9._-]+')
_SAFE_MODEL = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
_SAFE_DEVICE = re.compile(r'^(cpu|mps|cuda(?::\d+)?)$')


def _semaphores():
  global _analysis_slots, _gpu_slots
  if _analysis_slots is None:
    _analysis_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
  if _gpu_slots is None:
    _gpu_slots = asyncio.Semaphore(MAX_CONCURRENT_GPU_JOBS)
  return _analysis_slots, _gpu_slots


@asynccontextmanager
async def _job_slot(device: str):
  analysis_slots, gpu_slots = _semaphores()
  async with analysis_slots:
    if device.startswith('cuda'):
      async with gpu_slots:
        yield
    else:
      yield


def _sanitize_filename(filename: Optional[str], index: int) -> str:
  name = Path(filename or f'upload-{index}.bin').name
  name = _SAFE_FILENAME.sub('_', name).strip('._')
  if not name:
    name = f'upload-{index}.bin'
  return name[:220]


def _workspace_path(root: Path, value: str, field: str) -> Path:
  if not value.strip():
    raise HTTPException(422, f'{field} cannot be empty')
  candidate = Path(value)
  if candidate.is_absolute():
    raise HTTPException(422, f'{field} must be a relative path')
  resolved = (root / candidate).resolve()
  try:
    resolved.relative_to(root.resolve())
  except ValueError as exc:
    raise HTTPException(422, f'{field} must stay inside the request workspace') from exc
  return resolved


def _bool_or_workspace_path(root: Path, value: str, field: str, default_dir: str):
  normalized = value.strip().lower()
  if normalized in {'false', '0', 'no', 'off'}:
    return False
  if normalized in {'true', '1', 'yes', 'on'}:
    return root / default_dir
  if not value.strip():
    raise HTTPException(422, f'{field} cannot be empty')
  return _workspace_path(root, value, field)


def _jsonable_result(result: Any) -> Dict[str, Any]:
  data = asdict(result) if is_dataclass(result) else dict(result)
  data['path'] = Path(str(data['path'])).name
  data.pop('activations', None)
  data.pop('embeddings', None)
  return data


def _write_array_artifacts(result: Any, destination: Path, payload: Dict[str, Any]) -> None:
  activations = getattr(result, 'activations', None)
  embeddings = getattr(result, 'embeddings', None)
  stem = Path(str(getattr(result, 'path', 'result'))).stem
  if activations is not None or embeddings is not None:
    destination.mkdir(parents=True, exist_ok=True)
  if activations is not None:
    name = f'{stem}.activ.npz'
    np.savez_compressed(destination / name, **activations)
    payload['activations'] = {'format': 'npz', 'path': f'arrays/{name}'}
  else:
    payload['activations'] = None
  if embeddings is not None:
    name = f'{stem}.embed.npy'
    np.save(destination / name, embeddings)
    payload['embeddings'] = {'format': 'npy', 'path': f'arrays/{name}'}
  else:
    payload['embeddings'] = None


def _create_archive(root: Path, results: List[Any]) -> Path:
  result_payload = []
  for result in results:
    payload = _jsonable_result(result)
    _write_array_artifacts(result, root / 'arrays', payload)
    result_payload.append(payload)
  (root / 'results.json').write_text(
    json.dumps(result_payload, indent=2, allow_nan=False), encoding='utf-8'
  )
  archive = root / 'allin1-results.zip'
  excluded = {'uploads', archive.name}
  with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
      relative = path.relative_to(root)
      if path.is_file() and relative.parts[0] not in excluded:
        bundle.write(path, relative.as_posix())
  return archive


def _needs_archive(
  out_dir: Optional[str], visualize: str, sonify: str,
  include_activations: bool, include_embeddings: bool, keep_byproducts: bool,
) -> bool:
  return any((
    out_dir is not None,
    visualize.strip().lower() not in {'false', '0', 'no', 'off'},
    sonify.strip().lower() not in {'false', '0', 'no', 'off'},
    include_activations, include_embeddings, keep_byproducts,
  ))


app = FastAPI(
  title='All-In-One Music Structure Analyzer API',
  version=__version__,
  description=(
    'Analyze uploaded audio. Ordinary requests return JSON. Requests that produce files, '
    'activations, embeddings, or retained byproducts return application/zip.'
  ),
)


@app.middleware('http')
async def reject_oversized_declared_requests(request: Request, call_next):
  if request.url.path == '/analyze':
    content_length = request.headers.get('content-length')
    maximum = MAX_UPLOAD_BYTES * MAX_FILES + 1024 * 1024
    if content_length:
      try:
        if int(content_length) > maximum:
          return JSONResponse({'detail': 'Request body is too large'}, status_code=413)
      except ValueError:
        return JSONResponse({'detail': 'Invalid Content-Length header'}, status_code=400)
  return await call_next(request)


@app.get('/health', tags=['operations'])
def health() -> Dict[str, Any]:
  return {
    'status': 'ok',
    'version': __version__,
    'cuda_available': torch.cuda.is_available(),
    'default_device': DEFAULT_DEVICE,
  }


@app.post(
  '/analyze',
  tags=['analysis'],
  responses={
    200: {
      'description': 'JSON analysis result(s), or a ZIP containing results.json and generated files.',
      'content': {'application/json': {}, 'application/zip': {}},
    },
    413: {'description': 'An upload exceeded the configured size limit.'},
  },
)
async def analyze_audio(
  files: List[UploadFile] = File(..., description='One or more audio files.'),
  out_dir: Optional[str] = Form(None),
  visualize: str = Form('false', description='Boolean or relative output directory.'),
  sonify: str = Form('false', description='Boolean or relative output directory.'),
  model: str = Form('harmonix-all'),
  device: str = Form(DEFAULT_DEVICE),
  include_activations: bool = Form(False),
  include_embeddings: bool = Form(False),
  demix_dir: str = Form('./demix'),
  spec_dir: str = Form('./spec'),
  keep_byproducts: bool = Form(False),
  overwrite: bool = Form(False),
  multiprocess: bool = Form(True),
):
  if not files:
    raise HTTPException(422, 'At least one audio file is required')
  if len(files) > MAX_FILES:
    raise HTTPException(413, f'At most {MAX_FILES} files are allowed per request')
  if not _SAFE_MODEL.fullmatch(model):
    raise HTTPException(422, 'Invalid model name')
  if not _SAFE_DEVICE.fullmatch(device):
    raise HTTPException(422, 'device must be cpu, mps, cuda, or cuda:N')
  if device.startswith('cuda') and not torch.cuda.is_available():
    raise HTTPException(422, 'CUDA was requested but is not available')
  if device.startswith('cuda:') and int(device.split(':', 1)[1]) >= torch.cuda.device_count():
    raise HTTPException(422, f'{device} does not identify an available CUDA device')

  workspace = Path(tempfile.mkdtemp(prefix='allin1-api-'))
  uploads_dir = workspace / 'uploads'
  uploads_dir.mkdir()
  upload_paths: List[Path] = []
  used_names = set()
  try:
    for index, upload in enumerate(files, start=1):
      filename = _sanitize_filename(upload.filename, index)
      if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f'Unsupported audio extension for {filename}')
      if filename in used_names:
        filename = f'{index}-{filename}'
      used_names.add(filename)
      upload_path = uploads_dir / filename
      written = 0
      with upload_path.open('wb') as output:
        while chunk := await upload.read(1024 * 1024):
          written += len(chunk)
          if written > MAX_UPLOAD_BYTES:
            raise HTTPException(
              413, f'{filename} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit'
            )
          output.write(chunk)
      await upload.close()
      if written == 0:
        raise HTTPException(422, f'{filename} is empty')
      upload_paths.append(upload_path)

    api_out_dir = _workspace_path(workspace, out_dir, 'out_dir') if out_dir is not None else None
    api_visualize = _bool_or_workspace_path(workspace, visualize, 'visualize', 'viz')
    api_sonify = _bool_or_workspace_path(workspace, sonify, 'sonify', 'sonif')
    api_demix_dir = _workspace_path(workspace, demix_dir, 'demix_dir')
    api_spec_dir = _workspace_path(workspace, spec_dir, 'spec_dir')
    kwargs = {
      'paths': upload_paths, 'out_dir': api_out_dir, 'visualize': api_visualize,
      'sonify': api_sonify, 'model': model, 'device': device,
      'include_activations': include_activations, 'include_embeddings': include_embeddings,
      'demix_dir': api_demix_dir, 'spec_dir': api_spec_dir,
      'keep_byproducts': keep_byproducts, 'overwrite': overwrite,
      'multiprocess': multiprocess,
    }
    async with _job_slot(device):
      raw_results = await asyncio.to_thread(analyze, **kwargs)
    results = raw_results if isinstance(raw_results, list) else [raw_results]
    if _needs_archive(
      out_dir, visualize, sonify, include_activations, include_embeddings, keep_byproducts
    ):
      archive = _create_archive(workspace, results)
      return FileResponse(
        archive, media_type='application/zip', filename='allin1-results.zip',
        background=BackgroundTask(shutil.rmtree, workspace, ignore_errors=True),
      )
    payload = [_jsonable_result(result) for result in results]
    shutil.rmtree(workspace, ignore_errors=True)
    return JSONResponse(payload if len(files) > 1 else payload[0])
  except HTTPException:
    shutil.rmtree(workspace, ignore_errors=True)
    raise
  except Exception:
    LOGGER.exception('Analysis failed')
    shutil.rmtree(workspace, ignore_errors=True)
    raise HTTPException(500, 'Audio analysis failed')
  finally:
    for upload in files:
      await upload.close()
    if device.startswith('cuda') and torch.cuda.is_available():
      torch.cuda.empty_cache()
