import os
import json

"""
Details of models for the simulated eval.
"""

LOG_PATH = 'logs'
MANIFEST_PATH = f'{LOG_PATH}/manifest.json'

def check_args(cfg, **kwargs):
  good = all(val is not None for val in cfg.values())
  good = good and os.path.isfile(cfg['log'])
  if not good:
    raise ValueError(f"Invalid model parameters for {cfg['name']}: {kwargs}")

MODELS = {}

print(f'loading manifest {MANIFEST_PATH}...')
MANIFEST = json.load(open(MANIFEST_PATH, 'r'))
MANIFEST = {model['name']: model for model in MANIFEST['models']}

INVALID_MODELS = []
for model_name, model in MANIFEST.items():
  try:
    check_args(model)
  except ValueError:
    print(f'ignoring invalid model "{model_name}" from manifest')
    INVALID_MODELS.append(model_name)

for m in INVALID_MODELS:
  MANIFEST.pop(m)

print(f'found models: {list(MANIFEST.keys())}')
