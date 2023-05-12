import json
import time
import glob, os
from datetime import datetime

from simrd.runtime import *
from simrd.heuristic import *
from simrd.heuristic.ablation import *
from simrd.parse import parse_file, GRelease

from ...pareto import pareto
from ...util import get_output_dir, ensure_path, date_string

from ...eval import models as _models
from .definitions import *

def run_pareto(base_dir, model, heuristic, ratios, runtime, overhead_limit,
               num_trials=1, verbose=True, **kwargs):
  config = {
    'model': model,
    'heuristic': type(heuristic).__name__,
    'heuristic_features': list(heuristic.FEATURES),
    'ratios': ratios,
    'overhead_limit': overhead_limit,
    'runtime': runtime.ID,
    'runtime_features': list(runtime.FEATURES),
    'kwargs': kwargs
  }

  if verbose:
    print('starting pareto trial for config: {}...'.format(json.dumps(config, indent=2)))
  else:
    # we're probably running the full eval, let's say which heuristic
    print(
        f'  - running heuristic {type(heuristic).__name__}...',
        end='',
        flush=True,
    )

  # get log executor callback
  log_path = model['log']
  if verbose:
    print(f'parsing log [{log_path}]...')
  graph = None
  callback = None
  with open(log_path, 'r') as log_f:
    graph = parse_file(log_f, start=model['has_start'])
    callback = graph.get_closure()
  assert callback is not None
  if verbose: print('  done.')

  # run model with infinite budget to get baseline memory usage
  if verbose: print('getting baseline information...')
  rt = RuntimeV1(math.inf, Heuristic(), stats=False, trace=False)
  t = time.time()
  callback(rt)
  baseline_memory = rt.telemetry.summary['max_memory']
  baseline_compute = rt.telemetry.summary['model_compute']
  baseline_const = rt.telemetry.summary['model_const_memory']
  baseline_bottleneck = rt.telemetry.summary['bottleneck_memory']
  assert rt.telemetry.summary['remat_compute'] == 0
  if verbose:
    print(f'    - baseline compute:    {baseline_compute / 1000000} ms')
    print(f'    - baseline memory:     {baseline_memory / 1000000} MB')
    print(f'    - baseline const:     {baseline_const / 1000000} MB')
    print(f'    - baseline bottleneck: {baseline_bottleneck / 1000000} MB')
    print(f'  done, took {time.time() - t} seconds.')
  config['baseline_compute'] = baseline_compute
  config['baseline_memory'] = baseline_memory
  config['baseline_const'] = baseline_const
  config['baseline_bottleneck'] = baseline_bottleneck

  # run pareto, record results
  budgets = [int(baseline_memory * r) for r in ratios]
  remat_limit = baseline_compute * (overhead_limit - 1)
  assert remat_limit >= 0

  results = [{
    'ratio': ratios[i],
    'budget': budgets[i],
    'OOM': [],
    'remat_exceeded': [],
    'meta': None,
    'total_time': [],
    'overhead': [],
    'heuristic_eval_count': [],
    'heuristic_access_count': [],
    'num_trials': num_trials
  } for i in range(len(ratios))]

  if kwargs.get('no_dealloc', False):
    schedule = [op for op in graph.schedule if not isinstance(op, GRelease)]
    graph.schedule = schedule
    callback = graph.get_closure()

  # average numerical values over trials, pick the last meta (or the first that fails)
  for _ in range(num_trials):
    rts = pareto(callback, budgets, heuristic, runtime, verbose=verbose, \
      remat_limit=remat_limit, trace=False, stats=False, **kwargs)
    for i, rt in enumerate(rts):
      results[i]['OOM'].append(rt.OOM)
      results[i]['remat_exceeded'].append(rt.remat_exceeded)
      results[i]['total_time'].append(rt.meta['total_time'])
      results[i]['overhead'].append(rt.clock / baseline_compute)
      results[i]['heuristic_eval_count'].append(rt.telemetry.summary['heuristic_eval_count'])
      results[i]['heuristic_access_count'].append(rt.telemetry.summary['heuristic_access_count'])

  out_file = f"{date_string()}-{model['name']}-{type(heuristic).__name__}.json"
  out_path = f'{base_dir}/{out_file}'
  ensure_path(base_dir)

  with open(out_path, 'w') as out_f:
    try:
      out_f.write(json.dumps({'config': config, 'results': results}, indent=2))
    except:
      import pdb; pdb.set_trace()

  if verbose:
    print(f'-> done, saved to [{out_path}]')
  else: print('done', flush=True)

  return out_path

def run_pareto_heuristics(base_dir, model, heuristics, ratios, runtime, overhead_limit, **kwargs):
  for heuristic in heuristics:
    num_trials = heuristic.TRIALS
    run_pareto(base_dir, model, heuristic, ratios, runtime, overhead_limit, num_trials, **kwargs)

def run_pareto_paper(models=None, output_dir=None):
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 2.0
  heuristics = PAPER_PARETO_HEURISTICS
  runtime = RuntimeV2EagerOptimized
  if models is None:
    models = _models.MANIFEST.values()
  if output_dir is None:
    output_dir = get_output_dir(PARETO_MOD)

  base_dirs = []
  for model in models:
    print(f"running simulated pareto evaluation for {model['name']}...")
    base_dir = f'{output_dir}/{date_string()}-' + model['name']
    t = time.time()
    run_pareto_heuristics(base_dir, model, heuristics, ratios, runtime, overhead_limit, verbose=False)
    print(f'  done, saved to [{base_dir}], took {time.time() - t} seconds.')
    base_dirs.append(base_dir)

  return base_dirs

def run_ablation_paper(models=None, output_dir=None):
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 2.0
  runtime = RuntimeV2EagerOptimized
  heuristics = PAPER_ABLATION_HEURISTICS
  if models is None:
    models = _models.MANIFEST.values()
  if output_dir is None:
    output_dir = get_output_dir(PARETO_MOD)

  base_dirs = []
  for model in models:
    print(f"running simulated ablation evaluation for {model['name']}...")
    base_dir = f'{output_dir}/{date_string()}-' + model['name'] + '-ablate'
    t = time.time()
    run_pareto_heuristics(base_dir, model, heuristics, ratios, runtime, overhead_limit, \
      verbose=True)
    print(f'  done, saved to [{base_dir}], took {time.time() - t} seconds.')
    base_dirs.append(base_dir)

  return base_dirs

def run_banishing_paper(models=None, output_dir=None):
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 2.0
  if models is None:
    models = _models.MANIFEST.values()
  if output_dir is None:
    output_dir = get_output_dir(PARETO_MOD)

  base_dirs = []
  for model in models:
    print(f"running simulated banishing evaluation for {model['name']}...")
    base_dir = f'{output_dir}/{date_string()}-' + model['name'] + '-banish'
    t = time.time()
    run_pareto(base_dir, model, DTRUnopt(), ratios, RuntimeV1, overhead_limit, verbose=True)
    run_pareto(base_dir, model, DTR(), ratios, RuntimeV2EagerOptimized, overhead_limit, verbose=True)
    run_pareto(base_dir, model, DTR(), ratios, RuntimeV2EagerOptimized, overhead_limit, verbose=True, no_dealloc=True)
    print(f'  done, saved to [{base_dir}], took {time.time() - t} seconds.')
    base_dirs.append(base_dir)

  return base_dirs

def run_accesses_paper(models=None, output_dir=None):
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 2.0
  runtime = RuntimeV2EagerOptimized
  heuristics = PAPER_ACCESSES_HEURISTICS
  if models is None:
    models = _models.MANIFEST.values()
  if output_dir is None:
    output_dir = get_output_dir(PARETO_MOD)

  base_dirs = []
  for model in models:
    print(f"running simulated access overhead evaluation for {model['name']}...")
    base_dir = f'{output_dir}/{date_string()}-' + model['name'] + '-access'
    t = time.time()
    run_pareto_heuristics(base_dir, model, heuristics, ratios, runtime, overhead_limit, \
      verbose=True)
    print(f'  done, saved to [{base_dir}], took {time.time() - t} seconds.')
    base_dirs.append(base_dir)

  return base_dirs
