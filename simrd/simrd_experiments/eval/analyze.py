import pickle
import json
import time
from datetime import datetime
import pandas as pd
import cProfile

from simrd.runtime import *
from simrd.heuristic import *
from simrd.parse import parse_file

from simrd_experiments.util import ensure_path, get_output_dir, date_string
from simrd_experiments.execution_analysis.trace import *
import simrd_experiments.eval.models as models

"""
Analyze the execution of a model. This module helps investigate OOM cases.
"""

ANALYSIS_MOD = 'eval/analysis'

def run_baseline(callback, stats=False, trace=False):
  rt = RuntimeV1(math.inf, Heuristic(), stats=stats, trace=trace)
  t = time.time()
  callback(rt)
  t = time.time() - t
  result = {
    'baseline_memory': rt.telemetry.summary['max_memory'],
    'baseline_compute': rt.telemetry.summary['model_compute'],
    'baseline_const': rt.telemetry.summary['model_const_memory'],
    'baseline_bottleneck': rt.telemetry.summary['bottleneck_memory'],
    'baseline_time': t
  }
  return rt, result, None

def run_with_callback(callback, baseline, heuristic, ratio, runtime, 
                      overhead_limit=10, profile=False, rt_kwargs={}):
  budget = int(baseline['baseline_memory'] * ratio)
  bcompute = baseline['baseline_compute']
  remat_limit = bcompute * (overhead_limit - 1)
  assert remat_limit >= 0
  rt = runtime(budget, heuristic, remat_limit=remat_limit, **rt_kwargs)

  pr = None
  t = time.time()
  if profile:
    pr = cProfile.Profile()
    pr.enable()
  
  try:
    callback(rt)
  except (MemoryError, RematExceededError):
    pass
  except:
    raise

  t = time.time() - t
  if profile:
    pr.disable()

  result = {
    'ratio': ratio,
    'budget': budget,
    'OOM': rt.OOM, 'thrashed': rt.remat_exceeded,
    'overhead': (bcompute + rt.telemetry.summary['remat_compute']) / bcompute,
    'time': t
  }

  return rt, result, pr

def dump_run(rt, result, pr, analysis_dir=None):
  if analysis_dir is None:
    date_str = date_string()
    analysis_dir = get_output_dir(ANALYSIS_MOD) + '/' + date_str

  ensure_path(analysis_dir)
  with open(analysis_dir + '/runtime.bin', 'wb') as pf:
    rt._prepickle()
    pickle.dump(rt, pf)

  with open(analysis_dir + '/result.json', 'w') as jf:
    jf.write(json.dumps(result, indent=2))

  # TODO: dump config

  if pr is not None:
    pr.dump_stats(analysis_dir + '/profile.txt')

  return analysis_dir

# def run_model(model, heuristic, ratio, runtime, overhead_limit=10,
#               record=True, profile=False, baseline_only=False, rt_kwargs={}):
#   config = {
#     'model': model,
#     'heuristic': str(heuristic),
#     'heuristic_features': list(heuristic.FEATURES),
#     'ratio': ratio,
#     'overhead_limit': overhead_limit,
#     'runtime': runtime.ID,
#     'runtime_features': list(runtime.FEATURES),
#   }

#   # get log executor callback
#   log_path = model['log']
#   print('parsing log [{}]...'.format(log_path))
#   callback = None
#   with open(log_path, 'r') as log_f:
#     callback = parse_file(log_f, start=model['has_start']).get_closure()
#   assert callback is not None
#   print('  done.')

#   # run model with infinite budget to get baseline memory usage
#   print('getting baseline information...')
#   rt = RuntimeV1(math.inf, Heuristic(), stats=False, trace=False)
#   t = time.time()
#   callback(rt)
#   baseline_memory = rt.telemetry.summary['max_memory']
#   baseline_compute = rt.telemetry.summary['model_compute']
#   baseline_pinned = rt.telemetry.summary['model_pinned_memory']
#   baseline_bottleneck = rt.telemetry.summary['bottleneck_memory']
#   assert rt.telemetry.summary['remat_compute'] == 0
#   print('    - baseline compute:  {} ms'.format(baseline_compute / 1000000))
#   print('    - baseline memory:   {} MB'.format(baseline_memory / 1000000))
#   print('    - baseline pinned:   {} MB'.format(baseline_pinned / 1000000))
#   print('    - baseline bottleneck: {} MB'.format(baseline_bottleneck / 1000000))
#   print('  done, took {} seconds.'.format(time.time() - t))

#   if baseline_only: return None

#   # run model with given parameters
#   print('running with given config...')
#   budget = int(baseline_memory * ratio)
#   remat_limit = baseline_compute * (overhead_limit - 1)
#   assert remat_limit >= 0
#   rt = runtime(budget, heuristic, remat_limit=remat_limit, **rt_kwargs)

#   t = time.time()
#   pr = cProfile.Profile()
#   if profile:
#     pr.enable()

#   try:
#     callback(rt)
#   except (MemoryError, RematExceededError):
#     pass
#   except:
#     raise

#   t = time.time() - t
#   print('  done, took {} seconds.'.format(t))
#   if profile:
#     pr.disable()
#     pr.print_stats('cumtime')

#   result = {'ratio': ratio, 'budget': budget, 'OOM': rt.OOM, 'remat_exceeded': rt.remat_exceeded}
#   result['overhead'] = \
#     (baseline_compute + rt.telemetry.summary['remat_compute']) / baseline_compute
#   result['total_time'] = t
#   print('result:')
#   print(json.dumps(result, indent=2))
#   print('config:')
#   print(json.dumps(config, indent=2))

#   if not record:
#     return None

#   # pickle the whole runtime for later analysis
#   date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
#   basename = '{}-{}'.format(date_str, model['name'])
#   base_mod = ANALYSIS_MOD + '/' + basename
#   ensure_output_path(base_mod)
#   with open(get_output_path(base_mod, basename + '.bin'), 'wb') as pf:
#     rt._prepickle()
#     pickle.dump(rt, pf)

#   # might as well dump the config as well
#   with open(get_output_path(base_mod, 'config.json'), 'w') as jf:
#     jf.write(json.dumps(config, indent=2))

#   return basename

def dump_csv(analysis_dir):
  with open(analysis_dir + '/runtime.bin', 'rb') as pf:
    rt = pickle.load(pf)
    df = pd.DataFrame(rt.telemetry.tensor.values(), columns=Telemetry.TENSOR_STATS)
    df.to_csv(analysis_dir + '/tensor.csv')
    df = pd.DataFrame(rt.telemetry.storage.values(), columns=Telemetry.STORAGE_STATS)
    df.to_csv(analysis_dir + '/storage.csv')
    df = pd.DataFrame(rt.telemetry.operator.values(), columns=Telemetry.OPERATOR_STATS)
    df.to_csv(analysis_dir + '/operator.csv')

def analyze_memory(analysis_dir, start=0, end=None, render=True):
  with open(analysis_dir + '/runtime.bin', 'rb') as pf:
    plt.clf()
    rt = pickle.load(pf)
    trace_df = analyze_trace(rt.telemetry)
    memory_analysis(rt.telemetry, trace_df, start=start, end=end)
    plt.title(analysis_dir + ' Memory Analysis')
    plt.savefig(analysis_dir + '/memory_analysis.png', dpi=300)

    # analyze critical moment with maximum locked memory
    lock_df = analyze_max_locked(rt.telemetry, analysis_dir + '/max_locked_render', render)
    lock_df.to_csv(analysis_dir + '/max_locked.csv')

    # analyze max pinned memory
    pin_df = analyze_max_pinned(rt.telemetry, analysis_dir + '/max_pinned_render', render)
    pin_df.to_csv(analysis_dir + '/max_pinned.csv')

    if render:
      # render graph at final timestep
      s = State(rt.telemetry)
      while s.step():
        pass
      s.render_dot(analysis_dir + '/final_render')

# if __name__ == '__main__':
#   import sys; sys.setrecursionlimit(10000000)
#   import argparse
#   parser = argparse.ArgumentParser(description='Analyze a simulated execution.')
#   parser.add_argument('--runtime', default='RuntimeV2EagerOptimized', choices=RUNTIMES.keys())
#   parser.add_argument('--heuristic', default='DTREqClass', choices=HEURISTICS_NAMES.keys())
#   parser.add_argument('--ratio', default=1.0, type=float)
#   parser.add_argument('--overhead-limit', default=10, type=float)

#   parser.add_argument('--model', required=True, type=str, choices=models.ALL_MODELS.keys())
#   parser.add_argument('--layers', default=None, type=str)
#   parser.add_argument('--batch', default=None, type=str)
#   parser.add_argument('--fail', action='store_true')

#   parser.add_argument('--analyze', action='store_true')
#   parser.add_argument('--mem-start', default=0, type=int)
#   parser.add_argument('--mem-end', default=None, type=int)
#   parser.add_argument('--baseline', action='store_true')
#   parser.add_argument('--render', action='store_true')
#   parser.add_argument('--profile', action='store_true')

#   args = parser.parse_args()
#   rt_kwargs = {
#     'trace': args.analyze,
#     'stats': args.analyze
#   }

#   model_kwargs = {}
#   if args.fail: model_kwargs['fail'] = True
#   if args.layers: model_kwargs['layers'] = args.layers
#   if args.batch: model_kwargs['batch'] = args.batch
#   model = models.ALL_MODELS[args.model](**model_kwargs)

#   runtime = RUNTIMES[args.runtime]
#   heuristic = HEURISTICS_NAMES[args.heuristic]()

#   bn = run_model(
#     model,
#     heuristic,
#     args.ratio,
#     runtime,
#     overhead_limit=args.overhead_limit,
#     record=args.analyze,
#     profile=args.profile,
#     baseline_only=args.baseline,
#     rt_kwargs=rt_kwargs
#   )

#   if not args.baseline and args.analyze:
#     dump_csv(bn)
#     analyze_memory(bn, start=args.mem_start, end=args.mem_end, render=args.render)
