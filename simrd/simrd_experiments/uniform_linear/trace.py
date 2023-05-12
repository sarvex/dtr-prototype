import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from simrd.heuristic import *
from simrd.heuristic.ablation import AbESize
from simrd.runtime import *

import simrd_experiments.util as util
from simrd_experiments.bounds import *
from simrd_experiments.execution_analysis.trace import State

from simrd_experiments.uniform_linear.run import run, chop_failures

TRACE_MOD = 'uniform_linear/trace'

def render_array(data):
  ax = sns.heatmap(data, linewidths=0)#, cmap=plt.get_cmap('cool'))
  ax.invert_yaxis()
  ax.set_xlabel('Time')
  ax.set_ylabel('Tensor ID')

def render_trace(n, tel : Telemetry, dedup_grad=True):
  s = State(tel)
  n2 = n if dedup_grad else 2 * n
  slices = []
  while s.step():
    sl = [0] * n2
    for tid in s.material:
      if dedup_grad and tid < n or not dedup_grad:
        sl[tid] = 1
      else:
        sl[n - (tid % n) - 1] = 1.5
    slices.append(sl)
  data = np.array(slices).T
  render_array(data)
  plt.axvline(n+1, c='red', alpha=0.5, linewidth=1, label='begin backprop')

def plot_tq(n=80):
  bound = TQBound()
  heuristic = AbESize()
  runtime = RuntimeV2EagerOptimized
  rt_kwargs = {'stats': True, 'trace': True}

  rt = run(n, bound(n), heuristic, runtime, releases=True, rt_kwargs=rt_kwargs)
  render_trace(n, rt.telemetry, dedup_grad=True)
  plt.legend()
  plt.title(f'$n = {n}$, $B$ = {str(bound)}, Compute-Memory Heuristic')
  plt.savefig(util.get_output_path(TRACE_MOD, 'tq.png'), dpi=300)
  plt.clf()

def plot_treeverse(n=80):
  bound = Log2Bound()
  heuristic = DTR()
  runtime = RuntimeV2EagerOptimized
  rt_kwargs = {'stats': True, 'trace': True}

  rt = run(n, bound(n), heuristic, runtime, releases=True, rt_kwargs=rt_kwargs)
  render_trace(n, rt.telemetry, dedup_grad=True)
  plt.legend()
  plt.title(f'$n = {n}$, $B$ = {str(bound)}, DTR-Full Heuristic')
  plt.savefig(util.get_output_path(TRACE_MOD, 'treeverse.png'), dpi=300)
  plt.clf()

def plot_tq_local(n=80):
  bound = TQBound()
  heuristic = DTRLocal()
  runtime = RuntimeV2EagerOptimized
  rt_kwargs = {'stats': True, 'trace': True}

  rt = run(n, bound(n), heuristic, runtime, releases=True, rt_kwargs=rt_kwargs)
  render_trace(n, rt.telemetry, dedup_grad=True)
  plt.legend()
  plt.title(f'$n = {n}$, $B$ = {str(bound)}, DTR-Local Heuristic')
  plt.savefig(util.get_output_path(TRACE_MOD, 'tq_local.png'), dpi=300)
  plt.clf()

if __name__ == '__main__':
  plot_tq()
  plot_tq_local()
  plot_treeverse()
