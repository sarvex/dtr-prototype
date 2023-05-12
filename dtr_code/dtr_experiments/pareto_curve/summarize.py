from common import (invoke_main, render_exception, sort_data,
                    write_status, write_summary)

from validate_config import validate_trials_config
import numpy as np
from functools import reduce

def filter_params(specific_params):
    result = {
        'type' : specific_params['type'],
    }

    if specific_params.get('memory_budget') is not None and specific_params['memory_budget'] > 0:
        result['memory_budget'] = '{:.3g} MB'.format(specific_params['memory_budget'] * 1e-6)

    if result['type'] == 'dtr' and specific_params['kind'] == 'ratio':
        result['ratio'] = specific_params['ratio']

    return result


def summarize_results(stat):
    times = [
        np.median([entry['time']['mean'] for entry in stat['summary']]),
        np.median([entry['gpu_time']['mean'] for entry in stat['summary']])
    ]

    slowdown = '_'
    if 'slowdown' in stat['summary'][0]:
        slowdown = np.median([entry['slowdown']['mean'] for entry in stat['summary']])

    memory = [
        np.median([entry['input_mem']['mean'] for entry in stat['summary']]),
        np.median([entry['model_mem']['mean'] for entry in stat['summary']]),
        np.median([entry['total_mem']['mean'] for entry in stat['summary']])
    ]

    return [*times, slowdown, *memory]


def summarize(config, data):
    indent = ' ' * 3
    summary = 'Key: median (Wall clock time (ms), GPU time (ms), slowdown (X), input memory (MB), model memory (MB), and final memory (MB)) for each input\n'

    failed_models = {}

    for model in config['models']:
        model_data = data[model]
        result_by_settings = []
        for stat in model_data:
            if stat['summary'] == 'error':
                if model not in failed_models:
                    failed_models[model] = []
                result = filter_params(stat['specific_params'])
                result.update({'command_id' : stat['command_id']})
                if result not in failed_models[model]:
                    failed_models[model].append(result)
                continue

            configuration_str = '; '.join(
                [
                    f'{indent}_{k}_: {v}'
                    for k, v in filter_params(stat['specific_params']).items()
                ]
            )
            summaries = ', '.join([
                res if isinstance(res, str) else '{:.3f}'.format(res)
                for res in summarize_results(stat)
            ])

            result_by_settings.append(
                {
                    'heading': f'{indent}Configuration:\n{configuration_str}\n',
                    'summaries': [indent * 2 + f'*Results*: {summaries}\n'],
                }
            )

        if result_by_settings:
            summary += f'*{model}*:\n'
            for results in result_by_settings:
                summary += results['heading']
                for line in results['summaries']:
                    summary += line
                summary += '\n'
    if not failed_models:
        return summary
    error_summary = '*ERRORS CAUGHT AT:*\n'
    for model, settings in failed_models.items():
        error_summary += f'*{model}*:\n'
        for specific_params in settings:
            error_summary += (
                f';{indent}'.join(
                    [f'_{k}_: {v}' for k, v, in specific_params.items()]
                )
                + '\n'
            )
        error_summary += '\n'
    return error_summary + '\n' + summary


def main(data_dir, config_dir, output_dir):
    try:
        config, msg = validate_trials_config(config_dir)
        if config is None:
            write_status(output_dir, False, msg)
            return 1

        all_data = sort_data(data_dir)
        most_recent = all_data[-1]

        summary = summarize(config, most_recent)
        write_summary(output_dir, 'Pareto Curve Trial', summary)
        write_status(output_dir, True, 'success')

    except Exception as e:
        write_status(
            output_dir, False, f'Exception encountered: {render_exception(e)}'
        )
        return 1


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
