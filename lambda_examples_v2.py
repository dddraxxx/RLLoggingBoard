"""
Lambda function examples V2 for curriculum1 metrics in plot_lambda_metrics.py.
Optimized for single-pass processing with datasource grouping.
"""

import numpy as np
import re
from collections import defaultdict

# Task type to reward key mappings
TASK_REWARD_KEYS = {
    'zoom': ('all_zoom_reward', 'filter_zoom_reward', None),
    'rotflip': ('all_rotflip_reward', 'filter_rotflip_reward', 'rotflip_answer_index'),
    'draw': ('all_draw_reward', 'filter_draw_reward', 'draw_answer_index'),
    'compare': ('all_compare_reward', 'filter_compare_reward', 'compare_answer_index'),
}

def _detect_task_type(data_source, ability, index):
    """Fast task type detection prioritising datasource/ability hints."""
    ds_lower = str(data_source).lower()
    ability_lower = str(ability).lower() if ability else ''
    idx_lower = str(index).lower() if index else ''

    if any(key in ds_lower for key in ('sealvqa', 'visual_probe', 'visualprobe')):
        return 'zoom'
    if 'docvqa' in ds_lower:
        return 'rotflip'
    if any(key in ds_lower for key in ('read_value', 'curriculum_1_read')) or 'read_value' in ability_lower:
        return 'draw'
    if any(key in ds_lower for key in ('compare_value', 'compare_count', 'curriculum_1_compare')) or 'compare_value' in ability_lower:
        return 'compare'

    # Fallback to legacy index-based detection if datasource/ability are ambiguous.
    if 'read_value' in idx_lower:
        return 'draw'
    if 'compare_value' in idx_lower:
        return 'compare'
    return None


def _filter_by_datasource(step_data):
    """
    Single pass to group all data by datasource.
    Returns dict: {datasource: {key: [values for that datasource]}}
    """
    if 'data_source' not in step_data:
        return {}

    data_sources = step_data.get('data_source', [])
    num_samples = len(data_sources)

    # Group indices by datasource
    ds_groups = defaultdict(list)
    for i in range(num_samples):
        ds = str(data_sources[i]) if i < len(data_sources) else ''
        ds_groups[ds].append(i)

    # Build filtered step_data for each datasource
    result = {}
    for ds, indices in ds_groups.items():
        ds_data = {}
        for key, values in step_data.items():
            if isinstance(values, list):
                ds_data[key] = [values[i] for i in indices if i < len(values)]
            else:
                ds_data[key] = values
        result[ds] = ds_data

    return result


def _compute_tool_counts(step_data):
    """Compute tool count metrics for a single datasource group."""
    responses = step_data.get('response', [])
    if not responses:
        return {}

    tool_call_pattern = re.compile(r'</tool_response><\|im_end\|>\s*<\|im_start\|>assistant')
    tool_counts = []

    for response in responses:
        if isinstance(response, str):
            count = len(tool_call_pattern.findall(response))
            tool_counts.append(count)

    if not tool_counts:
        return {}

    return {
        'max_tool_count': max(tool_counts),
        'avg_tool_count': np.mean(tool_counts),
    }


def _compute_curriculum_rewards(step_data):
    """Compute curriculum reward metrics for a single datasource group."""
    data_sources = step_data.get('data_source', [])
    abilities = step_data.get('ability', [])
    indices = step_data.get('index', [])
    acc_rewards = step_data.get('acc_reward', [])

    if not data_sources:
        return {}

    # Detect task type (should be consistent within datasource)
    task_type = None
    for i in range(len(data_sources)):
        ds = data_sources[i] if i < len(data_sources) else ''
        ability = abilities[i] if i < len(abilities) else ''
        index = indices[i] if i < len(indices) else ''
        task_type = _detect_task_type(ds, ability, index)
        if task_type:
            break

    result = {}

    if acc_rewards:
        valid_acc = [v for v in acc_rewards if v is not None]
        if valid_acc:
            result['avg_acc_reward'] = np.mean(valid_acc)

    if not task_type:
        return result

    all_key, filter_key, answer_key = TASK_REWARD_KEYS[task_type]

    # Collect all_reward values
    if all_key in step_data:
        all_rewards = [v for v in step_data[all_key] if v is not None]
        if all_rewards:
            result[all_key] = all_rewards

    # Collect filter_reward values
    if filter_key in step_data:
        filter_rewards = [v for v in step_data[filter_key] if v is not None]
        if filter_rewards:
            result[filter_key] = filter_rewards

    return result


def _compute_answer_indices(step_data):
    """Compute answer index distribution for a single datasource group."""
    data_sources = step_data.get('data_source', [])
    abilities = step_data.get('ability', [])
    indices = step_data.get('index', [])

    if not data_sources:
        return {}

    # Detect task type and answer_key
    task_type = None
    for i in range(len(data_sources)):
        ds = data_sources[i] if i < len(data_sources) else ''
        ability = abilities[i] if i < len(abilities) else ''
        index = indices[i] if i < len(indices) else ''
        task_type = _detect_task_type(ds, ability, index)
        if task_type:
            break

    if not task_type:
        return {}

    all_key, filter_key, answer_key = TASK_REWARD_KEYS[task_type]
    if not answer_key or answer_key not in step_data:
        return {}

    # Count answer indices
    index_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for idx in step_data[answer_key]:
        try:
            idx_int = int(idx)
            if idx_int in index_counts:
                index_counts[idx_int] += 1
        except (TypeError, ValueError):
            pass

    total = sum(index_counts.values())
    if total == 0:
        return {}

    result = {}
    for idx in range(6):
        result[f'idx_{idx}_ratio'] = index_counts[idx] / total

    return result


# Main functions that will be called by plot_lambda_metrics
def datasource_tool_count_v2_function(step_data):
    """
    Extract tool count metrics per data source.
    Returns dict with keys: {datasource}__max_tool_count, {datasource}__avg_tool_count
    """
    ds_groups = _filter_by_datasource(step_data)
    result = {}

    for ds, ds_data in ds_groups.items():
        metrics = _compute_tool_counts(ds_data)
        ds_clean = str(ds).lower().replace(' ', '_').replace('-', '_')
        for metric_name, value in metrics.items():
            result[f"{ds_clean}__{metric_name}"] = value

    return result


def datasource_curriculum_reward_v2_function(step_data):
    """
    Extract curriculum reward metrics per data source.
    Returns dict with keys like: {datasource}__all_zoom_reward, {datasource}__avg_acc_reward
    """
    ds_groups = _filter_by_datasource(step_data)
    result = {}

    for ds, ds_data in ds_groups.items():
        metrics = _compute_curriculum_rewards(ds_data)
        ds_clean = str(ds).lower().replace(' ', '_').replace('-', '_')
        for metric_name, value in metrics.items():
            result[f"{ds_clean}__{metric_name}"] = value

    return result


def datasource_answer_index_v2_function(step_data):
    """
    Extract answer index distribution per data source.
    Returns dict with keys: {datasource}__idx_0_ratio, {datasource}__idx_1_ratio, ...
    """
    ds_groups = _filter_by_datasource(step_data)
    result = {}

    for ds, ds_data in ds_groups.items():
        metrics = _compute_answer_indices(ds_data)
        ds_clean = str(ds).lower().replace(' ', '_').replace('-', '_')
        for metric_name, value in metrics.items():
            result[f"{ds_clean}__{metric_name}"] = value

    return result


# Export list for plot_lambda_metrics.py
LAMBDA_EXAMPLES = [
    ("Datasource tool count v2", datasource_tool_count_v2_function),
    ("Datasource curriculum reward v2", datasource_curriculum_reward_v2_function),
    ("Datasource answer index v2", datasource_answer_index_v2_function),
]
