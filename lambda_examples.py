"""
Lambda function examples for custom metrics in RL Logging Board.

Each example should be a tuple of (name, expression).
For multi-line def functions, use triple quotes and name your function 'custom_function'.
"""

import numpy as np
import pandas as pd
from functools import partial

def tool_supervised_score_function(step_data):
    if 'tool_supervised_score' not in step_data:
        return {}
    tool_supervised_score = step_data['tool_supervised_score']
    # print(f"tool_supervised_score: {tool_supervised_score}")
    return tool_supervised_score

def datasource_count_function(step_data):
    datasource = {}
    if 'data_source' in step_data:
        np_arr = np.array(step_data['data_source'])
        datasource = {k: (np_arr == k).sum() for k in np.unique(np_arr)}
    return datasource

def all_reward_function(step_data):
    result = {}
    reward_keys = []
    for keys in step_data:
        if keys.endswith('zoom_reward'):
            reward_keys.append(keys)

    if 'data_source' not in step_data:
        return result

    data_sources = step_data['data_source']

    for reward_key in reward_keys:
        if reward_key not in step_data:
            continue
        reward_values = step_data[reward_key]

        for ds, reward_val in zip(data_sources, reward_values):
            combined_key = f"{ds}_{reward_key}"
            if combined_key not in result:
                result[combined_key] = []
            result[combined_key].append(reward_val)

    return result

def datasource_acc_reward_function(step_data):
    datasource = {}
    if 'data_source' not in step_data:
        return datasource
    for ds, reward, acc_reward in zip(step_data['data_source'], step_data['reward'], step_data['acc_reward']):
        if ds not in datasource:
            datasource[ds] = []
        datasource[ds].append(acc_reward)
    return datasource

def datasource_valid_tool_count_function(step_data):
    from collections import defaultdict
    datasource = defaultdict(list)
    if 'data_source' not in step_data:
        return datasource

    responses = step_data.get('response', [])
    data_sources = step_data.get('data_source', [])

    import re
    tool_call_pattern = re.compile(r'<tool_call>.*?</tool_call>', re.DOTALL)
    tool_response_pattern = re.compile(r'<tool_response>.*?</tool_response>', re.DOTALL)

    for ds, response in zip(data_sources, responses):
        if isinstance(response, str):
            tool_calls = tool_call_pattern.findall(response)
            tool_responses = tool_response_pattern.findall(response)
            valid_tool_count = min(len(tool_calls), len(tool_responses))
            datasource[ds].append(valid_tool_count/10.0)
        else:
            datasource[ds].append(0)

    return dict(datasource)

def datasource_reward_function(step_data):
    datasource = {}
    if 'data_source' not in step_data:
        return datasource
    for ds, reward in zip(step_data['data_source'], step_data['reward']):
        if ds not in datasource:
            datasource[ds] = []
        datasource[ds].append(reward)
    return datasource

def tool_calls_analysis_function(step_data):
    responses = step_data.get('response', [])
    import re
    tool_call_content_re = re.compile(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>')
    tool_counts = {'tool_call': 0}

    for response in responses:
        tool_call_content = tool_call_content_re.findall(response)
        if tool_call_content:
            for tool_call in tool_call_content:
                tool_counts['tool_call'] += 1

    for t in tool_counts:
        tool_counts[t] = tool_counts[t] / len(responses)

    return tool_counts

def tool_call_to_source_count_function(step_data):
    from collections import defaultdict
    responses = step_data.get('response', [])
    datasource = step_data.get('data_source', [])
    tool_counts = defaultdict(int)
    for response, datasource in zip(responses, datasource):
        if 'tool_call' in response.lower():
            tool_counts[datasource] += 1
    return tool_counts

def tool_call_to_source_percent_function(step_data):
    from collections import defaultdict
    responses = step_data.get('response', [])
    datasource = step_data.get('data_source', [])
    tool_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for response, datasource in zip(responses, datasource):
        if 'tool_call' in response.lower():
            tool_counts[datasource] += 1
        source_counts[datasource] += 1
    return {k: tool_counts[k]/(source_counts[k]+1e-6) for k in tool_counts}

def docvqa_aug_tool_call_rate_function(step_data):
    if 'data_source' not in step_data:
        return {}
    tools = ['image_mark_points', 'image_zoom_in', 'image_rotate', 'image_flip', 'image_draw_horizontal_line', 'image_draw_vertical_line']
    responses = step_data.get('response', [])
    from collections import defaultdict
    import re
    # inside <tool_call>...</tool_call>
    tool_call_content_re = re.compile(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>')
    tool_counts = defaultdict(int)
    source_counts = 0
    for ability, datasource, response in zip(step_data['ability'], step_data['data_source'], responses):
        if not ('docvqa' in datasource and ('rot' in ability or 'flip' in ability)):
            continue
        source_counts += 1
        tool_call_content = tool_call_content_re.findall(response)
        if tool_call_content:
            for tool_call in tool_call_content:
                for tool in tools:
                    if tool in tool_call:
                        tool_counts[tool] += 1
    return {k: tool_counts[k]/source_counts for k in tool_counts}

def docvqa_non_aug_tool_call_rate_function(step_data):
    if 'data_source' not in step_data:
        return {}
    tools = ['image_mark_points', 'image_zoom_in', 'image_rotate', 'image_flip', 'image_draw_horizontal_line', 'image_draw_vertical_line']
    responses = step_data.get('response', [])
    from collections import defaultdict
    import re
    # inside <tool_call>...</tool_call>
    tool_call_content_re = re.compile(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>')
    tool_counts = defaultdict(int)
    source_counts = 0
    for ability, datasource, response in zip(step_data['ability'], step_data['data_source'], responses):
        if 'docvqa' in datasource and ('rot' in ability or 'flip' in ability):
            continue
        source_counts += 1
        tool_call_content = tool_call_content_re.findall(response)
        if tool_call_content:
            for tool_call in tool_call_content:
                for tool in tools:
                    if tool in tool_call:
                        tool_counts[tool] += 1
    return {k: tool_counts[k]/source_counts for k in tool_counts}

def each_tool_avg_count_function(step_data):
    tools = ['image_mark_points', 'image_zoom_in', 'image_draw_line', 'draw_horizontal_line', 'draw_vertical_line', 'image_rotate', 'image_flip']
    responses = step_data.get('response', [])
    from collections import defaultdict
    import re
    # inside <tool_call>...</tool_call>
    tool_call_content_re = re.compile(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>')
    tool_counts = defaultdict(int)
    for response in responses:
        tool_call_content = tool_call_content_re.findall(response)
        if tool_call_content:
            for tool_call in tool_call_content:
                for tool in tools:
                    if tool in tool_call:
                        tool_counts[tool] += 1
    total_cases = len(responses)
    return {k: tool_counts[k]/total_cases for k in tool_counts}

def zoom_call_target_image_function(step_data):
    responses = step_data.get('response', [])
    from collections import defaultdict
    import re
    import json

    # Pattern to extract tool_call content
    tool_call_content_re = re.compile(r'<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>')
    target_image_counts = defaultdict(int)
    total_zoom_calls = 0

    for response in responses:
        if not isinstance(response, str):
            continue

        tool_call_content = tool_call_content_re.findall(response, re.DOTALL)
        if tool_call_content:
            for tool_call in tool_call_content:
                try:
                    tool_data = json.loads(tool_call)
                    # Check if this is a zoom_in tool call
                    if 'zoom_in' in tool_data.get('name').lower():
                        total_zoom_calls += 1
                        arguments = tool_data.get('arguments', {})
                        target_image = arguments.get('target_image')

                        # Count target_image values from -2 to 2
                        if target_image is not None and target_image in [-2, -1, 0, 1, 2]:
                            target_image_counts[target_image] += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    # Calculate distribution
    if total_zoom_calls == 0:
        return {str(i): 0 for i in range(-2, 3)}

    return {str(i): target_image_counts[i] / total_zoom_calls for i in range(-2, 3)}

def valid_tool_use_count_function(step_data):
    responses = step_data.get('response', [])
    turn_counts = []

    for response in responses:
        if isinstance(response, str):
            import re
            pattern = re.compile(r'</tool_response><\|im_end\|>\s*<\|im_start\|>assistant')
            # Count complete pairs
            matches = pattern.findall(response)
            turns = len(matches)
            turn_counts.append(turns)
        else:
            turn_counts.append(0)

    if not turn_counts:
        return {'max': 0, 'avg': 0}

    return {
        'max': max(turn_counts),
        'avg': sum(turn_counts) / len(turn_counts)
    }

def assertion_error_count_function(step_data):
    responses = step_data.get('response', [])
    assertion_error_count = 0
    total_responses = len(responses)

    max_assertion_error_count = 0
    for response in responses:
        if isinstance(response, str) and 'assertionerror' in response.lower():
            assertion_error_count += response.lower().count('assertionerror')
            max_assertion_error_count = max(max_assertion_error_count, response.lower().count('assertionerror'))

    return {
        'avg': assertion_error_count / total_responses if total_responses > 0 else -1,
        'max': max_assertion_error_count
    }

def answer_tag_count_function(step_data):
    responses = step_data.get('response', [])
    answer_count = 0
    total_responses = len(responses)
    import re
    for response in responses:
        think_answer_pattern = re.compile(r'</think>\s*<answer>', re.DOTALL)
        if isinstance(response, str) and think_answer_pattern.search(response):
            answer_count += 1
    return {
        'answer_rate': answer_count / total_responses if total_responses > 0 else -1
    }

def tool_error_penalty_applied_function(step_data):
    tool_error_penalty_applied = step_data.get('tool_error_penalty_applied', 0.0)
    return {
        'tool_error_penalty_applied': tool_error_penalty_applied
    }

def _generic_tool_count_function(step_data, filter_keywords, include=True, return_wprefix=False, extra_keys=None):
    """Generic function to count tool calls and compute average rewards for filtered data sources.

    Args:
        step_data: Dictionary containing response, data_source, and acc_reward data
        filter_keywords: List of keywords to match in data source (case-insensitive)
        include: If True, include matching sources; if False, exclude matching sources
        return_wprefix: If True, return metrics grouped by data_source with prefixed keys
        extra_keys: List of additional keys to extract and average from filtered items

    Returns:
        Dictionary with avg_tool_count and avg_acc_reward (optionally prefixed by data_source)
        Also includes averages for any extra_keys specified
    """
    if 'data_source' not in step_data:
        return {}

    responses = step_data.get('response', [])
    data_sources = step_data.get('data_source', [])
    acc_rewards = step_data.get('acc_reward', [])

    # Handle extra_keys
    if extra_keys is None:
        extra_keys = []
    extra_data = {}
    for key in extra_keys:
        data = step_data.get(key, [])
        if not isinstance(data, (list, tuple, np.ndarray)):
            data = [data] * len(responses)
        extra_data[key] = data

    import re
    from itertools import zip_longest
    from collections import defaultdict
    tool_call_content_re = re.compile(r'</tool_response><\|im_end\|>\s*<\|im_start\|>assistant', re.DOTALL)

    if not isinstance(acc_rewards, (list, tuple, np.ndarray)):
        acc_rewards = [acc_rewards] * len(responses)

    if return_wprefix:
        # Group by data_source after filtering by keywords
        def _default_stats():
            stats = {'total_tool_calls': 0, 'total_acc_reward': 0.0, 'count': 0}
            for key in extra_keys:
                stats[f'total_{key}'] = 0.0
            return stats
        ds_stats = defaultdict(_default_stats)

        for idx, (response, ds, acc_reward) in enumerate(zip_longest(responses, data_sources, acc_rewards, fillvalue=None)):
            if not isinstance(ds, str):
                continue
            ds_lower = ds.lower()

            # Check if any keyword matches
            matches = any(keyword in ds_lower for keyword in filter_keywords)

            # Include or exclude based on the include flag
            if matches == include:
                # Clean data_source name for use as prefix
                ds_prefix = ds.lower().replace(' ', '_').replace('-', '_')

                ds_stats[ds_prefix]['count'] += 1
                if isinstance(response, str):
                    tool_call_content = tool_call_content_re.findall(response)
                    ds_stats[ds_prefix]['total_tool_calls'] += len(tool_call_content) / 10.0
                try:
                    ds_stats[ds_prefix]['total_acc_reward'] += float(acc_reward)
                except (TypeError, ValueError):
                    ds_stats[ds_prefix]['total_acc_reward'] += 0.0

                # Accumulate extra keys
                for key in extra_keys:
                    try:
                        value = extra_data[key][idx] if idx < len(extra_data[key]) else 0.0
                        ds_stats[ds_prefix][f'total_{key}'] += float(value)
                    except (TypeError, ValueError, IndexError):
                        ds_stats[ds_prefix][f'total_{key}'] += 0.0

        # Build result dictionary with prefixed keys
        result = {}
        for ds_prefix, stats in ds_stats.items():
            if stats['count'] > 0:
                result[f'{ds_prefix}_avg_tool_count'] = stats['total_tool_calls'] / stats['count']
                result[f'{ds_prefix}_avg_acc_reward'] = stats['total_acc_reward'] / stats['count']
                for key in extra_keys:
                    result[f'{ds_prefix}_avg_{key}'] = stats[f'total_{key}'] / stats['count']

        return result
    else:
        # Original behavior: filter and aggregate
        total_tool_calls = 0
        filtered_count = 0
        total_acc_reward = 0.0
        total_extra = {key: 0.0 for key in extra_keys}

        for idx, (response, ds, acc_reward) in enumerate(zip_longest(responses, data_sources, acc_rewards, fillvalue=None)):
            if not isinstance(ds, str):
                continue
            ds_lower = ds.lower()

            # Check if any keyword matches
            matches = any(keyword in ds_lower for keyword in filter_keywords)

            # Include or exclude based on the include flag
            if matches == include:
                filtered_count += 1
                if isinstance(response, str):
                    tool_call_content = tool_call_content_re.findall(response)
                    total_tool_calls += len(tool_call_content) / 10.0
                try:
                    total_acc_reward += float(acc_reward)
                except (TypeError, ValueError):
                    total_acc_reward += 0.0

                # Accumulate extra keys
                for key in extra_keys:
                    try:
                        value = extra_data[key][idx] if idx < len(extra_data[key]) else 0.0
                        total_extra[key] += float(value)
                    except (TypeError, ValueError, IndexError):
                        total_extra[key] += 0.0

        if filtered_count == 0:
            result = {'avg_tool_count': 0, 'avg_acc_reward': 0}
            for key in extra_keys:
                result[f'avg_{key}'] = 0.0
            return result

        result = {
            'avg_tool_count': total_tool_calls / filtered_count,
            'avg_acc_reward': total_acc_reward / filtered_count
        }
        for key in extra_keys:
            result[f'avg_{key}'] = total_extra[key] / filtered_count
        return result

# Create specific functions using partial with serializable arguments
highres_tool_count_function = partial(_generic_tool_count_function, filter_keywords=['probe'], include=True)
seal_tool_count_function = partial(_generic_tool_count_function, filter_keywords=['sealvqa'], include=True)
rotflip_tool_count_function = partial(_generic_tool_count_function, filter_keywords=['docvqa'], include=True, extra_keys=['filter_rotflip_reward'])
draw_tool_count_function = partial(_generic_tool_count_function, filter_keywords=['read_value','compare'], include=True, return_wprefix=True, extra_keys=['filter_draw_reward', 'filter_compare_reward'])
zoomin_tool_count_function = partial(_generic_tool_count_function, filter_keywords=['sealvqa', 'visual_probe'], include=True)
non_zoomin_tool_count_function = partial(_generic_tool_count_function, filter_keywords=['sealvqa', 'visual_probe'], include=False)

def acc_reward_function(step_data):
    acc_reward = step_data.get('acc_reward', 0.0)
    return {
        'acc_reward': acc_reward
    }

def filter_rotflip_reward_function(step_data):
    """Extract filter_rotflip_reward and rotflip_answer_index ratios from curriculum1 reward output, filtered for docvqa."""
    if 'data_source' not in step_data or 'filter_rotflip_reward' not in step_data:
        return {}

    data_sources = step_data.get('data_source', [])
    filter_rotflip_rewards = step_data.get('filter_rotflip_reward', [])
    rotflip_answer_indices = step_data.get('rotflip_answer_index', [])

    # Ensure lists
    if not isinstance(data_sources, (list, tuple, np.ndarray)):
        data_sources = [data_sources]
    if not isinstance(filter_rotflip_rewards, (list, tuple, np.ndarray)):
        filter_rotflip_rewards = [filter_rotflip_rewards]
    if not isinstance(rotflip_answer_indices, (list, tuple, np.ndarray)):
        rotflip_answer_indices = [rotflip_answer_indices]

    # Filter for docvqa items
    filtered_rewards = []
    filtered_indices = []

    from itertools import zip_longest
    for ds, reward, idx in zip_longest(data_sources, filter_rotflip_rewards, rotflip_answer_indices, fillvalue=None):
        if isinstance(ds, str) and 'docvqa' in ds.lower():
            if reward is not None:
                filtered_rewards.append(reward)
            if idx is not None:
                filtered_indices.append(idx)

    if not filtered_rewards:
        return {}

    # Calculate average reward
    result = {'filter_rotflip_reward': np.mean(filtered_rewards)}

    # Calculate rotflip_answer_index ratios
    if filtered_indices:
        index_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        total = 0

        for idx in filtered_indices:
            if idx in index_counts:
                index_counts[idx] += 1
                total += 1

        # Calculate ratios
        if total > 0:
            result['idx_0_ratio'] = index_counts[0] / total
            result['idx_1_ratio'] = index_counts[1] / total
            result['idx_2_ratio'] = index_counts[2] / total
            result['idx_3_ratio'] = index_counts[3] / total
            result['idx_4_ratio'] = index_counts[4] / total
            result['idx_5_ratio'] = index_counts[5] / total

    return result

LAMBDA_EXAMPLES = [
    ("Filter rotflip reward", filter_rotflip_reward_function),
    ("Highres tool count", highres_tool_count_function),
    ("Seal tool count", seal_tool_count_function),
    ("Rotflip tool count", rotflip_tool_count_function),
    ("Draw tool count", draw_tool_count_function),
    ("Zoom call target image", zoom_call_target_image_function),
    ("Datasource acc reward", datasource_acc_reward_function),
    ("Datasource valid tool count", datasource_valid_tool_count_function),
    ("Zoom-in tool count", zoomin_tool_count_function),
    ("Non zoom-in tool count", non_zoomin_tool_count_function),
    ("Acc reward", acc_reward_function),
    ("All reward", all_reward_function),
    ("Valid tool use count", valid_tool_use_count_function),
    ("Tool error penalty applied", tool_error_penalty_applied_function),
    ("Answer tag count", answer_tag_count_function),
    ("Assertion error count", assertion_error_count_function),
    ("Tool supervised score", tool_supervised_score_function),
    ("Tool call count", tool_calls_analysis_function),
    ("Tool use percent", each_tool_avg_count_function),
    ("Tool call to source count", tool_call_to_source_count_function),
    ("Tool call to source percent", tool_call_to_source_percent_function),
    ("Datasource reward", datasource_reward_function),
    ("Datasource count", datasource_count_function),
    ("Docvqa aug tool call rate", docvqa_aug_tool_call_rate_function),
    ("Docvqa non aug tool call rate", docvqa_non_aug_tool_call_rate_function),
]
