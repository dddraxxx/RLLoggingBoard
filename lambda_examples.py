"""
Lambda function examples for custom metrics in RL Logging Board.

Each example should be a tuple of (name, expression).
For multi-line def functions, use triple quotes and name your function 'custom_function'.
"""

import numpy as np
import pandas as pd
import inspect

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

def acc_reward_function(step_data):
    acc_reward = step_data.get('acc_reward', 0.0)
    return {
        'acc_reward': acc_reward
    }

LAMBDA_EXAMPLES = [
    ["Acc reward", inspect.getsource(acc_reward_function)],
    ("Tool error penalty applied", inspect.getsource(tool_error_penalty_applied_function)),
    ("Valid tool use count", inspect.getsource(valid_tool_use_count_function)),
    ("Answer tag count", inspect.getsource(answer_tag_count_function)),
    ("Assertion error count", inspect.getsource(assertion_error_count_function)),
    ("Tool supervised score", inspect.getsource(tool_supervised_score_function)),
    ("Tool call count", inspect.getsource(tool_calls_analysis_function)),
    ("Tool use percent", inspect.getsource(each_tool_avg_count_function)),
    ("Tool call to source count", inspect.getsource(tool_call_to_source_count_function)),
    ("Tool call to source percent", inspect.getsource(tool_call_to_source_percent_function)),
    ("Datasource reward", inspect.getsource(datasource_reward_function)),
    ("Datasource count", inspect.getsource(datasource_count_function)),
    ("Docvqa aug tool call rate", inspect.getsource(docvqa_aug_tool_call_rate_function)),
    ("Docvqa non aug tool call rate", inspect.getsource(docvqa_non_aug_tool_call_rate_function)),
]
