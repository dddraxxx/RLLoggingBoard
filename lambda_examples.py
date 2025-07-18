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
    tool_counts = {'<tool_call>': 0, 'regular': 0}

    for response in responses:
        if 'tool_call' in response.lower():
            tool_counts['tool_call'] += 1
        else:
            tool_counts['regular'] += 1

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

def synthetic_chart_tool_call_rate_function(step_data):
    if 'data_source' not in step_data:
        return {}
    tools = ['image_mark_points', 'image_zoom_in', 'image_draw_line']
    responses = step_data.get('response', [])
    from collections import defaultdict
    import re
    # inside <tool_call>...</tool_call>
    tool_call_content_re = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tool_counts = defaultdict(int)
    source_counts = 0
    for datasource, response in zip(step_data['data_source'], responses):
        if datasource not in ['synthetic_chart']:
            continue
        source_counts += 1
        tool_call_content = tool_call_content_re.findall(response)
        if tool_call_content:
            for tool_call in tool_call_content:
                for tool in tools:
                    if tool in tool_call:
                        tool_counts[tool] += 1
    return {k: tool_counts[k]/source_counts for k in tool_counts}

def tool_use_percent_function(step_data):
    tools = ['image_mark_points', 'image_zoom_in', 'image_draw_line', 'draw_horizontal_line', 'draw_vertical_line']
    responses = step_data.get('response', [])
    from collections import defaultdict
    import re
    # inside <tool_call>...</tool_call>
    tool_call_content_re = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tool_call_content_list = []
    tool_counts = defaultdict(int)
    for response in responses:
        tool_call_content = tool_call_content_re.findall(response)
        if tool_call_content:
            for tool_call in tool_call_content:
                for tool in tools:
                    if tool in tool_call:
                        tool_counts[tool] += 1
    return tool_counts

LAMBDA_EXAMPLES = [
    ("Tool supervised score", inspect.getsource(tool_supervised_score_function)),
    ("Tool use percent", inspect.getsource(tool_use_percent_function)),
    ("Tool call to source count", inspect.getsource(tool_call_to_source_count_function)),
    ("Tool call to source percent", inspect.getsource(tool_call_to_source_percent_function)),
    ("Datasource reward", inspect.getsource(datasource_reward_function)),
    ("Datasource count", inspect.getsource(datasource_count_function)),
    ("Tool usage analysis", inspect.getsource(tool_calls_analysis_function)),
    ("Synthetic chart tool call rate", inspect.getsource(synthetic_chart_tool_call_rate_function)),
]
