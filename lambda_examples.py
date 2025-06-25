"""
Lambda function examples for custom metrics in RL Logging Board.

Each example should be a tuple of (name, expression).
For multi-line def functions, use triple quotes and name your function 'custom_function'.
"""

import numpy as np
import pandas as pd
import inspect


def datasource_count(step_data):
    datasource = {}
    for key, value in step_data.items():
        if key in ['data_source']:
            datasource[value] = datasource.get(value, 0) + 1
    return datasource

LAMBDA_EXAMPLES = [
    ("KL * Reward", "lambda step_data: [kl * r for kl, r in zip(step_data['avg_kl'], step_data['reward'])] if step_data['avg_kl'] else [0]"),

    ("Response length groups", """def custom_function(step_data):
    if not step_data.get('response'):
        return {'short': 0, 'medium': 0, 'long': 0}
    lengths = [len(r) for r in step_data['response']]
    return {
        'short': sum(1 for l in lengths if l < 50),
        'medium': sum(1 for l in lengths if 50 <= l < 200),
        'long': sum(1 for l in lengths if l >= 200)
    }"""),

    ("Reward categories", """def custom_function(step_data):
    rewards = step_data.get('reward', [])
    return {
        'high': sum(1 for r in rewards if r > 0.7),
        'medium': sum(1 for r in rewards if 0.3 <= r <= 0.7),
        'low': sum(1 for r in rewards if r < 0.3)
    }"""),

    ("Tool usage analysis", """def custom_function(step_data):
    responses = step_data.get('response', [])
    tool_counts = {'tool_call': 0, 'regular': 0}

    for response in responses:
        if 'tool_call' in response.lower():
            tool_counts['tool_call'] += 1
        else:
            tool_counts['regular'] += 1

    return tool_counts"""),

    ("Datasource count", inspect.getsource(datasource_count)),
]
