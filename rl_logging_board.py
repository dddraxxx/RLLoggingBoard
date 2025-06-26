"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

启动 web 页面可视化 RL 训练过程中间 metric 的数据状态。

Author: pankeyu
Date: 2023/10/31
"""
import os
import copy
from pathlib import Path
import time
import traceback
import argparse
import sys
import html
import re

import orjson as json


import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Import image actions
from image_actions import (
    get_image_action_registry,
    display_image_with_actions,
)


st.set_page_config(
    page_title="RL Logging Board",
    page_icon="📖",
    layout='wide'
)


# Get the global registry instance from the image_actions module
IMAGE_ACTION_REGISTRY = get_image_action_registry()


def load_common_filters():
    """
    Load common filter expressions from configuration file.

    Returns:
        list: List of filter expression strings, or empty list if file not found
    """
    try:
        # Look for common_filters.json in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filters_path = os.path.join(script_dir, 'common_filters.json')

        if os.path.exists(filters_path):
            with open(filters_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
                return data.get('common_filters', [])
        else:
            # Return default filters if file doesn't exist
            return [
                "response.str.contains('tool_call', case=False, na=False)",
                "response.str.contains('mark_point', case=False, na=False)",
                "response.str.contains('zoom_in', case=False, na=False)",
                "response.str.contains('draw_line', case=False, na=False)"
            ]
    except Exception as e:
        st.error(f"Error loading common filters: {e}")
        return []


def load_lambda_examples():
    """
    Load lambda function examples from Python configuration file.

    Returns:
        list: List of tuples (name, expression), or default examples if file not found
    """
    try:
        # Look for lambda_examples.py in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        examples_path = os.path.join(script_dir, 'lambda_examples.py')

        if os.path.exists(examples_path):
            # Import the examples from the Python file
            import importlib.util
            spec = importlib.util.spec_from_file_location("lambda_examples", examples_path)
            examples_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(examples_module)
            return examples_module.LAMBDA_EXAMPLES
        else:
            # Return default examples if file doesn't exist
            return [
                ("Average reward", "lambda step_data: [sum(step_data['reward']) / len(step_data['reward'])]"),
                ("Max reward", "lambda step_data: [max(step_data['reward'])]"),
                ("Reward variance", "lambda step_data: [np.var(step_data['reward'])]"),
                ("Reward distribution", "lambda step_data: step_data['reward']"),
                ("Response length avg", "lambda step_data: [np.mean(step_data['response_tokens_len'])] if step_data['response_tokens_len'] else [0]"),
                ("Reward improvement", "lambda step_data: [r - ref_r for r, ref_r in zip(step_data['reward'], step_data['ref_reward'])] if step_data['ref_reward'] else step_data['reward']"),
                ("High reward count", "lambda step_data: [sum(1 for r in step_data['reward'] if r > 0.5)]"),
                ("KL * Reward", "lambda step_data: [kl * r for kl, r in zip(step_data['avg_kl'], step_data['reward'])] if step_data['avg_kl'] else [0]")
            ]
    except Exception as e:
        st.error(f"Error loading lambda examples: {e}")
        return []


def save_lambda_examples(examples):
    """
    Save lambda function examples to Python configuration file.

    Args:
        examples (list): List of tuples (name, expression)
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        examples_path = os.path.join(script_dir, 'lambda_examples.py')

        # Create the Python file content
        content = '''"""
Lambda function examples for custom metrics in RL Logging Board.

Each example should be a tuple of (name, expression).
For multi-line def functions, use triple quotes and name your function 'custom_function'.
"""

import numpy as np
import pandas as pd

LAMBDA_EXAMPLES = [
'''

        for name, expression in examples:
            # Properly escape the expression for Python syntax
            if '\n' in expression:
                # Multi-line expression - use triple quotes
                content += f'    ("{name}", """{expression}"""),\n\n'
            else:
                # Single line expression - use regular quotes and escape
                escaped_expr = expression.replace('\\', '\\\\').replace('"', '\\"')
                content += f'    ("{name}", "{escaped_expr}"),\n\n'

        content += ']\n'

        with open(examples_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        st.error(f"Error saving lambda examples: {e}")
        return False


def process_content_for_display(content):
    """
    Process content for safe HTML display by:
    1. HTML escaping tags to prevent interpretation as HTML
    2. Compressing repeated <|image_pad|> tokens to <|image_pad|>*[n] format

    Args:
        content (str): Raw content string

    Returns:
        str: Processed content safe for HTML display
    """
    if not content:
        return content

    # HTML escape to prevent tag interpretation
    content = html.escape(content)

    # Compress repeated <|image_pad|> tokens
    # Find all consecutive occurrences of <|image_pad|>
    def compress_image_pads(match):
        repeated_token = match.group(0)
        count = repeated_token.count('&lt;|image_pad|&gt;')  # HTML escaped version
        if count > 1:
            return f'&lt;|image_pad|&gt;*[{count}]'
        return repeated_token

    # Pattern to match consecutive <|image_pad|> tokens (HTML escaped)
    pattern = r'(?:&lt;\|image_pad\|&gt;)+'
    content = re.sub(pattern, compress_image_pads, content)

    return content


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="RL Logging Board - Visualize RL training metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rl_logging_board.py
  python rl_logging_board.py --log-root ./my_logs
  python rl_logging_board.py --log-root /path/to/logs
        """
    )

    parser.add_argument(
        '--log-root',
        type=str,
        default='./rollout_samples',
        help='Root directory containing log files (default: ./rollout_samples)'
    )

    return parser.parse_args()


# Parse command line arguments
try:
    args = parse_args()
    DEFAULT_LOG_ROOT = args.log_root
except SystemExit:
    # This happens when streamlit processes its own arguments
    # Fall back to default value
    DEFAULT_LOG_ROOT = './rollout_samples'


def load_log_file(
    logdir: os.PathLike,
    max_samples_each_step: int,
    step_freq: int,
    start_step: int,
    end_step: int
):
    """
    解析本地log文件。

    Args:
        logdir (os.PathLike): _description_
        max_samples_each_step (int): _description_
        step_freq (int): 步骤采样频率，每隔step_freq步采样一次
        start_step (int): 开始步骤，只处理大于等于此步骤的数据
        end_step (int): 结束步骤，只处理小于等于此步骤的数据，-1表示不限制
    """
    st.session_state['logging_name'] = logdir
    st.session_state['max_samples_each_step'] = max_samples_each_step
    st.session_state['step_freq'] = step_freq
    st.session_state['start_step'] = start_step
    st.session_state['end_step'] = end_step
    st.session_state['logging_data'] = {}
    error_lines, success_lines = 0, 0

    # all_logs = os.listdir(logdir)
    # search for any jsonl files end with _rank0.jsonl, search for any depth
    all_logs = []
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.endswith('_rank0.jsonl'):
                all_logs.append(os.path.relpath(os.path.join(root, file), logdir))

    logs_names = [os.path.basename(log) for log in all_logs]
    progress_text = f"Processing all files..." + ','.join(logs_names)
    loading_files_bar = st.progress(0., text=progress_text)

    progress_text = f"Processing each file samples..."
    loading_samples_bar = st.progress(0., text=progress_text)


    for log_index in range(len(all_logs)):

        if not all_logs[log_index].endswith('.jsonl'):
            continue

        rl_log_file = os.path.join(
            logdir,
            all_logs[log_index]
        )

        # Get actual file size for accurate progress tracking
        file_size = os.path.getsize(rl_log_file)

        with open(rl_log_file, 'rb') as f:
            start_time = time.time()
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    data['step'] = int(data['step'])

                    # Apply step range filtering
                    if data['step'] < start_step:
                        continue
                    if end_step != -1 and data['step'] > end_step:
                        break

                    # Apply step frequency filtering
                    if data['step'] % step_freq != 0:
                        continue

                    if data['step'] not in st.session_state['logging_data']:
                        st.session_state['logging_data'][data['step']] = {
                            'prompt': [],
                            'response': [],
                            'ref_response': [],
                            'reward': [],
                            'ref_reward': [],
                            'response_tokens': [],
                            'logprobs': [],
                            'ref_logprobs': [],
                            'probs': [],
                            'ref_probs': [],
                            'values': [],
                            'token_rewards': [],
                            'kl': [],
                            'avg_kl': [],
                            'sum_kl': [],
                            'log_ratio': [],
                            'avg_log_ratio': [],
                            'sum_log_ratio': [],
                            'valid_reward': [],
                            'ref_valid_reward': [],
                            'response_tokens_len': [],
                            'ground_truth': [],
                            'image_path': [],
                        }
                    elif len(st.session_state['logging_data'][data['step']]['prompt']) >= max_samples_each_step and max_samples_each_step > 0:
                        continue

                    # for key in st.session_state['logging_data'][data['step']]:
                    #     if key in data:
                    #         st.session_state['logging_data'][data['step']][key].append(data[key])
                    # also update data keys
                    for key in data:
                        if key not in st.session_state['logging_data'][data['step']]:
                            st.session_state['logging_data'][data['step']][key] = []
                        st.session_state['logging_data'][data['step']][key].append(data[key])

                    if 'response_tokens' in data:
                        st.session_state['logging_data'][data['step']]['response_tokens_len'].append(len(data['response_tokens']))

                    if 'logprobs' in data and 'ref_logprobs' in data:
                        logp = np.array(data['logprobs'])
                        ref_logp = np.array(data['ref_logprobs'])
                        log_ratio = logp - ref_logp
                        kl = np.exp(log_ratio) - 1 - log_ratio
                        st.session_state['logging_data'][data['step']]['log_ratio'].append(log_ratio.tolist())
                        st.session_state['logging_data'][data['step']]['avg_log_ratio'].append(np.nanmean(log_ratio))
                        st.session_state['logging_data'][data['step']]['sum_log_ratio'].append(np.nansum(log_ratio))
                        st.session_state['logging_data'][data['step']]['kl'].append(kl.tolist())
                        st.session_state['logging_data'][data['step']]['avg_kl'].append(np.nanmean(kl))
                        st.session_state['logging_data'][data['step']]['sum_kl'].append(np.nansum(kl))
                        st.session_state['logging_data'][data['step']]['probs'].append(np.exp(logp).tolist())
                        st.session_state['logging_data'][data['step']]['ref_probs'].append(np.exp(ref_logp).tolist())

                    success_lines += 1

                except:
                    print(traceback.format_exc())
                    error_lines += 1

                # Update progress based on actual file position
                current_position = f.tell()
                percentage = current_position / file_size
                size_unit = 'MB' if file_size > 1_000_000 else 'KB' if file_size > 1_000 else 'bytes'
                size_factor = 1_000_000 if size_unit == 'MB' else 1_000 if size_unit == 'KB' else 1
                current_size = current_position / size_factor
                total_size = file_size / size_factor
                estiamte_finish_time = (time.time() - start_time) / percentage * (1 - percentage)
                loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {current_size:,.1f} / {total_size:,.1f} {size_unit} ({i + 1} lines)... {estiamte_finish_time:.1f}s")

    percentage = 1.0
    loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {(success_lines + error_lines)} / {(success_lines + error_lines)} samples...")

    file_percentage = (log_index + 1) / len(all_logs)
    loading_files_bar.progress(file_percentage, text=f"[{int(file_percentage * 100)}%] Loading {log_index + 1} / {len(all_logs)} files...")

    st.toast(
        f'Loaded {success_lines + error_lines} sample(s), sucess: {success_lines}, error: {error_lines}.',
        icon='🎉'
    )

    if not st.session_state['logging_data']:
        st.warning(f'No log file(s) found in {logdir}.', icon='⚠️')
        st.stop()

    all_steps = [int(s) for s in list(st.session_state["logging_data"].keys())]
    all_steps.sort()
    st.session_state['max_step_index'] = max(all_steps)
    st.session_state['min_step_index'] = min(all_steps)
    st.session_state['step_gap'] = 1 if len(all_steps) < 2 else all_steps[1] - all_steps[0]

    rewards_dict = {'step': [], 'reward': [], 'ref_reward': []}
    for step in st.session_state['logging_data']:
        st.session_state['logging_data'][step]['avg_reward'] = sum(st.session_state['logging_data'][step]['reward']) / len(st.session_state['logging_data'][step]['reward'])

        current_step_resp_length = [len(resp) for resp in st.session_state['logging_data'][step]['response']]
        st.session_state['logging_data'][step]['avg_length'] = int(sum(current_step_resp_length) / len(current_step_resp_length))

        current_step_ref_resp_length = [len(resp) for resp in st.session_state['logging_data'][step]['ref_response']]
        st.session_state['logging_data'][step]['avg_ref_length'] = int(sum(current_step_ref_resp_length) / len(current_step_ref_resp_length)) if len(current_step_ref_resp_length) else 0

        if len(st.session_state['logging_data'][step]['ref_reward']):
            st.session_state['logging_data'][step]['avg_ref_reward'] = sum(st.session_state['logging_data'][step]['ref_reward']) / len(st.session_state['logging_data'][step]['ref_reward']) if len(st.session_state['logging_data'][step]['ref_reward']) else 0
        else:
            st.session_state['logging_data'][step]['avg_ref_reward'] = 0
        rewards_dict['step'].append(step)
        rewards_dict['reward'].append(st.session_state['logging_data'][step]['avg_reward'])
        rewards_dict['ref_reward'].append(st.session_state['logging_data'][step]['avg_ref_reward'])

    rewards_df = pd.DataFrame.from_dict(rewards_dict)
    st.session_state['reward_df'] = rewards_df.set_index('step')


def plot_filled_line(
    x: list,
    y_list_list: list,
    data_names: list,
    colors: list,
    title=None
):
    """
    绘制带有阴影的折线图，阴影上下界为当前x对应的y列表中的最大、最小值。

    Args:
        x (list): step 横轴索引
        y_list_list (line_num, steps, step_wise): 可绘制多条直线，维度为：绘制折线条数，总共的step数，每个step对应几个y值
        data_names (list): 每条折线的名字列表
        colors (list): 每条折线的颜色列表（rgb）, e.g. -> ['255,171,171']
    """
    fig = go.Figure()

    x_rev = x[::-1]
    for i in range(len(y_list_list)):
        y_list = y_list_list[i]
        y_mean, y_lower, y_upper = [], [], []
        for y in y_list:
            y_arr = np.array(y)
            mean, std = float(y_arr.mean()), float(y_arr.std())
            y_mean.append(mean)
            y_lower.append(mean - std)
            y_upper.append(mean + std)
            # y_lower.append(min(y))
            # y_upper.append(max(y))
        y_lower = y_lower[::-1]

        fig.add_trace(go.Scatter(
            x=x + x_rev,
            y=y_upper + y_lower,
            fill='toself',
            fillcolor=f'rgba({colors[i]},0.1)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=data_names[i],
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            line_color=f'rgb({colors[i]})',
            name=data_names[i],
        ))

    fig.update_traces(mode='lines')

    if title:
        fig.update_layout(
            title=title,
            legend=dict(orientation="h")
        )

    return fig


@st.cache_data
def format_log_name(log_path, max_length=40):
    """
    Format log path for better display in selector.

    Args:
        log_path (str): Original log path
        max_length (int): Maximum display length

    Returns:
        str: Formatted log name for display
    """
    if len(log_path) <= max_length:
        return log_path

    # Split path and keep important parts
    parts = log_path.split('/')
    if len(parts) <= 2:
        return log_path

    # Keep first and last parts, truncate middle
    first_part = parts[0]
    last_part = parts[-1]

    if len(first_part) + len(last_part) + 5 <= max_length:  # 5 for ".../"
        return f"{first_part}/.../{last_part}"
    else:
        # If still too long, truncate the last part
        available_length = max_length - len(first_part) - 5
        if available_length > 10:
            return f"{first_part}/.../{last_part[:available_length-3]}..."
        else:
            return f".../{last_part}"


@st.cache_data
def get_log_directories(base_root_path):
    """
    Cached function to scan for log directories.

    Args:
        base_root_path (str): Base directory to scan

    Returns:
        list: List of log directory paths
    """
    base_path = Path(base_root_path)
    start_time = time.time()
    all_log_path_in_logdir = [
        file.parent.relative_to(base_path).as_posix()
        for file in base_path.glob('**/*_rank0.jsonl')
    ]
    elapsed_time = time.time() - start_time
    print(f"Time taken to scan log directories: {elapsed_time:.2f} seconds")
    # Remove duplicates and sort
    return sorted(list(set(all_log_path_in_logdir)))


def create_log_selector(all_log_paths, base_root_path, current_selection=None):
    """
    Create an improved log selector with expandable interface.

    Args:
        all_log_paths (list): List of available log paths
        base_root_path (str): Base root path for constructing full paths
        current_selection (str): Currently selected log path

    Returns:
        str: Selected log path
    """
    if not all_log_paths:
        return None

    # Initialize session state for log selector if not exists
    if 'selected_log_index' not in st.session_state:
        st.session_state.selected_log_index = len(all_log_paths) - 1

    # Ensure the index is valid
    if st.session_state.selected_log_index >= len(all_log_paths):
        st.session_state.selected_log_index = len(all_log_paths) - 1

    with st.sidebar.expander("📁 Choose Different Log Directory", expanded=True):
        st.markdown(f"**Available Logs ({len(all_log_paths)} found):**")

        # Create formatted options - cached for performance
        formatted_options = [f"{i+1:2d}. {format_log_name(log_path)}" for i, log_path in enumerate(all_log_paths)]

        # Radio button selection - this gives us the CURRENT selection immediately
        selected_option_index = st.radio(
            "Select log directory:",
            range(len(all_log_paths)),
            index=st.session_state.selected_log_index,
            format_func=lambda x: formatted_options[x],
            key="log_selector_radio"
        )

        # Update session state - no explicit rerun needed, Streamlit handles it
        st.session_state.selected_log_index = selected_option_index

        # Show preview of selected path
        preview_path = os.path.join(base_root_path, all_log_paths[selected_option_index])
        st.caption(f"**Preview:** `{preview_path}`")

    # Use the CURRENT selection (from radio button) for immediate path display
    current_log = all_log_paths[selected_option_index]
    full_current_path = os.path.join(base_root_path, current_log)

    st.sidebar.success(f"`{full_current_path}`")

    return all_log_paths[selected_option_index]


def init_sidebar():
    """
    侧边栏实例化。
    """
    st.sidebar.markdown(
        "<h1 style='text-align: center;'>📖 RL Logging Board</h1>",
        unsafe_allow_html=True
    )

    base_root_path = st.sidebar.text_input(
        "Log(s) Root Path",
        value=DEFAULT_LOG_ROOT,
        help=f"Default from command line: {DEFAULT_LOG_ROOT}"
    )

    if not os.path.exists(base_root_path):
        st.warning(f'Log(s) Root Path: `{base_root_path}` is not exists.', icon='⚠️')
        st.stop()

        # Use cached function for directory scanning - much faster on reruns
    all_log_path_in_logdir = get_log_directories(base_root_path)

    if not all_log_path_in_logdir:
        st.warning('No log files found.')
        st.code("""Logging Dir should be like:
Base Log Dir
    |__eval_topk_0_topp_1 (dir for evaluate logs)
    |   |__eval.jsonl
    |__topk_0_topp_1 (dir for training logs, only for rl logs)
        |__rollout_data_rank_0_1313.jsonl
    ...
""")
        st.stop()

    # Use the improved log selector
    log_name = create_log_selector(all_log_path_in_logdir, base_root_path)

    load_btn = st.sidebar.button(
        "Load & View",
        use_container_width=True
    )

    max_samples_each_step = st.sidebar.number_input(
        'Max Samples Each Step',
        help='当step batch size 过大时可能会造成平台卡顿，可设置阈值来下采样每个step的数据。',
        value=256,
        max_value=10240,
        min_value=1
    )

    step_freq = st.sidebar.number_input(
        'Step Frequency',
        help='每隔多少步采样一次，例如设置为2则只处理步骤0,2,4,6...，可以减少处理时间。',
        value=1,
        min_value=1
    )

    start_step = st.sidebar.number_input(
        'Start Step',
        help='开始处理的步骤编号，只处理大于等于此步骤的数据。',
        value=0,
        min_value=0
    )

    end_step = st.sidebar.number_input(
        'End Step',
        help='结束处理的步骤编号，只处理小于等于此步骤的数据。设置为-1表示不限制结束步骤。',
        value=-1,
        min_value=-1
    )

    if load_btn and (
        'logging_data' not in st.session_state
        or
        log_name != st.session_state['logging_name']
        or
        max_samples_each_step != st.session_state.get('max_samples_each_step', -1)
        or
        step_freq != st.session_state.get('step_freq', -1)
        or
        start_step != st.session_state.get('start_step', -1)
        or
        end_step != st.session_state.get('end_step', -1)
    ):
        load_log_file(
            os.path.join(base_root_path, log_name),
            max_samples_each_step,
            step_freq,
            start_step,
            end_step
        )

    with st.sidebar.expander('🧩 module setting', expanded=True):
        st.session_state['show_reward_logging'] = st.checkbox('Reward 曲线图', value=True)
        st.session_state['var_scaling'] = st.slider('Variance Scaling', min_value=0.1, max_value=1.0, value=0.2, help='Reward 曲线图阴影面积调整（对方差做 scaling）。')
        st.session_state['zero_shift'] = st.checkbox('Zero Shift', value=False, help='是否将所有reward曲线的第一项都平移到0（仅用于对比变化趋势）。')
        st.session_state['show_response'] = st.checkbox('Response 对比', value=True)

    with st.sidebar.expander('⚙️ show details setting', expanded=True):
        st.session_state['use_logp_as_kl'] = st.checkbox('Use LogP as KL', value=True, help='在 Reward 曲线图中用 LogProb 替代 KL 展示。')
        st.session_state['drop_pad'] = st.checkbox('Drop Padding Token', value=True)
        st.session_state['pad_token'] = st.text_input('Pad Token', value='<PAD>', disabled=not st.session_state['drop_pad'])
        st.session_state['drop_sys_prompt'] = st.checkbox('Drop System Prompt', value=True)
        st.session_state['end_token_of_sys_prompt'] = st.text_input('End Token of System Prompt', value='<endofsystem>', disabled=not st.session_state['drop_sys_prompt'])
        st.session_state['show_charts'] = st.checkbox('Show Charts', value=True)
        st.session_state['show_batch_samples'] = st.checkbox('Show Batch Samples', value=True)
        st.session_state['show_samples_pair'] = st.checkbox('Show Samples Pair', value=True)
        st.session_state['show_token_heat_map'] = st.checkbox('Show Heat Map', value=False)

def plot_filled_line(
    x: list,
    y_list_list: list,
    data_names: list,
    colors: list,
    title=None,
    var_scaling=1.
):
    """
    绘制带有阴影的折线图，阴影上下界为当前x对应的y列表中的最大、最小值。

    Args:
        x (list): step 横轴索引
        y_list_list (line_num, steps, step_wise): 可绘制多条直线，维度为：绘制折线条数，总共的step数，每个step对应几个y值
        data_names (list): 每条折线的名字列表
        colors (list): 每条折线的颜色列表（rgb）, e.g. -> ['255,171,171']
    """
    fig = go.Figure()

    x_rev = x[::-1]
    for i in range(len(y_list_list)):
        y_list = y_list_list[i]
        zero_shift_value = 0
        y_mean, y_lower, y_upper = [], [], []

        for idx, y in enumerate(y_list):
            y_arr = np.array(y)
            if idx == 0 and st.session_state['zero_shift']:
                zero_shift_value = np.nanmean(y_arr)

            y_arr = y_arr - zero_shift_value
            mean, std = float(np.nanmean(y_arr)), float(np.nanstd(y_arr))
            std *= var_scaling
            y_mean.append(mean)
            y_lower.append(mean - std)
            y_upper.append(mean + std)
        y_lower = y_lower[::-1]

        fig.add_trace(go.Scatter(
            x=x + x_rev,
            y=y_upper + y_lower,
            fill='toself',
            fillcolor=f'rgba({colors[i]},0.1)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=data_names[i],
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            line_color=f'rgb({colors[i]})',
            name=data_names[i],
        ))

    fig.update_traces(mode='lines')

    if title:
        fig.update_layout(
            title=title,
            legend=dict(orientation="h")
        )

    return fig


def main_page():
    """
    Metrics Page.
    """
    if "logging_data" not in st.session_state:
        st.info("Please Press 「Load & View」Button to load log.")
    else:
        if st.session_state['show_reward_logging']:
            # Create half-half layout for tabs
            left_col, right_col = st.columns([1, 1])

            with left_col:
                step_reward_tab, step_kl_tab, resp_len_tab = st.tabs([
                    'Step-Reward',
                    'Step-KL',
                    'Step-RespLen'
                ])

            with right_col:
                custom_tab = st.container()
                st.markdown("### Custom Metrics")
                custom_tab = st.container()

            with step_reward_tab:
                steps, reward, ref_reward, valid_reward, ref_valid_reward = [], [], [], [], []
                for step, value_dict in st.session_state['logging_data'].items():
                    steps.append(step)
                    reward.append(value_dict['reward'])

                    if value_dict['ref_reward']:
                        ref_reward.append(value_dict['ref_reward'])

                    if value_dict['valid_reward']:
                        valid_reward.append(value_dict['valid_reward'])

                    if value_dict['ref_valid_reward']:
                        ref_valid_reward.append(value_dict['ref_valid_reward'])

                all_curves = {
                    'ref_reward': {
                        'value': ref_reward,
                        'color': '132,201,255'
                    },
                    'reward': {
                        'value': reward,
                        'color': '255,171,171'
                    },
                    'ref_valid_reward': {
                        'value': ref_valid_reward,
                        'color': '132,155,200'
                    },
                    'valid_reward': {
                        'value': valid_reward,
                        'color': '200,155,200'
                    }
                }

                candidate_curves = [key for key in all_curves if all_curves[key]['value']]

                show_curves = st.multiselect(
                    'Show Rewards',
                    candidate_curves,
                    candidate_curves,
                    label_visibility='collapsed'
                )

                reward_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[all_curves[r]['value'] for r in show_curves],
                    data_names=show_curves,
                    colors=[all_curves[r]['color'] for r in show_curves],
                    title='👾 Rewards Logging (Step level)',
                    var_scaling=st.session_state['var_scaling']
                )

                st.plotly_chart(reward_fig, theme="streamlit", use_container_width=True)

            with step_kl_tab:
                steps, kl = [], []

                if st.session_state['use_logp_as_kl']:
                    for step, value_dict in st.session_state['logging_data'].items():
                        if all(value_dict['avg_log_ratio']):
                            steps.append(step)
                            kl.append(value_dict['avg_log_ratio'])
                else:
                    for step, value_dict in st.session_state['logging_data'].items():
                        if all(value_dict['kl']):
                            steps.append(step)
                            kl.append(value_dict['avg_kl'])

                reward_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[kl],
                    data_names=['KL'],
                    colors=['255,165,0'],
                    title='👾 KL Logging (Step level)'
                )
                st.plotly_chart(reward_fig, theme="streamlit", use_container_width=True)

            with resp_len_tab:
                steps, resp_len = [], []

                for step, value_dict in st.session_state['logging_data'].items():
                    if value_dict['response_tokens_len']:
                        steps.append(step)
                        resp_len.append(value_dict['response_tokens_len'])

                resp_len_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[resp_len],
                    data_names=['resp_len'],
                    colors=['255,165,0'],
                    title='👾 Response Length Logging (Step level)'
                )
                st.plotly_chart(resp_len_fig, theme="streamlit", use_container_width=True)

            with custom_tab:
                # Initialize session state for custom functions
                if 'custom_functions' not in st.session_state:
                    st.session_state['custom_functions'] = []

                # Calculate and plot custom metrics
                try:
                    steps = []
                    all_y_data = []
                    all_names = []
                    all_colors = []

                    # Prepare data for each function
                    for func_info in st.session_state['custom_functions']:
                        func_steps = []
                        func_results = []  # Store raw results first

                        # Collect all results for this function
                        for step, value_dict in st.session_state['logging_data'].items():
                            try:
                                # Handle both lambda and def functions during execution
                                if func_info.get('is_def', False):
                                    # Re-execute def function for each step
                                    namespace = {'np': np, 'pd': pd}
                                    exec(func_info['expression'], namespace)
                                    func_key = [k for k in namespace if k.endswith('function')][0]
                                    result = namespace[func_key](value_dict)
                                else:
                                    result = func_info['function'](value_dict)

                                func_steps.append(step)
                                func_results.append(result)

                            except Exception as e:
                                # Skip this step if function fails
                                continue

                        if func_results:  # Only process if we have data
                            if not steps:  # First function sets the steps
                                steps = func_steps

                            # Check if we have dict returns (grouped data)
                            first_result = func_results[0]
                            if isinstance(first_result, dict):
                                # Handle grouped data - create separate lines for each group
                                group_keys = first_result.keys()

                                for group_key in group_keys:
                                    group_y_data = []
                                    for result in func_results:
                                        if isinstance(result, dict) and group_key in result:
                                            group_y_data.append([result[group_key]])
                                        else:
                                            group_y_data.append([0])  # Default value if key missing

                                    all_y_data.append(group_y_data)
                                    all_names.append(f"{func_info['name']} - {group_key}")
                                    # Generate random color for each group
                                    random_color = f"{np.random.randint(50, 255)},{np.random.randint(50, 255)},{np.random.randint(50, 255)}"
                                    all_colors.append(random_color)

                            else:
                                # Handle non-grouped data
                                func_y_data = []
                                for result in func_results:
                                    if isinstance(result, (list, tuple, np.ndarray)):
                                        if len(result) > 0:
                                            func_y_data.append(list(result))
                                        else:
                                            func_y_data.append([0])
                                    else:
                                        # Single value
                                        func_y_data.append([result])

                                all_y_data.append(func_y_data)
                                all_names.append(func_info['name'])
                                # Generate random color for single function
                                random_color = f"{np.random.randint(50, 255)},{np.random.randint(50, 255)},{np.random.randint(50, 255)}"
                                all_colors.append(random_color)

                    # Plot if we have data
                    if all_y_data and steps:
                        custom_fig = plot_filled_line(
                            x=steps,
                            y_list_list=all_y_data,
                            data_names=all_names,
                            colors=all_colors,
                            title='🎨 Custom Metrics (Step level)',
                            var_scaling=st.session_state['var_scaling']
                        )
                        st.plotly_chart(custom_fig, theme="streamlit", use_container_width=True)
                    else:
                        st.info("📊 No custom functions added yet. Add functions from examples below to see plots.")

                except Exception as e:
                    st.error(f"Error plotting custom metrics: {str(e)}")

                # Function examples section with multi-line support
                with st.expander("📖 Function Examples & Quick Add", expanded=True):
                    # Load examples from file
                    examples = load_lambda_examples()

                    # Header with reset button
                    header_col1, header_col2 = st.columns([3, 1])
                    with header_col1:
                        st.markdown("**🚀 Ready-to-use Examples:**")

                    with header_col2:
                        if st.button("🔄 Reset All",
                                   help="Reset all examples to original file state",
                                   use_container_width=True,
                                   type="secondary"):
                            st.session_state.editing_examples = examples.copy()
                            st.success("🔄 All examples reset to original state!")
                            st.rerun()

                    # Initialize session state for editing
                    if 'editing_examples' not in st.session_state:
                        st.session_state.editing_examples = examples.copy()

                                        # Display examples in a more compact way
                    for i, (name, expr) in enumerate(st.session_state.editing_examples):
                        with st.container():
                            # Always editable name and expression
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                edited_name = st.text_input(
                                    "Name:",
                                    value=name,
                                    key=f"edit_name_{i}",
                                    label_visibility="collapsed"
                                )
                            with col2:
                                edited_expr = st.text_area(
                                    "Expression:",
                                    value=expr,
                                    height=100,
                                    key=f"edit_expr_{i}",
                                    help="Supports both lambda expressions and multi-line def functions",
                                    label_visibility="collapsed"
                                )

                            # Button row
                            col1, col2, col3 = st.columns([1, 1, 1])

                            with col1:
                                # Check if this function is already active
                                existing_names = [f['name'] for f in st.session_state.get('custom_functions', [])]
                                is_active = name in existing_names
                                if is_active:
                                    button_type = "primary"
                                    button_text = "🗑️ Remove"
                                else:
                                    button_type = "secondary"
                                    button_text = "➕ Add"

                                if st.button(button_text, key=f"add_{i}", use_container_width=True, type=button_type):
                                    if is_active:
                                        # Remove function if already active
                                        st.session_state['custom_functions'] = [
                                            f for f in st.session_state['custom_functions']
                                            if f['name'] != name
                                        ]
                                        st.success(f"🗑️ Removed '{name}' from chart!")
                                        st.rerun()
                                    else:
                                        # Use the edited expression for adding
                                        try:
                                            # Test with sample data
                                            sample_step_data = {
                                                'reward': [0.5, 0.3, 0.8],
                                                'ref_reward': [0.4, 0.2, 0.7],
                                                'response_tokens_len': [10, 15, 20],
                                                'avg_kl': [0.1, 0.2, 0.15],
                                                'avg_log_ratio': [0.05, 0.1, 0.08],
                                                'response': ['good response', 'good response', 'bad response']
                                            }

                                            # Handle both lambda and def functions
                                            if edited_expr.strip().startswith('def '):
                                                # Multi-line def function
                                                namespace = {'np': np, 'pd': pd}
                                                exec(edited_expr, namespace)
                                                func_key = [k for k in namespace if k.endswith('function')][0]
                                                test_func = namespace[func_key]
                                                is_def = True
                                            else:
                                                # Lambda function
                                                test_func = eval(edited_expr, {'np': np, 'pd': pd})
                                                is_def = False

                                            test_result = test_func(sample_step_data)

                                            # Add to session state with unique name if exists
                                            func_name = edited_name
                                            existing_names = [f['name'] for f in st.session_state['custom_functions']]
                                            counter = 1
                                            while func_name in existing_names:
                                                func_name = f"{edited_name}_{counter}"
                                                counter += 1

                                            new_func = {
                                                'name': func_name,
                                                'expression': edited_expr,
                                                'function': test_func,
                                                'is_def': is_def
                                            }
                                            st.session_state['custom_functions'].append(new_func)
                                            st.success(f"✅ Added '{func_name}' to chart!")
                                            st.rerun()

                                        except Exception as e:
                                            # print the error traceback
                                            import traceback
                                            traceback.print_exc()
                                            st.error(f"❌ Invalid function: {str(e)}")

                            with col2:
                                # Save button to update the example
                                if st.button("💾 Save", key=f"save_{i}", use_container_width=True):
                                    st.session_state.editing_examples[i] = (edited_name, edited_expr)
                                    st.success(f"💾 Saved changes to '{edited_name}'!")
                                    st.rerun()

                            with col3:
                                # Delete button to remove this example
                                if st.button("🗑️ Delete", key=f"delete_{i}", use_container_width=True):
                                    st.session_state.editing_examples.pop(i)
                                    st.success(f"🗑️ Deleted example '{name}'!")
                                    st.rerun()


        if st.session_state['show_response']:
            st.markdown('⚡️ **Each Step Response**')

            # Check if step indices are initialized
            if 'min_step_index' not in st.session_state or 'max_step_index' not in st.session_state:
                st.warning("Please load log data first using the 'Load & View' button in the sidebar.")
                return

            if st.session_state['min_step_index'] == st.session_state['max_step_index']:
                step_index = st.session_state['min_step_index']
            elif (
                len(st.session_state['logging_data']) > 2
                and
                list(st.session_state['logging_data'].keys())[2] - list(st.session_state['logging_data'].keys())[1] != list(st.session_state['logging_data'].keys())[1] - list(st.session_state['logging_data'].keys())[0]
            ):
                step_index = st.selectbox(
                    f"Step Index({st.session_state['max_step_index']} total steps):",
                    list(st.session_state['logging_data'].keys()),
                    index=0
                )
            else:
                step_index = st.slider(
                    f"Step Index({st.session_state['max_step_index']} total steps):",
                    min_value=st.session_state['min_step_index'],
                    max_value=st.session_state['max_step_index'],
                    value=st.session_state['min_step_index'],
                    step=st.session_state['step_gap']
                )

            cur_step_content_dict = st.session_state['logging_data'][step_index]
            cur_step_filtered_content_dict = copy.deepcopy(cur_step_content_dict)

            cur_step_filtered_content_dict['prompt'] = []
            for prompt in cur_step_content_dict['prompt']:
                if st.session_state['drop_pad']:
                    prompt = prompt.replace(st.session_state['pad_token'], '').strip()
                if st.session_state['drop_sys_prompt']:
                    prompt = prompt.split(st.session_state['end_token_of_sys_prompt'])[-1]
                cur_step_filtered_content_dict['prompt'].append(prompt)

            cur_step_filtered_content_dict['response'] = [c.replace(st.session_state['pad_token'], '').strip() if st.session_state['drop_pad'] else c for c in cur_step_content_dict['response']]
            cur_step_filtered_content_dict['reward_gap'] = [r - ref_r for r, ref_r in zip(cur_step_content_dict['reward'], cur_step_content_dict['ref_reward'])]
            cur_step_filtered_content_dict['valid_reward_gap'] = [r - ref_r for r, ref_r in zip(cur_step_content_dict['reward'], cur_step_content_dict['valid_reward'])]

            if st.session_state['show_charts']:

                if not cur_step_filtered_content_dict['ref_reward']:
                    cur_step_filtered_content_dict['ref_reward'] = [0] * len(cur_step_filtered_content_dict['reward'])

                # Chart selection controls
                chart_selection_cols = st.columns([2, 2, 2, 6])
                with chart_selection_cols[0]:
                    show_reward_dist = st.checkbox('Reward Distribution', value=True)
                with chart_selection_cols[1]:
                    show_reward_gap = st.checkbox('Reward Gap', value=True)
                with chart_selection_cols[2]:
                    show_reward_hist = st.checkbox('Reward Histogram', value=True)

                # Create columns only for selected charts
                selected_charts = []
                if show_reward_dist:
                    selected_charts.append('reward_dist')
                if show_reward_gap:
                    selected_charts.append('reward_gap')
                if show_reward_hist:
                    selected_charts.append('reward_hist')

                if selected_charts:
                    chart_cols = st.columns([6] * len(selected_charts))
                    chart_index = 0

                    if show_reward_dist:                                    # reward 分布
                        with chart_cols[chart_index]:
                            reward_distribution_dict = {
                                'sample_index': [],
                                'reward': [],
                                'tag': []
                            }
                            for sample_index, (reward, ref_reward) in enumerate(zip(cur_step_filtered_content_dict['reward'], cur_step_filtered_content_dict['ref_reward'])):
                                reward_distribution_dict['sample_index'].append(sample_index)
                                reward_distribution_dict['reward'].append(reward)
                                reward_distribution_dict['tag'].append('Reward')
                                reward_distribution_dict['sample_index'].append(sample_index)
                                reward_distribution_dict['reward'].append(ref_reward)
                                reward_distribution_dict['tag'].append('Ref Reward')

                            reward_distribution_df = pd.DataFrame.from_dict(reward_distribution_dict)
                            fig = px.bar(
                                reward_distribution_df,
                                x="sample_index",
                                y="reward",
                                color="tag",
                                barmode='group',
                                color_discrete_sequence=px.colors.diverging.Portland,
                                title="Reward in current batch samples"
                            )
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                        chart_index += 1

                    if show_reward_gap:                                     # reward gap 分布
                        with chart_cols[chart_index]:
                            reward_distribution_dict = {
                                'sample_index': [i for i in range(len(cur_step_filtered_content_dict['reward_gap']))],
                                'reward_gap': cur_step_filtered_content_dict['reward_gap']
                            }
                            reward_distribution_df = pd.DataFrame.from_dict(reward_distribution_dict)
                            fig = px.bar(
                                reward_distribution_df,
                                x="sample_index",
                                y="reward_gap",
                                color="reward_gap",
                                color_discrete_sequence=['red'],
                                title="Reward Gap (r - ref_r) in current batch"
                            )
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                        chart_index += 1

                    if show_reward_hist:                                    # reward 方差分布
                        with chart_cols[chart_index]:
                            if cur_step_filtered_content_dict['ref_reward']:
                                hist_data = [
                                    cur_step_filtered_content_dict['ref_reward'],
                                    cur_step_filtered_content_dict['reward'],
                                ]
                                group_labels = ['Ref Rewards', 'Rewards']
                            else:
                                hist_data = [cur_step_filtered_content_dict['reward']]
                                group_labels = ['Rewards']

                            fig = ff.create_distplot(
                                hist_data,
                                group_labels,
                                bin_size=[.02, .02],
                                curve_type='normal'
                            )
                            fig.update_layout(title="Reward Distribution in current batch")
                            st.plotly_chart(fig, use_container_width=True)

            showed_keys = [
                'prompt',
                'response',
                'reward',
                'ground_truth',
                'valid_reward',
                'avg_log_ratio',
                'sum_log_ratio',
                'avg_kl',
                'sum_kl',
                'ref_response',
                'ref_reward',
                'ref_valid_reward',
                'reward_gap',
                'valid_reward_gap',
                'image_path',
                'data_source',
                'ability',
            ]
            candidate_keys = [k for k in showed_keys if k in cur_step_filtered_content_dict and cur_step_filtered_content_dict[k]]
            content_dict = dict([(k, cur_step_filtered_content_dict[k]) for k in candidate_keys])
            content_df = pd.DataFrame.from_dict(content_dict)

            if st.session_state['show_batch_samples']:
                # Compact filter functionality
                st.markdown("**🔍 Filter Samples**")

                # Load common filters
                common_filters = load_common_filters()

                # Create filter interface with common filters
                filter_col1, filter_col2, filter_col3 = st.columns([3, 2, 1])

                with filter_col1:
                    filter_expression = st.text_input(
                        "Filter expression:",
                        value=st.session_state.get('last_filter_expression', ''),
                        help="Use pandas query syntax. Examples: reward > 0.5, response.str.len() > 100",
                        placeholder="e.g., reward > 0.5",
                        label_visibility="collapsed"
                    )
                    # Remember the last filter expression
                    if filter_expression != st.session_state.get('last_filter_expression', ''):
                        st.session_state.last_filter_expression = filter_expression

                with filter_col2:
                    st.write("")  # Spacer

                with filter_col3:
                    filter_enabled = st.checkbox(
                        "Enable Filter",
                        value=True,
                        key="filter_enabled_checkbox"
                    )

                # Common filter cards
                if common_filters:
                    cols = st.columns(len(common_filters))
                    for i, filter_expr in enumerate(common_filters):
                        with cols[i]:
                            if st.button(
                                filter_expr,
                                key=f"filter_btn_{i}",
                                use_container_width=True
                            ):
                                st.session_state.last_filter_expression = filter_expr
                                st.rerun()

                # Apply filter if enabled and expression is provided
                filtered_df = content_df
                filter_error = None
                filter_status_msg = ""

                if filter_enabled and filter_expression.strip():
                    try:
                        filtered_df = content_df.query(filter_expression)
                        if len(filtered_df) == 0:
                            filter_status_msg = f"⚠️ Filter returned 0 rows out of {len(content_df)} total"
                        else:
                            filter_status_msg = f"✅ Showing {len(filtered_df)} out of {len(content_df)} rows (filtered)"
                    except Exception as e:
                        filter_error = str(e)
                        filter_status_msg = f"❌ Filter error: {filter_error}"
                        filtered_df = content_df
                else:
                    filter_status_msg = f"📊 Showing all {len(content_df)} rows (no filter)"

                # Status and examples in one row
                status_col, examples_col = st.columns([2, 1])

                with status_col:
                    if filter_enabled and filter_expression.strip():
                        if filter_error:
                            st.error(filter_status_msg)
                            st.info("Showing all rows due to filter error.")
                        elif len(filtered_df) == 0:
                            st.warning(filter_status_msg)
                        else:
                            st.success(filter_status_msg)
                    else:
                        st.info(filter_status_msg)

                with examples_col:
                    with st.expander("📖 Filter Examples", expanded=False):
                        st.code("""
reward > 0.5
reward_gap > 0.1
response.str.len() > 100
response.str.contains('error', case=False, na=False)
(reward > 0.3) & (reward_gap > 0.05)
ground_truth.notna()
                        """, language="python")

                # Column selection functionality
                st.markdown("**🔧 Select Columns to Display**")
                available_columns = list(filtered_df.columns)

                # Default columns (exclude ref_reward from default)
                default_columns = [col for col in available_columns if col not in ['ref_reward', 'image_path']]

                selected_columns = st.multiselect(
                    "Choose columns to display:",
                    options=available_columns,
                    default=default_columns,
                    help="Select which columns to show in the dataframe below",
                    label_visibility="collapsed"
                )

                # Display dataframe with selected columns
                if selected_columns:
                    display_df = filtered_df[selected_columns]
                else:
                    display_df = filtered_df
                    st.warning("No columns selected. Showing all columns.")

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=350
                )

            if st.session_state['show_samples_pair']:

                c1, c2, c3 = st.columns([1, 1, 4])
                with c1:
                    if step_index == st.session_state['min_step_index']:
                        delta_char = 0
                    else:
                        try:
                            cur_avg_len = st.session_state['logging_data'][step_index]['avg_length']
                            last_avg_len = st.session_state['logging_data'][step_index-st.session_state['step_gap']]['avg_length']
                            delta_char = cur_avg_len - last_avg_len
                        except:
                            delta_char = 0
                    st.metric(                                                  # actor在当前step下的平均回复长度，delta为与上一个step的比较
                        'Response Average Length',
                        value=f"{st.session_state['logging_data'][step_index]['avg_length']} 字",
                        delta=f'{delta_char} 字'
                    )

                with c2:                                                        # ref_model在当前step下的平均回复长度，delta为与上一个step的比较
                    try:
                        delta_char = 0 if step_index == st.session_state['min_step_index'] else st.session_state['logging_data'][step_index]['avg_ref_length'] - st.session_state['logging_data'][step_index-st.session_state['step_gap']]['avg_ref_length']
                    except:
                        delta_char = 0
                    st.metric(
                        'Ref Response Average Length',
                        value=f"{st.session_state['logging_data'][step_index]['avg_ref_length']} 字",
                        delta=f'{delta_char} 字'
                    )

                with c3:
                    # Limit the width and use slider for better UX
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Get filtered indices if filter is enabled and has results
                        if filter_enabled and filter_expression.strip() and not filter_error and len(filtered_df) > 0:
                            # Use slider with snapping to nearest filtered index
                            filtered_indices = sorted(filtered_df.index.tolist())
                            min_idx, max_idx = min(filtered_indices), max(filtered_indices)

                            # Initialize session state for snapped slider if not exists
                            if 'snapped_sample_index' not in st.session_state:
                                st.session_state.snapped_sample_index = filtered_indices[0]

                            # Create slider with full range of filtered indices
                            raw_slider_value = st.slider(
                                f'Sample index (snaps to filtered samples: {len(filtered_indices)} available):',
                                min_value=min_idx,
                                max_value=max_idx,
                                value=st.session_state.snapped_sample_index,
                                key="filtered_sample_slider"
                            )

                            # Find nearest filtered index
                            def find_nearest_filtered_index(value, filtered_list):
                                return min(filtered_list, key=lambda x: abs(x - value))

                            snapped_index = find_nearest_filtered_index(raw_slider_value, filtered_indices)

                            # Update session state if snapped to different value
                            if snapped_index != st.session_state.snapped_sample_index:
                                st.session_state.snapped_sample_index = snapped_index
                                st.rerun()

                            sample_index = snapped_index

                            # Show which sample is selected if snapped
                            if raw_slider_value != snapped_index:
                                st.caption(f"Snapped to nearest filtered sample: {snapped_index}")
                        else:
                            # Use original slider behavior for unfiltered data
                            sample_index = st.slider(
                                f'Sample index (0-{len(cur_step_filtered_content_dict["response"]) - 1}):',
                                min_value=0,
                                max_value=len(cur_step_filtered_content_dict['response']) - 1,
                                value=0,
                                key="sample_index_slider"
                            )
                    with col2:
                        st.write("")  # Spacer for alignment

                # 单样本展示 response - ref_response 的回复
                # Column selection bar
                available_columns = {
                    'prompt': {'label': 'Prompt', 'color': '#B0C4DE', 'width': 4},
                    'response': {'label': 'Response', 'color': '#3DD56D', 'width': 3},
                    'ref_response': {'label': 'Ref Response', 'color': '#60B4FF', 'width': 3},
                    'reward_gap': {'label': 'Reward Gap', 'color': '#FFA500', 'width': 2},
                    'image_path': {'label': 'Image', 'color': '#FF69B4', 'width': 4},
                    'ground_truth': {'label': 'Ground Truth', 'color': '#000000', 'width': 1},
                    'reward': {'label': 'Reward', 'color': '#000000', 'width': 1},
                    # 'data_source': {'label': 'Data Source', 'color': '#000000', 'width': 1},
                    # 'ability': {'label': 'Ability', 'color': '#000000', 'width': 1}
                }

                # Filter available columns based on what data is present
                present_columns = []
                for col_name in available_columns.keys():
                    if col_name == 'reward_gap':
                        # reward_gap is calculated, so it's always available if we have reward data
                        if cur_step_filtered_content_dict.get('reward'):
                            present_columns.append(col_name)
                    elif col_name in cur_step_filtered_content_dict and cur_step_filtered_content_dict[col_name]:
                        present_columns.append(col_name)

                if not present_columns:
                    present_columns = ['prompt', 'response']  # fallback to basics
                print("Present columns: ", present_columns)

                selected_columns = st.multiselect(
                    'Select columns to display:',
                    options=present_columns,
                    default=[p for p in present_columns if p not in ['reward_gap']],
                    format_func=lambda x: available_columns[x]['label']
                )

                if selected_columns:
                    # Create dynamic columns based on selection
                    column_widths = [available_columns[col]['width'] for col in selected_columns]
                    columns = st.columns(column_widths)

                    for i, col_name in enumerate(selected_columns):
                        with columns[i]:
                            col_info = available_columns[col_name]

                            if col_name == 'prompt':
                                st.markdown(f'<font color="{col_info["color"]}">Prompt</font>', unsafe_allow_html=True)
                                content = cur_step_filtered_content_dict["prompt"][sample_index].replace('\n', '  \n').replace('~', '～')
                                content = process_content_for_display(content)
                                st.markdown(
                                    f'<font color="{col_info["color"]}">{content}</font>',
                                    unsafe_allow_html=True
                                )

                            elif col_name == 'response':
                                st.markdown(':green[Response]')
                                content = cur_step_filtered_content_dict["response"][sample_index].replace('\n', '  \n').replace('~', '～')
                                content = process_content_for_display(content)
                                st.markdown(
                                    f'<font color="{col_info["color"]}">{content}</font>',
                                    unsafe_allow_html=True
                                )

                            elif col_name == 'ref_response':
                                st.markdown(':blue[Ref Response]')
                                if (
                                        "ref_response" in cur_step_filtered_content_dict
                                        and
                                        cur_step_filtered_content_dict["ref_response"]
                                ):
                                    content = cur_step_filtered_content_dict["ref_response"][sample_index].replace('\n', '  \n').replace('~', '～')
                                    content = process_content_for_display(content)
                                    st.markdown(
                                        f'<font color="{col_info["color"]}">{content}</font>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.info('No `ref_response` found in log line data.')

                            elif col_name == 'image_path':
                                st.markdown('<font color="#FF69B4">Image</font>', unsafe_allow_html=True)
                                if (
                                        "image_path" in cur_step_filtered_content_dict
                                        and
                                        cur_step_filtered_content_dict["image_path"]
                                        and
                                        sample_index < len(cur_step_filtered_content_dict["image_path"])
                                ):
                                    image_path = cur_step_filtered_content_dict["image_path"][sample_index]
                                    display_image_with_actions(image_path, cur_step_filtered_content_dict["response"][sample_index])
                                else:
                                    st.info('No `image_path` found in log line data.')

                            else:
                                # Generic handling for other columns (ground_truth, data_source, ability, etc.)
                                st.markdown(f'<font color="{col_info["color"]}">{col_info["label"]}</font>', unsafe_allow_html=True)
                                if (
                                        col_name in cur_step_filtered_content_dict
                                        and
                                        cur_step_filtered_content_dict[col_name]
                                        and
                                        sample_index < len(cur_step_filtered_content_dict[col_name])
                                ):
                                    content = str(cur_step_filtered_content_dict[col_name][sample_index])
                                    # Apply text processing for long text fields like ground_truth
                                    if col_name in ['ground_truth']:
                                        content = content.replace('\n', '  \n').replace('~', '～')
                                        content = process_content_for_display(content)

                                    st.markdown(
                                        f'<font color="{col_info["color"]}">{content}</font>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.info(f'No `{col_name}` found in log line data.')

                # 展示更详细的 token-level 的信息
                if 'token_rewards' in cur_step_filtered_content_dict and cur_step_filtered_content_dict['token_rewards']:
                    # 检查 resp_tokens 的长度和 logprobs 的长度是否对齐
                    resp_token_len = len(cur_step_filtered_content_dict['response_tokens'][sample_index])
                    logp_len = len(cur_step_filtered_content_dict['logprobs'][sample_index])
                    if resp_token_len != logp_len:
                        st.info(
                            f'Note: `resp_tokens` (len: {resp_token_len}) is not equal to `logprobs` (len: {logp_len}), this may caused by &lt;PAD&gt; tokens, CLIP response tokens!',
                            icon='⚠️'
                        )
                        cur_step_filtered_content_dict['response_tokens'][sample_index] = cur_step_filtered_content_dict['response_tokens'][sample_index][:logp_len]

                    show_values = st.multiselect(
                        'Select show value(s)',
                        ['token_reward', 'log_ratio', 'kl', 'token_value', 'logp', 'ref_logp', 'prob', 'ref_prob'],
                        ['token_reward', 'log_ratio', 'kl', 'token_value', 'logp', 'ref_logp', 'prob', 'ref_prob']
                    )

                    new_dict, index_list = {}, []

                    if st.session_state['drop_pad'] and cur_step_filtered_content_dict['response_tokens'][sample_index][-1] == st.session_state['pad_token']:
                        first_pad_token_idx = cur_step_filtered_content_dict['response_tokens'][sample_index].index(st.session_state['pad_token'])
                        response_tokens_without_pad_token = cur_step_filtered_content_dict['response_tokens'][sample_index][:first_pad_token_idx]
                    else:
                        response_tokens_without_pad_token = cur_step_filtered_content_dict['response_tokens'][sample_index]

                    for token_idx in range(len(response_tokens_without_pad_token)):
                        if cur_step_filtered_content_dict['response_tokens']:
                            resp_token = cur_step_filtered_content_dict['response_tokens'][sample_index][token_idx]
                            resp_token = f'{token_idx} - {resp_token}'
                            if resp_token not in new_dict:
                                new_dict[resp_token] = []

                        if cur_step_filtered_content_dict['token_rewards']:
                            token_reward = cur_step_filtered_content_dict['token_rewards'][sample_index][token_idx]
                            if 'token_reward' in show_values:
                                new_dict[resp_token].append(token_reward)
                                if 'token_reward' not in index_list:
                                    index_list.append('token_reward')

                        if cur_step_filtered_content_dict['log_ratio']:
                            log_ratio = cur_step_filtered_content_dict['log_ratio'][sample_index][token_idx]
                            if 'log_ratio' in show_values:
                                new_dict[resp_token].append(log_ratio)
                                if 'log_ratio' not in index_list:
                                    index_list.append('log_ratio')

                        if cur_step_filtered_content_dict['kl']:
                            kl = cur_step_filtered_content_dict['kl'][sample_index][token_idx]
                            if 'kl' in show_values:
                                new_dict[resp_token].append(kl)
                                if 'kl' not in index_list:
                                    index_list.append('kl')

                        if cur_step_filtered_content_dict['values']:
                            value = cur_step_filtered_content_dict['values'][sample_index][token_idx]
                            if 'token_value' in show_values:
                                new_dict[resp_token].append(value)
                                if 'token_value' not in index_list:
                                    index_list.append('token_value')

                        if cur_step_filtered_content_dict['logprobs']:
                            logp = cur_step_filtered_content_dict['logprobs'][sample_index][token_idx]
                            if 'logp' in show_values:
                                new_dict[resp_token].append(logp)
                                if 'logp' not in index_list:
                                    index_list.append('logp')

                        if cur_step_filtered_content_dict['ref_logprobs']:
                            ref_logp = cur_step_filtered_content_dict['ref_logprobs'][sample_index][token_idx]
                            if 'ref_logp' in show_values:
                                new_dict[resp_token].append(ref_logp)
                                if 'ref_logp' not in index_list:
                                    index_list.append('ref_logp')

                        if cur_step_filtered_content_dict['probs']:
                            prob = cur_step_filtered_content_dict['probs'][sample_index][token_idx]
                            if 'prob' in show_values:
                                new_dict[resp_token].append(prob)
                                if 'prob' not in index_list:
                                    index_list.append('prob')

                        if cur_step_filtered_content_dict['ref_probs']:
                            ref_prob = cur_step_filtered_content_dict['ref_probs'][sample_index][token_idx]
                            if 'ref_prob' in show_values:
                                new_dict[resp_token].append(ref_prob)
                                if 'ref_prob' not in index_list:
                                    index_list.append('ref_prob')

                    try:
                        token_level_df = pd.DataFrame.from_dict(new_dict)
                        renamed_index_dict = dict((i, name) for i, name in enumerate(index_list))
                        token_level_df.rename(
                            index=renamed_index_dict,
                            inplace=True
                        )

                        st.dataframe(
                            token_level_df.style.background_gradient(axis=1, cmap="binary"),
                            use_container_width=True
                        )

                        if st.session_state['show_token_heat_map']:
                            fig = px.imshow(
                                token_level_df,
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="balance",
                            )
                            fig.update_xaxes(side="top")
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                    except Exception as e:
                        st.error(f'Error occured: {e}.')
                        st.write(new_dict)


if __name__ == '__main__':
    init_sidebar()
    main_page()
