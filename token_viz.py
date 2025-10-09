"""
Token-level visualization - text with bars and colors.
"""
import streamlit as st
import streamlit.components.v1 as components
import numpy as np


def get_color_for_value(value, metric_name):
    """Get background color based on metric value."""
    if value is None:
        return "#CCCCCC"

    # For logprobs: higher (closer to 0) is better
    if 'logp' in metric_name.lower():
        if value > -0.1:
            return "#90EE90"
        elif value > -0.5:
            return "#FFFFE0"
        elif value > -1.0:
            return "#FFD700"
        else:
            return "#FFB6C6"

    # For probabilities: closer to 1 is better
    elif 'prob' in metric_name.lower():
        if value is None:
            return "#CCCCCC"
        if value >= 0.8:
            return "#90EE90"
        elif value >= 0.5:
            return "#FFFFE0"
        elif value >= 0.2:
            return "#FFD700"
        else:
            return "#FFB6C6"

    # For rewards: positive is good
    elif 'reward' in metric_name.lower():
        if value > 0.3:
            return "#90EE90"
        elif value > 0:
            return "#FFFFE0"
        elif value == 0:
            return "#F0F0F0"
        else:
            return "#FFB6C6"

    # For KL: lower is better
    elif 'kl' in metric_name.lower():
        if value < 0.05:
            return "#90EE90"
        elif value < 0.1:
            return "#FFFFE0"
        else:
            return "#FFB6C6"

    else:
        if value > 0.5:
            return "#90EE90"
        elif value > 0:
            return "#FFFFE0"
        else:
            return "#FFB6C6"


def display_colored_tokens(tokens, logprobs):
    """
    Display tokens as text with bars above and colored backgrounds for logprobs only.

    Args:
        tokens: List of token strings
        logprobs: List of logprob values (same length as tokens)
    """
    if not tokens or not logprobs or len(tokens) != len(logprobs):
        st.warning("Invalid token or logprob data")
        return

    st.markdown("### 🎨 Token-Level LogProb Visualization")

    metric_name = "LogProb"
    values = logprobs

    # Normalize values for bar heights with tighter range to reduce variance
    values_array = np.array([v if v is not None else 0 for v in values], dtype=float)
    if len(values_array) > 0:
        val_min = values_array.min()
        val_max = values_array.max()
        if val_max != val_min:
            normalized = (values_array - val_min) / (val_max - val_min)
        else:
            normalized = np.full(len(values_array), 0.5)
    else:
        normalized = np.zeros(len(values))

    normalized = np.clip(normalized, 0.0, 1.0)

    # Convert normalized values to pixel heights with smaller range
    min_height, max_height = 25, 45
    bar_heights = normalized * (max_height - min_height) + min_height

    # Build complete HTML document
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {
                margin: 0;
                padding: 10px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            .token-container {
                line-height: 1.6;
                font-size: 14px;
                word-wrap: break-word;
            }
            .token-wrapper {
                display: inline;
            }
            .token-text {
                padding: 1px 2px;
                border-radius: 3px;
                white-space: pre-wrap;
            }
            .legend-container {
                margin-bottom: 15px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .legend-title {
                font-size: 12px;
                font-weight: bold;
                margin-bottom: 8px;
                color: #333;
            }
            .color-bar {
                display: flex;
                height: 25px;
                border-radius: 3px;
                overflow: hidden;
                margin-bottom: 5px;
            }
            .color-segment {
                flex: 1;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 10px;
                color: #333;
                font-weight: 500;
            }
            .legend-labels {
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <!-- Color Legend -->
        <div class="legend-container">
            <div class="legend-title">📊 LogProb Color Scale</div>
            <div class="color-bar">
                <div class="color-segment" style="background-color: #FFB6C6;">&lt; -1.0</div>
                <div class="color-segment" style="background-color: #FFD700;">-1.0 to -0.5</div>
                <div class="color-segment" style="background-color: #FFFFE0;">-0.5 to -0.1</div>
                <div class="color-segment" style="background-color: #90EE90;">&gt; -0.1</div>
            </div>
            <div class="legend-labels">
                <span>🔴 Low Confidence</span>
                <span>🟢 High Confidence</span>
            </div>
        </div>

        <!-- Tokens -->
        <div class="token-container">
    '''

    for i, (token, value) in enumerate(zip(tokens, values)):
        color = get_color_for_value(value, metric_name)
        value_str = f"{value:.3f}" if value is not None else "N/A"
        bar_height = bar_heights[i]
        normalized_value = normalized[i]

        # Escape HTML characters in token text
        token_text = str(token).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

        # Build tooltip and escape for HTML attribute
        tooltip = f"{metric_name}: {value_str} | Normalized: {normalized_value:.2f}"
        tooltip_escaped = tooltip.replace('"', '&quot;')

        html += f'''
        <span class="token-wrapper" title="{tooltip_escaped}">
            <span class="token-text" style="background-color: {color};">{token_text}</span>
        </span>
        '''

    html += '''
        </div>
    </body>
    </html>
    '''

    # Calculate height to show all content without scrolling
    estimated_rows = max(3, len(tokens) // 15)
    row_height = 30  # text height + padding
    legend_height = 100  # Height for the legend
    display_height = estimated_rows * row_height + legend_height

    components.html(html, height=display_height, scrolling=False)


# Test section
if __name__ == "__main__":
    st.set_page_config(page_title="Token Viz Test", layout='wide')
    st.title("🧪 Token Visualization Test")

    # 300-word realistic paragraph tokenized
    sample_tokens = [
        'In', 'the', 'rapidly', 'evolving', 'landscape', 'of', 'artificial', 'intelligence', ',',
        'large', 'language', 'models', 'have', 'emerged', 'as', 'one', 'of', 'the', 'most',
        'transformative', 'technologies', 'of', 'our', 'time', '.', 'These', 'sophisticated',
        'systems', ',', 'trained', 'on', 'vast', 'amounts', 'of', 'text', 'data', ',', 'demonstrate',
        'remarkable', 'capabilities', 'in', 'understanding', 'and', 'generating', 'human', '-',
        'like', 'text', 'across', 'a', 'wide', 'range', 'of', 'tasks', 'and', 'domains', '.',
        'From', 'answering', 'complex', 'questions', 'to', 'writing', 'creative', 'content', ',',
        'translating', 'between', 'languages', ',', 'and', 'even', 'assisting', 'with', 'code',
        'development', ',', 'language', 'models', 'have', 'proven', 'their', 'versatility', '.',
        'However', ',', 'the', 'development', 'and', 'deployment', 'of', 'these', 'models', 'also',
        'raise', 'important', 'considerations', 'about', 'safety', ',', 'reliability', ',', 'and',
        'ethical', 'use', '.', 'Researchers', 'and', 'developers', 'must', 'carefully', 'evaluate',
        'model', 'outputs', ',', 'implement', 'robust', 'safety', 'measures', ',', 'and', 'ensure',
        'that', 'these', 'powerful', 'tools', 'are', 'used', 'responsibly', '.', 'The', 'field',
        'continues', 'to', 'advance', 'rapidly', ',', 'with', 'new', 'techniques', 'for', 'training',
        ',', 'fine', '-', 'tuning', ',', 'and', 'aligning', 'models', 'with', 'human', 'values',
        'and', 'preferences', '.', 'Reinforcement', 'learning', 'from', 'human', 'feedback', ',',
        'for', 'example', ',', 'has', 'become', 'a', 'crucial', 'method', 'for', 'improving',
        'model', 'behavior', 'and', 'making', 'AI', 'systems', 'more', 'helpful', ',', 'harmless',
        ',', 'and', 'honest', '.', 'As', 'we', 'look', 'to', 'the', 'future', ',', 'the',
        'integration', 'of', 'these', 'technologies', 'into', 'everyday', 'applications', 'will',
        'likely', 'continue', 'to', 'grow', ',', 'bringing', 'both', 'exciting', 'opportunities',
        'and', 'new', 'challenges', 'that', 'require', 'careful', 'thought', 'and', 'collaborative',
        'effort', 'across', 'the', 'AI', 'community', '.'
    ]

    num_tokens = len(sample_tokens)

    # Generate realistic logprobs (most confident, some uncertain)
    np.random.seed(42)
    sample_logprobs = -np.abs(np.random.normal(0.05, 0.15, num_tokens))
    # Make some tokens less confident (punctuation, rare words)
    for i, tok in enumerate(sample_tokens):
        if tok in [',', '.', '-']:
            sample_logprobs[i] = -np.random.uniform(0.5, 1.5)
        elif tok in ['Reinforcement', 'sophisticated', 'versatility']:
            sample_logprobs[i] = -np.random.uniform(0.3, 0.8)

    display_colored_tokens(sample_tokens, sample_logprobs.tolist())

    st.markdown("---")
    st.info(f"""
    **Test Stats:**
    - Total tokens: {num_tokens}
    - Approx words: ~{num_tokens // 1.3:.0f}

    **Color Legend:**
    - 🟢 Green: Good values
    - 🟡 Yellow: Medium values
    - 🔴 Red: Poor values
    - Bars show relative metric values
    """)
