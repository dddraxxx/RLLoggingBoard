from typing import Union
import datasets
from pathlib import Path
from PIL import Image
import orjson
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor

charxiv_question_to_image = {}

def init_charxiv_question_to_image(charxiv_data: datasets.Dataset):
    print("Initializing charxiv_question_to_image...")
    for data in charxiv_data:
        charxiv_question_to_image[data['reasoning_q'].lower().strip()] = data['figure_path']

def find_image_in_charxiv(prompt: str, data_dir: str ='/scratch/doqihu/work/eval_data/hgf/charxiv', charxiv_data: Union[datasets.Dataset, str] = '/scratch/doqihu/work/eval_data/hgf/charxiv'):
    # find question, it is between <|vision_end|> and "Think first,"
    question = prompt.split('<|vision_end|>')[1].split('Think first,')[0].strip()
    # load data
    if isinstance(charxiv_data, str):
        charxiv_data = datasets.load_dataset(charxiv_data, split='validation')
    # data_dir
    data_dir = Path(data_dir)
    # find image
    if len(charxiv_question_to_image) == 0:
        init_charxiv_question_to_image(charxiv_data)
    figure_path = charxiv_question_to_image.get(question.lower())
    if figure_path is None:
        print(question)
    # read image
    image_path = data_dir / figure_path
    return image_path

def process_single_item(args):
    i, item = args
    image_path = find_image_in_charxiv(item['prompt'])
    return i, image_path

if __name__ == '__main__':
    jsonl_path = '/scratch/doqihu/work/verl_logs/agent_charxiv/charxiv_for_single_node/20250612_221427/logs/rl_logging_board/agent_charxiv/charxiv_for_single_node/rollout_data_rank0.jsonl'
    # read through orjson
    data = []
    with open(jsonl_path, 'rb') as f:
        for line in tqdm(f, desc="Loading data"):
            data.append(orjson.loads(line))
            # if len(data) >= 1000mage:
            #     break

    # Initialize charxiv data once before threading
    charxiv_data = datasets.load_dataset('princeton-nlp/CharXiv', split='validation')
    init_charxiv_question_to_image(charxiv_data)

    # Process with 16 threads
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(
            executor.map(process_single_item, enumerate(data)),
            total=len(data),
            desc="Processing images"
        ))

    # Update data with results
    for i, image_path in results:
        data[i]['image_path'] = image_path.as_posix()

    # save to jsonl
    # move to backup first
    shutil.move(jsonl_path, Path(jsonl_path).with_suffix('.bak.jsonl'))
    with open(jsonl_path, 'w') as f:
        for d in tqdm(data, desc="Saving data"):
            f.write(orjson.dumps(d).decode('utf-8') + '\n')