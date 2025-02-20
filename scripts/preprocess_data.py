import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple
import random
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import jsonlines

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('data/processed/preprocessing.log'),
              logging.StreamHandler()]
)


def load_raw_data(raw_dir: Path) -> List[Dict]:
    """Load all JSON files from raw directory."""
    data = []
    for file in raw_dir.glob('*_docs.json'):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
    return data


def clean_text(text: str) -> str:
    """Clean text by removing HTML tags and normalizing whitespace."""
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    return text.strip()


def extract_code_pairs(text: str) -> List[Dict]:
    """Extract code snippets and their explanations."""
    # Find Python code blocks (between triple backticks)
    code_blocks = re.finditer(r'```python(.*?)```', text, re.DOTALL)
    pairs = []

    for match in code_blocks:
        code = match.group(1).strip()
        # Get context before code block (limited to 500 chars)
        start_pos = max(0, match.start() - 500)
        context = text[start_pos:match.start()].strip()

        if code and context:
            pairs.append({
                'code': code,
                'explanation': clean_text(context),
                'language': 'python'
            })

    return pairs


def process_documents(docs: List[Dict]) -> List[Dict]:
    """Process all documents and extract code-explanation pairs."""
    processed_data = []

    for doc in docs:
        try:
            # Process both description and README
            text = f"{doc['description']}\n{doc.get('github_readme', '')}"
            pairs = extract_code_pairs(text)

            for pair in pairs:
                processed_data.append({
                    'package': doc['name'],
                    'version': doc['version'],
                    **pair
                })
        except Exception as e:
            logging.error(f"Error processing {doc['name']}: {e}")

    return processed_data


def save_splits(data: List[Dict], output_dir: Path) -> None:
    """Split data and save as JSONL files."""
    # Create train/val/test splits (70/15/15)
    train, temp = train_test_split(data, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    splits = {
        'train': train,
        'val': val,
        'test': test
    }

    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        try:
            with jsonlines.open(output_file, mode='w') as writer:
                writer.write_all(split_data)
            logging.info(f"Saved {len(split_data)} examples to {output_file}")
        except Exception as e:
            logging.error(f"Error saving {split_name} split: {e}")


def main():
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_docs = load_raw_data(raw_dir)
    processed_data = process_documents(raw_docs)
    save_splits(processed_data, processed_dir)


if __name__ == "__main__":
    main()