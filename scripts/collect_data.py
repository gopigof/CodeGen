import json
import logging
import os
from datetime import datetime
import requests
from typing import Dict, List, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/raw/collection.log'),
        logging.StreamHandler()
    ]
)

PACKAGES = [
    'requests',
    'numpy',
    'pandas',
    'tensorflow',
    'torch',
    'transformers',
    'scikit-learn',
    'matplotlib'
]


def fetch_package_docs(package_name: str) -> Optional[Dict]:
    """Fetch documentation for a given package from PyPI and GitHub."""
    try:
        # PyPI API endpoint
        pypi_url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(pypi_url, timeout=10)
        response.raise_for_status()

        pypi_data = response.json()

        # Get GitHub README if available
        github_url = pypi_data.get('info', {}).get('project_urls', {}).get('Source')
        github_content = ""
        if github_url and 'github.com' in github_url:
            raw_github_url = github_url.replace('github.com', 'raw.githubusercontent.com') + '/main/README.md'
            github_response = requests.get(raw_github_url, timeout=10)
            if github_response.status_code == 200:
                github_content = github_response.text

        return {
            'name': package_name,
            'version': pypi_data['info']['version'],
            'description': pypi_data['info']['description'],
            'github_readme': github_content,
            'collected_at': datetime.utcnow().isoformat()
        }
    except requests.RequestException as e:
        logging.error(f"Error fetching docs for {package_name}: {str(e)}")
        return None


def save_documentation(data: Dict, output_dir: Path) -> None:
    """Save package documentation to JSON file."""
    try:
        output_file = output_dir / f"{data['name']}_docs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved documentation for {data['name']}")
    except Exception as e:
        logging.error(f"Error saving docs for {data['name']}: {str(e)}")


def main():
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)

    for package in PACKAGES:
        logging.info(f"Fetching documentation for {package}")
        docs = fetch_package_docs(package)
        if docs:
            save_documentation(docs, output_dir)
        else:
            logging.warning(f"Skipping {package} due to fetch failure")


if __name__ == "__main__":
    main()