import logging
from dataclasses import dataclass
from enum import Enum
import requests
from typing import Dict, Optional, List
import black
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    COMPLETE = "complete"
    FORMAT = "format"
    SEARCH_DOCS = "search_docs"


@dataclass
class AgentConfig:
    llm_endpoint: str = "http://localhost:8000/complete"
    docs_endpoint: str = "http://localhost:8001/search"
    max_tokens: int = 256
    temperature: float = 0.7


class CodeAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools = {
            ActionType.FORMAT: self._format_code,
            ActionType.SEARCH_DOCS: self._search_documentation,
        }

    def process_query(self, query: str) -> Dict:
        action = self._determine_action(query)
        if action == ActionType.COMPLETE:
            return self._get_completion(query)
        return self.tools[action](query)

    def _determine_action(self, query: str) -> ActionType:
        formatting_keywords = {"format", "indent", "style", "pep8"}
        docs_keywords = {"docs", "documentation", "help", "example"}

        query_words = set(query.lower().split())
        if any(word in query_words for word in formatting_keywords):
            return ActionType.FORMAT
        if any(word in query_words for word in docs_keywords):
            return ActionType.SEARCH_DOCS
        return ActionType.COMPLETE

    def _get_completion(self, prompt: str) -> Dict:
        try:
            response = requests.post(
                self.config.llm_endpoint,
                json={
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Completion error: {str(e)}")
            raise

    def _format_code(self, code: str) -> Dict:
        try:
            formatted_code = black.format_str(
                self._extract_code(code),
                mode=black.FileMode()
            )
            return {"result": formatted_code}
        except Exception as e:
            logger.error(f"Formatting error: {str(e)}")
            return {"error": str(e)}

    def _search_documentation(self, query: str) -> Dict:
        try:
            response = requests.get(
                self.config.docs_endpoint,
                params={"q": query}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Documentation search error: {str(e)}")
            return {"error": str(e)}

    def _extract_code(self, text: str) -> str:
        code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        return code_match.group(1) if code_match else text


def main():
    config = AgentConfig()
    agent = CodeAgent(config)

    # Example usage
    query = "Help me format this code: def example(x,y): return x+y"
    result = agent.process_query(query)
    print(result)

    print(f"This is {result=}")

if __name__ == "__main__":
    main()