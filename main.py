import argparse
import logging
from pathlib import Path
import subprocess
import sys
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scripts.agent import CodeAgent, AgentConfig
from scripts.llama_cpp_wrapper import LlamaWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    query: str


app = FastAPI()
agent: Optional[CodeAgent] = None


def setup_environment() -> bool:
    try:
        scripts = [
            "scripts/collect_data.py",
            "scripts/preprocess_data.py",
            "scripts/fine_tune.py"
        ]

        for script in scripts:
            logger.info(f"Running {script}")
            result = subprocess.run([sys.executable, script], check=True)
            if result.returncode != 0:
                return False
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Setup failed: {str(e)}")
        return False


def init_agent() -> Optional[CodeAgent]:
    try:
        config = AgentConfig()
        return CodeAgent(config)
    except Exception as e:
        logger.error(f"Agent initialization failed: {str(e)}")
        return None


@app.on_event("startup")
async def startup_event():
    global agent
    if not Path("models/final/ggml-model-q4_k.bin").exists():
        if not setup_environment():
            raise RuntimeError("Environment setup failed")
    agent = init_agent()


@app.post("/complete")
async def complete_code(request: CompletionRequest):
    if not agent:
        raise HTTPException(500, "Agent not initialized")
    return agent.process_query(request.query)


def cli_interface():
    parser = argparse.ArgumentParser(description="Code Completion CLI")
    parser.add_argument("--setup", action="store_true", help="Run setup scripts")
    parser.add_argument("--query", type=str, help="Code completion query")
    args = parser.parse_args()

    if args.setup:
        setup_environment()

    if args.query:
        agent = init_agent()
        if agent:
            result = agent.process_query(args.query)
            print(result)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_interface()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)