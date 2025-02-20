from llama_cpp import Llama
import logging
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

class LlamaWrapper:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=-1  # Use all GPU layers
        )
        logger.info("Model loaded successfully")

    def generate(self, request: CompletionRequest) -> Dict:
        try:
            completion = self.llm.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=["```", "\n\n"]  # Stop at code block end
            )
            return {
                "completion": completion['choices'][0]['text'],
                "usage": completion['usage']
            }
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise

app = FastAPI()
model: Optional[LlamaWrapper] = None

@app.on_event("startup")
async def startup_event():
    global model
    model_path = Path("models/final/ggml-model-q4_k.bin")
    if not model_path.exists():
        raise RuntimeError("Model file not found")
    model = LlamaWrapper(str(model_path))

@app.post("/complete")
async def complete_code(request: CompletionRequest):
    if not model:
        raise HTTPException(500, "Model not initialized")
    return model.generate(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)