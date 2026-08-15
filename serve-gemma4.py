"""3th Floor Engine — OwnMyWork inference server.

Loads Gemma 4 31B on the TITAN RTX and serves OpenAI-compatible
/v1/chat/completions on port 8096. Secured with bearer token.

Start:
    python3 serve-gemma4.py

Stop:
    kill $(cat /home/jbench/3thfloor-engine/serve-gemma4.pid)
"""

import os
import sys
sys.path.insert(0, '/home/jbench/3thfloor-engine')

from thirthfloor.engine import Engine

MODEL_PATH = '/ai-models/gguf/google_gemma-4-31B-it-Q4_K_M.gguf'
PORT       = 8096
HOST       = '0.0.0.0'
TOKEN      = os.environ.get('ENGINE_TOKEN', 'omw-engine-2026')
GPU_LAYERS = -1   # all layers on TITAN RTX 24GB
CTX        = 8192

engine = Engine()
print(f'Loading Gemma 4 31B from {MODEL_PATH}...')
engine.load('gemma4', MODEL_PATH, ctx=CTX, gpu_layers=GPU_LAYERS)
print(f'Model loaded. Serving on {HOST}:{PORT}')
engine.serve(port=PORT, host=HOST, token=TOKEN)
