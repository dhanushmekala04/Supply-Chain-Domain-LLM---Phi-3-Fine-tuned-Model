# Local Inference with Ollama

Guide for running your fine-tuned Phi-3 GGUF model locally using Ollama.

---

## Overview

Ollama makes it easy to run large language models locally on your CPU. This guide shows how to load your `phi3_gguf_q8_0.gguf` model and use it for generating Cypher queries.

---

## Prerequisites

- **GGUF Model:** `phi3_gguf_q8_0.gguf` (downloaded from Kaggle)
- **RAM:** 8GB+ recommended
- **Storage:** 5GB free space
- **OS:** Windows, macOS, or Linux

---

## Installation

### Step 1: Install Ollama

**Windows/Mac:**
- Download from: https://ollama.ai
- Run the installer
- Ollama runs as a background service

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Verify Installation

```bash
ollama --version
```

Expected output: `ollama version 0.x.x`

---

## Setup Your Model

### Step 1: Create Model Directory

```bash
# Create a folder for your model
mkdir phi3-cypher-model
cd phi3-cypher-model
```

### Step 2: Copy GGUF File

Place your `phi3_gguf_q8_0.gguf` file in this directory:

```
phi3-cypher-model/
└── phi3_gguf_q8_0.gguf
```

### Step 3: Create Modelfile

Create a file named `Modelfile` (no extension) with this content:

```dockerfile
# Modelfile
FROM ./phi3_gguf_q8_0.gguf

SYSTEM "<|system|>You are a helpful AI assistant trained on supply chain query generation.<|end|>"

TEMPLATE """<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|end|>"
PARAMETER num_ctx 2048
```

**Modelfile explained:**
- `FROM` - Points to your GGUF file
- `SYSTEM` - Defines the assistant's role and behavior
- `TEMPLATE` - Phi-3 chat format for proper prompt formatting
- `temperature` - Controls randomness (0.7 = balanced creativity)
- `top_p` - Nucleus sampling threshold
- `stop` - Stops generation at end token
- `num_ctx` - Context window size

### Step 4: Import Model into Ollama

```bash
ollama create phi3-local -f Modelfile
```

**Expected output:**
```
transferring model data
using existing layer sha256:xxxxx
creating new layer sha256:xxxxx
writing manifest
success
```

### Step 5: Verify Model

```bash
ollama list
```

You should see:
```
NAME            ID              SIZE    MODIFIED
phi3-local      xxxxx           4.0 GB  X seconds ago
```

---

## Usage

### Command Line Interface

#### Basic Query

```bash
ollama run phi3-local "Top 5 products by total quantity"
```

**Output:**
```
MATCH (p:Product)-[:PART_OF]->(o:Order)
RETURN p.name, SUM(o.quantity) AS total_quantity
ORDER BY total_quantity DESC
LIMIT 5
```

#### Interactive Mode

```bash
ollama run phi3-local
```

Then type your queries:
```
>>> Find all customers in New York
MATCH (c:Customer) WHERE c.city = 'New York' RETURN c

>>> Show orders placed in the last 30 days
MATCH (o:Order) WHERE o.date >= date() - duration('P30D') RETURN o

>>> /bye
```

---

### Python API

#### Install Ollama Python Package

```bash
pip install ollama
```

#### Simple Script

```python
import ollama

def generate_cypher(instruction):
    response = ollama.generate(
        model='phi3-local',
        prompt=instruction
    )
    return response['response']

# Test
query = generate_cypher("Top 10 customers by order count")
print(query)
```

#### Advanced Script with Error Handling

```python
import ollama

def generate_cypher(instruction, temperature=0.1, max_tokens=150):
    """Generate Cypher query from natural language instruction"""
    try:
        response = ollama.generate(
            model='phi3-local',
            prompt=instruction,
            options={
                'temperature': temperature,
                'num_predict': max_tokens,
                'top_p': 0.9,
                'stop': ['<|end|>']
            }
        )
        return response['response'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Examples
queries = [
    "Find all suppliers in California",
    "Show products with price greater than 100",
    "List top 5 categories by product count"
]

for q in queries:
    cypher = generate_cypher(q)
    print(f"\nQuestion: {q}")
    print(f"Cypher: {cypher}")
```

#### Streaming Response

```python
import ollama

def stream_cypher(instruction):
    stream = ollama.generate(
        model='phi3-local',
        prompt=instruction,
        stream=True
    )
    
    print("Generating: ", end='', flush=True)
    for chunk in stream:
        print(chunk['response'], end='', flush=True)
    print()

stream_cypher("Show all orders from last month")
```

---

### REST API

Ollama runs a local API server at `http://localhost:11434`

#### Using curl

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "phi3-local",
  "prompt": "Find customers who ordered more than 5 times",
  "stream": false
}'
```

#### Using Python Requests

```python
import requests
import json

def generate_via_api(instruction):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "phi3-local",
        "prompt": instruction,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    
    response = requests.post(url, json=payload)
    return response.json()['response']

# Test
result = generate_via_api("Top 5 products by revenue")
print(result)
```

---

## File Structure

Your local setup should look like:

```
phi3-cypher-model/
├── Modelfile                      # Model configuration
├── phi3_gguf_q8_0.gguf           # Your GGUF model file (~4GB)
├── ollama                         # Ollama executable (if applicable)
└── inference.png                  # Screenshot of model output
```

---

## Resources

- **Ollama Documentation:** https://github.com/ollama/ollama
- **Ollama Python Library:** https://github.com/ollama/ollama-python
- **Model File Reference:** https://github.com/ollama/ollama/blob/main/docs/modelfile.md
- **API Documentation:** https://github.com/ollama/ollama/blob/main/docs/api.md

