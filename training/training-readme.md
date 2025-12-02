# Phi-3 QLoRA Fine-tuning Guide

Complete guide for fine-tuning Microsoft's Phi-3-mini-4k-instruct model using QLoRA (Quantized Low-Rank Adaptation) to generate Cypher queries from natural language instructions.

---

## Table of Contents
1. [What This Does](#what-this-does)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation Process](#installation-process)
5. [Dataset Preparation](#dataset-preparation)
6. [Configuration Explained](#configuration-explained)
7. [Training Pipeline](#training-pipeline)
8. [Model Output](#model-output)
---

## What This Does

This notebook transforms a general-purpose language model (Phi-3) into a specialized Cypher query generator. It teaches the model to understand natural language questions about graph databases and convert them into executable Cypher queries.

**Example:**
- **Input:** "Show me the top 5 products by total ordered quantity"
- **Output:** `MATCH (p:Product)-[:PART_OF]->(o:Order) RETURN p.name, SUM(o.quantity) AS total ORDER BY total DESC LIMIT 5`

**Why QLoRA?**
- **Q (Quantized):** Compresses model from 32-bit to 4-bit precision → Uses ~75% less memory
- **LoRA (Low-Rank Adaptation):** Only trains 1-2% of model parameters → Faster training, smaller files
- **Result:** Train a 3.8B parameter model on consumer GPUs with 10-16GB VRAM

---

## Project Structure

```
C:\Users\HP\OneDrive\Desktop\project 2\
└── training\
    ├── data\
    │   └── text_to_cypher_10000.csv        # Your training dataset (10,000 examples)
    │
    │
    └── phi3_qlora_final\                   # Final trained model
        ├── adapter_config.json             # LoRA configuration
        ├── adapter_model.safetensors       # Trained weights (small ~100MB)
        ├── tokenizer_config.json           # Tokenizer settings
        ├── tokenizer.model
        └── special_tokens_map.json
    |
    |___train_phi3_qlora.ipynb
```

**Important Notes:**
- The base Phi-3 model (~7.5GB) downloads to your cache, not this folder
- Only the small LoRA adapter (~100-200MB) saves to `phi3_qlora_final`
- Checkpoints in `phi3_qlora` help resume training if interrupted

---

## Requirements

### Hardware
- **GPU:** NVIDIA GPU with 10GB+ VRAM (T4, RTX 3080, A10, etc.)
- **RAM:** 16GB+ system RAM recommended
- **Storage:** 15GB free space (10GB model + 5GB checkpoints)

### Software
- **Python:** 3.8 - 3.11
- **CUDA:** 11.8 or 12.1
- **Platform:** Kaggle, Google Colab, or local Windows/Linux with NVIDIA GPU

### Package Versions (Auto-installed)
- PyTorch 2.1.2 with CUDA 12.1
- Transformers 4.40.2
- PEFT 0.10.0
- TRL 0.9.4
- BitsAndBytes 0.43.1
- Triton 2.1.0

---

## Installation Process

### Step 1: Clean Environment
```python
!pip uninstall -y torch torchvision torchaudio triton transformers trl peft bitsandbytes accelerate datasets tokenizers pyarrow
```
Removes any conflicting package versions that might cause errors.

### Step 2: Install PyTorch Stack
```python
!pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
!pip install -q triton==2.1.0
```
Installs PyTorch with CUDA 12.1 support and Triton for optimized operations.

### Step 3: Install Training Libraries
```python
!pip install -q transformers==4.40.2 tokenizers==0.19.1
!pip install -q trl==0.9.4 peft==0.10.0 accelerate==0.28.0 bitsandbytes==0.43.1
!pip install -q pyarrow==19.0.1 datasets==2.20.0
```

### Step 4: Restart Runtime
**CRITICAL:** After installation, you MUST restart your Python kernel/runtime. The notebook does this automatically:
```python
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
```

### Step 5: Verify Installation
Run the verification cell to confirm all packages loaded correctly:
```
ENVIRONMENT VERIFICATION
============================================================
Python: 3.10.12
PyTorch: 2.1.2
Triton: 2.1.0
CUDA Available: True
Transformers: 4.40.2
TRL: 0.9.4
PEFT: 0.10.0
BitsAndBytes: 0.43.1
============================================================
✅ ALL IMPORTS SUCCESSFUL - NO ERRORS!
```

---

## Dataset Preparation

### Required Format
Your CSV must have exactly two columns:

| instruction | cypher |
|-------------|--------|
| Top 5 products by quantity | `MATCH (p:Product) RETURN p ORDER BY p.quantity DESC LIMIT 5` |
| Find customers in New York | `MATCH (c:Customer) WHERE c.city = 'New York' RETURN c` |
| Orders placed last month | `MATCH (o:Order) WHERE o.date >= date() - duration('P1M') RETURN o` |

### Column Names
- **`instruction`**: Natural language question/command
- **`cypher`**: Target Cypher query to generate

If your columns have different names, update these lines in the notebook:
```python
IN_COL = "instruction"  # Change to your input column name
OUT_COL = "cypher"      # Change to your output column name
```

### Data Quality Tips
1. **Variety:** Include simple and complex queries
2. **Consistency:** Use consistent Cypher syntax and formatting
3. **Balance:** Mix different query types (MATCH, CREATE, UPDATE, DELETE)
4. **Length:** Queries should typically be 10-200 tokens long

### Dataset Split
The notebook automatically splits your data:
- **85%** for training (8,500 examples from your 10,000)
- **15%** for evaluation (1,500 examples)

---

## Configuration Explained

```python
config = {
    # Base model from Hugging Face
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    
    # Where to find your CSV (tries in order until found)
    "dataset_candidates": [
        r"C:\Users\HP\OneDrive\Desktop\project 2\training\data\text_to_cypher_10000.csv",
        "./data/text_to_cypher_10000.csv",  # Relative path fallback
    ],
    
    # Where to save checkpoints during training
    "output_dir": r"C:\Users\HP\OneDrive\Desktop\project 2\training\phi3_qlora",
    
    # Where to save final model
    "final_model_dir": r"C:\Users\HP\OneDrive\Desktop\project 2\training\phi3_qlora_final",
    
    # Maximum sequence length (instruction + cypher combined)
    "max_seq_length": 2048,
    
    # Percentage of data used for validation
    "train_test_split": 0.15,  # 15% for evaluation
    
    # Training duration
    "num_epochs": 2,  # Full passes through the dataset
    
    # Batch processing
    "batch_size": 2,                    # Samples per GPU at once
    "gradient_accumulation": 8,         # Effective batch = 2 × 8 = 16
    
    # Learning rate (how fast model updates)
    "learning_rate": 2e-4,  # 0.0002
    
    # LoRA hyperparameters
    "lora_r": 64,           # Rank: higher = more capacity, more memory
    "lora_alpha": 32,       # Scaling factor: typically r/2
    "lora_dropout": 0.1,    # Prevents overfitting
    
    # Reproducibility
    "seed": 42,
}
```

### Key Parameter Trade-offs

**If you have limited GPU memory:**
- Reduce `batch_size` to 1
- Increase `gradient_accumulation` to 16
- Reduce `lora_r` to 32

**If training is too slow:**
- Increase `batch_size` to 4
- Reduce `gradient_accumulation` to 4
- Reduce `num_epochs` to 1

**If model is underfitting (poor performance):**
- Increase `num_epochs` to 3-4
- Increase `lora_r` to 128
- Increase `learning_rate` to 3e-4

---

## Training Pipeline

### 1. Model Loading (30-90 seconds)
```python
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,  # 4-bit quantization
    device_map="auto",                # Auto-distribute across GPUs
)
```

**What happens:**
- Downloads Phi-3 model (~7.5GB) to cache
- Quantizes weights from 32-bit → 4-bit
- Reduces memory from ~15GB → ~4GB

### 2. LoRA Preparation
```python
lora_cfg = LoraConfig(
    r=64,                          # Low-rank dimension
    target_modules=[               # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
)
model = get_peft_model(model, lora_cfg)
```

**Output:**
```
trainable params: 41,943,040 || all params: 3,820,084,224 || trainable%: 1.09%
```
Only 1% of parameters are being trained!

### 3. Data Formatting
Converts your CSV into Phi-3's chat template:
```
<|user|>
Show me the top 5 products by quantity<|end|>
<|assistant|>
MATCH (p:Product) RETURN p ORDER BY p.quantity DESC LIMIT 5<|end|>
```

### 4. Training Execution
```python
trainer.train()
```

**What you'll see:**
```
Step    Loss     Eval Loss    Time
50      2.456    -            0:45
100     1.823    1.654        1:30
150     1.456    -            2:15
200     1.234    1.198        3:00
...
```

**Metrics explained:**
- **Loss:** Training error (lower is better)
- **Eval Loss:** Validation error (should decrease with Loss)
- **Time:** Elapsed since start

**Training duration:** ~30-60 minutes for 10,000 examples with 2 epochs

### 5. Early Stopping
Automatically stops if validation loss doesn't improve for 3 evaluations (300 steps):
```python
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
```

### 6. Saving
```python
trainer.model.save_pretrained(config["final_model_dir"])
tokenizer.save_pretrained(config["final_model_dir"])
```

Creates a ZIP file you can download:
```
phi3_qlora_final_zip.zip (100-200MB)
```

---

## Model Output



### Final Model Structure
```
phi3_qlora_final/
├── adapter_config.json              # LoRA configuration
├── adapter_model.safetensors        # Trained LoRA weights (~100-200MB)
├── tokenizer_config.json            # How text is converted to tokens
├── tokenizer.model                  # Vocabulary file
├── special_tokens_map.json          # Special tokens like <|end|>
└── tokenizer.json                   # Fast tokenizer data
```

**File sizes:**
- `adapter_model.safetensors`: 100-200MB (your trained weights)
- All tokenizer files combined: ~5MB
- **Total:** ~105-205MB (very portable!)

---

## Using Your Trained Model

### Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model (downloads to cache if not present)
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer from your trained folder
tokenizer = AutoTokenizer.from_pretrained(
    r"C:\Users\HP\OneDrive\Desktop\project 2\training\phi3_qlora_final"
)

# Load your trained LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    r"C:\Users\HP\OneDrive\Desktop\project 2\training\phi3_qlora_final"
)

model.eval()  # Set to evaluation mode
```

### Generating Queries

```python
def generate_cypher(instruction, max_length=150):
    """Generate a Cypher query from natural language instruction"""
    
    # Format input in Phi-3 template
    prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=0.1,           # Low = more deterministic
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    
    # Decode and extract just the Cypher query
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|assistant|>" in full_response:
        cypher = full_response.split("<|assistant|>")[-1].strip()
    else:
        cypher = full_response.strip()
    
    return cypher

# Example usage
instruction = "Show me all customers who placed orders in the last 30 days"
cypher_query = generate_cypher(instruction)
print(f"Instruction: {instruction}")
print(f"Generated Cypher: {cypher_query}")
```

**Example output:**
```
Instruction: Show me all customers who placed orders in the last 30 days
Generated Cypher: MATCH (c:Customer)-[:PLACED]->(o:Order) 
WHERE o.date >= date() - duration('P30D') 
RETURN c.name, COUNT(o) AS order_count 
ORDER BY order_count DESC
```


---

