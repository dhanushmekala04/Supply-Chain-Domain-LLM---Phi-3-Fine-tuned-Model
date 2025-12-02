# Phi-3 Model Inference & GGUF Conversion

Quick guide for loading your fine-tuned Phi-3 model in Kaggle and converting it to GGUF format for local CPU inference.

---

## Overview

This notebook performs three tasks:
1. **Test Inference** - Load and test your fine-tuned model in Kaggle
2. **Merge LoRA** - Combine LoRA adapters with base model into a single FP16 model
3. **Convert to GGUF** - Create a quantized model for CPU inference with llama.cpp

---

## Part 1: Test Inference in Kaggle

### Install Dependencies

```python
!pip install -q torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
!pip install -q pyarrow==19.0.1 datasets==2.20.0
!pip install -q transformers==4.40.2 tokenizers==0.19.1
!pip install -q trl==0.9.4 peft==0.10.0 accelerate==0.28.0 bitsandbytes==0.43.1
```

### Load Model & Test

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 4-bit quantization config
bnb = BitsAndBytesConfig(load_in_4bit=True)

# Load base model (downloads ~7.5GB)
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb,
    device_map="auto"
)

# Load your fine-tuned adapter
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/fine-tune-phi/other/default/1")
model = PeftModel.from_pretrained(base_model, "/kaggle/input/fine-tune-phi/other/default/1")
model.eval()

# Test generation
prompt = "<|user|>\nTop 5 products by total ordered quantity.\n<|end|>\n<|assistant|>\n"
tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**tokens, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Expected Output:**
```
<|user|>
Top 5 products by total ordered quantity.
<|end|>
<|assistant|>
MATCH (p:Product)-[:PART_OF]->(o:Order) 
RETURN p.name, SUM(o.quantity) AS total_quantity 
ORDER BY total_quantity DESC 
LIMIT 5
```

---

## Part 2: Merge LoRA into Base Model

### Why Merge?
- LoRA adapters require loading base model + adapter separately
- Merged model = single file, easier to deploy and convert
- Keeps FP16 precision for best quality

### Installation

```bash
!pip install transformers==4.40.2 peft==0.10.0 accelerate bitsandbytes
!apt-get -y install git-lfs
```

### Configuration

```python
LORA_DIR = "/kaggle/input/fine-tune-phi/other/default/1"  # Your uploaded adapter
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
MERGED_DIR = "/kaggle/working/phi3_merged_fp16"
```

### Merge Process

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,    # FP16 for quality
    device_map="cpu",             # Use CPU to save GPU memory
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading LoRA adapter:", LORA_DIR)
peft_model = PeftModel.from_pretrained(
    base,
    LORA_DIR,
    torch_dtype=torch.float16
)

print("Merging LoRA → base model...")
merged = peft_model.merge_and_unload()

print("Saving merged FP16 model to:", MERGED_DIR)
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print("✅ Merge complete.")
```

**Output:**
```
phi3_merged_fp16/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors  (~4GB)
├── model-00002-of-00002.safetensors  (~3.5GB)
├── model.safetensors.index.json
└── tokenizer files...
```

**Total size:** ~7.5GB merged model in FP16

---

## Part 3: Convert to GGUF for CPU Inference

### What is GGUF?
- **GGUF** = GPT-Generated Unified Format
- Optimized for CPU inference with llama.cpp
- Supports various quantization levels (Q4, Q5, Q8)
- Enables fast inference on laptops without GPU

### Clone llama.cpp

```bash
!git clone https://github.com/ggerganov/llama.cpp /kaggle/working/llama.cpp
```

### Convert to GGUF (Q8_0)

```bash
!python3 /kaggle/working/llama.cpp/convert_hf_to_gguf.py \
    /kaggle/working/phi3_merged_fp16 \
    --outfile /kaggle/working/phi3_gguf_q8_0.gguf \
    --outtype q8_0
```

**Quantization Options:**
- `q8_0` - 8-bit (best quality, ~4GB)
- `q5_k_m` - 5-bit medium (balanced, ~2.5GB)
- `q4_k_m` - 4-bit medium (smaller, ~2GB)
- `q4_0` - 4-bit (smallest, ~1.8GB, lower quality)

**Conversion time:** 2-5 minutes

### Create Download ZIP

```python
import shutil

zip_path = "/kaggle/working/phi3_gguf_q8_0.zip"
shutil.make_archive(
    zip_path.replace(".zip",""), 
    "zip", 
    "/kaggle/working", 
    "phi3_gguf_q8_0.gguf"
)
print("✅ ZIP created:", zip_path)

# Create download link
from IPython.display import FileLink
FileLink('/kaggle/working/phi3_gguf_q8_0.zip')
```

**Output:** `phi3_gguf_q8_0.zip` (~4GB)

---

## File Structure

After running the notebook:

```
/kaggle/working/
├── llama.cpp/                      # Conversion tools
├── phi3_merged_fp16/               # Merged FP16 model (~7.5GB)
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── tokenizer files...
├── phi3_gguf_q8_0.gguf            # GGUF model (~4GB)
└── phi3_gguf_q8_0.zip             # Downloadable ZIP
```

---

## Performance Comparison

| Format | Size | Device | Speed | Quality |
|--------|------|--------|-------|---------|
| LoRA Adapter | 200MB | GPU | Fast | Best |
| Merged FP16 | 7.5GB | GPU | Fast | Best |
| GGUF Q8 | 4GB | CPU | Medium | Excellent |
| GGUF Q5 | 2.5GB | CPU | Fast | Good |
| GGUF Q4 | 2GB | CPU | Very Fast | Fair |

**Recommendation:**
- **GPU available:** Use LoRA adapter or merged FP16
- **CPU only:** Use GGUF Q8 for best quality, Q5 for balance
- **Mobile/edge:** Use GGUF Q4 for smallest size



## Summary

### Workflow Overview

1. **Kaggle (GPU):**
   - Test fine-tuned model with LoRA
   - Merge LoRA → FP16 model
   - Convert FP16 → GGUF Q8

2. **Local (CPU):**
   - Download GGUF file
   - Ready for deployment

### Commands Cheatsheet

```bash
# Test in Kaggle
python inference_notebook.ipynb

# Download GGUF
# (Use Kaggle UI → Output → Download phi3_gguf_q8_0.zip)
```

---

## Next Steps

- **Production deployment:** Integrate with your application
- **Monitoring:** Add logging and error handling
- **Optimization:** Profile and optimize for your hardware
- **Fine-tuning:** Iterate with more data if needed

---

## Resources

- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **GGUF spec:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Phi-3 Model:** https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

