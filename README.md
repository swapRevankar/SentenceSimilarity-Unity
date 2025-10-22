## Model Setup
This demo uses a lightweight **MiniLMv6** sentence-embedding model and a BERT-style **vocabulary file** for tokenization.  
Due to size and licensing, these files are **not included** in the repository — you’ll need to download them manually.

### Expected folder structure
Assets/
├─ Models/
│ └─ MiniLMv6/
│ └─ model.onnx
│
├─ Data/
│ └─ Vocab/
│ └─ vocab.txt

- **`model.onnx`** – exported MiniLMv6 ONNX model  
- **`vocab.txt`** – tokenizer vocabulary file

### How to add the model

1. Download the MiniLMv6 ONNX model and vocab from a public source such as  
   [Hugging Face → MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
2. Copy:
   - `model.onnx` → `Assets/Models/MiniLMv6/`
   - `vocab.txt` → `Assets/Data/vocab/`