## Model Setup
This demo uses a lightweight **MiniLMv6** sentence-embedding model and a BERT-style **vocabulary file** for tokenization.  
Due to size and licensing, these files are **not included** in the repository — you’ll need to download them manually.

### Expected folder structure
```text
Assets/
├─ Models/
│ └─ MiniLMv6/
│ └─ model.onnx
│
├─ Data/
│ └─ Vocab/
│ └─ vocab.txt
```
- **`model.onnx`** – exported MiniLMv6 ONNX model  
- **`vocab.txt`** – tokenizer vocabulary file

### How to add the model

1. Download the MiniLMv6 ONNX model and vocab from a public source such as  
   [Hugging Face → MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
2. Copy:
   - `model.onnx` → `Assets/Models/MiniLMv6/`
   - `vocab.txt` → `Assets/Data/vocab/`

### Run the Demo

1. Open the project in **Unity 2023 LTS**.  
2. Open the scene:
   ```text
   Assets/Scenes/SentenceSimilarityDemo.unity
   ```
3. In the Inspector, assign:
    - Model Asset → Assets/Models/MiniLMv6/model.onnx
    - Vocab Asset → Assets/Data/vocab/vocab.txt
4. Press **Play** type a source sentence, add comparisons, and click **Compute** to see results.

### Example Output

When you run the demo, you can type any source sentence and compare it against multiple others in real time.

Example:

| Source | Comparison | Similarity |
|---------|-------------|------------|
| "Welcome to Jumanji!" | "Don't just stand there, hop in." | **0.39** |
| "Welcome to Jumanji!" | "I love pizza." | **0.17** |

These values will vary slightly depending on your model backend (CPU vs GPU) but should follow the same pattern — higher values for semantically related sentences.


