# main.py
import logging
import torch

from metrics import metrics
from preprocessing import preprocess_documents
from embedder import Embedder, EmbedderConfig
from vectorstore import VectorStore, VectorStoreConfig
from memory import MemoryConfig, MemoryModule
from agents import RagAgent, AgentConfig, LLMConfig, LLMGenerator

'''
a) Default
python main.py --use_faiss --llm gpt2


This means:

FAISS on

LLM = gpt2

embedding model = default MiniLM

device auto-detected
b) Use FAISS with inner-product (cosine-like), on GPU
python main.py \
  --data_dir ./data/papers \
  --use_faiss \
  --faiss_ip \
  --device cuda \
  --llm gpt2 \
  --embed_model sentence-transformers/all-MiniLM-L6-v2

c) CPU-only, larger embedding model, longer chunks
python main.py \
  --device cpu \
  --data_dir ./data/docs \
  --max_chars 1500 \
  --overlap_chars 300 \
  --embed_model sentence-transformers/all-mpnet-base-v2 \
  --embed_batch_size 16 \
  --llm gpt2 \
  --llm_max_input 768 \
  --llm_max_new_tokens 128

d) Turn off FAISS, use NumPy fallback (useful for debugging)
python main.py \
  --data_dir ./data/toy \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --llm gpt2 \
  --device cpu


(no --use_faiss flag → pure NumPy search)

e) Lower temperature, more deterministic answers
python main.py \
  --use_faiss \
  --llm gpt2 \
  --temperature 0.1

python main.py --data_dir data/wikitext2_train --use_faiss --llm gpt2
'''

def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Preprocess corpus
    chunks, metadatas = preprocess_documents(input_path="data/wikitext2_train")

    # 2. Embed corpus
    emb_config = EmbedderConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=512,
        batch_size=32,
    )
    embedder = Embedder(emb_config)
    corpus_embeddings = embedder.embed_corpus(chunks)

    # 3. Vector store
    dim = corpus_embeddings.shape[1]
    vs_config = VectorStoreConfig(
        dim=dim,
        use_faiss=True,
        faiss_index_type="IndexFlatIP",
        normalize_embeddings=True,
    )
    vectorstore = VectorStore(vs_config)
    vectorstore.add_documents(corpus_embeddings, chunks, metadatas)

    # 4. Memory
    mem_config = MemoryConfig(dim=dim)
    memory = MemoryModule(mem_config)

    # 5. LLM
    llm_config = LLMConfig(
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    llm = LLMGenerator(llm_config)

    # 6. Agent
    agent_config = AgentConfig()
    agent = RagAgent(embedder, vectorstore, memory, llm, agent_config)

    # 7. Simple loop
    while True:
        query = input("Enter query (or 'exit')> ").strip()
        if query.lower() == "exit":
            break
        answer, debug = agent.generate_answer(query)
        print("\n[Answer]:", answer, "\n")

    # 8. Print/save metrics
    metrics.log_summary()
    metrics.dump_csv("latencies.csv")

if __name__ == "__main__":
    main()
