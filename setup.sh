#!/bin/bash
# Download large files from Hugging Face
# These files are too large for GitHub and are hosted on Hugging Face Hub

set -e

echo "=== TimeAware LLM Role-Play: Downloading large files ==="

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

python3 -c "
from huggingface_hub import hf_hub_download
import os
import shutil

REPO_ID = 'Kkonjeong/TimeAware-LLM-Demo'
FILES = [
    ('databases/knowledge_embeddings.pkl', 'knowledge_enriched_rag/databases/'),
    ('databases/knowledge_embeddings_array.npy', 'knowledge_enriched_rag/databases/'),
    ('databases/memory_vectors.npy', 'knowledge_enriched_rag/databases/'),
    ('databases/search_index_embeddings.npy', 'knowledge_enriched_rag/databases/'),
    ('models/roberta_200q_best_20251010_064944.pt', 'knowledge_enriched_rag/models/'),
]

for hf_path, local_dir in FILES:
    local_file = os.path.join(local_dir, os.path.basename(hf_path))
    if os.path.exists(local_file) and os.path.getsize(local_file) > 1000:
        print(f'[SKIP] {local_file} already exists ({os.path.getsize(local_file)} bytes)')
        continue
    print(f'[DOWNLOAD] {hf_path} -> {local_file}')
    os.makedirs(local_dir, exist_ok=True)
    downloaded = hf_hub_download(repo_id=REPO_ID, filename=hf_path, local_dir='.')
    # If hf_hub_download saves to a different location, copy it
    if os.path.abspath(downloaded) != os.path.abspath(local_file):
        shutil.copy2(downloaded, local_file)
    print(f'[DONE] {local_file} ({os.path.getsize(local_file)} bytes)')

print()
print('=== All large files downloaded successfully ===')
"
