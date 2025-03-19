Be aware of the following context, AI.

# Notes

These are my thoughts and live logs I like to record stream of thought style while coding.

## Installation

Open the project and create the virtualenv with `uv`:
```
uv init --lib rag267
```

Add all the deps from the Setup section of the assignment notebook:
```
source .venv/bin/activate
uv add transformers
uv add datasets
uv add loralib
uv add bitsandbytes
uv add accelerate
uv add langchain
uv add einops
uv add faiss-gpu
uv add langchain_community
uv add chromadb bs4 qdrant-client
uv add langchainhub
uv add langchain-huggingface
uv add langchain-cohere
uv add wikipedia arxiv
uv add pymupdf
uv add xmltodict
uv add cohere
 ```

Ran into issues installing `faiss-gpu` - ignoring until it becomes an issue:
```
❯ uv add faiss-gpu
  × No solution found when resolving dependencies for split (python_full_version >= '3.12.4'):
  ╰─▶ Because all versions of faiss-gpu have no wheels with a matching Python version tag (e.g., `cp311`) and your project depends
      on faiss-gpu, we can conclude that your project's requirements are unsatisfiable.

      hint: Wheels are available for `faiss-gpu` (v1.7.2) with the following Python ABI tags: `cp36m`, `cp37m`, `cp38`, `cp39`,
      `cp310`
  help: If you want to add the package regardless of the failed resolution, provide the `--frozen` flag to skip locking and syncing.
```

Verified langchain installed and version matches notebook:
```
❯ uvx --with langchain -p 3.11 ipython
Installed 43 packages in 179ms
/Users/seansica/.cache/uv/archive-v0/CjKwjoHn1MlaekUIgbdkk/lib/python3.11/site-packages/IPython/core/interactiveshell.py:955: UserWarning: Attempting to work in a virtualenv. If you encounter problems, please install IPython inside the virtualenv.
  warn(
Python 3.11.11 (main, Feb 12 2025, 15:06:01) [Clang 19.1.6 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 9.0.2 -- An enhanced Interactive Python. Type '?' for help.
Tip: You can find how to type a latex symbol by back completing it `\θ<tab>` will expand to `\theta`.

In [1]: import langchain

In [2]: langchain.__version__
Out[2]: '0.3.20'
```

We need a way to track environment variables for `COHERE_API_KEY` and `OPENAI_API_KEY`:
```
uv add python-dotenv
touch .env
echo 'COHERE_API_KEY=\nOPENAI_API_KEY=' >> .env
```

## Tracing/Profiling

Configure environment to connect to LangSmith:

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<your-api-key>"
LANGSMITH_PROJECT="w267-final-project-rag-pipeline"
OPENAI_API_KEY="<your-openai-api-key>"
```

Now we can run any LLM, Chat model, or Chain. Its trace will be sent to this project.


## Planning

Let's think through what we need for a RAG pipeline. We need to split things up into two phases:

1. Building the components of the RAG system
2. Actually running and using the RAG system

### Building the components of the RAG system

We need:
- [ ] an embedding model to generate vectors
- [ ] a loading and chunking mechanism to process our documents
- [ ] a vector store to store the chunks and/or embedding vectors
- [ ] a retrieval mechanism to:
  - [ ] fetch vectors/chunks that are relevant to the input text
  - [ ] pre-process the vectors/chunks (cleaning the text for suitability with LLM)
- [ ] an augmentation mechanism to process/format the inference prompt

### Actually running and using the RAG system
