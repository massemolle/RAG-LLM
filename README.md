# LLM Assistant

The aim of this README is to introduce you on how to use, maintain and improve the RAG assistant.

# 1. General Introduction

## Running the app
The main UI is in `streamv3.py`. Run it with:
```bash
streamlit run streamv3.py --server.port 8502
```
The 'User' mode uses the BM25 method as it is currently the fastest available. The 'developper' mode allows the user to choose the embedding method and the LLM to be used.

## Guardrails Modes

The application supports three guardrails modes, selectable via dropdown in the UI:

| Mode | Description | Latency |
|------|-------------|---------|
| **Off** | No guardrails, direct LLM API call | Fastest |
| **Classic** | LLM-as-judge only (5 judges: 3 input, 2 output) | Fast (~500-800ms) |
| **Complete** | Full multi-layer pipeline with speculative parallel execution | Thorough (~1-2s) |

### Complete Mode Architecture (Speculative Parallel)

In Complete mode, **fast guards** and **LLM judges** run in parallel simultaneously:

```
┌──────────────────────────────────────────────────────────────┐
│                   PARALLEL EXECUTION                          │
│  ┌─────────────────────────┐  ┌────────────────────────────┐ │
│  │ Fast Guards (~500ms)    │  │ LLM Judges (speculative)   │ │
│  │ - Embedding similarity  │  │ - Input sentimental        │ │
│  │ - LLM Guard            │  │ - Input security           │ │
│  │ - NeMo Guardrails      │  │ - Input topic              │ │
│  │ - Topic taxonomy       │  │                            │ │
│  └─────────────────────────┘  └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Fast Guards Result? │
              └─────────────────────┘
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   All ALLOWED   Any BLOCKED   Any ESCALATE
       │             │             │
       ▼             ▼             ▼
    VALID         BLOCKED      Use LLM Judge
 (ignore LLM)                   Results
```

This architecture eliminates latency from LLM judge escalation since the judges are already running.

## Deployment and HTTPS

### Self-Signed Certificates (Development)

Self-signed certificates are pre-generated in `.ssl/` and configured in `.streamlit/config.toml`. To regenerate:

```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout .ssl/private.key \
  -out .ssl/certchain.pem \
  -days 365 -nodes \
  -subj "/CN=localhost/O=RAG-LLM/C=LU"
```

> **Note:** Browsers will show a security warning for self-signed certificates. Accept the warning to proceed.

### Production HTTPS Options

**Option A – Streamlit native SSL**  
Edit `.streamlit/config.toml` and set the paths to your PEM certificate and key (e.g. Let's Encrypt):
```toml
[server]
sslCertFile = "/path/to/certchain.pem"
sslKeyFile = "/path/to/private.key"
enableCORS = false
enableXsrfProtection = true
```
Then run `streamlit run streamv3.py`; the app will serve over HTTPS.

**Option B – Reverse proxy (recommended for production)**  
Run Streamlit on HTTP (e.g. `localhost:8502`) and put a reverse proxy (nginx or Caddy) in front with TLS termination, proxying to `http://127.0.0.1:8502`. Add secure headers (e.g. `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options`) in the proxy config.

# 2. Using and choosing a LLM model

## BeamStudio (Cloud LLMs)

The application supports BeamStudio cloud-hosted LLMs via the `llm_client.py` abstraction:

| Model | Description | Notes |
|-------|-------------|-------|
| **gpt-5.1** | Latest reasoning model (default) | High token limit, best quality |
| **gpt-4o** | Fast, efficient model | Good balance of speed/quality |
| **gpt-5-mini** | Lightweight reasoning model | Temperature must be 1 (API constraint) |

Select a model in the UI sidebar under "LLM Selection". BeamStudio requires setting the `BEAMSTUDIO_API_KEY` environment variable.

## Local LLMs

In this part, I will introduce both LM-studio and HuggingFace's transformers library to access frozen LLM model (i.e. fixed weights).

## LM-Studio
You can find LM-Studio on their website [here](https://lmstudio.ai/). Their product is a local AI toolkit meant for easy acess to multiple LLM models. The main strengh of LM-Studio was it's flexibility (it's possible to run multiple LLM at the same time) and it's ease of use, calling the LLM through a simple API.

However, during the testing phase of our models we found that the LM-Studio local server would juste crash and stop giving answers to the API calls. This would stop the flow of the python scripts, as they would wait for an answer that wasn't coming. You can find similar issues like [this](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/267) on their github pages. While waiting for a fix, we decided to switch to another solutions for accessing the LLMs.

## Hugging Face

Pipeline is a function from the transformers library developped by HuggingFace. It works like an API, and allows us to use pretrained model for inference (i.e. we can't retrain them). This is perfect as all the public model from the HuggingFace website can be used this way.
You can find the Pipeline url [here](https://huggingface.co/docs/transformers/en/main_classes/pipelines) but I will explain here how it works and how to use it. 

In order to access and use a particular model, you first need to create a pipeline object like this :  
```python
pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B", trust_remote_code = True)
```
Here the type of model and the model name can both be found on the 'Model' page of the HuggingFace website ([here](https://huggingface.co/models)). Then, depending on the type of model, you need to feed the 'pipe' object different type of messages. Here, for the 'Qwen2-0.0.5B' model the message would look like this : 

```python
messages = [
            {"role": "user", "content": ...},
            ]
answer=pipe(messages)
```

In terms of the actual code, the pipeline object is stored as an attribute of the RAG class, and the message format is defined in the answer method of the same RAG class. 

### Using custom LLM Models

You can still use custom models within the pipeline libraries, using a custom Pipeline class. You can get more info [here](https://huggingface.co/docs/transformers/v4.30.0/add_new_pipeline). 
This hasn't been tested, but you should still be able to add a pipeline object in the RAG. 


# 3. Embedding functions 

All embedding functions must be present in the 'embedding.py' file as a class with 2 main methods : retrieve and process. Retrieve is used to get the phrases from the documents relevant to the query. It must return a dict with the documents as a list of strings, and the score of said documents. You must have some way of sending back only relevant doc with a threshold on the minimal score to send back documents.
The second method is used to preprocess documents and store them locally, either as an index of as a file with all the embeddings.
The following class is a template on what input to take in and what to output for newer methods, but you don't need to use it as a parent class.

```python
class Method():
    def __init__(self, k=5):
        self.k = k

    def retrieve(self, query : str , path = None, doc = None) -> dict:
        '''
        Input : 
        query : str -> Answer to give answer to
        path : str -> Where documents or embeddings are stored locally
        doc : ... -> list of documents or document to extract context from

        Return : 
        dict with 2 keys : 
            - 'doc' with a list of str, corresponding to the context
            - 'score' the similarity scores, used to set a threshold of minimum similarity to accept the context
        ''' 
        return ['']
    
    def process(self, doc = None, path = None, get = False) : 
        '''
        Input : 
        path : str -> Where to load the documents from. 
        doc : ... -> list of documents or document to pre-process
        Return : 
        None if get==False else (documents embeddings, docs)
        ''' 
        return None
```

Note that you also need to add a line in the 'RagV2.py' file, in the '__init__' method of the RAG class, in order to be able to use your method from the front.


## Common Errors

### Short answer / repetitive answer from the LLM

I couldn't find any related issue on the hugging face forum or on stack overflow. I believe it comes from the small size of the LLMs used, as too much context could explain the poor performance. I could also be because of the lack of information in the training dataset about the question asked. If this is the issue, a bigger model would solve it.

What was tried : 
- Using a QA model
- Increasing the 'max_token' pipeline argument
 

## TODO and Ideas moving forward

One of the main limitation of this work (and of RAG in general) is the lack of image processing in the whole pipeline. We had a few ideas, like generating an image description through a third-party (as unrelated to the rest of the RAG) model to add it to the document databases (either in the document index or in the embedding vector file). Another way to treat images   


        
