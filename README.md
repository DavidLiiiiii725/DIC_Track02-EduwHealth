# EduwHealth
Repository for NYUSH DIC 2025 mental health and education track

## Overview
This project is a fully local tutoring system for machine learning beginners.  
You run everything on your own machine.  
No cloud services.  
No remote APIs.

The system combines retrieval augmented generation, multi agent reasoning, and safety awareness.  
You control the models, the data, and the behavior.

## What you get
- A local tutor for machine learning fundamentals  
- Retrieval augmented generation using your own knowledge base  
- Multiple agents with clear roles  
- Risk detection powered by a local language model  
- Offline execution from end to end  

## How it works
You ask a question.  
The system retrieves relevant knowledge from a vector store.  
Agents reason in parallel.  
A risk model evaluates the message.  
You receive a grounded response.

## System architecture
- Vector store built with FAISS  
- Knowledge stored as embedded text chunks  
- RAG node retrieves context before generation  
- LangGraph coordinates agent execution  
- Local language model performs reasoning and analysis  
- Risk model evaluates emotional and safety signals  

## Agents
### Tutor
- Explains concepts clearly  
- Prioritizes retrieved knowledge  

### Coach
- Supports motivation and learning persistence  
- Encourages autonomy, competence, and relatedness  

### Critic
- Reviews responses for safety  
- Flags risky patterns  

## Risk model
Risk scoring runs on every user message.  
Feature extraction uses a local language model.  
No keyword lists.  
No hand written lexicons.

The model outputs:
- risk_score from 0 to 1  
- risk_level as low, medium, or high  
- reasons for transparency and debugging  

## Knowledge base
You store learning material in a single raw text file.  
You build embeddings once.  
You reuse the vector index on every run.

## Workflow
1. Write learning content into kb_raw/ml_intro.md  
2. Run the build script to create the vector index  
3. Start the tutor application  
4. Ask questions  

## Key files
- kb_raw/ml_intro.md  
- scripts/build_vector_kb.py  
- memory/vector_store.py  
- agents/rag_node.py  
- analytics/feature_extractor_llm.py  
- analytics/risk_model.py  
- core/orchestrator.py  

## Local models
You choose the models.  
Ollama works well.  
CUDA acceleration supported.  
CPU fallback supported.

## Hardware support
- GPU support for embedding and inference  
- Large RAM support for fast indexing  
- NPU optional for future extensions  

## Design goals
- Full local control  
- Transparent reasoning  
- Deterministic data flow  
- Reproducible behavior  
- Research friendly structure  

## Who this is for
You study machine learning.  
You build agent systems.  
You want local execution.  
You care about safety and grounding.

## Next steps
- Add more course content  
- Expand the knowledge base  
- Train a learned risk model  
- Add evaluation scripts  
- Add a user interface  
