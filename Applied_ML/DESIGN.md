#Healthcare RAG Pipeline

#Overview and Why I Built It This Way 

I built this RAG pipeline to run entirely locally. Sending unencrypted medical queries to a cloud API like OpenAI is a massive privacy risk, so the whole architecture is designed to stay on-device. The embedding generation (all-MiniLM-L6-v2) and the actual text generation (llama3.2) both execute on my local RTX 5050. It removes network latency and means there is zero data exfiltration.

1. How the Retrieval Actually Works Instead of using basic keyword matching, I used ChromaDB to map the MedQuAD dataset into a dense vector space.
•	It works by calculating the cosine similarity between the user's prompt and the document chunks.
•	During validation, this got a 100% precision hit rate (8 for 8 on the test queries) because it understands semantic context, so it doesn't get confused between things like Type 1 and Type 2 diabetes.
•	The system pulls the top 5 most relevant chunks and forces the LLM to only answer using that specific context to prevent hallucinations.

2. The Safety "Kill Switch" LLMs are terrible at handling active emergencies, so I didn't even let them try. I built a deterministic intercept layer that runs before the prompt even reaches Ollama. If someone types in high-risk keywords (like stroke symptoms, heart attack, or overdose), the pipeline hits a hard block. It zeroes out the confidence score to 0.0000 and immediately spits out a hardcoded warning to call 911. In the evaluation suite, this blocked 5 out of 5 adversarial emergency prompts perfectly. It saves compute time and stops the AI from giving bad advice during an actual crisis.

3. Hardware Constraints Running a transformer embedding model and a local LLM simultaneously is heavy on memory. I routed Llama 3.2 through the Ollama backend so it utilizes quantized weights. If I didn't do this, running inference while querying the database would have completely blown past the 8GB VRAM limit on my laptop GPU and crashed the script.

4. Where the System Fails (Limitations) The system works for the baseline tests, but I know it has a few weak points:

•	Static Data: The ChromaDB is just a snapshot of the CSV. If medical guidelines change tomorrow, the database is blind to it until I write a script to re-ingest everything.
•	Chunking issues: Because the text is split into fixed chunks, it's totally possible that a chunk cuts off right in the middle of a list of medication side effects. The LLM would only see half the context.
•	Hardcoded Confidence: The threshold for what counts as a "high confidence" match (0.75) is just an arbitrary number I picked based on testing. It really should be dynamic depending on how serious the medical question is.
