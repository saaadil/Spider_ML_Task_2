# Spider ML Task 2 Submission

Hey, this is Aadil's repo for the Machine Learning Task 2 induction. 

I split the codebase into two main parts: the Base ML track (where I built the neural nets manually without high-level wrappers) and the Applied ML track (the local Healthcare RAG pipeline). 

Everything here was built, trained, and tested locally on my laptop (RTX 5050, 8GB VRAM). Because of that, you'll notice a lot of my design choices revolve around surviving that memory limit without crashing.

## Where to find my analysis
Instead of dumping a massive wall of text here, I wrote a specific breakdown for each level. They explain my math, why I chose certain architectures, and where my code actually fails. Please read these before looking at the raw code:

### 1. Base ML 
* **Level 1 (Custom ResNet):** [Read my analysis here](./Base_ML/Level_1_ResNet/ANALYSIS.md)
* **Level 2 (Custom LSTM):** [Read my analysis here](./Base_ML/Level_2_LSTM/ANALYSIS.md)
* **Level 3 (Transformer vs LSTM):** [Read the comparison here](./Base_ML/Level_3_Transformer/ANALYSIS.md)

### 2. Applied ML
* **Healthcare RAG Pipeline:** [Read the system design here](./Applied_ML/DESIGN.md)

## How to run it
Make sure you activate the virtual environment first so the dependencies don't mess up your system.

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt