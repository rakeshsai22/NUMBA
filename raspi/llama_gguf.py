import logging
# import llama_cpp
from llama_cpp import Llama
import time
import json
import inspect

# Configure logging

def run_inference(sequence, echo, temperature, max_tokens=4096, top_p=0.95, top_k=40):
    start_time = time.time()
    
    try:
        output = llm(
            sequence,
            echo=echo,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        completion_tokens = output.get('usage', {}).get('completion_tokens', 0)
        prompt_tokens = output.get('usage', {}).get('prompt_tokens', 0)
        total_tokens = output.get('usage', {}).get('total_tokens', 0)
        
        # Print inference details to terminal
        print(json.dumps({
            "sequence": sequence[:50] + "...",
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "elapsed_time": elapsed_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }, indent=4))
        
        print(f"\nGenerated Response:")
        print(output['choices'][0]['text'])
        
        return output
    
    except Exception as e:
        print("Error during inference:", e)
        return None

print("Loading the model...")
try:
    llm = Llama(
        model_path="/home/rise/Downloads/llama-2-7b-chat.Q4_0.gguf",
        n_ctx=4096,
        # n_batch=256,
        n_threads=4
    )
    print("Model Loaded Successfully")
    
    # Print model parameters to terminal
    print(json.dumps({
        "model_path": "/home/rise/Downloads/llama-2-7b-chat.Q4_0.gguf",
        "n_ctx": llm.n_ctx,
        "n_batch": llm.n_batch,
        "n_threads": llm.n_threads
    }, indent=4))
    
except Exception as e:
    print("Error loading the model:", e)

# Define test sequences
test_sequences = [
    "Q: What states that treaties are to be interpreted in good faith according to the ordinary meaning given to the terms of the treaty in their context and in light of its object and purpose? Answer:",
    # "Q: Can you tell me about the theory of uncertainty in detail? Answer:",
    "Q: Can you tell me about the theory of uncertainty in detail? Answer:",
    "Q: Explain the laws of thermodynamics in simple terms. Answer:",
    "Q: Describe the differences between supervised and unsupervised learning. Answer:",
]

temperatures = [1.0]

for idx, sequence in enumerate(test_sequences):
    print(f"\n{'='*50}")
    print(f"Running Test Sequence {idx + 1}")
    print(f"{'='*50}")
    
    for temp in temperatures:
        run_inference(
            sequence=sequence,
            echo=True,
            temperature=temp,
            max_tokens=4096,
            top_p=0.95,
            top_k=40
        )
