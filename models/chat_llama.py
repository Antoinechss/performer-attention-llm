"""
Simple chat interface for TinyLlama model
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def chat():
    print("Loading TinyLlama model...")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Create a streamer for real-time output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("TinyLlama Chat (type 'quit' to exit)")
    print("="*50 + "\n")
    
    # Chat loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Format the prompt for chat
        prompt = f"<|user|>\n{user_input}</s>\n<|assistant|>\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response with streaming
        print("Assistant: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer  # Enable streaming output
            )
        
        print()  # New line after generation


if __name__ == "__main__":
    chat()
