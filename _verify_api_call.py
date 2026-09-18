#!/usr/bin/env python3
"""Test: Make one real API call to see actual model output format."""
import re, json, requests, sys

API_URL = "http://127.0.0.1:8802/v1/completions"

FEW_SHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6 trees planted.
Final answer: 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
Final answer: 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have total in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
Final answer: 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave 20 - 12 = 8 lollipops to Denny.
Final answer: 8"""

TEST_QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
]

def extract_answer(text):
    m = re.search(r'Final answer:\s*\$?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if m: return m.group(1).rstrip('.')
    m = re.search(r'The answer is\s*\$?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if m: return m.group(1).rstrip('.')
    return None

print("=" * 60)
print("REAL API CALL TEST - Model Output Format Check")
print("=" * 60)

for i, q in enumerate(TEST_QUESTIONS):
    prompt = f"{FEW_SHOT}\n\nQuestion: {q}\nAnswer:"
    payload = {
        "model": "Qwen3-4B-Base",
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }
    print(f"\n--- Question {i+1} ---")
    print(f"Q: {q[:60]}...")
    
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["text"]
        finish = data["choices"][0].get("finish_reason", "unknown")
        
        print(f"Finish reason: {finish}")
        print(f"Output length: {len(text)} chars = ~{len(text)//4} tokens")
        print(f"Output (first 500 chars):")
        print("-" * 50)
        print(text[:500])
        print("-" * 50)
        
        ans = extract_answer(text)
        print(f"Extracted answer: {ans}")
        
        # Check if output contains "Final answer:" pattern
        if "Final answer:" in text.lower():
            idx = text.lower().index("final answer:")
            print(f"Final answer position: char {idx}")
        elif "The answer is" in text:
            idx = text.index("The answer is")
            print(f"The answer is position: char {idx}")
        else:
            print("WARNING: No recognizable answer pattern found!")
            
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)