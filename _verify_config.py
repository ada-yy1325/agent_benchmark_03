#!/usr/bin/env python3
"""Verify the prompt format and answer extraction."""
import re

FEW_SHOT_EXAMPLES = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
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

def build_prompt(question: str) -> str:
    return f"{FEW_SHOT_EXAMPLES}\n\nQuestion: {question}\nAnswer:"

def extract_answer(text: str) -> str | None:
    # Priority 1: "Final answer: X"
    match = re.search(r'Final answer:\s*\$?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).rstrip(".")
    # Priority 2: "The answer is X"
    match = re.search(r'The answer is\s*\$?\s*([+-]?\d+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).rstrip(".")
    # Priority 3: \boxed{X}
    match = re.search(r'\\?boxed\{\$?\s*([+-]?\d+\.?\d*)\}', text)
    if match:
        return match.group(1)
    return None

# Test: verify prompt
test_q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
prompt = build_prompt(test_q)

print("=== PROMPT VERIFICATION ===")
print(f"Total prompt length: {len(prompt)} chars = ~{len(prompt)//4} tokens")
print(f"Number of examples: {prompt.count('Question:')} (should be 5 = 4 shots + 1 test)")
print(f"Test question format ends with: ...{prompt[-40:]}")
print()

# Test: extract_answer on simulated model outputs
test_outputs = [
    ("Final answer: 72", "72"),
    ("Answer: April = 48, May = 24, total = 72\nFinal answer: 72", "72"),
    ("Let's think step by step. 48 + 24 = 72. The answer is 72.", "72"),
    ("The answer is $72.", "72"),
    ("\\boxed{42}", "42"),
    ("Final answer: 8.", "8"),
    ("no answer here", None),
]

print("=== ANSWER EXTRACTION VERIFICATION ===")
all_pass = True
for output, expected in test_outputs:
    result = extract_answer(output)
    status = "✓" if result == expected else f"✗ (got {result!r}, expected {expected!r})"
    if result != expected:
        all_pass = False
    print(f"  {status}: {output[:60]:60s} → {result}")

print()
print(f"ALL TESTS PASS: {all_pass}")