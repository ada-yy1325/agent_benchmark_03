import sys

with open('/Users/yy1325/创智学院实习/test2/agent_benchmark_03/_start_dsv4.py') as f:
    lines = f.readlines()

# Lines 93-132 currently have the monkey-patch in wrong location (after LLM(**config))
# We need to restructure so:
# 1. Keep lines 0-92 (imports, check_download, dummy_load)
# 2. Replace lines 93-132 with correct ordering
# 3. Keep lines 133+ 

# Find the try_load function's line range
# try_load starts at line where 'def try_load():' appears
# try_load ends before 'if __name__ == "__main__":'

# Current structure of try_load (lines 93-index in original):
# Actually, let me just check if the monkey-patch is in the wrong place
# by looking for '# Monkey-patch:' that lacks indentation

for i, l in enumerate(lines):
    if l.strip().startswith('# Monkey-patch:') and not l.startswith(' ' * 4):
        print(f"Found bad monkey-patch at line {i+1}: {repr(l)}")
        # Find the block boundaries
        # The bad block starts at this line
        bad_start = i
        # Find the end - it's the line after 'print("[INFO] No patch needed: {e}")'
        for j in range(i, min(i + 30, len(lines))):
            if 'No patch needed' in lines[j]:
                bad_end = j + 1  # include this line
                break
        else:
            print("Could not find end of bad block")
            sys.exit(1)
        
        print(f"Bad block: lines {bad_start+1} to {bad_end}")
        
        # Remove the bad block
        del lines[bad_start:bad_end]
        
        # Now find where to insert: after 'from vllm import LLM, SamplingParams'
        for j, l2 in enumerate(lines):
            if 'from vllm import LLM, SamplingParams' in l2 and 'try_load' not in l2:
                # Find the try_load version
                # Count from here to find the right context
                pass
        
        # Better approach: find the line with 'from vllm import' inside try_load
        # by checking context around it
        for j in range(len(lines)):
            if 'from vllm import LLM, SamplingParams' in lines[j]:
                # Check if this is inside try_load by looking at preceding lines
                # dummy_load and try_load both have this import
                # Let's look a few lines before
                if j > 0 and '    print()' in lines[j-1] and '    \\n' in lines[j]:
                    # This is the try_load version (dummy_load has different preceding context)
                    pass
        
        break
else:
    print("No bad monkey-patch found")

# Let me just use a simpler approach: read the whole file and replace the bad section
with open('/Users/yy1325/创智学院实习/test2/agent_benchmark_03/_start_dsv4.py') as f:
    content = f.read()

# Find the problematic section
old = '''    llm = LLM(**config)\n# Monkey-patch'''
if old in content:
    print("Found problem - no indentation on monkey-patch line")
    content = content.replace(
        '# Monkey-patch: skip Float4 format_cast unsupported on 910B2C',
        '    # Monkey-patch: skip Float4 format_cast unsupported on 910B2C'
    )
    with open('/Users/yy1325/创智学院实习/test2/agent_benchmark_03/_start_dsv4.py', 'w') as f:
        f.write(content)
    print("Fixed indentation")
else:
    print("Old pattern not found, trying alternative...")
    # Check if the monkey-patch is at end of try_load
    idx = content.find('from vllm import LLM, SamplingParams')
    # Find second occurrence (try_load)
    idx2 = content.find('from vllm import LLM, SamplingParams', idx + 1)
    print(f"Second import at position {idx2}")
    print(f"Next 200 chars: {repr(content[idx2:idx2+200])}")
