#!/usr/bin/env python3
"""Fix: Replace `print(flush=True, ...)` with `print(..., flush=True)` everywhere."""
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("print(flush=True,"):
        indent = line[: len(line) - len(stripped)]
        # Content after "print(flush=True, "
        rest = stripped[len("print(flush=True,") :].lstrip()
        # Find matching closing paren
        depth = 0
        split_idx = None
        for i, ch in enumerate(rest):
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    split_idx = i
                    break
                depth -= 1
        if split_idx is not None:
            inner = rest[:split_idx]
            closing = rest[split_idx]
            new_line = indent + "print(" + inner + ", flush=True)" + rest[split_idx + 1 :]
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(path, "w") as f:
    f.writelines(new_lines)

print(f"Fixed: {path}")