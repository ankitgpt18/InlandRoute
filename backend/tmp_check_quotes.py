import sys

path = r'c:\Users\ankit\OneDrive\Desktop\InlandRoute\backend\app\services\model_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

with open('line_quotes.txt', 'w') as out:
    for i, line in enumerate(content.splitlines()):
        if '"""' in line:
            out.write(f"Line {i+1}: {line.strip()}\n")

triple_q = content.count('"""')
triple_s = content.count("'''")
single_q = content.count('"')
single_s = content.count("'")

print(f"Triple Double Quotes: {triple_q}")
print(f"Triple Single Quotes: {triple_s}")
print(f"Single Double Quotes: {single_q}")
print(f"Single Single Quotes: {single_s}")

if triple_q % 2 != 0:
    print("WARNING: Imbalanced Triple Double Quotes")
if triple_s % 2 != 0:
    print("WARNING: Imbalanced Triple Single Quotes")

# For single quotes, we need to be careful as they might be inside triple quotes
# But usually, it shouldn't be too hard to find the culprit.
