text = open('_tmp_dep/build_plan.py', encoding='utf-8').read()
stack = []
in_string = None
escape = False

for i, ch in enumerate(text):
    if in_string:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == in_string:
            in_string = None
        continue
    if ch in ('"', "'"):
        in_string = ch
        continue
    if ch == '#':
        # comment until end of line (in python source)
        nl = text.find('\n', i)
        if nl == -1:
            break
        i = nl
        continue
    if ch == '{':
        stack.append(i)
    elif ch == '}':
        if stack:
            stack.pop()

if stack:
    print("Unclosed opens at chars:")
    for c in stack:
        ln = text[:c].count(chr(10))+1
        ln_text_start = text.rfind('\n', 0, c) + 1
        print(f"  line {ln} (col {c - ln_text_start}): ...{text[max(0,c-80):c+50]}...")
else:
    print("All balanced")
