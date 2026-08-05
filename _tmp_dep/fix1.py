text = open('_tmp_dep/build_plan.py', encoding='utf-8').read()
lines = text.split('\n')

# Find the exact problematic pattern
# Line 1635: '                            "вывод по аналогии"},'
# Line 1636: '            },'
# Replace: remove line 1636 entirely

# Find by pattern
target_idx = None
for i, l in enumerate(lines):
    if 'вывод по аналогии' in l:
        print(f"Found at {i+1}: {l!r}")
        # Check next
        if i+1 < len(lines):
            print(f"  Next: {lines[i+1]!r}")
            if lines[i+1].strip() == '},':
                # delete next line
                lines[i+1] = ''  # blank out
                target_idx = i+1
                print(f"  -> removed next line")

text2 = '\n'.join(lines)
open('_tmp_dep/build_plan.py', 'w', encoding='utf-8').write(text2)
print("OK")
