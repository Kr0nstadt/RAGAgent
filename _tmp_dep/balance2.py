text = open('_tmp_dep/build_plan.py', encoding='utf-8').read()
lines = text.split('\n')

# Find lines with double `},` patterns where they appear immediately after another close
import re

# Just regex for },  followed by , or newline
# Look for },  } on adjacent lines (more reliably: }, on two consecutive lines where the inner is just whitespace)
for i in range(len(lines) - 1):
    cur = lines[i].rstrip()
    nxt = lines[i+1].rstrip()
    if cur == '},' and nxt == '},':
        print(f"L{i+1}-{i+2}: double '}},' sequence")
        # show context
        for k in range(max(0, i-2), min(len(lines), i+4)):
            print(f"  {k+1}: {lines[k]!r}")
        print()
