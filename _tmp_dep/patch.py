text = open('_tmp_dep/build_plan.py', encoding='utf-8').read()
lines = text.split('\n')

# Remove the extra '},' on line 1637 (index 1636)
# line 1637: '            },'
if lines[1636].strip() == '},':
    lines[1636] = ''  # remove entirely
    print("Removed extra '},' at line 1637")
else:
    print(f"Line 1637 was: {lines[1636]!r} — not removed")

text2 = '\n'.join(lines)
open('_tmp_dep/build_plan.py', 'w', encoding='utf-8').write(text2)
print("OK")
