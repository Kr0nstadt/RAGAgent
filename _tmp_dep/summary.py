import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open('task_data/подробное_описание_БП_по_блокам_с_дополнением_1-0b72f01f4f/agent_process_plan.json', encoding='utf-8'))

out = []
out.append(f"Top-level keys: {list(d.keys())}")
out.append(f"blocks count: {len(d['blocks'])}")
for b in d['blocks']:
    out.append(f"  block {b['block_id']}: {b['title']} ({len(b['steps'])} steps)")
out.append(f"open_gaps_summary ({len(d['open_gaps_summary'])}):")
for x in d['open_gaps_summary']:
    out.append(f"  - {x}")
out.append(f"graph_gaps_summary ({len(d['graph_gaps_summary'])}):")
for x in d['graph_gaps_summary']:
    out.append(f"  - {x}")
out.append(f"open_assumptions ({len(d['confirmed_decisions_reference']['open_assumptions'])}):")
for x in d['confirmed_decisions_reference']['open_assumptions']:
    out.append(f"  - {x}")
out.append(f"ui_path_vocabulary ({len(d['ui_path_vocabulary'])}):")
for x in d['ui_path_vocabulary']:
    out.append(f"  - {x}")

open('_tmp_dep/summary.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out)[:5000])
