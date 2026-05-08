"""Quick analysis of character_cache.json to find aliases."""
import json, sys, collections
sys.stdout.reconfigure(encoding="utf-8")

cache = json.loads(open("chroma_novel/character_cache.json", encoding="utf-8").read())

counter = collections.Counter()
for names in cache.values():
    for n in names:
        counter[n] += 1

print(f"Total unique names: {len(counter)}\n")
print("All names (count >= 5), sorted by count:")
for name, cnt in counter.most_common():
    if cnt < 5:
        break
    print(f"  {cnt:5d}  {name}")
