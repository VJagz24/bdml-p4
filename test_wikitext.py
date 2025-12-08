from datasets import load_dataset

print("Loading WikiText-103...")
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

print(f"✓ Dataset loaded!")
print(f"  Examples: {len(dataset):,}")
print(f"  First 200 chars: {dataset[0]['text'][:200]}")