from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

print("Checking first 10 examples:")
print("="*50)

for i in range(10):
    text = dataset[i]['text']
    print(f"\nExample {i}:")
    print(f"  Length: {len(text)}")
    print(f"  Content: '{text[:100]}'")
    if len(text) > 0:
        print(f"  ✓ Has content!")
        break