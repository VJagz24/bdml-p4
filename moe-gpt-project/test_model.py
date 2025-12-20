import torch
from model import GPT, GPTConfig

print("Creating 5M parameter model...")
config = GPTConfig()  # Uses default 5M config
model = GPT(config)

print(f"\n✓ Model created successfully!")

# Test forward pass
print("\nTesting forward pass...")
x = torch.randint(0, config.vocab_size, (2, 32))
with torch.no_grad():
    logits, loss = model(x, targets=x)

print(f"✓ Forward pass works!")
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Loss: {loss.item():.4f}")
print("\n🚀 Ready to train!")