
import torch
from spike_mamba.main import create_spiking_mamba_model

# Create SpikingMamba model
model = create_spiking_mamba_model(
    d_model=512,
    n_layer=6,
    vocab_size=1000,
    spike_mode="pre_post",
    threshold=1.0,
    tau_mem=20.0,
    adaptive_threshold=True,
    spike_regularization=0.01
).to('cuda')

# Test forward pass
batch_size, seq_len = 2, 64
input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to('cuda')

with torch.no_grad():
    output = model(input_ids, return_spike_stats=True)
    print(output)
