#!/usr/bin/env python3
"""
Train Multi‑Layer Time Models, measure the energy injected by the multiplicative
term γ·x/σ in ScalarScaleLayerNorm, and plot accuracy versus raw and modified
energy.  Also saves each trained model and writes results to JSON.
"""
# ───────────────────────────────────────────────────────── imports & globals
import json, math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

import network_definitions_noenc as nd
from physics_copy import PhysicalConstants

# Globals used by ScalarScaleLayerNorm to accumulate ΔE_gain
ENERGY_FACTOR  = 1.0        # set before evaluation  (t_end/t1_steps)/R
ENERGY_ACCUM   = 0.0        # running sum of ΔE_gain over all samples

# ───────────────────────────── patch ScalarScaleLayerNorm (simpler than hooks)
def ssln_forward(self, x):
    """
    Original behaviour: y = γ·x̂ + β  where x̂=(x-μ)/σ.
    Extra bookkeeping: measure energy change due to the scaling term γx/σ,
    multiply by ENERGY_FACTOR, and accumulate into globals.
    """
    global ENERGY_ACCUM, ENERGY_FACTOR

    # stats over the last `n` dims
    n = len(self.normalized_shape)
    axes = tuple(range(-n, 0))
    mu   = x.mean(dim=axes, keepdim=True)
    var  = x.var (dim=axes, unbiased=False, keepdim=True)
    sigma = torch.sqrt(var + self.eps)

    # --- multiplicative part only
    scaled = (self.gamma / sigma) * x
    pre_E  = (x     ** 2).flatten(1).sum(1)   # (B,)
    post_E = (scaled** 2).flatten(1).sum(1)
    delta  = (post_E - pre_E) * ENERGY_FACTOR # (B,)

    ENERGY_ACCUM += delta.sum().item()

    # --- normal LayerNorm output
    x_hat = (x - mu) / sigma
    return self.gamma * x_hat + self.beta

# Monkey‑patch
nd.ScalarScaleLayerNorm.forward = ssln_forward

# ───────────────────────────────────────────────────────── evaluation function
def evaluate(model, dataloader, E_target, t_end, t1_steps, const, device):
    """
    Runs model on `dataloader`, returns (accuracy, ⟨ΔE_gain⟩).
    The global accumulator inside ScalarScaleLayerNorm does the ΔE bookkeeping.
    """
    global ENERGY_FACTOR, ENERGY_ACCUM
    ENERGY_FACTOR = (t_end / t1_steps) / const.R
    ENERGY_ACCUM  = 0.0

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs = imgs.view(imgs.size(0), -1)
            
            # interpolate
            imgs_inter = actenc(imgs, 1)

            s2 = torch.sum(imgs_inter**2, 1)
            alpha = torch.sqrt(E_target / ((s2 / const.R) * (t_end / t1_steps)))
            imgs_scaled = imgs * alpha.unsqueeze(1)

            out = model(imgs_scaled, alpha)
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)

    accuracy       = 100. * correct / total
    avg_delta_E    = ENERGY_ACCUM / total
    return accuracy, avg_delta_E
    
# ─────────────────────────────────────────────────────────────── main program

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
const        = PhysicalConstants()

if int(sys.argv[1])==49:
    net_configs = [[49, 32, 16, 10]]
elif int(sys.argv[1])==196:
    net_configs = [[196, 64, 32, 10]]
elif int(sys.argv[1])==784:
    net_configs = [[784, 100, 100, 10]]

# net_configs  = [[49, 32, 16, 10]]

run_start = int(sys.argv[3])

energy_lvls  = np.array([10**(-float(sys.argv[2]))])
# energy_lvls = np.logspace(-14, -10, 5)
batch_size   = 64
lr           = 3e-4
epochs       = 10
runs_per_E   = 1
bandwidth    = 1e9

train_raw, test_raw = nd.get_full_mnist_datasets()
out_dir = Path("models"); out_dir.mkdir(exist_ok=True)
results = {}

for dims in net_configs:
    in_dim, first_hid = dims[0], dims[1]
    t_steps = in_dim * first_hid
    trainloader, testloader = nd.build_dataloaders_for_inputdim(
        train_raw, test_raw, in_dim, batch_size
    )
    t_end = t_steps / bandwidth
    
    actenc = nd.ActivationEncoding(n_t=t_steps, input_dim=in_dim)

    raw_Es, mod_Es, accs = [], [], []

    for E in energy_lvls:
        for run in range(run_start, run_start+1):
            model = nd.MultiLayerTimeModel(dims, t_steps, bandwidth).to(device)
            torch.manual_seed(42 + run)

            opt  = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            for ep in range(1, epochs+1):
                model.train(); epoch_loss = n_seen = 0
                for imgs, lbl in trainloader:
                    imgs, lbl = imgs.to(device), lbl.to(device)
                    imgs = imgs.view(imgs.size(0), -1)
                    
                    # interpolate
                    imgs_inter = actenc(imgs, 1)

                    s2 = torch.sum(imgs_inter**2, 1)
                    alpha = torch.sqrt(E / ((s2 / const.R) * (t_end / t_steps)))
                    imgs_scaled = imgs * alpha.unsqueeze(1)

                    opt.zero_grad()
                    out = model(imgs_scaled, alpha)
                    loss = loss_fn(out, lbl)
                    loss.backward(); opt.step()

                    epoch_loss += loss.item() * imgs.size(0); n_seen += imgs.size(0)

                # quick val accuracy each epoch
                acc_ep, _ = evaluate(model, testloader, E, t_end, t_steps, const, device)
                print(f"{dims} | E={E:.1e} run={run} "
                      f"epoch {ep}/{epochs}  loss={epoch_loss/n_seen:.4f}  acc={acc_ep:.2f}%")

            # final evaluation (also gets ΔE_gain)
            acc, dE = evaluate(model, testloader, E, t_end, t_steps, const, device)
            mod_E = E + dE
            raw_Es.append(E); mod_Es.append(mod_E); accs.append(acc)
            print(f"Final  ⟨ΔE_gain⟩={dE:.3e}  Modified_E={mod_E:.3e}  Acc={acc:.2f}%")

            # ---------- save model ----------
            
            fname = (f"model_layers{len(dims)-1}_dims{'-'.join(map(str,dims))}"
                     f"_E{E:.2e}_run{run}.pth")
            torch.save(model.state_dict(), out_dir / fname)

            results["-".join(map(str, dims))] = (raw_Es, mod_Es, accs)

            with open(f"testaccjsons/testaccs{len(dims)-1}_dims{'-'.join(map(str,dims))}_E{E:.2e}_run{run}.json", 'w') as f:
                json.dump(results, f, indent=2)


