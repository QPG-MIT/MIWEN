#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import network_definitions as nd  # (Your file with MultiLayerTimeModel, etc.)
from physics_copy import PhysicalConstants

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Settings
    bandwidth = 1e9
    opamp_power = 1e-5
    timebin = 1 / bandwidth
    numoflayers_list = [3, 3]  # for 49 and 196 inputs
    numtimebins_list = [49 * 32, 196 * 64]
    layer_dims_list = ["49-32-16-10", "196-64-32-10"]
    # colors = ['tab:blue', 'tab:orange']
    colors = ['tab:blue', 'tab:orange', '#004080', '#cc5500']

    # Energies to load
    E_targets = np.logspace(-21, -5, 17)
    run_indices = range(1, 6)

    # -----------------------
    # NEW PLOTTING ROUTINE
    # -----------------------
    # Create two subplots: top = no-amp (vs E_targets), bottom = with-amp (vs all_Es)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # side-by-side

    for idx, (numoflayers, numtimebins, layer_dims_str) in enumerate(
            zip(numoflayers_list, numtimebins_list, layer_dims_list)):
        opamp_energy = opamp_power * numoflayers * numtimebins * timebin
        layer_dims = list(map(int, layer_dims_str.split('-')))

        # Store values per energy level
        median_accs, q1_accs, q3_accs = [], [], []
        all_Es = []

        for E_target in E_targets:
            accs = []
            toteners = []
            for run_idx in run_indices:
                filename = f"testaccjsons/testaccs{len(layer_dims) - 1}_dims{'-'.join(map(str, layer_dims))}_E{E_target:.2e}_run{run_idx}.json"
                if not os.path.exists(filename):
                    print(f"Missing file: {filename}")
                    continue

                # with open(filename, 'r') as f:
                #     dicttosave = json.load(f)
                #     accs.append(dicttosave["accuracies"][0])

                with open(filename, 'r') as f:
                    dicttosave = json.load(f)
                    accs.append(dicttosave[f"{'-'.join(map(str, layer_dims))}"][2][0])
                    toteners.append(dicttosave[f"{'-'.join(map(str, layer_dims))}"][1][0])

            if accs:
                median_accs.append(np.median(accs))
                q1_accs.append(np.percentile(accs, 25))
                q3_accs.append(np.percentile(accs, 75))
                totenermed = np.median(np.array(toteners))
                all_Es.append(totenermed + opamp_energy)

        all_Es = np.array(all_Es)
        median_accs = np.array(median_accs)
        q1_accs = np.array(q1_accs)
        q3_accs = np.array(q3_accs)

        # Bottom subplot: with op-amp energy (accs vs all_Es)
        ax2.errorbar(
            all_Es, median_accs,
            yerr=[median_accs - q1_accs, q3_accs - median_accs],
            fmt='o', linestyle='none', label=f"{layer_dims_str}",
            capsize=4, color=colors[idx+2], markersize=10
        )

        # Top subplot: no op-amp energy (accs vs E_targets)
        line = ax1.errorbar(
            E_targets, median_accs,
            yerr=[median_accs - q1_accs, q3_accs - median_accs],
            fmt='s', linestyle='none', capsize=4, color=colors[idx], markersize=10
        )
        line[0].set_label(f"{layer_dims_str}\n(no amp)")

    # Final axes settings
    # Top subplot (no amp): semilog-x with requested limits
    ax1.set_xscale("log")
    ax1.set_xlim(1e-21, 1e-4)
    ax1.set_xlabel("Input Energy (J)", fontsize=18)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=18)
    ax1.set_yticks(np.arange(0, 110, 10))
    ax1.grid(True)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.legend(fontsize=16)

    # Bottom subplot (with amp): semilog-x with requested limits
    ax2.set_xscale("log")
    ax2.set_xlim(1e-8, 1e-4)
    ax2.set_xlabel("Input Energy (J)", fontsize=18)
    # ax2.set_ylabel("Test Accuracy (%)", fontsize=18)
    ax2.set_yticks(np.arange(0, 110, 10))
    ax2.grid(True)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig("median_accuracy_with_errorbars_layernorm_Aug31_nointerpol.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
