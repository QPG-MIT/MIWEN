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

    # Loop over both configurations
    # plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 6))
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

        plt.errorbar(all_Es, median_accs, yerr=[median_accs - q1_accs, q3_accs - median_accs],
                     fmt='o-', label=f"{layer_dims_str}",
                     capsize=4, color=colors[idx+2], markersize=10, linewidth=3)

        # Plot without opamp energy shift
        line = plt.errorbar(E_targets, median_accs, yerr=[median_accs - q1_accs, q3_accs - median_accs],
                     fmt='s', linestyle='--', capsize=4, color=colors[idx], markersize=10, linewidth=3)
        line[0].set_label(f"{layer_dims_str}\n(no amp)")

    # Final plot settings
    plt.xscale("log")
    plt.xlabel("Input Energy (J)", fontsize=18)
    plt.ylabel("Test Accuracy (%)", fontsize=18)
    plt.title("Test Accuracy vs. Input Energy (Median ± IQR)", fontsize=18)
    plt.yticks(np.arange(0, 110, 10))
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # custom_lines = [
    #     Line2D([0], [0], color=colors[0], linestyle='-', marker='o', linewidth=3, markersize=10,
    #            label=f"{layer_dims_list[0]}"),
    #     Line2D([0], [0], color=colors[0], linestyle='--', marker='s', linewidth=3, markersize=10,
    #            label=f"{layer_dims_list[0]}\n(no opamp)"),
    #     Line2D([0], [0], color=colors[1], linestyle='-', marker='o', linewidth=3, marker size=10,
    #            label=f"{layer_dims_list[1]}"),
    #     Line2D([0], [0], color=colors[1], linestyle='--', marker='s', linewidth=3, markersize=10,
    #            label=f"{layer_dims_list[1]}\n(no opamp)")
    # ]
    # plt.legend(handles=custom_lines)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("median_accuracy_with_errorbars_layernorm_May19.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
