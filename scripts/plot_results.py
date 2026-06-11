"""Plot SINR ECDF, relative data rate, and HOF/PP probability comparisons."""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# IEEE compatible colors
COLOR_3GPP = 'royalblue'
COLOR_PPO = 'orange'
COLOR_LLM = 'green'

# Plot ECDF of SINR before and after handover
def plot_sinr_ecdf(ppo_data, gpp_data, llm_data, save_path):
    plt.figure(figsize=(6, 5))
    
    # Helper to clean data (remove NaNs and -inf)
    def clean(data):
        d = data[~np.isnan(data)]
        return d[d > -100] # Filter out -inf or extremely low values
    
    def ecdf_values(data, x):
        data = np.asarray(data)
        if data.size == 0:
            return np.zeros_like(x, dtype=float)
        sorted_data = np.sort(data)
        return np.searchsorted(sorted_data, x, side="right") / float(sorted_data.size)

    # 3GPP Data
    gpp_before = clean(gpp_data['sinr_before_ho'])
    gpp_after = clean(gpp_data['sinr_after_ho'])
    
    # PPO Data
    ppo_before = clean(ppo_data['sinr_before_ho'])
    ppo_after = clean(ppo_data['sinr_after_ho'])

    # LLM Data
    llm_before = clean(llm_data['sinr_before_ho'])
    llm_after = clean(llm_data['sinr_after_ho'])

    x = np.linspace(-15, 10, 500)
    
    plt.plot(x, ecdf_values(gpp_before, x), label='3GPP before HO', color=COLOR_3GPP)
    plt.plot(x, ecdf_values(gpp_after, x), label='3GPP after HO', color=COLOR_3GPP, linestyle='--')
    plt.plot(x, ecdf_values(ppo_before, x), label='PPO before HO', color=COLOR_PPO)
    plt.plot(x, ecdf_values(ppo_after, x), label='PPO after HO', color=COLOR_PPO, linestyle='--')
    plt.plot(x, ecdf_values(llm_before, x), label='LLM-RULE before HO', color=COLOR_LLM)
    plt.plot(x, ecdf_values(llm_after, x), label='LLM-RULE after HO', color=COLOR_LLM, linestyle='--')

    # Add thresholds lines
    plt.axvline(x=-8, color='firebrick', linestyle='-', label='$Q_{out}$')
    plt.axvline(x=-6, color='firebrick', linestyle='--', label='$Q_{in}$')

    plt.xlabel('SINR (dB)', fontsize=12)
    plt.ylabel('ECDF', fontsize=12)
    plt.xlim([-15, 10])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=8)
    plt.title('SINR Distribution before/after Handover')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(os.path.splitext(save_path)[0] + '.eps', format='eps', dpi=300)
    print(f"Saved SINR plot to {save_path}")

def plot_rate_vs_velocity(ppo_data, gpp_data, llm_data, save_path):
    plt.figure(figsize=(6, 5))

    speeds = ppo_data['speeds']

    def get_r_rel(data):
        return np.asarray(data['r_rel'], dtype=float) * 100

    ppo_rate = get_r_rel(ppo_data)
    gpp_rate = get_r_rel(gpp_data)
    llm_rate = get_r_rel(llm_data)

    plt.plot(speeds, ppo_rate, '--o', label='PPO', color=COLOR_PPO, markerfacecolor='none', markersize=8, linewidth=1.5)
    plt.plot(speeds, gpp_rate, '-^', label='3GPP', color=COLOR_3GPP, markerfacecolor='none', markersize=8, linewidth=1.5)
    plt.plot(speeds, llm_rate, ':D', label='LLM-RULE', color=COLOR_LLM, markerfacecolor='none', markersize=8, linewidth=1.5)

    plt.xlabel('UE velocity (km/h)', fontsize=12)
    plt.ylabel('Relative data rate (%)', fontsize=12)
    
    all_rates = np.concatenate([ppo_rate, gpp_rate, llm_rate])
    min_rate = np.min(all_rates)
    max_rate = np.max(all_rates)
    margin = (max_rate - min_rate) * 0.1
    y_min = max(0, min_rate - margin)
    y_max = max_rate + margin
    plt.ylim([y_min, y_max])

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.title('Relative Data Rate vs Velocity')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(os.path.splitext(save_path)[0] + '.eps', format='eps', dpi=300)
    print(f"Saved Rate plot to {save_path}")

def plot_hof_pp_probability(ppo_data, gpp_data, llm_data, save_path):
    speeds = ppo_data['speeds']
    x = np.arange(len(speeds))
    width = 0.15

    # Data
    hof_ppo = ppo_data['mean_rlf_prob']
    pp_ppo = ppo_data['mean_pp_prob']
    hof_3gpp = gpp_data['mean_rlf_prob']
    pp_3gpp = gpp_data['mean_pp_prob']
    hof_llm = llm_data['mean_rlf_prob']
    pp_llm = llm_data['mean_pp_prob']

    fig, ax1 = plt.subplots(figsize=(6, 5))

    # HOF bars (left axis)
    ax1.bar(x - 1.5*width, hof_3gpp, width, label='HOF$_{3GPP}$', color=COLOR_3GPP, edgecolor='black')
    ax1.bar(x - 0.5*width, hof_ppo,   width, label='HOF$_{PPO}$',   color=COLOR_PPO, edgecolor='black')
    ax1.bar(x + 0.5*width, hof_llm,   width, label='HOF$_{LLM-RULE}$',   color=COLOR_LLM, edgecolor='black')
    
    ax1.set_xlabel('UE velocity (km/h)', fontsize=12)
    ax1.set_ylabel('HOF probability', fontsize=12)
    ax1.set_ylim([0, 0.15])
    ax1.set_xticks(x)
    ax1.set_xticklabels(speeds)
    
    # PP bars (right axis)
    ax2 = ax1.twinx()
    ax2.bar(x + 1.5*width, pp_3gpp, width, label='PP$_{3GPP}$', color='white', edgecolor=COLOR_3GPP, hatch='//')
    ax2.bar(x + 2.5*width, pp_ppo,   width, label='PP$_{PPO}$',   color='white', edgecolor=COLOR_PPO, hatch='//')
    ax2.bar(x + 3.5*width, pp_llm,   width, label='PP$_{LLM-RULE}$',   color='white', edgecolor=COLOR_LLM, hatch='//')
    
    ax2.set_ylabel('PP probability', fontsize=12)
    ax2.set_ylim([0, 0.6])

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2,
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=8, frameon=True, edgecolor='black', fontsize=8)
    
    plt.title('HOF and PP Probability vs Velocity')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # More space for legend
    fig.savefig(save_path, dpi=300)
    fig.savefig(os.path.splitext(save_path)[0] + '.eps', format='eps', dpi=300)
    print(f"Saved HOF/PP plot to {save_path}")

def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    ppo_path = os.path.join(results_dir,"ppo", "ppo_results.npz")
    gpp_path = os.path.join(results_dir,"3gpp", "3gpp_results.npz")
    llm_path = os.path.join(results_dir,"rule", "rule_llm_agent_results.npz")

    if (
        not os.path.exists(ppo_path)
        or not os.path.exists(gpp_path)
        or not os.path.exists(llm_path)
    ):
        print("Error: required data files not found!")
        print("Please run validate_ppo.py, validate_3gpp.py, and validate_ppo_llm_rule.py first.")
        return

    ppo_data = np.load(ppo_path)
    gpp_data = np.load(gpp_path)
    llm_data = np.load(llm_path)

    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_sinr_ecdf(ppo_data, gpp_data, llm_data, os.path.join(figures_dir, "sinr_ecdf.png"))
    plot_rate_vs_velocity(ppo_data, gpp_data, llm_data, os.path.join(figures_dir, "rate_vs_velocity.png"))
    plot_hof_pp_probability(ppo_data, gpp_data, llm_data, os.path.join(figures_dir, "hof_pp_probability.png"))

if __name__ == "__main__":
    main()
