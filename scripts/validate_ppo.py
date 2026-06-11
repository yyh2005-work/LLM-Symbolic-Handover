"""Validate PPO protocol on the handover environment."""

import os
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

try:
    from .symbolic_module_llm import RuleSymbolicExplainerLLM
except ImportError:
    from symbolic_module_llm import RuleSymbolicExplainerLLM


ENABLE_LLM_RULE_SYMBOLIC = True


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def main(root_path: str) -> int:
    """Main function."""
    config = Config()

    # Load data files
    data_dir = os.path.join(root_path, "data", "processed")
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")

    # Speed filter
    use_speed_list = [30, 50, 70, 90]
    rsrp_files, sinr_files, speeds = ut.filenames_speed_filter(
        rsrp_files, sinr_files, use_speed_list
    )

    # Preprocess datasets
    rsrp_list = []
    sinr_list = []
    sinr_norm_list = []
    for rsrp_fname_i, sinr_fname_i in zip(rsrp_files, sinr_files):
        rsrp, sinr = dl.load_preprocess_dataset(
            config, data_dir, rsrp_fname_i, sinr_fname_i
        )

        if config.clip_sinr:
            sinr_norm = ut.clipnorm(
                sinr, config.sinr_lower_clip, config.sinr_upper_clip
            )
        else:
            sinr_norm = sinr

        sinr_list.append(sinr)
        rsrp_list.append(rsrp)
        sinr_norm_list.append(sinr_norm)

    # Create environment and load PPO model
    env = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    check_env(env, warn=True)

    model_dir = os.path.join(root_path, "models", "model")

    model = PPO.load(
        model_dir,
        env=env,
        tensorboard_log=None,
        device="cpu",
    )

    # Result containers
    result_container = ut.get_result_container(speeds)
    aggregated_stats = {key: [] for key in env.ho_procedure.get_stats_dict()}

    # Rule symbolic explainer
    use_llm_rule_symbolic = bool(getattr(config, "use_llm_rule_symbolic_in_validate_ppo", False)) or ENABLE_LLM_RULE_SYMBOLIC
    if use_llm_rule_symbolic:
        explainer = RuleSymbolicExplainerLLM(n_bs=env.n_bs, root_path=root_path)
    else:
        explainer = RuleSymbolicExplainer(n_bs=env.n_bs)

    enable_rule_symbolic = True

    rule_symbolic_stats = {
        "total_episodes": 0,
        "total_steps": 0,
        "symbolic_states_recorded": 0,
    }

    env.set_test_mode(True)
    if config.test_deterministic_actions:
        print("[PPO] Test with deterministic actions.")
    else:
        print("[PPO] Test with actions sampled from the policy distribution.")

    print(f"[Env] HO preparation abort permitted: {config.permit_ho_prep_abort}")
    if use_llm_rule_symbolic:
        print(f"[LLM-RULE] 规则提取模块已启用")
    else:
        print(f"[RULE] 规则提取模块已启用")

    # Run the PPO handover protocol
    for i in range(env.n_datasets):
        env.set_dataset_idx(i)

        obs, _ = env.reset()

        truncated = False
        episode_rewards = []

        dataset_rule_symbolic_info = {
            "symbolic_states": [],
            "symbolic_actions": [],
            "ppo_actions": [],
            "rewards": [],
        }

        while not truncated:
            ppo_action, _ = model.predict(obs, deterministic=config.test_deterministic_actions)

            final_action = ppo_action

            next_obs, reward, _, truncated, _ = env.step(final_action)

            if enable_rule_symbolic:
                symbolic_state, symbolic_action = explainer.update_kg(obs, final_action, reward)

                dataset_rule_symbolic_info["symbolic_states"].append(symbolic_state)
                dataset_rule_symbolic_info["symbolic_actions"].append(symbolic_action)
                rule_symbolic_stats["symbolic_states_recorded"] += 1

            dataset_rule_symbolic_info["ppo_actions"].append(ppo_action)
            dataset_rule_symbolic_info["rewards"].append(reward)

            episode_rewards.append(reward)
            obs = next_obs

        rule_symbolic_stats["total_episodes"] += 1
        rule_symbolic_stats["total_steps"] += len(episode_rewards)

        print(
            f"Testing PPO HO protocol with dataset {i + 1:3d}/{env.n_datasets}. "
            f"Episode reward: {np.sum(episode_rewards):.2f}"
        )

        # Save statistics
        stats = env.get_statistics()
        for key, val in stats.items():
            aggregated_stats[key].append(val)

        speed = speeds[i]
        result_container["sinr_connected"][speed].extend(
            env.ho_procedure.sinr_timeline
        )
        result_container["sinr_max"][speed].extend(
            list(np.max(env.sinr_list[env.dataset_idx], axis=1))
        )
        result_container["sinr_at_ho_exe_pcell"].extend(
            env.ho_procedure.sinr_at_ho_exe_pcell
        )
        result_container["sinr_after_ho_exe_tcell"].extend(
            env.ho_procedure.sinr_after_ho_exe_tcell
        )
        result_container["n_ho"][speed].append(stats["num_ho_exe_started"])
        result_container["n_pp"][speed].append(stats["num_pp"])
        result_container["n_rlf"][speed].append(stats["num_rlf"])

    # SINR (all speeds combined)
    sinr_at_ho_exe_pcell_db = np.array(result_container["sinr_at_ho_exe_pcell"])
    sinr_after_ho_exe_tcell_db = np.array(result_container["sinr_after_ho_exe_tcell"])
    sinr_at_ho_exe_pcell_db[np.isnan(sinr_at_ho_exe_pcell_db)] = -np.inf
    sinr_after_ho_exe_tcell_db[np.isnan(sinr_after_ho_exe_tcell_db)] = -np.inf

    # Results (all speeds individually)
    rate_mbps = []
    r_rel = []
    mean_pp_prob = []
    mean_rlf_prob = []
    for speed in np.unique(speeds):
        sinr_connected_db = np.array(result_container["sinr_connected"][speed])
        sinr_connected_lin = 10 ** (sinr_connected_db / 10)
        sinr_connected_lin[np.isnan(sinr_connected_lin)] = 0

        sinr_max_lin = 10 ** (np.array(result_container["sinr_max"][speed]) / 10)
        sinr_max_lin[np.isnan(sinr_max_lin)] = 0

        r_mean = np.mean(config.bw * np.log2(1 + sinr_connected_lin))
        r_max = np.mean(config.bw * np.log2(1 + sinr_max_lin))

        rate_mbps.append(float(r_mean / 1e6))
        r_rel.append(_safe_ratio(r_mean, r_max))
        n_ho_sum = float(np.sum(result_container["n_ho"][speed]))
        n_pp_sum = float(np.sum(result_container["n_pp"][speed]))
        n_rlf_sum = float(np.sum(result_container["n_rlf"][speed]))
        mean_pp_prob.append(_safe_ratio(n_pp_sum, n_ho_sum))
        mean_rlf_prob.append(_safe_ratio(n_rlf_sum, n_ho_sum))

    # Print aggregated statistics
    aggregated_stats["speeds"] = np.unique(speeds).tolist()
    aggregated_stats["rate_mbps"] = rate_mbps
    aggregated_stats["r_rel"] = r_rel
    aggregated_stats["mean_pp_prob"] = mean_pp_prob
    aggregated_stats["mean_rlf_prob"] = mean_rlf_prob
    ut.print_aggregated_stats(aggregated_stats)

    # Save symbolic rules
    if enable_rule_symbolic:
        kg_save_path = os.path.join(
            root_path,
            "results",
            "rule",
            "rule_symbolic_llm_knowledge_base.json" if use_llm_rule_symbolic else "rule_symbolic_knowledge_base.json",
        )
        os.makedirs(os.path.dirname(kg_save_path), exist_ok=True)
        explainer.save_knowledge_base(kg_save_path)

        decision_list_path, decision_list = explainer.save_decision_list_to_results(
            root_path=root_path,
            min_state_count=8,
            min_confidence=0.4,
            min_action_count=1,
            reward_weight=0.2,
        )
        print(f"决策列表已保存至: {decision_list_path}")

    # Save results for plotting
    save_path = os.path.join(root_path, "results","ppo", "ppo_results.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        speeds=aggregated_stats["speeds"],
        rate_mbps=aggregated_stats["rate_mbps"],
        r_rel=aggregated_stats["r_rel"],
        mean_pp_prob=aggregated_stats["mean_pp_prob"],
        mean_rlf_prob=aggregated_stats["mean_rlf_prob"],
        sinr_before_ho=result_container["sinr_at_ho_exe_pcell"],
        sinr_after_ho=result_container["sinr_after_ho_exe_tcell"],
    )
    print(f"[PPO]绘图数据已保存至: {save_path}")

    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()
    raise SystemExit(main(os.path.abspath(args.root_path)))
