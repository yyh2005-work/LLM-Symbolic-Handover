"""Deploy LLM-generated symbolic rules on the handover environment."""

import json
import os
import numpy as np
from stable_baselines3.common.env_checker import check_env

from ho_optim_drl.config import Config
import ho_optim_drl.dataloader as dl
from ho_optim_drl.gym_env import HandoverEnvPPO
import ho_optim_drl.utils as ut

try:
    from .symbolic_module_llm import RuleSymbolicExplainerLLM
except ImportError:
    from symbolic_module_llm import RuleSymbolicExplainerLLM

USE_SPEED_LIST = [30, 50, 70, 90]
SHUFFLE_DATASETS = True
SHUFFLE_SEED = 42


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _load_symbol_vocabulary(root_path: str) -> dict:
    """Load LLM-generated symbol vocabulary from stage1 summary."""
    summary_path = os.path.join(
        root_path, "results", "llm_generated_symbxrl", "stage1",
        "stage1_generation_summary.json"
    )
    if not os.path.exists(summary_path):
        return {"dim_names": [], "state_symbols": {}, "action_symbols": []}

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    vocab_raw = summary.get("vocabulary", {})
    if not vocab_raw:
        return {"dim_names": [], "state_symbols": {}, "action_symbols": []}

    if "state_dimensions" in vocab_raw:
        dim_names = vocab_raw.get("state_dimensions", [])
        state_symbols = vocab_raw.get("state_symbols") or vocab_raw.get("vocabulary", {})
        if not isinstance(state_symbols, dict):
            state_symbols = {}
        action_symbols = vocab_raw.get("action_symbols", ["Stay", "HandoverToBest", "HandoverToOther"])
    else:
        dim_names = list(vocab_raw.keys())
        state_symbols = {k: v for k, v in vocab_raw.items()}
        action_symbols = summary.get("metadata", {}).get(
            "action_symbols", ["Stay", "HandoverToBest", "HandoverToOther"]
        )

    return {"dim_names": dim_names, "state_symbols": state_symbols, "action_symbols": action_symbols}


def _identify_important_indices(dim_names: list[str], state_symbols: dict, top_k: int = 4) -> list[int]:
    """Select top-k most discriminative dimensions for coarse matching."""
    if not dim_names or not state_symbols:
        return list(range(min(top_k, len(dim_names)))) if dim_names else []

    dim_cardinality = []
    for i, dim in enumerate(dim_names):
        values = state_symbols.get(dim, [])
        card = len(values) if isinstance(values, (list, tuple)) else 999
        dim_cardinality.append((i, card))

    dim_cardinality.sort(key=lambda x: x[1], reverse=True)
    selected = [idx for idx, _ in dim_cardinality[:min(top_k, len(dim_cardinality))]]
    return sorted(selected)


def _build_coarse_index(decision_list: list[dict], dim_names: list[str],
                        state_symbols: dict) -> dict:
    """Build coarse-match index keyed by important dimension values."""
    important_indices = _identify_important_indices(dim_names, state_symbols)
    coarse_index: dict = {}

    for rule in decision_list:
        state_key = rule.get("state_key", "")
        parts = state_key.split(",")
        if len(parts) != len(dim_names):
            continue
        action = rule.get("symbolic_action")
        if not action:
            continue
        score = rule.get("score", rule.get("confidence", 0))

        coarse_key_parts = []
        for idx in important_indices:
            if idx < len(parts):
                coarse_key_parts.append(parts[idx])
        coarse_key = ",".join(coarse_key_parts)

        if coarse_key not in coarse_index:
            coarse_index[coarse_key] = []
        coarse_index[coarse_key].append((state_key, action, score))

    for key in coarse_index:
        coarse_index[key].sort(key=lambda x: x[2], reverse=True)

    return coarse_index, important_indices

def _build_rule_indexes(decision_list: list[dict]) -> dict:
    """Build exact-match rule index from state key to symbolic action."""
    full_map = {}
    for rule in decision_list:
        state_key = rule.get("state_key")
        action = rule.get("symbolic_action")
        if not state_key or not action:
            continue
        full_map[state_key] = action
    return full_map

def main(root_path: str) -> int:
    """Main function."""

    config = Config()

    # Load data files
    data_dir = os.path.join(root_path, "data", "processed")
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")

    # Speed filter
    use_speed_list = USE_SPEED_LIST
    rsrp_files, sinr_files, speeds = ut.filenames_speed_filter(rsrp_files, sinr_files, use_speed_list)

    # Preprocess datasets
    rsrp_list, sinr_list, sinr_norm_list = [], [], []
    for rsrp_f, sinr_f in zip(rsrp_files, sinr_files):
        rsrp, sinr = dl.load_preprocess_dataset(config, data_dir, rsrp_f, sinr_f)
        sinr_norm = ut.clipnorm(sinr, config.sinr_lower_clip, config.sinr_upper_clip) if config.clip_sinr else sinr
        rsrp_list.append(rsrp)
        sinr_list.append(sinr)
        sinr_norm_list.append(sinr_norm)

    # Create environment
    env = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    check_env(env, warn=True)
    env.set_test_mode(True)

    explainer = RuleSymbolicExplainerLLM(n_bs=env.n_bs, root_path=root_path)

    # Load deployed rule decision list
    rules_path = os.path.join(root_path, "results", "rule","rule_symbolic_llm_decision_list.json")
    if not os.path.exists(rules_path):
        print(f"[LLM-RULE] missing decision list: {rules_path}")
        print("[LLM-RULE] 请先运行 validate_ppo.py 生成可部署规则")
        return 1
    with open(rules_path, "r", encoding="utf-8") as f:
        decision_list = json.load(f)
    decision_map = _build_rule_indexes(decision_list)

    # Build symbol vocabulary and matching indexes
    symbol_vocab = _load_symbol_vocabulary(root_path)
    dim_names = symbol_vocab["dim_names"]
    state_symbols = symbol_vocab["state_symbols"]
    if not dim_names:
        vocab = explainer.mapper.get_symbol_vocabulary()
        if isinstance(vocab, dict):
            dim_names = vocab.get("state_dimensions", list(vocab.keys()))
            state_symbols = vocab.get("state_symbols", {})
        elif isinstance(vocab, (list, tuple)):
            dim_names = list(vocab)
        if not dim_names:
            dim_names = ["dim_%d" % i for i in range(8)]

    coarse_index, important_indices = _build_coarse_index(decision_list, dim_names, state_symbols)
    coarse_predicate_names = [dim_names[i] if i < len(dim_names) else "dim_%d" % i for i in important_indices]
    print(f"[LLM-RULE] 维度数: {len(dim_names)}, 粗匹配谓词: {coarse_predicate_names}")

    # Result containers
    result_container = ut.get_result_container(speeds)
    aggregated_stats = {k: [] for k in env.ho_procedure.get_stats_dict()}

    # Shuffle datasets
    dataset_order = list(range(env.n_datasets))
    if SHUFFLE_DATASETS:
        rng = np.random.default_rng(SHUFFLE_SEED)
        dataset_order = rng.permutation(env.n_datasets).tolist()

    # Deployment log
    deployment_log = {
        "total_steps": 0,
        "exact_matched_count": 0,
        "coarse_matched_count": 0,
        "fallback_count": 0,
        "rule_count": len(decision_map),
        "coarse_rule_count": len(coarse_index),
        "coarse_predicates": coarse_predicate_names,
        "shuffle_datasets": SHUFFLE_DATASETS,
        "shuffle_seed": SHUFFLE_SEED if SHUFFLE_DATASETS else None,
        "dataset_order": dataset_order,
        "action_counts": {"Stay": 0, "HandoverToBest": 0, "HandoverToOther": 0},
        "dataset_rewards": [],
    }

    # Run the LLM-rule agent
    for order_idx, dataset_idx in enumerate(dataset_order):
        env.set_dataset_idx(dataset_idx)
        obs, _ = env.reset()
        truncated = False
        ep_reward = 0.0

        while not truncated:
            # Identify serving cell and best neighbor
            serving_cell = int(np.argmax(obs[: env.n_bs]))
            sinr_values = obs[env.n_bs : 2 * env.n_bs]
            sinr_copy = sinr_values.copy()
            sinr_copy[serving_cell] = -np.inf
            best_target = int(np.argmax(sinr_copy))
            
            # State translation and rule matching
            symbolic_state = explainer.translate_state_and_commit(obs)
            state_key = ",".join(symbolic_state)
            state_parts = state_key.split(",")

            # Exact match -> coarse match -> fallback
            symbolic_action = decision_map.get(state_key)
            if symbolic_action is not None:
                deployment_log["exact_matched_count"] += 1
            else:
                coarse_key_parts = []
                for idx in important_indices:
                    if idx < len(state_parts):
                        coarse_key_parts.append(state_parts[idx])
                coarse_key = ",".join(coarse_key_parts)
                candidates = coarse_index.get(coarse_key, [])
                if candidates:
                    symbolic_action = candidates[0][1]
                    deployment_log["coarse_matched_count"] += 1
                else:
                    symbolic_action = "Stay"
                    deployment_log["fallback_count"] += 1

            # Map symbolic action to environment action
            feature_dict = explainer.mapper.decode_state(obs)
            action = explainer.mapper.symbolic_action_to_action_id(symbolic_action, feature_dict)

            # Track action type
            if action == serving_cell:
                deployment_log["action_counts"]["Stay"] += 1
            elif action == best_target:
                deployment_log["action_counts"]["HandoverToBest"] += 1
            else:
                deployment_log["action_counts"]["HandoverToOther"] += 1

            next_obs, reward, _, truncated, _ = env.step(action)
            ep_reward += float(reward)
            deployment_log["total_steps"] += 1
            obs = next_obs

        deployment_log["dataset_rewards"].append(ep_reward)

        # Save statistics
        stats = env.get_statistics()
        for key, val in stats.items():
            aggregated_stats[key].append(val)

        speed = speeds[dataset_idx]
        result_container["sinr_connected"][speed].extend(env.ho_procedure.sinr_timeline)
        result_container["sinr_max"][speed].extend(list(np.max(env.sinr_list[env.dataset_idx], axis=1)))
        result_container["sinr_at_ho_exe_pcell"].extend(env.ho_procedure.sinr_at_ho_exe_pcell)
        result_container["sinr_after_ho_exe_tcell"].extend(env.ho_procedure.sinr_after_ho_exe_tcell)
        result_container["n_ho"][speed].append(stats["num_ho_exe_started"])
        result_container["n_pp"][speed].append(stats["num_pp"])
        result_container["n_rlf"][speed].append(stats["num_rlf"])

        print(f"[LLM-RULE] dataset {order_idx + 1:3d}/{env.n_datasets} (idx={dataset_idx}), episode reward: {ep_reward:.2f}")

    # Results (all speeds individually)
    unique_speeds = np.unique(speeds).tolist()
    rate_mbps, r_rel, mean_pp_prob, mean_rlf_prob = [], [], [], []
    for speed in unique_speeds:
        sinr_connected_db = np.array(result_container["sinr_connected"][speed], dtype=float)
        sinr_connected_lin = 10 ** (sinr_connected_db / 10)
        sinr_connected_lin[np.isnan(sinr_connected_lin)] = 0

        sinr_max_lin = 10 ** (np.array(result_container["sinr_max"][speed], dtype=float) / 10)
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
    aggregated_stats["speeds"] = unique_speeds
    aggregated_stats["rate_mbps"] = rate_mbps
    aggregated_stats["r_rel"] = r_rel
    aggregated_stats["mean_pp_prob"] = mean_pp_prob
    aggregated_stats["mean_rlf_prob"] = mean_rlf_prob
    ut.print_aggregated_stats(aggregated_stats)

    results_dir = os.path.join(root_path, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save results for plotting
    save_path = os.path.join(results_dir, "rule", "rule_llm_agent_results.npz")
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

    # Save deployment log
    log_path = os.path.join(results_dir, "rule", "rule_llm_agent_deployment_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(deployment_log, f, ensure_ascii=False, indent=2)

    print(f"[LLM-RULE] 绘图数据已保存至: {save_path}")
    print(f"[LLM-RULE] 部署日志已保存至: {log_path}")

    total_steps = deployment_log["total_steps"]
    exact = deployment_log.get("exact_matched_count", 0)
    coarse = deployment_log.get("coarse_matched_count", 0)
    fallback = deployment_log.get("fallback_count", 0)
    if total_steps > 0:
        print(f"[LLM-RULE] 匹配统计: 精确={exact} "
              f"粗匹配={coarse} "
              f"回退={fallback}")

    return 0

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main(ROOT))
