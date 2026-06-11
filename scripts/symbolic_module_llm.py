"""LLM-assisted symbolic rule extraction and deployment."""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
import json
import os

from ho_optim_drl.config import Config
try:
    from .generate_llm_symbol_mapper import LLMMetaProgrammer, LLMSettings
except ImportError:
    from generate_llm_symbol_mapper import LLMMetaProgrammer, LLMSettings

class RuleSymbolicExplainerLLM:
    """Explains PPO behaviour by mapping to symbolic states and extracting rules."""

    def __init__(self, n_bs: int, root_path: str):
        self.n_bs = n_bs
        self.knowledge_base: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Load pre-generated LLM mapper
        config = Config()
        settings = LLMSettings.from_config(config)
        output_dir = os.path.join(root_path, "results", "llm_generated_symbxrl", "stage1")
        mapper_path = os.path.join(output_dir, "stage1_generated_mapper.py")
        meta_path = os.path.join(output_dir, "stage1_mapper_meta.json")

        if not os.path.exists(mapper_path):
            raise FileNotFoundError(
                f"LLM mapper not found: {mapper_path}. "
                "Please run scripts/generate_llm_symbol_mapper.py first."
            )

        self.programmer = LLMMetaProgrammer(settings, output_dir=output_dir)
        mapper_code = open(mapper_path, "r", encoding="utf-8").read()
        self.mapper = self.programmer._load_mapper_from_code(mapper_code, n_bs=n_bs)

        mode = "pre_generated"
        if os.path.exists(meta_path):
            try:
                mode = json.loads(open(meta_path, "r", encoding="utf-8").read()).get(
                    "mode", mode
                )
            except Exception:
                mode = "pre_generated"

        print(f"[LLM-RULE] Initialised: n_bs={n_bs}, mode={mode}")

    def translate_to_symbolic(self, state: np.ndarray, action: int) -> Tuple[Tuple[str, ...], str]:
        symbolic_state, symbolic_action, _ = self.mapper.translate_to_symbolic(state, action)
        return symbolic_state, symbolic_action

    def translate_state_and_commit(self, state: np.ndarray) -> Tuple[str, ...]:
        symbolic_state, _, _ = self.mapper.translate_to_symbolic(state, None)
        return symbolic_state

    def update_kg(self, state: np.ndarray, action: int, reward: float) -> Tuple[Tuple[str, ...], str]:
        """Update knowledge graph with a (state, action, reward) observation."""
        try:
            if not 0 <= action < self.n_bs:
                raise ValueError(f"Action out of range: expected 0 to {self.n_bs-1}, got {action}")

            symbolic_state, symbolic_action, _ = self.mapper.translate_to_symbolic(state, action)
            state_key = ','.join(symbolic_state)

            if state_key not in self.knowledge_base:
                self.knowledge_base[state_key] = {}

            if symbolic_action not in self.knowledge_base[state_key]:
                self.knowledge_base[state_key][symbolic_action] = {
                    'count': 0,
                    'total_reward': 0.0
                }

            self.knowledge_base[state_key][symbolic_action]['count'] += 1
            self.knowledge_base[state_key][symbolic_action]['total_reward'] += reward
            return symbolic_state, symbolic_action

        except Exception as e:
            print(f"[LLM-RULE] Knowledge graph update failed: {e}")
            raise

    def save_knowledge_base(self, filepath: str) -> None:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
        print(f"[LLM-RULE] Knowledge graph saved to: {filepath}")

    def load_knowledge_base(self, filepath: str) -> None:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"[LLM-RULE] Knowledge graph loaded from {filepath}")
        except Exception as e:
            print(f"[LLM-RULE] Failed to load knowledge graph: {e}")

    def _select_rule_action(self, state_parts: list[str], action_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        return action_candidates[0]

    def _state_key_to_fol(self, state_key: str, action: str, confidence: float, support: int) -> str:
        """Convert a symbolic state-action pair to a first-order logic rule string."""
        parts = state_key.split(",")
        vocab = self.mapper.get_symbol_vocabulary()
        dims = vocab.get("state_dimensions", [])
        if not dims and isinstance(vocab, dict):
            dims = list(vocab.keys())
        
        if len(parts) == len(dims):
            conditions = [f"{dim}({val})" for dim, val in zip(dims, parts)]
            antecedent = " ∧ ".join(conditions)
        else:
            antecedent = f"State({state_key})"
            
        fol = f"∀state: {antecedent} ⇒ Action({action}) [support={support}, confidence={confidence:.2f}]"
        return fol

    def extract_decision_list(
        self,
        min_state_count: int = 8,
        min_confidence: float = 0.4,
        min_action_count: int = 1,
        reward_weight: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Extract a scored, ranked decision list from the knowledge graph."""
        decision_list: List[Dict[str, Any]] = []

        for state_key, actions_dict in self.knowledge_base.items():
            total_count = sum(action_data['count'] for action_data in actions_dict.values())
            if total_count < min_state_count:
                continue

            action_candidates: List[Dict[str, Any]] = []
            for action_name, action_data in actions_dict.items():
                support = int(action_data['count'])
                if support < min_action_count:
                    continue
                confidence = support / total_count
                avg_reward = action_data['total_reward'] / max(support, 1)
                score = avg_reward + reward_weight * confidence
                action_candidates.append(
                    {
                        'symbolic_action': action_name,
                        'support': support,
                        'confidence': float(confidence),
                        'avg_reward': float(avg_reward),
                        'score': float(score),
                    }
                )

            if not action_candidates:
                continue

            action_candidates.sort(key=lambda x: (x['score'], x['confidence'], x['support']), reverse=True)
            state_parts = state_key.split(',')
            best = self._select_rule_action(state_parts, action_candidates)
            
            required_conf = min_confidence if best['symbolic_action'] == 'Stay' else min(0.12, min_confidence)
            if best['confidence'] < required_conf:
                continue

            fol_representation = self._state_key_to_fol(
                state_key,
                best['symbolic_action'],
                best['confidence'],
                int(best['support'])
            )
            
            vocab = self.mapper.get_symbol_vocabulary()
            dims = vocab.get("state_dimensions", [])
            if not dims and isinstance(vocab, dict):
                dims = list(vocab.keys())
            predicates = {}
            if len(state_parts) == len(dims):
                for dim, val in zip(dims, state_parts):
                    predicates[dim] = val
            
            decision_list.append(
                {
                    'state_key': state_key,
                    'fol_representation': fol_representation,
                    'predicates': predicates,
                    'symbolic_action': best['symbolic_action'],
                    'confidence': best['confidence'],
                    'support': best['support'],
                    'total_count': int(total_count),
                    'avg_reward': best['avg_reward'],
                    'score': best['score'],
                }
            )

        decision_list.sort(
            key=lambda x: (x['score'], x['confidence'], x['support']),
            reverse=True,
        )
        return decision_list

    def save_decision_list(
        self,
        filepath: str,
        min_state_count: int = 8,
        min_confidence: float = 0.4,
        min_action_count: int = 1,
        reward_weight: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Extract and save decision list to a JSON file."""
        decision_list = self.extract_decision_list(
            min_state_count=min_state_count,
            min_confidence=min_confidence,
            min_action_count=min_action_count,
            reward_weight=reward_weight,
        )
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(decision_list, f, ensure_ascii=False, indent=2)
        return decision_list

    def save_decision_list_to_results(
        self,
        root_path: str,
        min_state_count: int = 8,
        min_confidence: float = 0.4,
        min_action_count: int = 1,
        reward_weight: float = 0.2,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        results_dir = os.path.join(root_path, 'results')
        output_path = os.path.join(results_dir, "rule", "rule_symbolic_llm_decision_list.json")
        decision_list = self.save_decision_list(
            filepath=output_path,
            min_state_count=min_state_count,
            min_confidence=min_confidence,
            min_action_count=min_action_count,
            reward_weight=reward_weight,
        )
        return output_path, decision_list

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        total_states = len(self.knowledge_base)
        total_actions = sum(len(actions) for actions in self.knowledge_base.values())
        total_samples = sum(
            sum(action_data['count'] for action_data in actions.values())
            for actions in self.knowledge_base.values()
        )

        return {
            'total_symbolic_states': total_states,
            'total_symbolic_actions': total_actions,
            'total_samples': total_samples,
        }
