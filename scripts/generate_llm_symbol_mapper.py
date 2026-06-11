"""LLM-assisted symbolic mapper generation.

Calls an LLM to generate Python code for a BaseSymbolicMapper subclass,
and falls back to a built-in mapper when the LLM is unavailable.
"""

from __future__ import annotations

import abc
import argparse
import json
import math
import os
import re
import sys
import time
import types
from collections import deque, OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib import error, request

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

try:
    from ho_optim_drl.config import Config
except ImportError:
    from ho_optim_drl.config import Config

@dataclass
class LLMSettings:

    enabled: bool = True
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    api_base: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.1
    timeout_s: int = 180
    max_tokens: int = 4000
    auto_overwrite_mapper: bool = False
    allow_builtin_fallback: bool = True
    reasoning_effort: str = ""
    enable_thinking: bool = False

    @classmethod
    def from_config(cls, config: Config) -> "LLMSettings":
        """Read LLM-related settings from the project Config object."""
        return cls(
            enabled=bool(getattr(config, "enable_llm_symbolic_design", True)),
            provider=str(getattr(config, "llm_provider", "deepseek")),
            model=str(getattr(config, "llm_model", "deepseek-v4-pro")),
            api_base=str(getattr(config, "llm_api_base", "https://api.deepseek.com")),
            api_key_env=str(getattr(config, "llm_api_key_env", "DEEPSEEK_API_KEY")),
            temperature=float(getattr(config, "llm_temperature", 0.1)),
            timeout_s=int(getattr(config, "llm_timeout_s", 180)),
            max_tokens=int(getattr(config, "llm_max_tokens", 4000))
            if hasattr(config, "llm_max_tokens")
            else 4000,
            auto_overwrite_mapper=bool(
                getattr(config, "llm_auto_overwrite_symbolic_config", False)
            ),
            allow_builtin_fallback=bool(
                getattr(config, "llm_allow_builtin_fallback", True)
            )
            if hasattr(config, "llm_allow_builtin_fallback")
            else True,
        )


@dataclass
class StageArtifact:
    """File paths and metadata for a pipeline stage."""

    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SymbxrlError(RuntimeError):
    """Base exception for the symbolic RL pipeline."""


class MapperGenerationError(SymbxrlError):
    """Mapper source-code generation failed."""


class MapperValidationError(SymbxrlError):
    """Generated mapper failed validation."""


class BaseSymbolicMapper(abc.ABC):
    """Abstract base class for LLM-generated symbolic mappers.

    Actions are restricted to: Stay, HandoverToBest, HandoverToOther.
    """

    ACTION_SYMBOLS: tuple[str, ...] = (
        "Stay",
        "HandoverToBest",
        "HandoverToOther",
    )

    def __init__(self, n_bs: int) -> None:
        self.n_bs = int(n_bs)
        if self.n_bs <= 1:
            raise ValueError("n_bs must be > 1.")

    @abc.abstractmethod
    def get_symbol_vocabulary(self) -> dict[str, Any]:
        """Return symbol vocabulary dict, or a list/tuple of dimension names."""

    @abc.abstractmethod
    def translate_to_symbolic(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> Any:
        """Map continuous state to discrete symbols.

        Returns (symbolic_state_tuple, symbolic_action_str) or
        (symbolic_state_tuple, symbolic_action_str, feature_dict).
        """

    def decode_state(self, state: np.ndarray) -> dict[str, Any]:
        """Decode environment state into physically meaningful features."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        expected_dim = 2 * self.n_bs + 1
        if state.shape[0] != expected_dim:
            raise ValueError(
                f"Invalid state dimension, expected {expected_dim}, got {state.shape[0]}."
            )

        serving_onehot = state[: self.n_bs]
        sinr_norm_all = state[self.n_bs : 2 * self.n_bs]
        pp_pending = bool(float(state[-1]) > 0.5)

        if np.any(serving_onehot > 0.5):
            serving_bs = int(np.argmax(serving_onehot))
        else:
            serving_bs = -1

        rank_desc = np.argsort(-sinr_norm_all)
        best_bs = int(rank_desc[0])
        second_best_bs = int(rank_desc[1]) if len(rank_desc) > 1 else best_bs

        if 0 <= serving_bs < self.n_bs:
            best_target_bs = next(
                (int(bs) for bs in rank_desc.tolist() if int(bs) != serving_bs),
                best_bs,
            )
        else:
            best_target_bs = best_bs

        serving_sinr = (
            float(sinr_norm_all[serving_bs])
            if 0 <= serving_bs < self.n_bs
            else float("nan")
        )
        best_sinr = float(sinr_norm_all[best_bs])
        second_best_sinr = float(sinr_norm_all[second_best_bs])
        best_target_sinr = float(sinr_norm_all[best_target_bs])
        gap_best_serving = (
            float(best_sinr - serving_sinr) if not math.isnan(serving_sinr) else 0.0
        )
        gap_second_serving = (
            float(second_best_sinr - serving_sinr)
            if not math.isnan(serving_sinr)
            else 0.0
        )
        gap_target_serving = (
            float(best_target_sinr - serving_sinr)
            if not math.isnan(serving_sinr)
            else 0.0
        )

        return {
            "serving_bs": serving_bs,
            "best_bs": best_bs,
            "second_best_bs": second_best_bs,
            "best_target_bs": best_target_bs,
            "serving_onehot": serving_onehot.astype(float).tolist(),
            "sinr_norm_all": sinr_norm_all.astype(float).tolist(),
            "pp_pending": pp_pending,
            "serving_sinr": serving_sinr,
            "best_sinr": best_sinr,
            "second_best_sinr": second_best_sinr,
            "best_target_sinr": best_target_sinr,
            "gap_best_serving": gap_best_serving,
            "gap_second_serving": gap_second_serving,
            "gap_target_serving": gap_target_serving,
            "sinr_rank_desc": [int(x) for x in rank_desc.tolist()],
        }

    def action_to_symbolic(
        self, action: int, feature_dict: dict[str, Any]
    ) -> str:
        """Map numeric action id to symbolic action string."""
        action = int(action)
        serving_bs = int(feature_dict["serving_bs"])
        best_bs = int(feature_dict.get("best_bs", feature_dict.get("sinr_rank_desc", [0])[0]))
        best_target_bs = int(feature_dict.get("best_target_bs", best_bs))

        if action == serving_bs:
            return "Stay"
        if action == best_target_bs:
            return "HandoverToBest"
        return "HandoverToOther"

    def symbolic_action_to_action_id(
        self, symbolic_action: str, feature_dict: dict[str, Any]
    ) -> int:
        """Decode symbolic action string back to environment action id."""
        symbolic_action = str(symbolic_action)
        serving_bs = int(feature_dict["serving_bs"])
        best_bs = int(feature_dict.get("best_bs", feature_dict.get("sinr_rank_desc", [0])[0]))
        best_target_bs = int(feature_dict.get("best_target_bs", best_bs))
        rank_desc = [int(x) for x in feature_dict.get("sinr_rank_desc", [best_bs])]

        if symbolic_action == "Stay":
            return serving_bs if serving_bs >= 0 else best_bs
        if symbolic_action == "HandoverToBest":
            return best_target_bs if serving_bs >= 0 else best_bs
        if symbolic_action == "HandoverToOther":
            for bs in rank_desc:
                if bs != serving_bs and bs != best_target_bs:
                    return bs
            return best_target_bs if serving_bs >= 0 else best_bs

        return serving_bs if serving_bs >= 0 else best_bs


class DelegatingSymbolicMapper(BaseSymbolicMapper):
    """Wraps a non-standard LLM mapper to satisfy the BaseSymbolicMapper interface."""

    def __init__(self, n_bs: int, delegate: Any) -> None:
        super().__init__(n_bs=n_bs)
        self._delegate = delegate

    def get_symbol_vocabulary(self) -> Any:
        return self._delegate.get_symbol_vocabulary()

    def translate_to_symbolic(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> Any:
        return self._delegate.translate_to_symbolic(state, action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)





def _json_safe(value: Any) -> Any:
    """Convert numpy / illegal floats to JSON-safe types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        if math.isinf(float(value)):
            return "inf" if float(value) > 0 else "-inf"
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(file_path: str, payload: Any) -> None:
    """Write payload as JSON to disk."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def _write_text(file_path: str, text: str) -> None:
    """Write plain text to disk."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


def _sanitize_python_code(raw_text: str) -> str:
    """Strip Markdown fences from LLM response, returning pure Python."""
    raw_text = raw_text.strip()
    fence_match = re.search(
        r"```(?:python)?\s*([\s\S]*?)```", raw_text, flags=re.IGNORECASE
    )
    if fence_match:
        raw_text = fence_match.group(1).strip()
    return raw_text


def _extract_chat_content(response_json: dict[str, Any]) -> str:
    """Extract message.content from an OpenAI-compatible response."""
    choices = response_json.get("choices", [])
    if not choices:
        raise MapperGenerationError("LLM response missing 'choices' field.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _build_bootstrap_mapper_code() -> str:
    """Built-in fallback mapper (used when LLM is unavailable)."""
    return '''class LLM_AutoMapper(BaseSymbolicMapper):
    """Built-in fallback automatic mapper."""

    def __init__(self, n_bs: int):
        super().__init__(n_bs=n_bs)
        self.serving_sinr_hist = deque(maxlen=8)
        self.best_sinr_hist = deque(maxlen=8)
        self.rank_hist = deque(maxlen=6)
        self.a3_hysteresis = 0.10
        self.a3_strong = 0.25
        self.trend_eps = 0.03

    def get_symbol_vocabulary(self):
        return {
            "state_dimensions": [
                "serving_quality",
                "best_quality",
                "a3_relation",
                "serving_trend",
                "best_trend",
                "handover_guard",
                "dominance_gap",
                "rank_stability",
            ],
            "state_symbols": {
                "serving_quality": ["poor", "fair", "good", "excellent"],
                "best_quality": ["poor", "fair", "good", "excellent"],
                "a3_relation": ["serving_best", "near_tie", "a3_candidate", "strong_a3_candidate"],
                "serving_trend": ["falling_fast", "falling", "stable", "rising"],
                "best_trend": ["falling", "stable", "rising"],
                "handover_guard": ["pp_guard", "idle"],
                "dominance_gap": ["tiny_gap", "medium_gap", "large_gap"],
                "rank_stability": ["unstable", "stable"],
            },
            "action_symbols": list(self.ACTION_SYMBOLS),
        }

    def _quality_bucket(self, value: float) -> str:
        if value < 0.25:
            return "poor"
        if value < 0.50:
            return "fair"
        if value < 0.75:
            return "good"
        return "excellent"

    def _trend_label(self, hist):
        if len(hist) < 2:
            return "stable"
        delta = float(hist[-1] - hist[0]) / max(len(hist) - 1, 1)
        if delta <= -2 * self.trend_eps:
            return "falling_fast"
        if delta < -self.trend_eps:
            return "falling"
        if delta >= self.trend_eps:
            return "rising"
        return "stable"

    def _best_trend_label(self, hist):
        if len(hist) < 2:
            return "stable"
        delta = float(hist[-1] - hist[0]) / max(len(hist) - 1, 1)
        if delta < -self.trend_eps:
            return "falling"
        if delta > self.trend_eps:
            return "rising"
        return "stable"

    def translate_to_symbolic(self, state: np.ndarray, action=None):
        feature_dict = self.decode_state(state)
        serving_sinr = float(feature_dict["serving_sinr"])
        best_sinr = float(feature_dict["best_target_sinr"])
        gap = float(feature_dict["gap_target_serving"])
        rank_tuple = tuple(int(x) for x in feature_dict["sinr_rank_desc"])

        self.serving_sinr_hist.append(serving_sinr if not np.isnan(serving_sinr) else best_sinr)
        self.best_sinr_hist.append(best_sinr)
        self.rank_hist.append(rank_tuple)

        serving_quality = self._quality_bucket(serving_sinr if not np.isnan(serving_sinr) else 0.0)
        best_quality = self._quality_bucket(best_sinr)

        if gap <= 0.01:
            a3_relation = "serving_best"
        elif gap <= self.a3_hysteresis:
            a3_relation = "near_tie"
        elif gap <= self.a3_strong:
            a3_relation = "a3_candidate"
        else:
            a3_relation = "strong_a3_candidate"

        serving_trend = self._trend_label(self.serving_sinr_hist)
        best_trend = self._best_trend_label(self.best_sinr_hist)
        handover_guard = "pp_guard" if feature_dict["pp_pending"] else "idle"

        if gap <= 0.08:
            dominance_gap = "tiny_gap"
        elif gap <= 0.20:
            dominance_gap = "medium_gap"
        else:
            dominance_gap = "large_gap"

        if len(self.rank_hist) < 3:
            rank_stability = "stable"
        else:
            rank_stability = "stable" if len(set(self.rank_hist)) <= 2 else "unstable"

        symbolic_state = (
            serving_quality,
            best_quality,
            a3_relation,
            serving_trend,
            best_trend,
            handover_guard,
            dominance_gap,
            rank_stability,
        )

        symbolic_action = ""
        if action is not None:
            symbolic_action = self.action_to_symbolic(int(action), feature_dict)

        feature_dict.update(
            {
                "serving_quality": serving_quality,
                "best_quality": best_quality,
                "a3_relation": a3_relation,
                "serving_trend": serving_trend,
                "best_trend": best_trend,
                "handover_guard": handover_guard,
                "dominance_gap": dominance_gap,
                "rank_stability": rank_stability,
            }
        )
        return symbolic_state, symbolic_action, feature_dict
'''


class LLMMetaProgrammer:
    """Generate mapper source code via LLM call."""

    def __init__(self, settings: LLMSettings, output_dir: str) -> None:
        self.settings = settings
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_prompt(self, n_bs: int) -> str:
        return f"""
你现在是一位 3GPP 无线通信专家兼高级 Python 工程师。我正在进行基站切换的可解释强化学习研究。

请你直接输出一份完整、可执行、无 Markdown 包裹的 Python 代码。
你必须继承已经存在的 BaseSymbolicMapper 基类，编写一个名为 LLM_AutoMapper 的子类。

物理环境定义：
- n_bs：基站总数。
- state：一维 numpy 数组，维度 2*n_bs + 1。
- state[0:n_bs]：当前服务基站的 One-hot 编码。
- state[n_bs:2*n_bs]：所有基站的归一化 SINR 信号质量。
- state[-1]：浮点数，>0.5 表示处于切换挂起保护期，否则为空闲。
- action：整数 0 到 n_bs-1，表示智能体选择接入的目标基站 ID。
- 动作语义约定：`HandoverToBest` 指排除当前服务基站后的最优目标基站，而不是包含服务基站在内的全局最优小区。

你的任务：
1. __init__ 必须接受 n_bs 参数，且必须调用 super().__init__(n_bs=n_bs) 或 super().__init__(n_bs)。
   示例：def __init__(self, n_bs: int): super().__init__(n_bs=n_bs)
   可选参数应放在 n_bs 之后，例如：def __init__(self, n_bs: int, window_size=10): ...
2. 在 __init__ 中设计历史状态缓存，因为判断信号趋势需要历史数据。
3. 在 get_symbol_vocabulary 中返回你自行设计的离散符号词汇定义。
4. 在 translate_to_symbolic 中实现具体数学映射逻辑，将连续状态转化为离散的字符串元组。
5. 你可以根据 PPO 的输入输出、切换场景机理和 3GPP 专业知识自由设计状态符号维度、命名和分类逻辑。
6. 必须充分考虑通信协议中的 A3 事件（迟滞门限）和多普勒频移。
7. 鼓励输出具有通信物理意义、可用于后续规则蒸馏与部署的中间特征。
8. 不要把状态维度写死为我给你的示例名字，也不要照抄任何预定义的人工规则方案。
9. 如果你认为有更合理的抽象方式，可以自由决定状态维度数量与命名。
10. 尽量复用已有的 self.decode_state(...) 和 self.action_to_symbolic(...) 能力，但不要重新定义 BaseSymbolicMapper。
11. 如果 action 为 None，也必须正确返回 symbolic_state_tuple 和 feature_dict，
    此时 symbolic_action_string 返回空字符串即可。
12. 代码必须兼容 n_bs={n_bs}，但不要写死为固定基站数。
13. 你设计的状态词汇、动作词汇和中间特征必须具备良好的可读性，命名应让通信研究者一眼看懂其物理语义。
14. 你设计的符号应尽量适合后续提取为一阶逻辑规则，请优先采用"谓词化、离散化、可组合"的命名方式。
15. 词汇设计应尽量使规则最终能自然表达为类似
    IF serving_quality_is_poor AND neighbor_advantage_is_strong AND guard_state_is_idle
    THEN action_is_HandoverToBest
    这样的可读规则。
16. 状态维度名称、状态取值名称、动作语义名称都应避免含糊缩写，尽量使用可直接映射为一阶逻辑原子或谓词的英文短语。
17. 请优先设计"属性-取值"式离散语义
    每个取值都应能直接作为逻辑条件使用，而不是仅作为难以解释的编号。
18. feature_dict 中建议保留足够多的、可支撑后续一阶逻辑解释的中间特征
接口约束：
- 必须定义类名 `LLM_AutoMapper`。
- translate_to_symbolic 必须返回：
  (symbolic_state_tuple, symbolic_action_string, feature_dict)
- symbolic_state_tuple 中每个元素都必须是字符串。
- symbolic_action_string 只能从以下集合中选择：
  {list(BaseSymbolicMapper.ACTION_SYMBOLS)}
- get_symbol_vocabulary 推荐返回 dict；如果你更倾向于直接返回 list/tuple，
  系统会将其解释为 state_dimensions。
- feature_dict 不强制固定字段名，但必须包含足够的可解释中间量，
  以便后续规则统计、诊断和部署。

基类方法说明（可直接调用，不要重新实现）：
- self.decode_state(state) 返回 dict，包含以下键：
  serving_bs, best_bs, best_target_bs, serving_sinr, best_target_sinr,
  best_sinr, second_best_sinr, gap_target_serving, pp_pending, sinr_rank_desc
  使用示例：feature_dict = self.decode_state(state); serving_bs = feature_dict["serving_bs"]
- self.action_to_symbolic(action, feature_dict) 返回动作语义字符串
  使用示例：symbolic_action = self.action_to_symbolic(action, feature_dict)
  注意：必须传入 feature_dict 参数，不能只传 action

一阶逻辑友好性约束：
- 你设计的 state_dimensions 应尽量对应“可被谓词化的属性”。
- 你设计的每个符号值应尽量能读成逻辑原子
- 避免使用难以解释的匿名标签、纯编号标签、无语义缩写标签。
- 避免只返回过度连续化或过度细碎的词汇，尽量保证规则可以被人类阅读、比较、合并和蒸馏。
- 如果多个维度都很重要，优先让它们语义独立、组合后可形成清晰的条件合取。

实现要求：
- 只能输出 Python 代码，不能输出 Markdown，不能输出解释。
- 不要再定义 BaseSymbolicMapper，直接使用已有基类。
- 不要 import symbolic_mapper（也不要从任何模块导入 BaseSymbolicMapper）；直接使用环境提供的 BaseSymbolicMapper。
- 允许使用 numpy、math、deque。
- 代码应具备防御性，能处理异常状态或无效 one-hot。

请直接开始输出完整代码，不要额外解释。
""".strip()

    def _call_openai_compatible_llm(self, prompt: str) -> dict[str, Any]:
        """Call an OpenAI-compatible chat-completions API."""
        api_key = os.getenv(self.settings.api_key_env, "").strip()
        if not api_key:
            raise MapperGenerationError(
                f"Environment variable `{self.settings.api_key_env}` not set; cannot call LLM."
            )

        endpoint = self.settings.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是顶级 3GPP 通信专家和 Python 工程师。"
                        "你只能返回纯 Python 代码，不允许返回 Markdown。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }

        if self.settings.reasoning_effort:
            payload["reasoning_effort"] = self.settings.reasoning_effort

        extra_body = {}
        if self.settings.enable_thinking:
            extra_body["thinking"] = {"type": "enabled"}
        if extra_body:
            payload["extra_body"] = extra_body

        req = request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.settings.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MapperGenerationError(
                f"LLM HTTP request failed, status {exc.code}, detail: {detail}"
            ) from exc
        except error.URLError as exc:
            raise MapperGenerationError(f"LLM network request failed: {exc.reason}") from exc

        try:
            response_json = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MapperGenerationError("LLM response is not valid JSON.") from exc

        return {
            "request_payload": payload,
            "response_json": response_json,
            "raw_content": _extract_chat_content(response_json),
        }

    def _normalize_mapper_output(
        self, mapper: BaseSymbolicMapper, state: np.ndarray, action: int
    ) -> tuple[tuple[str, ...], str, dict[str, Any]]:
        """Normalize mapper output to a 3-tuple (state, action, features)."""
        output = mapper.translate_to_symbolic(state, action)
        if not isinstance(output, tuple):
            raise MapperValidationError("translate_to_symbolic must return a tuple.")

        if len(output) == 2:
            symbolic_state, symbolic_action = output
            feature_dict = mapper.decode_state(state)
        elif len(output) == 3:
            symbolic_state, symbolic_action, feature_dict = output
        else:
            raise MapperValidationError(
                "translate_to_symbolic must return a 2- or 3-tuple."
            )

        if not isinstance(symbolic_state, tuple):
            raise MapperValidationError("symbolic_state must be a tuple of strings.")
        if not all(isinstance(x, str) for x in symbolic_state):
            raise MapperValidationError("All elements in symbolic_state must be strings.")
        if symbolic_action is None:
            symbolic_action = ""
        if not isinstance(symbolic_action, str):
            raise MapperValidationError("symbolic_action must be a string.")
        if not isinstance(feature_dict, dict):
            raise MapperValidationError("feature_dict must be a dict.")
        return symbolic_state, symbolic_action, feature_dict

    @staticmethod
    def _normalize_vocabulary(vocab: Any) -> dict[str, Any]:
        """Normalize LLM vocabulary to dict format."""
        if isinstance(vocab, dict):
            return vocab
        if isinstance(vocab, (list, tuple)) and all(
            isinstance(x, str) for x in vocab
        ):
            return {
                "state_dimensions": list(vocab),
                "state_symbols": {},
                "action_symbols": list(BaseSymbolicMapper.ACTION_SYMBOLS),
            }
        raise MapperValidationError(
            "get_symbol_vocabulary must return a dict or a list/tuple of strings."
        )

    @staticmethod
    def _instantiate_mapper_class(mapper_cls: type, n_bs: int) -> Any:
        """Instantiate mapper class."""
        try:
            return mapper_cls(n_bs=n_bs)
        except TypeError:
            return mapper_cls(n_bs)

    def _load_mapper_from_code(
        self, code_text: str, n_bs: int
    ) -> BaseSymbolicMapper:
        """Dynamically load mapper class via exec()."""
        code_text = _sanitize_python_code(code_text)
        compile(code_text, "<llm_mapper>", "exec")

        exec_globals: dict[str, Any] = {
            "__builtins__": __builtins__,
            "np": np,
            "math": math,
            "deque": deque,
            "Optional": Optional,
            "Any": Any,
            "OrderedDict": OrderedDict,
            "BaseSymbolicMapper": BaseSymbolicMapper,
        }
        exec_locals: dict[str, Any] = {}
        symbolic_mapper_prev = sys.modules.get("symbolic_mapper")
        symbolic_mapper_shim = types.ModuleType("symbolic_mapper")
        symbolic_mapper_shim.BaseSymbolicMapper = BaseSymbolicMapper
        symbolic_mapper_shim.np = np
        symbolic_mapper_shim.math = math
        symbolic_mapper_shim.deque = deque
        symbolic_mapper_shim.Optional = Optional
        symbolic_mapper_shim.Any = Any
        sys.modules["symbolic_mapper"] = symbolic_mapper_shim
        try:
            exec(code_text, exec_globals, exec_locals)
        finally:
            if symbolic_mapper_prev is None:
                sys.modules.pop("symbolic_mapper", None)
            else:
                sys.modules["symbolic_mapper"] = symbolic_mapper_prev

        mapper_cls = exec_locals.get("LLM_AutoMapper") or exec_globals.get(
            "LLM_AutoMapper"
        )
        if mapper_cls is None:
            raise MapperValidationError("Class `LLM_AutoMapper` not found.")
        if not isinstance(mapper_cls, type):
            raise MapperValidationError("`LLM_AutoMapper` is not a class.")
        raw_mapper = self._instantiate_mapper_class(mapper_cls, n_bs=n_bs)
        if issubclass(mapper_cls, BaseSymbolicMapper):
            mapper = raw_mapper
        else:
            if not hasattr(raw_mapper, "get_symbol_vocabulary") or not hasattr(
                raw_mapper, "translate_to_symbolic"
            ):
                raise MapperValidationError(
                    "`LLM_AutoMapper` does not implement the required mapper interface."
                )
            mapper = DelegatingSymbolicMapper(n_bs=n_bs, delegate=raw_mapper)

        if not hasattr(mapper, "get_symbol_vocabulary"):
            raise MapperValidationError("Mapper missing get_symbol_vocabulary method.")
        if not hasattr(mapper, "translate_to_symbolic"):
            raise MapperValidationError("Mapper missing translate_to_symbolic method.")

        self._normalize_vocabulary(mapper.get_symbol_vocabulary())

        mock_state = np.zeros(2 * n_bs + 1, dtype=np.float32)
        mock_state[0] = 1.0
        mock_state[n_bs : 2 * n_bs] = np.linspace(0.1, 0.9, n_bs, dtype=np.float32)
        self._normalize_mapper_output(mapper, mock_state, 0)
        return mapper

    def generate_mapper_code(self, n_bs: int) -> tuple[str, dict[str, Any], str]:
        """Generate mapper source code. Returns (code, meta, mode)."""
        prompt = self.build_prompt(n_bs=n_bs)
        prompt_path = os.path.join(self.output_dir, "stage1_llm_prompt.txt")
        _write_text(prompt_path, prompt)

        # Reuse existing code if present and overwrite is disabled.
        mapper_path = os.path.join(self.output_dir, "stage1_generated_mapper.py")
        raw_path = os.path.join(self.output_dir, "stage1_llm_raw_response.json")
        meta_path = os.path.join(self.output_dir, "stage1_mapper_meta.json")
        if os.path.exists(mapper_path) and not self.settings.auto_overwrite_mapper:
            code_text = _sanitize_python_code(
                open(mapper_path, "r", encoding="utf-8").read()
            )
            return (
                code_text,
                {
                    "stage": "stage1",
                    "mode": "reuse_existing",
                    "prompt_path": prompt_path,
                    "mapper_path": mapper_path,
                    "meta_path": meta_path,
                },
                "reuse_existing",
            )

        if self.settings.enabled:
            try:
                llm_resp = self._call_openai_compatible_llm(prompt)
                code_text = _sanitize_python_code(llm_resp["raw_content"])
                _write_json(raw_path, llm_resp)
                _write_text(mapper_path, code_text)
                _write_json(
                    meta_path,
                    {
                        "stage": "stage1",
                        "mode": "llm",
                        "provider": self.settings.provider,
                        "model": self.settings.model,
                        "api_base": self.settings.api_base,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                return (
                    code_text,
                    {
                        "stage": "stage1",
                        "mode": "llm",
                        "prompt_path": prompt_path,
                        "mapper_path": mapper_path,
                        "raw_path": raw_path,
                        "meta_path": meta_path,
                    },
                    "llm",
                )
            except SymbxrlError as exc:
                _write_json(
                    meta_path,
                    {
                        "stage": "stage1",
                        "mode": "llm_failed_fallback",
                        "provider": self.settings.provider,
                        "model": self.settings.model,
                        "api_base": self.settings.api_base,
                        "llm_error": str(exc),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                if not self.settings.allow_builtin_fallback:
                    raise

        # Fall back to built-in expert bootstrap code when LLM is unavailable.
        code_text = _build_bootstrap_mapper_code()
        _write_text(mapper_path, code_text)
        if not self.settings.enabled:
            _write_json(
                meta_path,
                {
                    "stage": "stage1",
                    "mode": "builtin_fallback",
                    "reason": "LLM call disabled.",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        elif not os.path.exists(meta_path):
            _write_json(
                meta_path,
                {
                    "stage": "stage1",
                    "mode": "builtin_fallback",
                    "reason": "LLM not enabled or request branch not reached.",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        return (
            code_text,
            {
                "stage": "stage1",
                "mode": json.loads(open(meta_path, "r", encoding="utf-8").read()).get(
                    "mode", "builtin_fallback"
                ),
                "prompt_path": prompt_path,
                "mapper_path": mapper_path,
                "meta_path": meta_path,
            },
            "builtin_fallback",
        )

    def build_mapper(self, n_bs: int) -> tuple[BaseSymbolicMapper, StageArtifact]:
        """Run mapper generation end-to-end."""
        code_text, meta, mode = self.generate_mapper_code(n_bs=n_bs)
        try:
            mapper = self._load_mapper_from_code(code_text, n_bs=n_bs)
        except Exception as exc:
            if mode == "builtin_fallback":
                raise
            if not self.settings.allow_builtin_fallback:
                raise MapperValidationError(
                    f"LLM-generated code failed validation and built-in fallback is disabled: {exc}"
                ) from exc

            # Fall back to built-in bootstrap mapper.
            code_text = _build_bootstrap_mapper_code()
            mapper_path = os.path.join(self.output_dir, "stage1_generated_mapper.py")
            meta_path = os.path.join(self.output_dir, "stage1_mapper_meta.json")
            _write_text(mapper_path, code_text)
            _write_json(
                meta_path,
                {
                    "stage": "stage1",
                    "mode": "builtin_fallback_after_validation_error",
                    "reason": str(exc),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            mapper = self._load_mapper_from_code(code_text, n_bs=n_bs)
            meta["mode"] = "builtin_fallback_after_validation_error"

        return mapper, StageArtifact(files=meta, metadata={"stage": "stage1"})


def _parse_speed_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_project_env(root_path: str, config: "Config", use_speed_list: list[int]):
    import ho_optim_drl.dataloader as dl
    import ho_optim_drl.utils as ut
    from ho_optim_drl.gym_env import HandoverEnvPPO

    data_dir = os.path.join(root_path, "data", "processed")
    rsrp_files = dl.get_filenames(data_dir, "rsrp")
    sinr_files = dl.get_filenames(data_dir, "sinr")
    rsrp_files, sinr_files, _ = ut.filenames_speed_filter(
        rsrp_files, sinr_files, use_speed_list
    )

    rsrp_list = []
    sinr_list = []
    sinr_norm_list = []
    for rsrp_f, sinr_f in zip(rsrp_files, sinr_files):
        rsrp, sinr = dl.load_preprocess_dataset(config, data_dir, rsrp_f, sinr_f)
        if config.clip_sinr:
            sinr_norm = ut.clipnorm(
                sinr, config.sinr_lower_clip, config.sinr_upper_clip
            )
        else:
            sinr_norm = sinr
        rsrp_list.append(rsrp)
        sinr_list.append(sinr)
        sinr_norm_list.append(sinr_norm)

    env = HandoverEnvPPO(config, rsrp_list, sinr_list, sinr_norm_list)
    env.set_test_mode(True)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: LLM meta-programming - generate symbolic mapper")
    parser.add_argument(
        "--root-path",
        type=str,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--n-bs",
        type=int,
        default=0,
        help="Optional: specify n_bs directly to skip environment construction (useful if scipy is missing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "llm_generated_symbxrl", "stage1"),
    )
    parser.add_argument(
        "--speed-list",
        type=str,
        default="30,50,70,90",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
    )
    args = parser.parse_args()

    root_path = os.path.abspath(args.root_path)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    settings = LLMSettings.from_config(config)
    if args.disable_llm:
        settings.enabled = False
    if args.force_regenerate:
        settings.auto_overwrite_mapper = True

    n_bs = int(args.n_bs)
    if n_bs <= 0:
        try:
            env = _build_project_env(root_path, config, _parse_speed_list(args.speed_list))
            n_bs = int(env.n_bs)
        except ModuleNotFoundError as exc:
            print(f"[Stage1] Environment construction failed: {exc}")
            print("[Stage1] Likely cause: scipy not installed (required for .mat files).")
            print("[Stage1] Solutions:")
            print("  - Run: pip install -r requirements.txt")
            print("  - Or use: --n-bs <num_bs> to skip environment construction")
            return 1
        except Exception as exc:
            print(f"[Stage1] Environment construction failed: {exc}")
            return 1
    programmer = LLMMetaProgrammer(settings=settings, output_dir=output_dir)
    prompt_text = programmer.build_prompt(n_bs=n_bs)
    if args.print_prompt:
        print(prompt_text)

    mapper, artifact = programmer.build_mapper(n_bs=n_bs)
    summary_path = os.path.join(output_dir, "stage1_generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_bs": n_bs,
                "files": artifact.files,
                "metadata": artifact.metadata,
                "vocabulary": mapper.get_symbol_vocabulary(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Stage1] done. summary={summary_path}")
    print(json.dumps({"files": artifact.files, "metadata": artifact.metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
