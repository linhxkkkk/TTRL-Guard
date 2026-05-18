# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Dict
from collections import Counter, deque
import torch
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return "None", 0.0
    
    counter = Counter(model_answers)
    
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)
    
    return majority_answer, majority_ratio


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric

# =============================================================================
# TTRL-Guard: Unified Framework for FRS + MPS + RCSU
# Paper §3.2 / §3.3 / §3.4
# =============================================================================


class TTRLGuard:
    """
    TTRL-Guard unified framework: FRS + MPS + RCSU.

    FRS (Flip-Rate-Aware Reward Scaling):    Down-weights unreliable samples during
                                              high flip-rate periods.
    MPS (Minority-Preserving Sampling):      Protects minority answers during high
                                              flip-rate periods to delay collapse.
    RCSU (Risk-Conditioned Sparse Updating): Stochastically skips gradient updates
                                              for high-risk prompts; fully unsupervised
                                              (no GT/LA dependency).

        High-risk trigger conditions (FR-history pattern, fully unsupervised):
          1. had_competition  : FR > tau_fr observed at least once in history
          2. history_len >= rcsu_window : sufficient history steps accumulated (>= W)
          3. mr_win > rcsu_theta_mr    : recent windowed MR is high (high confidence)
        All three conditions met → classified as high-risk; gradient update skipped
        with probability p_skip.

        Note: stable_steps is NOT used because on hard datasets (e.g., MATH-L5)
        FR fluctuates frequently, causing stable_steps to reset to 0 repeatedly.
        history_len is monotonically increasing and reliably marks "seen enough rounds".

        Skip cap protection: at most 25% of prompts are skipped per step (floor,
        minimum 1), to avoid excessive skipping when had_comp triggers at scale.
        For small batches (e.g., AIME 8 prompts), 25% = at most 2 skipped.

        FRS term-3 trigger cap: each prompt triggers the stable-wrong down-weight
        at most rcsu_window times cumulatively; further triggers are suppressed to
        prevent always-wrong prompts from being indefinitely down-weighted (which
        would starve the gradient signal).
    """

    def __init__(
        self,
        tau_fr: float = 0.3,
        tau_mr: float = 0.6,
        lambda1: float = 0.5,
        lambda2: float = 0.3,
        beta_max: float = 0.3,
        mps_min_votes_denom: int = 4,
        mps_steady: int = 3,
        rcsu_window: int = 5,
        rcsu_theta_mr: float = 0.7,
        rcsu_p_skip: float = 0.7,
        enable_frs: bool = True,
        enable_mps: bool = True,
        enable_rcsu: bool = True,
    ):
        self.tau_fr = tau_fr
        self.tau_mr = tau_mr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.beta_max = beta_max
        self.mps_min_votes_denom = mps_min_votes_denom
        self.mps_steady = mps_steady
        self.rcsu_window = rcsu_window
        self.rcsu_theta_mr = rcsu_theta_mr
        self.rcsu_p_skip = rcsu_p_skip
        self.enable_frs = enable_frs
        self.enable_mps = enable_mps
        self.enable_rcsu = enable_rcsu

        # Per-prompt history: deque of (mv_label, mr)
        self._history: Dict[str, deque] = {}
        self._mps_low_fr_count: Dict[str, int] = {}
        self._mps_active: Dict[str, bool] = {}
        self._rcsu_high_risk: Dict[str, bool] = {}
        # RCSU pattern tracking: FR history shape (fully unsupervised, no LA/GT)
        self._rcsu_had_competition: Dict[str, bool] = {}   # True if FR > tau_fr was ever observed
        self._rcsu_stable_steps: Dict[str, int] = {}       # consecutive steps with FR=0
        # FRS term-3 trigger cap: max rcsu_window cumulative triggers per prompt
        # to avoid starving always-wrong prompts of gradient signal
        self._frs_stable_wrong_cnt: Dict[str, int] = {}    # cumulative trigger count (never reset)

        self._step_count = 0
        self._rng = np.random.default_rng(42)

    def _get_fr(self, key: str) -> float:
        history = self._history.get(key)
        if history is None or len(history) < 2:
            return 0.0
        prev_mv = history[-2][0]
        curr_mv = history[-1][0]
        return 1.0 if prev_mv != curr_mv else 0.0

    def _get_windowed_mr(self, key: str) -> float:
        """Sliding mean of MR over the last rcsu_window steps (unsupervised, no LA/GT)."""
        history = self._history.get(key)
        if history is None or len(history) == 0:
            return 0.5
        recent = list(history)[-self.rcsu_window:]
        mr_mean = float(np.mean([x[1] for x in recent]))
        return mr_mean

    def step(
        self,
        prompt_indices: List,
        mr_list: List[float],
        mv_labels: List[str],
        la_list: Optional[List[float]] = None,   # kept for API compatibility; unused by RCSU (unsupervised)
        answer_vote_counts: Optional[List[Dict]] = None,
        n_samples: int = 32,
    ) -> tuple:
        """
        Called once per training step.

        Returns:
            frs_weights  : np.ndarray[float], shape=[n_prompts]
            rcsu_mask    : np.ndarray[bool],  shape=[n_prompts], False=skip update
            mps_bonus    : List[Dict[str,float]], per-prompt minority answer bonus
            guard_metrics: dict
        """
        n_prompts = len(prompt_indices)
        frs_weights = np.ones(n_prompts, dtype=float)
        rcsu_mask   = np.ones(n_prompts, dtype=bool)
        mps_bonus: List[Dict[str, float]] = [{} for _ in range(n_prompts)]

        # Update history first, then compute FR
        for i, pidx in enumerate(prompt_indices):
            key = str(pidx)
            if key not in self._history:
                self._history[key] = deque(maxlen=max(self.rcsu_window * 3, 20))
                self._mps_low_fr_count[key] = 0
                self._mps_active[key] = False
                self._rcsu_high_risk[key] = False
            self._history[key].append((str(mv_labels[i]), float(mr_list[i])))
            # Always maintain FR history state regardless of enable_rcsu (also needed by FRS)
            if key not in self._rcsu_had_competition:
                self._rcsu_had_competition[key] = False
                self._rcsu_stable_steps[key] = 0
            if key not in self._frs_stable_wrong_cnt:
                self._frs_stable_wrong_cnt[key] = 0

        # Compute current FR based on history
        fr_list = [self._get_fr(str(pidx)) for pidx in prompt_indices]

        # Update FR history state (maintained regardless of RCSU, required by FRS term-3)
        for i, pidx in enumerate(prompt_indices):
            key = str(pidx)
            fr = fr_list[i]
            if fr > self.tau_fr:
                self._rcsu_had_competition[key] = True
                self._rcsu_stable_steps[key] = 0
            else:
                self._rcsu_stable_steps[key] = self._rcsu_stable_steps.get(key, 0) + 1

        # ── FRS ──────────────────────────────────────────────────────────────
        frs_trigger_count = 0
        frs_double_trigger_count = 0
        frs_stable_wrong_count = 0  # stably high-MR but never had competition (suspected wrong steady-state)
        if self.enable_frs:
            for i, pidx in enumerate(prompt_indices):
                fr = fr_list[i]
                mr = float(mr_list[i])
                key = str(pidx)
                # Term 1: higher FR -> overall down-weight
                w = 1.0 - self.lambda1 * fr
                # Term 2: high-MR AND high-FR contradiction -> extra penalty (pathological-B signal)
                if mr > self.tau_mr and fr > self.tau_fr:
                    w *= (1.0 - self.lambda2)
                    frs_double_trigger_count += 1
                if fr > self.tau_fr:
                    frs_trigger_count += 1
                # Term 3: never had competition (had_competition=False) but sufficient history
                # and consistently high MR -> model "always stable on one answer", likely wrong
                # steady-state (pathological-C). Use history_len >= rcsu_window instead of
                # stable_steps for consistency with RCSU logic.
                had_comp = self._rcsu_had_competition.get(key, False)
                history_len = len(self._history.get(key, []))
                mr_win = self._get_windowed_mr(key)
                if (not had_comp and history_len >= self.rcsu_window and mr_win > self.tau_mr):
                    # Never flipped + enough history + high confidence -> possible wrong steady-state
                    # Cap: at most rcsu_window cumulative triggers to avoid indefinite down-weighting
                    _sw_cnt = self._frs_stable_wrong_cnt.get(key, 0)
                    if _sw_cnt < self.rcsu_window:
                        w *= (1.0 - self.lambda2 * 0.5)
                        self._frs_stable_wrong_cnt[key] = _sw_cnt + 1
                        frs_stable_wrong_count += 1
                frs_weights[i] = max(0.1, w)

        # ── RCSU ─────────────────────────────────────────────────────────────
        # Pattern 4 (FR history shape, fully unsupervised):
        #   HighRisk = had_competition AND history_len >= W AND mr_win > theta_MR
        # history_len (monotonically increasing) is used instead of stable_steps
        # (frequently reset to 0 on hard datasets like MATH-L5, making it unreliable).
        # had_comp=True confirms the prompt once showed competition (model not always
        # locked on one answer); combined with high mr_win -> "competed then re-locked"
        # = high risk.
        #
        # Skip cap: limit skips to at most 25% of prompts per step to avoid
        # skip_rate > 0.5 when had_comp triggers at scale.
        # FR history state is already updated above (_rcsu_had_competition / _rcsu_stable_steps).
        rcsu_skip_count = 0
        rcsu_highrisk_count = 0
        _rcsu_max_skip = max(1, int(n_prompts * 0.25))  # max 25% skipped per step (AIME 8 prompts -> max 2)
        if self.enable_rcsu:
            for i, pidx in enumerate(prompt_indices):
                key = str(pidx)
                # Windowed MR mean
                mr_win = self._get_windowed_mr(key)
                # High-risk condition: had competition + enough history + currently high confidence
                had_comp    = self._rcsu_had_competition.get(key, False)
                history_len = len(self._history.get(key, []))
                history_ok  = history_len >= self.rcsu_window
                mr_ok       = mr_win > self.rcsu_theta_mr
                is_high_risk = had_comp and history_ok and mr_ok
                self._rcsu_high_risk[key] = is_high_risk
                if is_high_risk:
                    rcsu_highrisk_count += 1
                    # Skip condition: random sample + below per-step skip cap
                    if self._rng.random() < self.rcsu_p_skip and rcsu_skip_count < _rcsu_max_skip:
                        rcsu_mask[i] = False
                        rcsu_skip_count += 1

        # ── MPS ──────────────────────────────────────────────────────────────
        mps_active_count = 0
        mps_protected_prompts = 0
        mps_bonus_total = 0.0    # total bonus across all prompts this step, for mps_bonus_mean metric
        mps_bonus_prompts = 0    # number of prompts that actually received bonus this step
        if self.enable_mps and answer_vote_counts is not None:
            min_votes = max(1, n_samples // self.mps_min_votes_denom)
            for i, pidx in enumerate(prompt_indices):
                key = str(pidx)
                fr = fr_list[i]
                mv = str(mv_labels[i])

                # Update MPS active state: activate on high FR; deactivate after mps_steady consecutive low-FR steps
                if fr > self.tau_fr:
                    self._mps_active[key] = True
                    self._mps_low_fr_count[key] = 0
                elif self._mps_active[key]:
                    self._mps_low_fr_count[key] += 1
                    if self._mps_low_fr_count[key] >= self.mps_steady:
                        self._mps_active[key] = False

                if self._mps_active[key]:
                    mps_active_count += 1
                    vote_dict = (answer_vote_counts[i]
                                 if answer_vote_counts is not None and i < len(answer_vote_counts)
                                 else {})
                    if not isinstance(vote_dict, dict):
                        vote_dict = {}
                    # Minority candidates: vote count >= min_votes and not the MV answer
                    minority_candidates = {
                        ans: cnt for ans, cnt in vote_dict.items()
                        if cnt >= min_votes and ans != mv
                    }
                    if minority_candidates:
                        mps_protected_prompts += 1
                        total_minority = sum(minority_candidates.values())
                        # beta_t scales positively with FR
                        beta_t = self.beta_max * min(fr / max(self.tau_fr, 1e-6), 1.0)
                        prompt_bonus_sum = 0.0
                        for ans, cnt in minority_candidates.items():
                            b = beta_t * (cnt / max(total_minority, 1))
                            mps_bonus[i][ans] = b
                            prompt_bonus_sum += b
                        mps_bonus_total += prompt_bonus_sum
                        mps_bonus_prompts += 1

        # ── Metrics ──────────────────────────────────────────────────────────
        fr_arr = np.array(fr_list)
        mr_arr = np.array(mr_list, dtype=float)

        guard_metrics = {
            "guard/frs_weight_mean":           float(np.mean(frs_weights)),
            "guard/frs_weight_min":            float(np.min(frs_weights)),
            "guard/frs_trigger_rate":          float(frs_trigger_count) / max(n_prompts, 1),
            "guard/frs_double_trigger_rate":   float(frs_double_trigger_count) / max(n_prompts, 1),
            "guard/frs_stable_wrong_rate":     float(frs_stable_wrong_count) / max(n_prompts, 1),
            "guard/rcsu_highrisk_rate":        float(rcsu_highrisk_count) / max(n_prompts, 1),
            "guard/rcsu_skip_rate":            float(rcsu_skip_count) / max(n_prompts, 1),
            "guard/mps_active_rate":           float(mps_active_count) / max(n_prompts, 1),
            "guard/mps_protected_rate":        float(mps_protected_prompts) / max(n_prompts, 1),
            "guard/mps_bonus_rate":            float(mps_bonus_prompts) / max(n_prompts, 1),
            "guard/mps_bonus_mean":            float(mps_bonus_total / max(mps_bonus_prompts, 1)),
            "guard/fr_mean":                   float(np.mean(fr_arr)),
            "guard/fr_high_rate":              float(np.mean(fr_arr > self.tau_fr)),
            "guard/mr_mean":                   float(np.mean(mr_arr)),
            "guard/rcsu_had_competition_rate": float(sum(self._rcsu_had_competition.get(str(p), False) for p in prompt_indices)) / max(n_prompts, 1),
            "guard/rcsu_stable_mean":          float(np.mean([self._rcsu_stable_steps.get(str(p), 0) for p in prompt_indices])),
            "guard/frs_enabled":               float(self.enable_frs),
            "guard/mps_enabled":               float(self.enable_mps),
            "guard/rcsu_enabled":              float(self.enable_rcsu),
            "guard/step_count":                float(self._step_count),
        }
        self._step_count += 1
        return frs_weights, rcsu_mask, mps_bonus, guard_metrics

    def is_high_risk(self, prompt_index) -> bool:
        return self._rcsu_high_risk.get(str(prompt_index), False)


def compute_guard_advantage(
    advantages: "torch.Tensor",
    frs_weights: np.ndarray,
    rcsu_mask: np.ndarray,
    n: int,
) -> "torch.Tensor":
    """
    Apply FRS weights and RCSU mask to the advantage tensor.

    advantages is a 2D token-level tensor, shape = (bs, response_length),
    where bs = n_prompts * n_samples_per_prompt.
    frs_weights / rcsu_mask are per-prompt 1D arrays, shape = (n_prompts,).

    Processing:
      1. Repeat per-prompt weights n times -> per-sample weights, shape = (bs,)
      2. unsqueeze(-1) -> shape = (bs, 1), broadcasts to (bs, response_length)

    Args:
        advantages  : shape=(bs, response_length) or (bs,), token-level advantage
        frs_weights : shape=(n_prompts,)
        rcsu_mask   : shape=(n_prompts,), False=skip update for that prompt
        n           : n_samples_per_prompt
    """
    import torch
    # per-prompt -> per-sample: each prompt has n samples
    frs_expanded  = np.repeat(frs_weights, n)           # (bs,)
    rcsu_expanded = np.repeat(rcsu_mask.astype(float), n)  # (bs,)

    frs_tensor  = torch.tensor(frs_expanded,  dtype=advantages.dtype, device=advantages.device)
    rcsu_tensor = torch.tensor(rcsu_expanded, dtype=advantages.dtype, device=advantages.device)

    # If advantages is 2D (bs, response_length), unsqueeze to broadcast over token dim
    if advantages.dim() == 2:
        frs_tensor  = frs_tensor.unsqueeze(-1)   # (bs, 1) -> broadcast to (bs, response_length)
        rcsu_tensor = rcsu_tensor.unsqueeze(-1)

    return advantages * frs_tensor * rcsu_tensor


def apply_mps_bonus_to_advantage(
    advantages: "torch.Tensor",
    mps_bonus: List[Dict[str, float]],
    batch,
    n: int,
    tokenizer,
) -> "torch.Tensor":
    """
    Add MPS minority-preserving bonus to the advantage tensor.

    MPS design intent: apply a positive advantage bias to samples with minority
    answers, to delay the collapse of correct minority answers.

    For each prompt with minority candidates (mps_bonus[i] is non-empty):
      - Decode each sample's response for that prompt
      - If the extracted answer matches a minority candidate, add the bonus
      - bonus = beta_t x prompt_adv_scale, where prompt_adv_scale is based on
        the average absolute advantage magnitude for that prompt

    advantages shape: (bs, response_length), bs = n_prompts * n_samples_per_prompt

    Note on batch layout (verl DataProto.repeat(interleave=True) uses repeat_interleave):
       layout = [p0_s0, p0_s1, ..., p0_sN, p1_s0, p1_s1, ..., p1_sN, ...]
       -> sample k of prompt rank r has batch index r * n + k
       (NOT k * n_prompts + r, which is the interleave=False layout)

    Args:
        advantages : (bs, seq_len) or (bs,)
        mps_bonus  : per-prompt list, each element is {minority_answer: beta}
        batch      : DataProto, used to decode sample responses
        n          : n_samples_per_prompt
        tokenizer  : used to decode response ids
    """
    import torch
    if not any(mps_bonus):
        return advantages  # no MPS bonus at all, return unchanged

    adv_mod = advantages.clone()
    bs = advantages.shape[0]
    # verl DataProto.repeat(interleave=True) uses torch.repeat_interleave:
    # layout = [p0_s0, p0_s1,..., p0_sN, p1_s0,..., p1_sN, ...]
    # -> sample k of prompt rank r has index = r * n + k

    for r, bonus_dict in enumerate(mps_bonus):
        if not bonus_dict:
            continue
        # Collect batch indices for all samples of this prompt
        sample_indices = [r * n + k for k in range(n) if r * n + k < bs]
        if not sample_indices:
            continue

        # Use the average absolute advantage of this prompt as scale reference
        prompt_adv = advantages[sample_indices]  # (n, seq_len) or (n,)
        adv_scale = float(prompt_adv.abs().mean().item()) + 1e-6

        for s_idx in sample_indices:
            try:
                item = batch[s_idx]
                # Decode this sample's response to extract the answer
                resp_ids  = item.batch["responses"]
                attn_mask = item.batch["attention_mask"]
                plen      = item.batch["prompts"].shape[-1]
                vlen      = int(attn_mask[plen:].sum().item())
                resp_str  = tokenizer.decode(resp_ids[:vlen], skip_special_tokens=True)
                ans = extract_answer(resp_str)
                if ans is not None:
                    try:
                        ans = simplify_expression_string(ans)
                    except Exception:
                        pass
                # Check whether the answer is a minority candidate
                bonus_val = 0.0
                if ans is not None and ans in bonus_dict:
                    bonus_val = bonus_dict[ans] * adv_scale
                elif ans is None and bonus_dict:
                    # When answer extraction fails, give a tiny bonus to preserve exploration
                    bonus_val = float(np.mean(list(bonus_dict.values()))) * adv_scale * 0.05

                if bonus_val > 0:
                    adv_mod[s_idx] = adv_mod[s_idx] + bonus_val
            except Exception:
                continue

    return adv_mod


def compute_guard_from_batch(
    batch,
    n: int,
    guard: "TTRLGuard",
) -> tuple:
    """
    Extract information from a DataProto batch and call guard.step().
    This is the direct entry point called from ray_trainer.py.

    Returns:
        frs_weights, rcsu_mask, mps_bonus, guard_metrics
    """
    n_prompts = len(batch) // n

    prompt_indices: List[str] = []
    mr_list: List[float] = []
    mv_labels: List[str] = []
    answer_vote_counts: List[Dict] = []

    raw_mr = batch.non_tensor_batch.get("majority_ratio_list", None)
    if raw_mr is not None and len(raw_mr) != n_prompts:
        raw_mr = raw_mr[::n][:n_prompts]

    seen_prompts: set = set()
    prompt_order: List[int] = []

    for i in range(len(batch)):
        item = batch[i]
        pidx = str(item.non_tensor_batch.get("extra_info", {}).get("index", i // n))
        if pidx not in seen_prompts:
            seen_prompts.add(pidx)
            prompt_order.append(i)
            prompt_indices.append(pidx)

    if len(prompt_order) != n_prompts:
        prompt_order = [i * n for i in range(n_prompts)]
        prompt_indices = [str(i) for i in range(n_prompts)]

    for rank, i in enumerate(prompt_order):
        item = batch[i]
        mv_gt = item.non_tensor_batch["reward_model"].get("majority_gt", "")
        mr    = float(raw_mr[rank]) if raw_mr is not None and rank < len(raw_mr) else 0.5
        mv_labels.append(str(mv_gt))
        mr_list.append(mr)
        vote_dict = item.non_tensor_batch.get("reward_model", {}).get("vote_counts", {})
        answer_vote_counts.append(vote_dict if isinstance(vote_dict, dict) else {})

    return guard.step(
        prompt_indices=prompt_indices,
        mr_list=mr_list,
        mv_labels=mv_labels,
        la_list=None,   # RCSU is unsupervised (pattern 4), does not use LA/GT
        answer_vote_counts=answer_vote_counts if any(answer_vote_counts) else None,
        n_samples=n,
    )



