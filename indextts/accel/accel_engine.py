import copy
import sys
from typing import List, Optional

import torch
from torch import nn

from .attention import (
    ForwardContext,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
)
from .kv_manager import KVCacheManager, Seq


class AccelInferenceEngine:
    # GPT2InferenceModel assigns the start-of-speech token mel position 0.
    TTS_START_POSITION = 0

    def __init__(
        self,
        model,
        lm_head,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 256,
        num_blocks: int = 128,
        use_cuda_graph: bool = True,
        kv_manager: Optional[KVCacheManager] = None,
    ):
        """
        Args:
            model: The GPT transformer model (should have accel attention)
            lm_head: Language model head for generating logits
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension per head
            block_size: KV cache block size
            num_blocks: Total number of KV cache blocks
            use_cuda_graph: Whether to use CUDA Graph for decode optimization
        """
        self.model = model
        self.lm_head = lm_head
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        self.hidden_size = (
            model.config.hidden_size
            if hasattr(model, "config")
            else head_dim * num_heads
        )
        model_dtype = next(model.parameters()).dtype
        if kv_manager is None:
            kv_manager = KVCacheManager(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                block_size=block_size,
                num_blocks=num_blocks,
                dtype=model_dtype,
            )
        else:
            expected = (num_layers, num_heads, head_dim, block_size, num_blocks)
            actual = (
                kv_manager.num_layers,
                kv_manager.num_heads,
                kv_manager.head_dim,
                kv_manager.block_size,
                kv_manager.num_blocks,
            )
            if actual != expected:
                raise ValueError(
                    f"shared KV cache shape is incompatible: expected {expected}, got {actual}"
                )
            if kv_manager.dtype != model_dtype:
                raise ValueError(
                    "shared KV cache dtype is incompatible: "
                    f"expected {model_dtype}, got {kv_manager.dtype}"
                )
        self.kv_manager = kv_manager
        self.kv_manager.wire_kv_cache_to_model(model)
        # Sampling is sensitive to logit rounding. Lazily retain an fp32 copy
        # of the final norm/head when accelerated inference runs in fp16/bf16.
        self._lm_head_fp32: Optional[nn.Module] = None
        self.current_sequences = []
        self.graphs = {}
        self.graph_vars = None
        self.graph_pool = None
        self.graph_captured = False

    def _prepare_prefill(self, requests: List[Seq]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for req in requests:
            seqlen = len(req)
            input_ids.extend(req[req.num_cached_tokens :])
            positions.extend(list(range(req.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - req.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if req.block_table:
                num_cached = req.num_cached_tokens
                num_total = len(req)

                for token_idx in range(num_cached, num_total):
                    block_idx = token_idx // self.block_size
                    block_offset = token_idx % self.block_size
                    block_id = req.block_table[block_idx]
                    slot_idx = block_id * self.block_size + block_offset
                    slot_mapping.append(slot_idx)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            max_len = max(len(req.block_table) for req in requests)
            block_tables_list = []
            for req in requests:
                table = req.block_table + [-1] * (max_len - len(req.block_table))
                block_tables_list.append(table)
            block_tables = torch.tensor(
                block_tables_list, dtype=torch.int32, pin_memory=True
            ).cuda(non_blocking=True)

        set_forward_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )

        return input_ids, positions

    def _prepare_decode(self, requests: List[Seq]):
        if not requests:
            raise RuntimeError("FATAL: No requests provided to _prepare_decode!")

        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for req in requests:
            input_ids.append(req.last_token)

            pos = len(req) - 1
            if hasattr(self, "_tts_mode") and self._tts_mode:
                pos = self._tts_decode_position(len(req))
            positions.append(pos)

            context_lens.append(len(req))
            slot_mapping.append(
                req.block_table[-1] * self.block_size + req.last_block_num_tokens - 1
            )

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        max_len = max(len(req.block_table) for req in requests)
        block_tables_list = []
        for req in requests:
            table = req.block_table + [-1] * (max_len - len(req.block_table))
            block_tables_list.append(table)
        block_tables = torch.tensor(
            block_tables_list, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        assert block_tables.dim() == 2, (
            f"block_tables must be 2D, got shape {block_tables.shape}"
        )
        assert block_tables.size(0) == len(requests), (
            f"block_tables batch size mismatch: {block_tables.size(0)} vs {len(requests)}"
        )

        set_forward_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        return input_ids, positions

    def _tts_decode_position(self, sequence_length: int) -> int:
        """Match GPT2InferenceModel's generated mel-token positions.

        The stock path uses position 0 for the start token and position 2 for
        the first generated token. ``sequence_length`` already includes the
        token about to be decoded, while ``_tts_prompt_len`` includes the
        start token.
        """
        return sequence_length - self._tts_prompt_len + 1

    def _prepare_sample(self, requests: List[Seq], temperature: float):
        temperatures = [temperature] * len(requests)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    def _compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states in fp32 before applying sampling controls."""
        if self.lm_head is None:
            return self.model.compute_logits(hidden.float())

        head_dtype = next(self.lm_head.parameters()).dtype
        if head_dtype == torch.float32:
            return self.lm_head(hidden.float())

        if self._lm_head_fp32 is None:
            self._lm_head_fp32 = copy.deepcopy(self.lm_head).float()
        return self._lm_head_fp32(hidden.float())

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, sequences: List[Seq]
    ) -> torch.Tensor:
        """Match transformers' repetition-penalty logits processor."""
        penalty = self._gen_rep_penalty
        if penalty == 1.0:
            return logits
        for row_index, sequence in enumerate(sequences):
            if not sequence.token_ids:
                continue
            token_ids = torch.as_tensor(
                sequence.token_ids, device=logits.device, dtype=torch.long
            )
            row = logits[row_index]
            scores = row.gather(0, token_ids)
            scores = torch.where(scores < 0, scores * penalty, scores / penalty)
            row.scatter_(0, token_ids, scores)
        return logits

    def _warp_logits(
        self, logits: torch.Tensor, min_tokens_to_keep: int = 1
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p in transformers order."""
        temperature = self._gen_temperature
        top_k = self._gen_top_k
        top_p = self._gen_top_p

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            keep = min(max(int(top_k), min_tokens_to_keep), logits.size(-1))
            threshold = torch.topk(logits, keep, dim=-1)[0][..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=False, dim=-1
            )
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_remove = cumulative_probs <= (1.0 - top_p)
            sorted_remove[..., -min_tokens_to_keep:] = False
            remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove, float("-inf"))

        return logits

    def _sample_tokens(
        self, logits: torch.Tensor, sequences: List[Seq]
    ) -> torch.Tensor:
        """Apply processors/warpers and sample like single-beam HF generation."""
        logits = self._apply_repetition_penalty(logits.float(), sequences)
        if not self._gen_do_sample:
            return logits.argmax(dim=-1)
        logits = self._warp_logits(logits)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # Match Transformers' single-beam sampling path. Although the previous
        # exponential-race sampler represented the same categorical
        # distribution, it consumed random numbers differently and therefore
        # selected different tokens for the same seed, making direct parity
        # comparisons unnecessarily difficult.
        return torch.multinomial(probabilities, num_samples=1).squeeze(1)

    def _beam_candidates(
        self,
        logits: torch.Tensor,
        sequences: List[Seq],
        beam_scores: torch.Tensor,
        num_beams: int,
    ) -> List[tuple[float, int, int]]:
        """Return ranked ``(score, parent beam, token)`` candidates."""
        # Transformers' beam loop converts logits to log probabilities before
        # applying its LogitsProcessorList. This differs from its single-beam
        # sampling loop and matters substantially for large repetition
        # penalties because log-softmax scores are non-positive.
        processed = torch.log_softmax(logits.float(), dim=-1)
        processed = self._apply_repetition_penalty(processed, sequences)
        if self._gen_do_sample:
            # Transformers keeps at least two tokens for beam sampling so one
            # low-ranked token cannot collapse all beam diversity.
            processed = self._warp_logits(processed, min_tokens_to_keep=2)

        token_scores = processed + beam_scores[:, None]
        flat_scores = token_scores.reshape(-1)
        candidate_count = min(2 * num_beams, flat_scores.numel())

        if self._gen_do_sample:
            probabilities = torch.softmax(flat_scores, dim=0)
            support = int((probabilities > 0).sum().item())
            candidate_count = min(candidate_count, support)
            if candidate_count == 0:
                raise RuntimeError("beam sampling has no finite token candidates")
            candidate_indices = torch.multinomial(
                probabilities, num_samples=candidate_count, replacement=False
            )
            candidate_scores = flat_scores[candidate_indices]
            order = torch.argsort(candidate_scores, descending=True)
            candidate_indices = candidate_indices[order]
            candidate_scores = candidate_scores[order]
        else:
            candidate_scores, candidate_indices = torch.topk(
                flat_scores, k=candidate_count, largest=True, sorted=True
            )

        vocab_size = logits.size(-1)
        parent_indices = torch.div(
            candidate_indices, vocab_size, rounding_mode="floor"
        )
        token_ids = candidate_indices.remainder(vocab_size)
        return list(
            zip(
                candidate_scores.tolist(),
                parent_indices.tolist(),
                token_ids.tolist(),
            )
        )

    @staticmethod
    def _normalized_beam_score(
        score: float, generated_length: int, length_penalty: float
    ) -> float:
        length = max(1, generated_length)
        return score / (length ** length_penalty)

    def _branch_beam_sequences(
        self,
        parents: List[Seq],
        candidates: List[tuple[float, int, int]],
    ) -> tuple[List[Seq], List[float]]:
        """Fork only parents with multiple winners and release discarded beams."""
        selections_by_parent: dict[int, List[tuple[int, float, int]]] = {}
        for output_index, (score, parent_index, token_id) in enumerate(candidates):
            selections_by_parent.setdefault(parent_index, []).append(
                (output_index, score, token_id)
            )

        child_slots: List[Optional[Seq]] = [None] * len(candidates)
        child_scores = [0.0] * len(candidates)
        for parent_index, parent in enumerate(parents):
            selections = selections_by_parent.get(parent_index, [])
            if not selections:
                self.kv_manager.remove_seq(parent)
                continue

            # Reuse the parent for one winner. Only additional children need a
            # cache fork; copy-on-write then duplicates a partial block exactly
            # when two surviving hypotheses truly diverge.
            branches = [parent]
            branches.extend(
                self.kv_manager.fork_sequence(parent)
                for _ in range(len(selections) - 1)
            )
            for branch, (output_index, score, token_id) in zip(
                branches, selections
            ):
                branch.append_token(token_id)
                self.kv_manager.append_to_seq(branch)
                child_slots[output_index] = branch
                child_scores[output_index] = score

        if any(child is None for child in child_slots):
            raise RuntimeError("failed to construct every selected beam branch")
        children = [child for child in child_slots if child is not None]

        self.current_sequences = children
        return children, child_scores

    def _generate_beams(
        self,
        logits: torch.Tensor,
        sequence: Seq,
        prompt_tokens: List[int],
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        stop_tokens: Optional[List[int]],
        tts_mel_embedding: Optional[torch.nn.Module],
        tts_text_pos_embedding: Optional[torch.nn.Module],
        device: torch.device,
    ) -> torch.Tensor:
        """Decode one prompt with paged-cache beam search or beam sampling."""
        stop_token_set = set(stop_tokens or [])
        completed: List[tuple[float, float, List[int]]] = []
        active_sequences = [sequence]
        active_scores = [0.0]

        for generated_step in range(max_new_tokens):
            score_tensor = torch.tensor(
                active_scores, dtype=torch.float32, device=logits.device
            )
            ranked = self._beam_candidates(
                logits, active_sequences, score_tensor, num_beams
            )
            best_step_score = ranked[0][0]

            active_candidates: List[tuple[float, int, int]] = []
            for rank, (score, parent_index, token_id) in enumerate(ranked):
                parent = active_sequences[parent_index]
                parent_generated = parent.token_ids[parent.num_prompt_tokens :]
                if token_id in stop_token_set:
                    # Match Transformers: only EOS candidates ranked within the
                    # beam width become completed hypotheses.
                    if rank < num_beams:
                        generated_length = len(parent_generated) + 1
                        normalized = self._normalized_beam_score(
                            score, generated_length, length_penalty
                        )
                        completed.append(
                            (normalized, score, copy.copy(parent_generated))
                        )
                    continue

                if len(active_candidates) < num_beams:
                    active_candidates.append((score, parent_index, token_id))

            completed.sort(key=lambda item: item[0], reverse=True)
            if len(completed) > num_beams:
                completed = completed[:num_beams]

            if not active_candidates:
                for active in active_sequences:
                    self.kv_manager.remove_seq(active)
                self.current_sequences = []
                active_sequences = []
                active_scores = []
                break

            active_sequences, active_scores = self._branch_beam_sequences(
                active_sequences, active_candidates
            )

            if generated_step + 1 >= max_new_tokens:
                break

            if len(completed) >= num_beams:
                current_length = generated_step + 1
                best_active = self._normalized_beam_score(
                    best_step_score, current_length, length_penalty
                )
                worst_completed = completed[-1][0]
                if worst_completed >= best_active:
                    break

            decode_ids, decode_pos = self._prepare_decode(active_sequences)
            context = get_forward_context()
            hidden_states = self._run_decode_with_graph(
                decode_ids,
                decode_pos,
                context,
                tts_mel_embedding=tts_mel_embedding,
                tts_text_pos_embedding=tts_text_pos_embedding,
            )
            logits = self._compute_logits(hidden_states)
            reset_forward_context()

        for active, score in zip(active_sequences, active_scores):
            generated = active.token_ids[active.num_prompt_tokens :]
            normalized = self._normalized_beam_score(
                score, len(generated), length_penalty
            )
            completed.append((normalized, score, copy.copy(generated)))

        for active in active_sequences:
            if active.block_table:
                self.kv_manager.remove_seq(active)
        self.current_sequences = []

        if not completed:
            best_tokens: List[int] = []
        else:
            best_tokens = max(completed, key=lambda item: item[0])[2]
        return torch.tensor(
            [prompt_tokens + best_tokens], dtype=torch.long, device=device
        )

    def reset_model_state(self, *, weights_changed: bool = True) -> None:
        """Invalidate state tied to the currently loaded GPT checkpoint."""
        if self.current_sequences:
            raise RuntimeError("cannot reset accelerated model state during generation")
        self.kv_manager.reset()
        if weights_changed:
            self._lm_head_fp32 = None

    def _capture_cuda_graphs(self, tts_mel_embedding=None, tts_text_pos_embedding=None):
        print("Capturing CUDA graphs for decode optimization...")
        max_bs = 8  # Support up to batch size 8
        max_num_blocks = (2048 + self.block_size - 1) // self.block_size
        model_dtype = next(self.model.parameters()).dtype
        input_ids = torch.ones(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.ones(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(
            max_bs, max_num_blocks, dtype=torch.int32, device="cuda"
        )
        outputs = torch.zeros(
            max_bs, self.hidden_size, dtype=model_dtype, device="cuda"
        )
        inputs_embeds_buffer = torch.zeros(
            max_bs, self.hidden_size, dtype=model_dtype, device="cuda"
        )

        self.graph_bs = [1, 2, 4, 8]

        use_tts = tts_mel_embedding is not None and tts_text_pos_embedding is not None

        def run_decode(bs):
            # Warmup and capture must execute exactly the same tensor operations.
            if use_tts:
                assert tts_mel_embedding is not None
                assert tts_text_pos_embedding is not None
                emb = tts_mel_embedding(input_ids[:bs])
                pos_clamped = torch.clamp(positions[:bs], min=0)
                pos_emb = tts_text_pos_embedding.emb(pos_clamped)
                inputs_embeds_buffer[:bs] = emb + pos_emb
                out = self.model(
                    inputs_embeds=inputs_embeds_buffer[:bs].unsqueeze(1),
                    return_dict=True,
                ).last_hidden_state
            else:
                out = self.model(
                    input_ids=input_ids[:bs].unsqueeze(1), return_dict=True
                ).last_hidden_state
            outputs[:bs] = out.squeeze(1) if out.dim() == 3 else out

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            slot_mapping[:bs] = torch.arange(bs, dtype=torch.int32, device="cuda")
            context_lens[:bs] = bs + 1
            block_tables[:bs, :] = 0

            set_forward_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            run_decode(bs)
            with torch.cuda.graph(graph, self.graph_pool):
                run_decode(bs)

            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_forward_context()

        self.graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "outputs": outputs,
            "inputs_embeds": inputs_embeds_buffer,
        }
        print(f"CUDA graphs captured for batch sizes: {self.graph_bs}")

    def _run_decode_with_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        context: ForwardContext,
        tts_mel_embedding: Optional[torch.nn.Module] = None,
        tts_text_pos_embedding: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        bs = input_ids.size(0)
        use_tts_embedding = hasattr(self, "_tts_mode") and self._tts_mode

        if not self.use_cuda_graph or not self.graphs:
            if use_tts_embedding:
                assert tts_mel_embedding is not None
                assert tts_text_pos_embedding is not None
                inputs_embeds = tts_mel_embedding(input_ids)
                pos_clamped = torch.clamp(positions, min=0)
                pos_emb = tts_text_pos_embedding.emb(pos_clamped)
                inputs_embeds = inputs_embeds + pos_emb
                out = self.model(
                    inputs_embeds=inputs_embeds.unsqueeze(1), return_dict=True
                ).last_hidden_state
            else:
                out = self.model(
                    input_ids=input_ids.unsqueeze(1), return_dict=True
                ).last_hidden_state
            return out.squeeze(1) if out.dim() == 3 else out

        graph_bs = next((x for x in self.graph_bs if x >= bs), None)
        if graph_bs is None:
            if use_tts_embedding:
                assert tts_mel_embedding is not None
                assert tts_text_pos_embedding is not None
                inputs_embeds = tts_mel_embedding(input_ids)
                pos_clamped = torch.clamp(positions, min=0)
                pos_emb = tts_text_pos_embedding.emb(pos_clamped)
                inputs_embeds = inputs_embeds + pos_emb
                out = self.model(
                    inputs_embeds=inputs_embeds.unsqueeze(1), return_dict=True
                ).last_hidden_state
            else:
                out = self.model(
                    input_ids=input_ids.unsqueeze(1), return_dict=True
                ).last_hidden_state
            return out.squeeze(1) if out.dim() == 3 else out

        graph = self.graphs[graph_bs]
        graph_vars = self.graph_vars

        if graph_vars is None:
            raise RuntimeError("Graph variables not initialized")

        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :].fill_(-1)
        graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
            context.block_tables
        )
        graph.replay()

        return graph_vars["outputs"][:bs]

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tts_embeddings: Optional[
            torch.Tensor
        ] = None,  # TTS: [pad][cond][text] embeddings (87 tokens, NO start_mel)
        tts_mel_embedding: Optional[torch.nn.Module] = None,  # TTS: mel_embedding layer
        tts_text_pos_embedding: Optional[
            torch.nn.Module
        ] = None,  # TTS: text_pos_embedding layer
    ) -> torch.Tensor:
        """
        Generate tokens.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            num_beams: Number of beam hypotheses to retain
            length_penalty: Exponent used to normalize completed beam scores
            stop_tokens: List of token IDs that stop generation

        Returns:
            Generated token IDs [batch_size, total_len]
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must be between zero and one")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero")
        if int(num_beams) < 1:
            raise ValueError("num_beams must be at least one")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least one")
        num_beams = int(num_beams)

        self._gen_temperature = float(temperature)
        self._gen_top_k = top_k
        self._gen_top_p = top_p
        self._gen_rep_penalty = float(repetition_penalty)
        self._gen_do_sample = bool(do_sample)

        batch_size = input_ids.size(0)
        device = input_ids.device
        if num_beams > 1 and batch_size != 1:
            raise NotImplementedError(
                "accelerated beam search currently supports one prompt at a time"
            )

        self._tts_mode = tts_embeddings is not None
        self._tts_prompt_len = input_ids.size(1) if self._tts_mode else 0

        if self.use_cuda_graph and not self.graph_captured:
            print(
                f"[CAPTURE] use_cuda_graph={self.use_cuda_graph}, graph_captured={self.graph_captured}",
                file=sys.stderr,
                flush=True,
            )
            self._capture_cuda_graphs(
                tts_mel_embedding=tts_mel_embedding,
                tts_text_pos_embedding=tts_text_pos_embedding,
            )
            self.graph_captured = True
            print(
                f"[CAPTURE] Completed! graphs={list(self.graphs.keys())}",
                file=sys.stderr,
                flush=True,
            )

        if tts_embeddings is not None:
            actual_seq_len = tts_embeddings.size(1) + 1  # embeddings + start_mel_token
        else:
            actual_seq_len = input_ids.size(1)

        is_varlen_batch = (
            tts_embeddings is not None
            and attention_mask is not None
            and batch_size > 1
            and (attention_mask.sum(dim=1) != attention_mask.size(1)).any()
        )

        if is_varlen_batch:
            seq_lens = [attention_mask[i].sum().item() for i in range(batch_size)]
        else:
            seq_lens = [actual_seq_len] * batch_size

        sequences = []
        for i in range(batch_size):
            seq_len = seq_lens[i]
            token_ids = [1] * seq_len
            if tts_embeddings is not None and seq_len > 0:
                token_ids[-1] = input_ids[i, -1].item() if input_ids.size(1) > 0 else 1
            else:
                token_ids = input_ids[i].tolist()
            req = Seq(token_ids)
            self.kv_manager.allocate(req)
            sequences.append(req)

        self.current_sequences = sequences

        prefill_ids, prefill_pos = self._prepare_prefill(sequences)

        if (
            tts_embeddings is not None
            and tts_mel_embedding is not None
            and tts_text_pos_embedding is not None
        ):
            start_token_id = input_ids[0, -1] if input_ids.size(1) > 0 else 8192

            start_emb = tts_mel_embedding(
                torch.tensor([[start_token_id]], device="cuda")
            )  # [1, 1, hidden_dim]

            start_pos = torch.full_like(
                start_token_id.reshape(1, 1), self.TTS_START_POSITION
            )
            pos_emb = tts_text_pos_embedding.emb(start_pos)
            start_emb = start_emb + pos_emb
            start_emb = start_emb.repeat(batch_size, 1, 1)

            if is_varlen_batch:
                valid_embeddings = []
                for i in range(batch_size):
                    emb_len = seq_lens[i] - 1
                    padding_len = tts_embeddings.size(1) - emb_len
                    valid_emb = tts_embeddings[i, padding_len:].unsqueeze(
                        0
                    )  # [1, emb_len, hidden_dim]
                    valid_embeddings.append(
                        torch.cat([valid_emb, start_emb[i : i + 1]], dim=1)
                    )
                full_embeddings = torch.cat(
                    valid_embeddings, dim=1
                )  # [1, total_tokens, hidden_dim]
            else:
                full_embeddings = torch.cat(
                    [tts_embeddings, start_emb], dim=1
                )  # [batch_size, seq_len, hidden_dim]

            model_dtype = next(self.model.parameters()).dtype
            if full_embeddings.dtype != model_dtype:
                full_embeddings = full_embeddings.to(model_dtype)

            hidden_states = self.model(
                inputs_embeds=full_embeddings, return_dict=True
            ).last_hidden_state

        else:
            hidden_states = self.model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state

        if is_varlen_batch:
            context = get_forward_context()
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            last_hidden = torch.stack(
                [hidden_states[0, cu_seqlens[i + 1] - 1] for i in range(batch_size)]
            )
        else:
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        reset_forward_context()

        logits = self._compute_logits(last_hidden)
        if num_beams > 1:
            return self._generate_beams(
                logits=logits,
                sequence=sequences[0],
                prompt_tokens=input_ids[0].tolist(),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=float(length_penalty),
                stop_tokens=stop_tokens,
                tts_mel_embedding=tts_mel_embedding,
                tts_text_pos_embedding=tts_text_pos_embedding,
                device=device,
            )
        first_token = self._sample_tokens(logits, sequences)

        first_token_list = first_token.tolist()

        generated_tokens = [[] for _ in range(batch_size)]
        is_finished = [False] * batch_size

        for i, token_id in enumerate(first_token_list):
            if stop_tokens and token_id in stop_tokens:
                is_finished[i] = True
            else:
                generated_tokens[i].append(token_id)
                sequences[i].append_token(token_id)
                self.kv_manager.append_to_seq(sequences[i])

        if all(is_finished):
            for req in sequences:
                self.kv_manager.remove_seq(req)
            self.current_sequences = []

            output_ids = []
            for i in range(batch_size):
                full_sequence = input_ids[i].tolist() + generated_tokens[i]
                output_ids.append(full_sequence)

            output = torch.tensor(output_ids, dtype=torch.long, device=device)
            return output

        remaining_tokens = max_new_tokens - 1

        for step in range(remaining_tokens):
            decode_ids, decode_pos = self._prepare_decode(sequences)

            context = get_forward_context()
            hidden_states = self._run_decode_with_graph(
                decode_ids,
                decode_pos,
                context,
                tts_mel_embedding=tts_mel_embedding,
                tts_text_pos_embedding=tts_text_pos_embedding,
            )

            logits = self._compute_logits(hidden_states)

            reset_forward_context()

            next_token = self._sample_tokens(logits, sequences)
            next_token_list = next_token.tolist()

            for i, token_id in enumerate(next_token_list):
                if is_finished[i]:
                    continue
                elif stop_tokens and token_id in stop_tokens:
                    is_finished[i] = True
                else:
                    sequences[i].append_token(token_id)
                    self.kv_manager.append_to_seq(sequences[i])
                    generated_tokens[i].append(token_id)

            if all(is_finished):
                break

        for req in sequences:
            self.kv_manager.remove_seq(req)
        self.current_sequences = []

        pad_token = stop_tokens[0] if stop_tokens else 0

        if is_varlen_batch:
            max_prompt_len = attention_mask.size(1)
            output_ids = []

            for i in range(batch_size):
                padding_len = max_prompt_len - seq_lens[i]
                initial_tokens = sequences[i].token_ids[
                    : sequences[i].num_prompt_tokens
                ]
                padded_prompt = [pad_token] * padding_len + initial_tokens
                full_sequence = padded_prompt + generated_tokens[i]
                output_ids.append(full_sequence)
        else:
            output_ids = [
                sequences[i].token_ids[: sequences[i].num_prompt_tokens]
                + generated_tokens[i]
                for i in range(batch_size)
            ]

        max_length = max(len(seq) for seq in output_ids)
        padded_output_ids = [
            seq + [pad_token] * (max_length - len(seq)) for seq in output_ids
        ]

        output = torch.tensor(padded_output_ids, dtype=torch.long, device=device)

        assert output.size(0) == batch_size, (
            f"Output batch size mismatch: {output.size(0)} != {batch_size}"
        )

        return output
