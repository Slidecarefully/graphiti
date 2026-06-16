"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

# 这个文件是 Graphiti 节点去重链路中的“确定性去重辅助模块”。
# 它不调用 LLM，也不访问数据库，而是为上层 resolve_extracted_nodes 提供快速、可预测的去重判断。
# 整体逻辑是：
# 1. 先把实体名称标准化；
# 2. 对已有候选节点建立 exact / fuzzy 索引；
# 3. 对新抽取节点先做精确名称匹配；
# 4. 精确匹配不成功时，再对足够可靠的名称做 MinHash/LSH 模糊匹配；
# 5. 如果仍无法确定，就把该节点标记为 unresolved，交给上层 LLM 去重流程处理。

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import blake2b
from typing import TYPE_CHECKING

# 只在类型检查阶段导入 EntityNode，运行时不导入，避免循环依赖。
# 这是因为该模块会被节点维护逻辑导入，而 EntityNode 所在模块可能又间接依赖维护工具。
if TYPE_CHECKING:
    from graphiti_core.nodes import EntityNode


# 下面这些常量控制“什么时候可以相信模糊匹配”。
# 精确匹配不受这些阈值影响；这些阈值只保护 MinHash/LSH 模糊路径。
_NAME_ENTROPY_THRESHOLD = 1.5
_MIN_NAME_LENGTH = 6
_MIN_TOKEN_COUNT = 2
_FUZZY_JACCARD_THRESHOLD = 0.9
_MINHASH_PERMUTATIONS = 32
_MINHASH_BAND_SIZE = 4


# 最保守的名称标准化：只做小写和空白折叠。
# 这个函数用于 exact-name matching，也就是判断两个名称在规范化后是否完全一致。
def _normalize_string_exact(name: str) -> str:
    """Lowercase text and collapse whitespace so equal names map to the same key."""
    normalized = re.sub(r'[\s]+', ' ', name.lower())
    return normalized.strip()


# 更宽松的名称标准化：在 exact 标准化基础上，去掉大多数非字母数字字符，只保留撇号和空格。
# 这个函数用于 fuzzy matching，因为 MinHash 需要从更干净的字符串里生成 n-gram shingles。
def _normalize_name_for_fuzzy(name: str) -> str:
    """Produce a fuzzier form that keeps alphanumerics and apostrophes for n-gram shingles."""
    normalized = re.sub(r"[^a-z0-9' ]", ' ', _normalize_string_exact(name))
    normalized = normalized.strip()
    return re.sub(r'[\s]+', ' ', normalized)


# 用字符级 Shannon entropy 粗略估计一个名称是否“足够具体”。
# 低熵名称通常太短、太重复或信息量太低，例如 "aa", "bob", "LLC"。
# 这类名称如果走模糊匹配，很容易误合并，所以后面会交给 LLM 判断。
def _name_entropy(normalized_name: str) -> float:
    """Approximate text specificity using Shannon entropy over characters.

    We strip spaces, count how often each character appears, and sum
    probability * -log2(probability). Short or repetitive names yield low
    entropy, which signals we should defer resolution to the LLM instead of
    trusting fuzzy similarity.
    """
    if not normalized_name:
        return 0.0

    counts: dict[str, int] = {}
    for char in normalized_name.replace(' ', ''):
        counts[char] = counts.get(char, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


# 判断一个名称是否适合进入 fuzzy matching。
# 这里同时考虑长度、token 数和 entropy：
# 太短且 token 太少的名字直接不信任；
# 通过基本长度检查后，还要 entropy 达标才允许走 MinHash/LSH。
def _has_high_entropy(normalized_name: str) -> bool:
    """Filter out very short or low-entropy names that are unreliable for fuzzy matching."""
    token_count = len(normalized_name.split())
    if len(normalized_name) < _MIN_NAME_LENGTH and token_count < _MIN_TOKEN_COUNT:
        return False

    return _name_entropy(normalized_name) >= _NAME_ENTROPY_THRESHOLD


# 把名称变成 3-gram shingles。
# 例如 "openai" 会变成 {"ope", "pen", "ena", "nai"}。
# shingle 集合用于后面的 MinHash 近似相似度计算。
def _shingles(normalized_name: str) -> set[str]:
    """Create 3-gram shingles from the normalized name for MinHash calculations."""
    cleaned = normalized_name.replace(' ', '')
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()

    return {cleaned[i : i + 3] for i in range(len(cleaned) - 2)}


# 对一个 shingle 加 seed 后做稳定哈希。
# seed 模拟 MinHash 中的不同排列；blake2b 保证同样输入在不同进程中也能得到确定性结果。
def _hash_shingle(shingle: str, seed: int) -> int:
    """Generate a deterministic 64-bit hash for a shingle given the permutation seed."""
    digest = blake2b(f'{seed}:{shingle}'.encode(), digest_size=8)
    return int.from_bytes(digest.digest(), 'big')


# 计算一个 shingle 集合的 MinHash 签名。
# 对每个 seed，都取该 seed 下所有 shingle hash 的最小值。
# 最终得到一个固定长度的签名，用来近似表示原始 shingle 集合。
def _minhash_signature(shingles: Iterable[str]) -> tuple[int, ...]:
    """Compute the MinHash signature for the shingle set across predefined permutations."""
    if not shingles:
        return tuple()

    seeds = range(_MINHASH_PERMUTATIONS)
    signature: list[int] = []
    for seed in seeds:
        min_hash = min(_hash_shingle(shingle, seed) for shingle in shingles)
        signature.append(min_hash)

    return tuple(signature)


# 把 MinHash 签名切成多个 band，用于 LSH 候选召回。
# 只要两个名字在某个 band 上完全相同，就会被放到同一个桶里，成为潜在相似候选。
def _lsh_bands(signature: Iterable[int]) -> list[tuple[int, ...]]:
    """Split the MinHash signature into fixed-size bands for locality-sensitive hashing."""
    signature_list = list(signature)
    if not signature_list:
        return []

    bands: list[tuple[int, ...]] = []
    for start in range(0, len(signature_list), _MINHASH_BAND_SIZE):
        band = tuple(signature_list[start : start + _MINHASH_BAND_SIZE])
        if len(band) == _MINHASH_BAND_SIZE:
            bands.append(band)
    return bands


# 精确计算两个 shingle 集合的 Jaccard 相似度。
# LSH 只是用来快速缩小候选范围，最终是否通过 fuzzy matching 仍然要看这个 Jaccard 分数。
def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity between two shingle sets, handling empty edge cases."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union else 0.0


# 缓存名称到 shingles 的转换结果。
# 去重过程中，同一个候选节点可能被多个抽取节点比较；缓存可以减少重复计算。
@lru_cache(maxsize=512)
def _cached_shingles(name: str) -> set[str]:
    """Cache shingle sets per normalized name to avoid recomputation within a worker."""
    return _shingles(name)


# 候选节点索引集合。
# 上层会先通过语义搜索召回一批 existing_nodes，这里把它们预处理成多种查询结构：
# - nodes_by_uuid：通过 uuid 找候选节点；
# - normalized_existing：支持 exact-name lookup；
# - shingles_by_candidate：支持 Jaccard 计算；
# - lsh_buckets：支持 fuzzy 候选快速召回。
@dataclass
class DedupCandidateIndexes:
    """Precomputed lookup structures that drive entity deduplication heuristics."""

    existing_nodes: list[EntityNode]
    nodes_by_uuid: dict[str, EntityNode]
    normalized_existing: defaultdict[str, list[EntityNode]]
    shingles_by_candidate: dict[str, set[str]]
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]]


# 去重解析过程中的可变状态。
# resolved_nodes 与 extracted_nodes 一一对应；
# uuid_map 记录“新抽取节点 uuid -> 最终节点 uuid”；
# unresolved_indices 记录确定性规则无法判断、需要 LLM 继续处理的节点；
# duplicate_pairs 记录被判定为重复的节点对，便于上层调试或后续处理。
@dataclass
class DedupResolutionState:
    """Mutable resolution bookkeeping shared across deterministic and LLM passes."""

    resolved_nodes: list[EntityNode | None]
    uuid_map: dict[str, str]
    unresolved_indices: list[int]
    duplicate_pairs: list[tuple[EntityNode, EntityNode]] = field(default_factory=list)


# 当新抽取节点被判定为已有节点的重复时，有时不能简单丢弃新节点信息。
# 例如已有节点只有 ["Entity"]，但新抽取节点带有 ["Entity", "Person"]；
# 这时要把更具体的类型提升到已有 canonical node 上。
def _promote_resolved_node(
    extracted_node: EntityNode,
    resolved_node: EntityNode,
) -> EntityNode:
    """Upgrade a generic canonical node when a duplicate carries a specific type."""
    # 如果已有节点本身已经有具体类型，就保持已有节点不变。
    resolved_specific_labels = [label for label in resolved_node.labels if label != 'Entity']
    if resolved_specific_labels:
        return resolved_node

    # 如果新抽取节点也没有具体类型，就没有可提升的信息。
    extracted_specific_labels = [label for label in extracted_node.labels if label != 'Entity']
    if not extracted_specific_labels:
        return resolved_node

    # 合并标签时保持顺序，并避免重复 label。
    # 'Entity' 始终放在前面，随后保留已有标签，再追加新抽取节点的具体标签。
    promoted_labels: list[str] = []
    for label in ['Entity', *resolved_node.labels, *extracted_specific_labels]:
        if label not in promoted_labels:
            promoted_labels.append(label)

    resolved_node.labels = promoted_labels
    return resolved_node


# 为一批候选节点构建 exact 和 fuzzy 两类索引。
# 这个函数通常在每次去重判断前调用一次，避免后续对每个 extracted_node 重复预处理候选节点。
def _build_candidate_indexes(existing_nodes: list[EntityNode]) -> DedupCandidateIndexes:
    """Precompute exact and fuzzy lookup structures once per dedupe run."""
    normalized_existing: defaultdict[str, list[EntityNode]] = defaultdict(list)
    nodes_by_uuid: dict[str, EntityNode] = {}
    shingles_by_candidate: dict[str, set[str]] = {}
    lsh_buckets: defaultdict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)

    for candidate in existing_nodes:
        # exact 索引：规范化后的名称 -> 一个或多个候选节点。
        # 如果同名候选只有一个，可以直接命中；如果同名候选有多个，就会被视为歧义，交给 LLM。
        normalized = _normalize_string_exact(candidate.name)
        normalized_existing[normalized].append(candidate)

        # uuid 索引：LSH 只存 uuid，最终需要通过 uuid 找回节点对象。
        nodes_by_uuid[candidate.uuid] = candidate

        # fuzzy 索引：候选名称先做 fuzzy 标准化，再生成 shingles。
        shingles = _cached_shingles(_normalize_name_for_fuzzy(candidate.name))
        shingles_by_candidate[candidate.uuid] = shingles

        # LSH 索引：将候选节点的 MinHash 签名按 band 放入桶中。
        # 之后新节点只需要查同 band 的桶，就能得到一小批可能相似的候选。
        signature = _minhash_signature(shingles)
        for band_index, band in enumerate(_lsh_bands(signature)):
            lsh_buckets[(band_index, band)].append(candidate.uuid)

    return DedupCandidateIndexes(
        existing_nodes=existing_nodes,
        nodes_by_uuid=nodes_by_uuid,
        normalized_existing=normalized_existing,
        shingles_by_candidate=shingles_by_candidate,
        lsh_buckets=lsh_buckets,
    )


# 确定性去重主函数。
# 它只在“候选集合”内部判断，不负责候选召回；候选召回发生在上层 node_similarity_search。
# 判断顺序很重要：
# 1. 先做 exact-name matching，命中唯一候选则直接 resolve；
# 2. 如果 exact 命中多个候选，说明有歧义，交给 LLM；
# 3. exact 没命中时，检查名称是否足够可靠；
# 4. 可靠名称才进入 MinHash/LSH fuzzy matching；
# 5. fuzzy 分数不够高，仍然交给 LLM。
def _resolve_with_similarity(
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
) -> None:
    """Attempt deterministic resolution using exact name hits and fuzzy MinHash comparisons.

    Exact normalized-name matching runs first for *all* names regardless of
    length or entropy.  The entropy gate only guards the fuzzy (MinHash/LSH)
    path where short or low-entropy names produce unreliable shingle sets.
    """
    for idx, node in enumerate(extracted_nodes):
        normalized_exact = _normalize_string_exact(node.name)
        normalized_fuzzy = _normalize_name_for_fuzzy(node.name)

        # --- exact-name matching (always attempted) ---
        # 第一优先级永远是精确名称匹配。
        # 只要规范化名称完全相同，而且候选唯一，就可以确定为同一节点。
        existing_matches = indexes.normalized_existing.get(normalized_exact, [])
        if len(existing_matches) == 1:
            match = _promote_resolved_node(node, existing_matches[0])
            state.resolved_nodes[idx] = match
            state.uuid_map[node.uuid] = match.uuid
            if match.uuid != node.uuid:
                state.duplicate_pairs.append((node, match))
            continue

        # 如果同一个规范化名称对应多个候选，确定性规则无法知道该选哪个。
        # 这时不冒险合并，而是把它交给 LLM 结合上下文判断。
        if len(existing_matches) > 1:
            # Ambiguous: multiple candidates share the same normalized name.
            # Escalate to LLM so it can pick the best match.
            state.unresolved_indices.append(idx)
            continue

        # --- entropy gate (protects fuzzy matching only) ---
        # 精确匹配没有命中后，才考虑 fuzzy matching。
        # 但短名称、低熵名称的 shingle 很容易误导相似度，因此先用 entropy gate 保护。
        if not _has_high_entropy(normalized_fuzzy):
            state.unresolved_indices.append(idx)
            continue

        # --- fuzzy matching via MinHash/LSH ---
        # 对新节点名称生成 shingles 和 MinHash signature；
        # 再通过 LSH band 找到共享桶的候选 uuid。
        shingles = _cached_shingles(normalized_fuzzy)
        signature = _minhash_signature(shingles)
        candidate_ids: set[str] = set()
        for band_index, band in enumerate(_lsh_bands(signature)):
            candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))

        # 对 LSH 召回的候选计算真实 Jaccard 相似度，选择分数最高的候选。
        best_candidate: EntityNode | None = None
        best_score = 0.0
        for candidate_id in candidate_ids:
            candidate_shingles = indexes.shingles_by_candidate.get(candidate_id, set())
            score = _jaccard_similarity(shingles, candidate_shingles)
            if score > best_score:
                best_score = score
                best_candidate = indexes.nodes_by_uuid.get(candidate_id)

        # 只有超过高阈值的 fuzzy 命中才会被自动合并。
        # 这里阈值是 0.9，说明 fuzzy 路径非常保守，主要处理拼写/标点/轻微变体。
        if best_candidate is not None and best_score >= _FUZZY_JACCARD_THRESHOLD:
            best_candidate = _promote_resolved_node(node, best_candidate)
            state.resolved_nodes[idx] = best_candidate
            state.uuid_map[node.uuid] = best_candidate.uuid
            if best_candidate.uuid != node.uuid:
                state.duplicate_pairs.append((node, best_candidate))
            continue

        # 所有确定性路径都不能可靠判断时，不强行决策，交给上层 LLM 去重。
        state.unresolved_indices.append(idx)


# 显式导出这些内部工具。
# 虽然很多名字以下划线开头，但 __all__ 表示这些函数会被其他维护模块有意引用。
__all__ = [
    'DedupCandidateIndexes',
    'DedupResolutionState',
    '_normalize_string_exact',
    '_normalize_name_for_fuzzy',
    '_has_high_entropy',
    '_minhash_signature',
    '_lsh_bands',
    '_jaccard_similarity',
    '_cached_shingles',
    '_FUZZY_JACCARD_THRESHOLD',
    '_build_candidate_indexes',
    '_promote_resolved_node',
    '_resolve_with_similarity',
]
