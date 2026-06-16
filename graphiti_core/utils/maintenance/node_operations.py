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

# 这个文件是 Graphiti 写入链路中的“实体节点维护”模块：
# 负责从 episode 中抽取实体、把新实体和已有图谱节点去重对齐，并补全节点属性、摘要与 embedding。
# 阅读顺序可以按：抽取节点 -> 构造 EntityNode -> 候选召回 -> 确定性/LLM 去重 -> 属性补全 -> 摘要生成。
import logging
from collections.abc import Awaitable, Callable
from time import time
from typing import Any

from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
    create_entity_node_embeddings,
)
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate, NodeResolutions
from graphiti_core.prompts.extract_nodes import (
    ExtractedEntities,
    ExtractedEntity,
    SummarizedEntities,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import node_similarity_search
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.attribute_utils import apply_capped_attributes
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _normalize_string_exact,
    _promote_resolved_node,
    _resolve_with_similarity,
)
from graphiti_core.utils.text_utils import (
    MAX_SUMMARY_CHARS,
    concatenate_episodes,
    truncate_at_sentence,
)

# logger 贯穿抽取、去重和摘要流程，用来记录耗时、异常响应以及 LLM 返回不规范时的保护性日志。
logger = logging.getLogger(__name__)

# Maximum number of nodes to summarize in a single LLM call
# 下面几个常量控制性能和质量的平衡：摘要批量大小、去重候选数量、向量相似度下限。
MAX_NODES = 30
NODE_DEDUP_CANDIDATE_LIMIT = 15
NODE_DEDUP_COSINE_MIN_SCORE = 0.6

# 节点摘要过滤器是一个异步钩子：外部可以决定某些节点是否需要重新生成 summary。
NodeSummaryFilter = Callable[[EntityNode], Awaitable[bool]]


# 主入口之一：从一个或多个 EpisodicNode 中抽取实体节点。
# 它只负责“抽取并初步整理”，并不负责和数据库已有节点做语义去重；真正的图谱级去重在 resolve_extracted_nodes。
async def extract_nodes(
    clients: GraphitiClients,
    episode: EpisodicNode | list[EpisodicNode],
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    custom_extraction_instructions: str | None = None,
) -> tuple[list[EntityNode], dict[str, list[int]]]:
    """Extract entity nodes from one or more episodes.

    Parameters
    ----------
    episode : EpisodicNode | list[EpisodicNode]
        A single episode or a list of episodes to extract entities from.
        When a list is provided, their contents are concatenated for extraction
        and the first episode is used for metadata (source type, group_id, etc.).

    Returns
    -------
    tuple[list[EntityNode], dict[str, list[int]]]
        A tuple of (extracted_nodes, node_episode_index_map) where
        node_episode_index_map maps node UUID to a list of 0-indexed episode
        positions that the node was extracted from.
    """
    # 统一把单 episode 和多 episode 输入转成列表，后续逻辑都按列表处理；第一个 episode 负责提供 group/source 等元数据。
    episodes = episode if isinstance(episode, list) else [episode]
    primary_episode = episodes[0]

    start = time()
    llm_client = clients.llm_client

    # Build entity types context
    # 先把 ontology 信息整理成 prompt 上下文，后面 LLM 抽取实体时会用 entity_type_id 指向这些类型。
    entity_types_context = _build_entity_types_context(entity_types)

    # Build episode attribution instructions for multi-episode extraction
    # 多 episode 合并抽取时，需要让 LLM 标注每个实体来自哪些 episode，方便后面建立正确的 MENTIONS 边。
    episode_attribution = ''
    if len(episodes) > 1:
        episode_attribution = (
            '\n7. **Episode Attribution**: The content contains multiple episodes labeled '
            '[Episode 0], [Episode 1], etc. Each episode header includes a timestamp indicating '
            'when that episode occurred. For each extracted entity, set `episode_indices` '
            'to the 0-based list of episode numbers where that entity is mentioned. '
            'An entity appearing in Episodes 0 and 2 should have `episode_indices: [0, 2]`.'
        )

    # Build base context
    # LLM 抽取实体所需上下文由三部分组成：当前 episode 内容、历史 episode 作为背景、自定义抽取指令/实体类型约束。
    context = {
        'episode_content': concatenate_episodes(episodes),
        'episode_timestamp': primary_episode.valid_at.isoformat(),
        'previous_episodes': [
            {
                'content': ep.content,
                'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
            }
            for ep in previous_episodes
        ],
        'custom_extraction_instructions': (custom_extraction_instructions or '')
        + episode_attribution,
        'entity_types': entity_types_context,
        'source_description': primary_episode.source_description,
    }

    # Extract entities
    # 进入 LLM 抽取：这里得到的还是 prompt schema 层的 ExtractedEntity，不是最终图节点。
    extracted_entities = await _extract_nodes_single(llm_client, primary_episode, context)

    # Filter empty names
    # 空名字实体没有图谱意义，直接过滤，避免后续生成无效节点或污染去重。
    filtered_entities = [e for e in extracted_entities if e.name.strip()]

    end = time()
    logger.debug(f'Extracted {len(filtered_entities)} entities in {(end - start) * 1000:.0f} ms')

    # Convert to EntityNode objects with episode attribution
    # 把抽取结果转成 EntityNode，同时记录每个节点对应的 episode index，给后续 episodic edge 使用。
    extracted_nodes, node_episode_index_map = _create_entity_nodes(
        filtered_entities, entity_types_context, excluded_entity_types, episodes
    )

    # 清理同一轮抽取中的完全重复节点；这不是最终去重，只是本批结果内部的轻量规整。
    extracted_nodes = _collapse_exact_duplicate_extracted_nodes(
        extracted_nodes, node_episode_index_map
    )

    logger.debug(f'Extracted nodes: {[n.uuid for n in extracted_nodes]}')
    return extracted_nodes, node_episode_index_map


# 把外部传入的自定义实体类型转换成 LLM prompt 可读的结构。
# 默认总会放入 Entity 类型，避免没有自定义 ontology 时无法抽取通用实体。
def _build_entity_types_context(
    entity_types: dict[str, type[BaseModel]] | None,
) -> list[dict]:
    """Build entity types context with ID mappings."""
    entity_types_context = [
        {
            'entity_type_id': 0,
            'entity_type_name': 'Entity',
            'entity_type_description': (
                'A specific, identifiable entity that does not fit any of the other listed '
                'types. Must still be a concrete, meaningful thing — specific enough to be '
                'uniquely identifiable. GOOD: a named entity not covered by the other types. '
                'BAD: "luck", "ideas", "tomorrow", "things", "them", "everybody", '
                '"a sense of wonder", "great times". '
                'When in doubt, do not extract the entity.'
            ),
        }
    ]

    if entity_types is not None:
        entity_types_context += [
            {
                'entity_type_id': i + 1,
                'entity_type_name': type_name,
                'entity_type_description': type_model.__doc__,
            }
            for i, (type_name, type_model) in enumerate(entity_types.items())
        ]

    return entity_types_context


# 在节点去重 prompt 中，需要给 LLM 一个实体类型说明，帮助它判断两个名字相似的节点是否真的同一实体。
def _get_entity_type_description(
    labels: list[str], entity_types: dict[str, type[BaseModel]] | None
) -> str:
    type_name = next((item for item in labels if item != 'Entity'), '')
    type_model = entity_types.get(type_name) if entity_types is not None else None
    return (type_model.__doc__ if type_model is not None else None) or 'Default Entity Type'


# 类型 docstring 可能包含大量 GOOD/BAD 示例，摘要阶段不需要这些抽取指导，所以这里压缩成更干净的类型描述。
def _truncate_type_description(docstring: str) -> str:
    """Extract a concise type description from a docstring for summary prompts.

    Returns the first paragraph (up to the first blank line), capped at 3
    sentences.  This strips GOOD/BAD examples, trigger patterns, and other
    extraction-specific guidance that is irrelevant to summarization.
    """
    # Take only the first paragraph.
    paragraph_lines: list[str] = []
    for line in docstring.splitlines():
        if not line.strip():
            if paragraph_lines:
                break
            continue  # skip leading blank lines
        paragraph_lines.append(line)

    text = ' '.join(line.strip() for line in paragraph_lines)

    # Cap at 3 sentences.
    sentences: list[str] = []
    remaining = text
    for _ in range(3):
        idx = _find_sentence_end(remaining)
        if idx == -1:
            sentences.append(remaining)
            remaining = ''
            break
        sentences.append(remaining[: idx + 1])
        remaining = remaining[idx + 1 :].lstrip()
    return ' '.join(sentences).strip()


# 这个小工具为上面的 docstring 截断服务：尽量找到真正的句子边界，而不是误切 e.g.、Dr.、2.0 这类缩写或小数。
def _find_sentence_end(text: str) -> int:
    """Return the index of the first sentence boundary.

    A sentence ends at `.`, `!`, or `?` when followed by end-of-string or
    a space then an uppercase letter.  This avoids splitting on abbreviations
    like "e.g.", "Dr.", or decimals like "2.0".
    """
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in '.!?':
            continue
        # End of string counts as a sentence boundary.
        if i + 1 >= n:
            return i
        # Space followed by an uppercase letter is a sentence boundary.
        if text[i + 1] == ' ' and i + 2 < n and text[i + 2].isupper():
            return i
    return -1


# 单次 LLM 抽取封装：统一把 LLM 的 dict 响应转换成 ExtractedEntities 模型，再返回实体列表。
async def _extract_nodes_single(
    llm_client: LLMClient,
    episode: EpisodicNode,
    context: dict,
) -> list[ExtractedEntity]:
    """Extract entities using a single LLM call."""
    llm_response = await _call_extraction_llm(llm_client, episode, context)
    response_object = ExtractedEntities(**llm_response)
    return response_object.extracted_entities


# 根据 episode 的来源类型选择不同 prompt。
# message/text/json 的信息结构不同，因此抽取提示词也分开，以减少 LLM 误读输入格式。
async def _call_extraction_llm(
    llm_client: LLMClient,
    episode: EpisodicNode,
    context: dict,
) -> dict:
    """Call the appropriate extraction prompt based on episode type."""
    if episode.source == EpisodeType.message:
        prompt = prompt_library.extract_nodes.extract_message(context)
        prompt_name = 'extract_nodes.extract_message'
    elif episode.source == EpisodeType.text:
        prompt = prompt_library.extract_nodes.extract_text(context)
        prompt_name = 'extract_nodes.extract_text'
    elif episode.source == EpisodeType.json:
        prompt = prompt_library.extract_nodes.extract_json(context)
        prompt_name = 'extract_nodes.extract_json'
    else:
        # Fallback to text extraction
        prompt = prompt_library.extract_nodes.extract_text(context)
        prompt_name = 'extract_nodes.extract_text'

    # 所有抽取 prompt 最终都走统一的 LLMClient.generate_response，并用 Pydantic response_model 约束返回结构。
    return await llm_client.generate_response(
        prompt,
        response_model=ExtractedEntities,
        group_id=episode.group_id,
        prompt_name=prompt_name,
    )


# 把 LLM 抽出的轻量 ExtractedEntity 转成图谱层的 EntityNode。
# 这里会处理实体类型、排除类型、group_id 归属，以及多 episode 场景下的来源归因。
def _create_entity_nodes(
    extracted_entities: list[ExtractedEntity],
    entity_types_context: list[dict],
    excluded_entity_types: list[str] | None,
    episodes: list[EpisodicNode],
) -> tuple[list[EntityNode], dict[str, list[int]]]:
    """Convert ExtractedEntity objects to EntityNode objects.

    Returns
    -------
    tuple[list[EntityNode], dict[str, list[int]]]
        A tuple of (nodes, node_episode_index_map) where node_episode_index_map
        maps each node UUID to 0-indexed episode positions the node was extracted from.
    """
    primary_episode = episodes[0]
    extracted_nodes = []
    node_episode_index_map: dict[str, list[int]] = {}

    # 每个 ExtractedEntity 都会被转成一个临时 EntityNode；最终是否复用已有节点，要等 resolve_extracted_nodes 决定。
    for extracted_entity in extracted_entities:
        type_id = extracted_entity.entity_type_id
        if 0 <= type_id < len(entity_types_context):
            entity_type_name = entity_types_context[type_id].get('entity_type_name')
        else:
            entity_type_name = 'Entity'

        # Check if this entity type should be excluded
        # 如果调用方排除了某些实体类型，这里直接跳过，不进入后续去重和保存链路。
        if excluded_entity_types and entity_type_name in excluded_entity_types:
            logger.debug(f'Excluding entity of type "{entity_type_name}"')
            continue

        # labels 至少包含 Entity，同时包含具体类型；用 set 去掉 Entity 与自定义类型重名时的重复。
        labels: list[str] = list({'Entity', str(entity_type_name)})

        # 新建的是“候选节点”：有 name/type/group/created_at，但 summary 和属性还未补全。
        new_node = EntityNode(
            name=extracted_entity.name,
            group_id=primary_episode.group_id,
            labels=labels,
            summary='',
            created_at=utc_now(),
        )
        extracted_nodes.append(new_node)

        # Map node to 0-indexed episode positions (LLM returns 0-indexed).
        # Clamp to valid range; fall back to all episodes if empty.
        # LLM 给出的 episode_indices 也要防御性校验；越界索引会被丢弃。
        indices = [i for i in extracted_entity.episode_indices if 0 <= i < len(episodes)]
        if not indices:
            indices = list(range(len(episodes)))
        node_episode_index_map[new_node.uuid] = indices

        logger.debug(f'Created new node: {new_node.uuid}')

    return extracted_nodes, node_episode_index_map


# 第一层去重：只合并同一批抽取结果里“规范化名称完全相同”的重复实体。
# 这一步很窄，目的是清理 LLM 在同一次抽取中的明显重复；跨图谱的语义去重留给 resolve_extracted_nodes。
def _collapse_exact_duplicate_extracted_nodes(
    extracted_nodes: list[EntityNode],
    node_episode_index_map: dict[str, list[int]] | None = None,
) -> list[EntityNode]:
    """Collapse same-message duplicates with the same normalized name.

    This is intentionally narrow: it only merges exact normalized-name duplicates that the
    extraction prompt should already have emitted once. When duplicates disagree on specificity,
    keep the more specific node (for example, `Person` over bare `Entity`).

    When node_episode_index_map is provided, episode indices from discarded nodes are merged
    into the canonical node's entry so attribution is preserved.
    """
    # 少于两个节点时没有本批内部重复的可能，直接返回。
    if len(extracted_nodes) < 2:
        return extracted_nodes

    # canonical_by_name 保存每个规范化名称下保留下来的代表节点；ordered_names 用来保持原始顺序。
    canonical_by_name: dict[str, EntityNode] = {}
    ordered_names: list[str] = []

    for node in extracted_nodes:
        normalized_name = _normalize_string_exact(node.name)

        # 规范化名称相同才会进入本函数的合并逻辑，避免过度合并语义相近但不相同的实体。
        existing = canonical_by_name.get(normalized_name)
        if existing is None:
            canonical_by_name[normalized_name] = node
            ordered_names.append(normalized_name)
            continue

        existing_specific_labels = {label for label in existing.labels if label != 'Entity'}
        node_specific_labels = {label for label in node.labels if label != 'Entity'}

        # 如果重复项有更具体的类型，或同等类型下名字更完整，就替换为新的 canonical 节点。
        if len(node_specific_labels) > len(existing_specific_labels) or (
            len(node_specific_labels) == len(existing_specific_labels)
            and len(node.name.strip()) > len(existing.name.strip())
        ):
            old_canonical = existing
            canonical_by_name[normalized_name] = node
            # Merge episode indices: old canonical -> new canonical
            if node_episode_index_map is not None:
                old_indices = node_episode_index_map.pop(old_canonical.uuid, [])
                new_indices = node_episode_index_map.get(node.uuid, [])
                node_episode_index_map[node.uuid] = sorted(set(new_indices + old_indices))
        elif node_episode_index_map is not None:
            # Discard this node; merge its indices into the existing canonical
            discarded_indices = node_episode_index_map.pop(node.uuid, [])
            canonical_indices = node_episode_index_map.get(existing.uuid, [])
            node_episode_index_map[existing.uuid] = sorted(
                set(canonical_indices + discarded_indices)
            )

    return [canonical_by_name[name] for name in ordered_names]


# 去重候选可能来自语义搜索，也可能由调用者显式传入 override；这里合并两类候选，同时保持顺序和唯一性。
def _merge_candidate_nodes(
    candidate_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[EntityNode]:
    """Deduplicate candidate nodes while preserving search order and overrides."""
    merged_candidates = list(candidate_nodes)
    if existing_nodes_override is not None:
        merged_candidates.extend(existing_nodes_override)

    seen_candidate_uuids: set[str] = set()
    ordered_candidates: list[EntityNode] = []
    for candidate in merged_candidates:
        if candidate.uuid in seen_candidate_uuids:
            continue
        seen_candidate_uuids.add(candidate.uuid)
        ordered_candidates.append(candidate)

    return ordered_candidates


# 为每个新抽取节点收集可能的已有节点候选。
# 后续去重判断只在这些候选范围内进行，避免把整个图谱都塞给 LLM。
async def _collect_candidate_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    existing_nodes_override: list[EntityNode] | None,
) -> list[list[EntityNode]]:
    """Search per extracted name and return ordered candidates for each extracted node."""
    # 先用向量相似度从图中召回候选，再把 override 候选并入每个节点的候选列表。
    search_results = await _semantic_candidate_search(clients, extracted_nodes)

    return [_merge_candidate_nodes(result, existing_nodes_override) for result in search_results]


# 候选召回阶段：先把抽取节点名称转成向量，再用余弦相似度在同一 group_id 的图空间里找相似节点。
# 这里不做 rerank，只提供一批可能重复的候选，真正是否重复由后面的规则和 LLM 判断。
async def _semantic_candidate_search(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
) -> list[list[EntityNode]]:
    """Run direct cosine similarity search per extracted node without reranking."""
    if not extracted_nodes:
        return []

    # 去重检索只使用节点名称，不带 summary/attributes，因为此时新节点通常还没有这些信息。
    queries = [node.name.replace('\n', ' ') for node in extracted_nodes]

    # 优先批量生成 embedding；如果某些 embedder 不支持 batch，就退回并发单条创建。
    try:
        query_vectors = await clients.embedder.create_batch(queries)
    except NotImplementedError:
        query_vectors = list(
            await semaphore_gather(
                *[clients.embedder.create(input_data=[query]) for query in queries]
            )
        )

    # 每个抽取节点独立做相似节点搜索；限制在同 group_id，保证租户/命名空间隔离。
    return list(
        await semaphore_gather(
            *[
                node_similarity_search(
                    clients.driver,
                    query_vector,
                    SearchFilters(),
                    [node.group_id],
                    NODE_DEDUP_CANDIDATE_LIMIT,
                    NODE_DEDUP_COSINE_MIN_SCORE,
                )
                for node, query_vector in zip(extracted_nodes, query_vectors, strict=True)
            ]
        )
    )


# 把单个节点的去重结果写回批处理状态：包括最终节点、uuid 映射、重复节点对。
def _commit_resolution(
    state: DedupResolutionState,
    resolved_node: EntityNode | None,
    uuid_map: dict[str, str],
    duplicate_pairs: list[tuple[EntityNode, EntityNode]],
    index: int,
) -> None:
    """Commit a single-node resolution result into the batch-level state."""
    if resolved_node is not None:
        state.resolved_nodes[index] = resolved_node
    state.uuid_map.update(uuid_map)
    state.duplicate_pairs.extend(duplicate_pairs)


# 第二层去重：当确定性相似度规则无法决策时，把未决节点和候选节点交给 LLM 判断。
# 这里的重点不是“让 LLM 全权决定”，而是给 LLM 一个受控候选集，并对返回 id 做严格校验。
async def _resolve_with_llm(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    indexes: DedupCandidateIndexes,
    state: DedupResolutionState,
    episode: EpisodicNode | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_types: dict[str, type[BaseModel]] | None,
) -> None:
    """Escalate unresolved nodes to the dedupe prompt so the LLM can select or reject duplicates.

    The guardrails below defensively ignore malformed or duplicate LLM responses so the
    ingestion workflow remains deterministic even when the model misbehaves.
    """
    # 如果前面的确定性去重已经解决全部节点，就无需调用 LLM，节省成本并降低不确定性。
    if not state.unresolved_indices:
        return

    entity_types_dict: dict[str, type[BaseModel]] = entity_types if entity_types is not None else {}

    # 只把未决节点发给 LLM，而不是整批节点，减少 token 并让 LLM 任务更聚焦。
    llm_extracted_nodes = [extracted_nodes[i] for i in state.unresolved_indices]

    # 给 LLM 的 extracted_nodes 使用相对 id，从 0 开始；返回结果也必须引用这些相对 id。
    extracted_nodes_context = [
        {
            'id': i,
            'name': node.name,
            'entity_type': node.labels,
            'entity_type_description': _get_entity_type_description(node.labels, entity_types_dict),
        }
        for i, node in enumerate(llm_extracted_nodes)
    ]

    sent_ids = [ctx['id'] for ctx in extracted_nodes_context]
    logger.debug(
        'Sending %d entities to LLM for deduplication with IDs 0-%d (actual IDs sent: %s)',
        len(llm_extracted_nodes),
        len(llm_extracted_nodes) - 1,
        sent_ids if len(sent_ids) < 20 else f'{sent_ids[:10]}...{sent_ids[-10:]}',
    )
    if llm_extracted_nodes:
        sample_size = min(3, len(extracted_nodes_context))
        logger.debug(
            'First %d entity IDs: %s',
            sample_size,
            [ctx['id'] for ctx in extracted_nodes_context[:sample_size]],
        )
        if len(extracted_nodes_context) > 3:
            logger.debug(
                'Last %d entity IDs: %s',
                sample_size,
                [ctx['id'] for ctx in extracted_nodes_context[-sample_size:]],
            )

    # 候选节点上下文包含 candidate_id、name、types、summary 和 attributes，LLM 只能从这些候选里选重复对象。
    existing_nodes_context = [
        {
            **candidate.attributes,
            'candidate_id': i,
            'name': candidate.name,
            'entity_types': candidate.labels,
            'summary': candidate.summary[:120] if candidate.summary else '',
        }
        for i, candidate in enumerate(indexes.existing_nodes)
    ]

    # Build candidate_id -> node mapping for resolving duplicates by ID
    # 建立 candidate_id 到 EntityNode 的映射，方便把 LLM 返回的 candidate id 转成真实节点对象。
    candidates_by_id: dict[int, EntityNode] = {
        i: node for i, node in enumerate(indexes.existing_nodes)
    }

    context = {
        'extracted_nodes': extracted_nodes_context,
        'existing_nodes': existing_nodes_context,
        'episode_content': episode.content if episode is not None else '',
        'previous_episodes': (
            [
                {
                    'content': ep.content,
                    'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
                }
                for ep in previous_episodes
            ]
            if previous_episodes is not None
            else []
        ),
    }

    # 节点去重的 LLM 调用只处理未决节点，并要求返回 NodeResolutions 结构。
    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.nodes(context),
        response_model=NodeResolutions,
        prompt_name='dedupe_nodes.nodes',
    )

    # 再次用 Pydantic 解析响应，确保后面的逻辑面对的是结构化的 resolution 列表。
    node_resolutions: list[NodeDuplicate] = NodeResolutions(**llm_response).entity_resolutions

    valid_relative_range = range(len(state.unresolved_indices))
    processed_relative_ids: set[int] = set()

    received_ids = {r.id for r in node_resolutions}
    expected_ids = set(valid_relative_range)
    missing_ids = expected_ids - received_ids
    extra_ids = received_ids - expected_ids

    logger.debug(
        'Received %d resolutions for %d entities',
        len(node_resolutions),
        len(state.unresolved_indices),
    )

    if missing_ids:
        logger.warning('LLM did not return resolutions for IDs: %s', sorted(missing_ids))

    if extra_ids:
        logger.warning(
            'LLM returned invalid IDs outside valid range 0-%d: %s (all returned IDs: %s)',
            len(state.unresolved_indices) - 1,
            sorted(extra_ids),
            sorted(received_ids),
        )

    # 逐条应用 LLM 决策；所有 id 都会被校验，避免模型返回越界/重复 id 破坏状态。
    for resolution in node_resolutions:
        relative_id: int = resolution.id
        duplicate_candidate_id: int = resolution.duplicate_candidate_id

        if relative_id not in valid_relative_range:
            logger.warning(
                'Skipping invalid LLM dedupe id %d (valid range: 0-%d, received %d resolutions)',
                relative_id,
                len(state.unresolved_indices) - 1,
                len(node_resolutions),
            )
            continue

        if relative_id in processed_relative_ids:
            logger.warning('Duplicate LLM dedupe id %s received; ignoring.', relative_id)
            continue
        processed_relative_ids.add(relative_id)

        original_index = state.unresolved_indices[relative_id]
        extracted_node = extracted_nodes[original_index]

        resolved_node: EntityNode

        # duplicate_candidate_id < 0 表示 LLM 判断没有重复候选，保留为新节点。
        if duplicate_candidate_id < 0:
            resolved_node = extracted_node

        # 找到重复候选时，不是简单丢弃新节点，而是用 _promote_resolved_node 合并/提升已有节点信息。
        elif duplicate_candidate_id in candidates_by_id:
            resolved_node = _promote_resolved_node(
                extracted_node, candidates_by_id[duplicate_candidate_id]
            )
        else:
            logger.warning(
                'Invalid duplicate_candidate_id %d for extracted node %s; treating as no duplicate.',
                duplicate_candidate_id,
                extracted_node.uuid,
            )
            resolved_node = extracted_node

        state.resolved_nodes[original_index] = resolved_node
        state.uuid_map[extracted_node.uuid] = resolved_node.uuid
        if resolved_node.uuid != extracted_node.uuid:
            state.duplicate_pairs.append((extracted_node, resolved_node))


# 图谱级节点去重入口：把刚抽出的 EntityNode 解析到最终节点。
# 整体顺序是：候选召回 -> 确定性相似度去重 -> LLM 兜底判断 -> 未命中者作为新节点保留。
async def resolve_extracted_nodes(
    clients: GraphitiClients,
    extracted_nodes: list[EntityNode],
    episode: EpisodicNode | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    existing_nodes_override: list[EntityNode] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]]]:
    """Resolve nodes with semantic retrieval first, then deterministic and LLM dedup."""
    llm_client = clients.llm_client

    # 先为每个抽取节点准备候选列表；没有候选的节点大概率会作为新节点保留。
    candidate_nodes_by_extracted = await _collect_candidate_nodes(
        clients,
        extracted_nodes,
        existing_nodes_override,
    )

    # state 统一记录最终解析结果、uuid 映射和仍需 LLM 处理的节点索引。
    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=[],
    )

    # 第一轮先走确定性/相似度规则，能自动解决的就不交给 LLM。
    for idx, (node, candidates) in enumerate(
        zip(extracted_nodes, candidate_nodes_by_extracted, strict=True)
    ):
        if not candidates:
            continue

        # 为当前节点的候选建立索引，方便相似度规则按名称、uuid、标准化字符串等方式快速判断。
        indexes = _build_candidate_indexes(candidates)
        local_state = DedupResolutionState(
            resolved_nodes=[None], uuid_map={}, unresolved_indices=[]
        )

        # 局部状态只解析当前一个节点；解析成功后再提交到全局 state。
        _resolve_with_similarity([node], indexes, local_state)
        if local_state.resolved_nodes[0] is not None:
            _commit_resolution(
                state,
                local_state.resolved_nodes[0],
                local_state.uuid_map,
                local_state.duplicate_pairs,
                idx,
            )
            continue

        state.unresolved_indices.append(idx)

    # 确定性规则解决不了的节点，再合并它们的候选并交给 LLM 做最终判断。
    if state.unresolved_indices:
        llm_candidate_nodes = _merge_candidate_nodes(
            [
                candidate
                for idx in state.unresolved_indices
                for candidate in candidate_nodes_by_extracted[idx]
            ],
            None,
        )
        await _resolve_with_llm(
            llm_client,
            extracted_nodes,
            _build_candidate_indexes(llm_candidate_nodes),
            state,
            episode,
            previous_episodes,
            entity_types,
        )

    if not state.unresolved_indices and not any(candidate_nodes_by_extracted):
        logger.debug('No semantic dedup candidates found; keeping all extracted nodes as new')

    # 兜底：既没有候选也没有被 LLM 解析的节点，保留自身 uuid，作为新节点进入图谱。
    for idx, node in enumerate(extracted_nodes):
        if state.resolved_nodes[idx] is None:
            state.resolved_nodes[idx] = node
            state.uuid_map[node.uuid] = node.uuid

    logger.debug(
        'Resolved nodes: %s',
        [node.uuid for node in state.resolved_nodes if node is not None],
    )

    return (
        [node for node in state.resolved_nodes if node is not None],
        state.uuid_map,
        state.duplicate_pairs,
    )


# 为摘要阶段预处理边：从 edge 列表构建 node_uuid -> connected edges 的索引，避免每个节点都线性扫描所有边。
def _build_edges_by_node(edges: list[EntityEdge] | None) -> dict[str, list[EntityEdge]]:
    """Build a dictionary mapping node UUIDs to their connected edges."""
    # 同一条边要同时挂到 source 和 target 两侧，因为任一端节点摘要都可能需要这条事实。
    edges_by_node: dict[str, list[EntityEdge]] = {}
    if not edges:
        return edges_by_node
    for edge in edges:
        if edge.source_node_uuid not in edges_by_node:
            edges_by_node[edge.source_node_uuid] = []
        if edge.target_node_uuid not in edges_by_node:
            edges_by_node[edge.target_node_uuid] = []
        edges_by_node[edge.source_node_uuid].append(edge)
        edges_by_node[edge.target_node_uuid].append(edge)
    return edges_by_node


# 节点补全入口：在节点已完成去重/解析之后，为最终节点抽属性、生成摘要，并更新 name embedding。
# 它通常接在 add_episode 的 edge resolve 之后，因此可以利用 new_edges 避免重复事实污染 summary。
async def extract_attributes_from_nodes(
    clients: GraphitiClients,
    nodes: list[EntityNode],
    episode: EpisodicNode | list[EpisodicNode] | None = None,
    previous_episodes: list[EpisodicNode] | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    should_summarize_node: NodeSummaryFilter | None = None,
    edges: list[EntityEdge] | None = None,
    skip_fact_appending: bool = False,
    include_type_descriptions: bool = False,
) -> list[EntityNode]:
    llm_client = clients.llm_client
    embedder = clients.embedder

    # Pre-build edges lookup for O(E + N) instead of O(N * E)
    # 先把边按节点分组，后面摘要阶段可以快速拿到与某个节点有关的新事实。
    edges_by_node = _build_edges_by_node(edges)

    # Extract attributes in parallel (per-entity calls)
    # 属性抽取是 per-node 的独立 LLM 调用，因此可以并发执行。
    attribute_results: list[dict[str, Any]] = await semaphore_gather(
        *[
            _extract_entity_attributes(
                llm_client,
                node,
                episode,
                previous_episodes,
                (
                    entity_types.get(next((item for item in node.labels if item != 'Entity'), ''))
                    if entity_types is not None
                    else None
                ),
            )
            for node in nodes
        ]
    )

    # _extract_entity_attributes returns the already-merged attribute dict
    # (overlay of prior + cap-kept fields), so direct assignment is the merge.
    # 属性抽取函数内部已经处理了旧属性合并，这里直接覆盖为合并后的结果。
    for node, attributes in zip(nodes, attribute_results, strict=True):
        node.attributes = attributes

    # Extract summaries in batch
    # 属性之后再生成 summary，这样 summary prompt 能看到更新后的 attributes。
    await _extract_entity_summaries_batch(
        llm_client,
        nodes,
        episode,
        previous_episodes,
        should_summarize_node,
        edges_by_node,
        skip_fact_appending=skip_fact_appending,
        entity_types=entity_types if include_type_descriptions else None,
    )

    # summary/name 等文本信息更新后，重新生成节点 embedding，保证后续搜索能命中新状态。
    await create_entity_node_embeddings(embedder, nodes)

    return nodes


# 针对单个节点抽取结构化属性。
# 只有定义了自定义 entity_type 且该类型有字段时才会调用 LLM；普通 Entity 没有 schema，就直接返回空属性。
async def _extract_entity_attributes(
    llm_client: LLMClient,
    node: EntityNode,
    episode: EpisodicNode | list[EpisodicNode] | None,
    previous_episodes: list[EpisodicNode] | None,
    entity_type: type[BaseModel] | None,
) -> dict[str, Any]:
    # 没有 schema 就无法做结构化属性抽取；普通默认 Entity 会走这条快速路径。
    if entity_type is None or len(entity_type.model_fields) == 0:
        return {}

    # 属性抽取不使用 summary，避免已有摘要中的推断性内容影响结构化字段。
    attributes_context = _build_episode_context(
        # should not include summary
        node_data={
            'name': node.name,
            'entity_types': node.labels,
            'attributes': node.attributes,
        },
        episode=episode,
        previous_episodes=previous_episodes,
    )

    # 属性抽取通常用小模型即可，因为输出受 Pydantic schema 约束，任务相对窄。
    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_attributes(attributes_context),
        response_model=entity_type,
        model_size=ModelSize.small,
        group_id=node.group_id,
        prompt_name='extract_nodes.extract_attributes',
        attribute_extraction=True,
    )

    # Overlay merge: cap-dropped or LLM-omitted fields keep prior values.
    # See attribute_utils for the merge_mode contract; the edge path uses 'replace'.
    # overlay 合并表示：LLM 这次没有返回的字段沿用旧值，避免一次抽取把已有属性清空。
    merged, _ = apply_capped_attributes(
        llm_response,
        entity_type,
        node.attributes,
        merge_mode='overlay',
        prompt_name='extract_nodes.extract_attributes',
        entity_uuid=node.uuid,
        group_id=node.group_id,
    )

    # Shape validation only — we discard the validated instance because returning
    # `model_dump()` would expand defaults across all fields and clobber prior
    # values that the merge above just preserved.
    # 这里只做形状校验，不使用 model_dump，避免 Pydantic 默认值展开后覆盖旧属性。
    entity_type(**merged)

    return merged


# 批量生成/更新节点摘要。
# 设计上优先把新 edge facts 直接追加到短 summary，只有内容过长或需要 episode prompt 时才调用 LLM 压缩总结。
async def _extract_entity_summaries_batch(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | list[EpisodicNode] | None,
    previous_episodes: list[EpisodicNode] | None,
    should_summarize_node: NodeSummaryFilter | None,
    edges_by_node: dict[str, list[EntityEdge]],
    *,
    skip_fact_appending: bool = False,
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> None:
    """Extract summaries for multiple entities in batched LLM calls.

    When skip_fact_appending is False (default), nodes with short summaries get edge
    facts appended directly without an LLM call.  Nodes needing summarization are
    partitioned into flights of MAX_NODES and processed with separate LLM calls.

    When skip_fact_appending is True, the raw fact-append shortcut is bypassed and all
    nodes are routed through LLM summarization using an episode-based prompt that
    matches the async graph summary worker.
    """
    # Determine which nodes need LLM summarization vs direct edge fact appending
    # 先筛选哪些节点真的需要 LLM 摘要；能直接追加事实的就不调用模型。
    nodes_needing_llm: list[EntityNode] = []

    for node in nodes:
        # Check if node should be summarized at all
        # 外部过滤器可以跳过某些节点，例如已经有异步 worker 负责总结的节点。
        if should_summarize_node is not None and not await should_summarize_node(node):
            continue

        # skip_fact_appending=True 时，强制走 episode-based LLM 摘要，不直接把 fact 文本拼到 summary 后面。
        if skip_fact_appending:
            # Always route through LLM — no raw fact concatenation.
            if episode is not None or node.summary:
                nodes_needing_llm.append(node)
            continue

        # 默认路径下，先拿到与当前节点相关的新边事实，尝试轻量更新 summary。
        node_edges = edges_by_node.get(node.uuid, [])

        # Build summary with edge facts appended
        summary_with_edges = node.summary
        if node_edges:
            edge_facts = '\n'.join(edge.fact for edge in node_edges if edge.fact)
            summary_with_edges = f'{summary_with_edges}\n{edge_facts}'.strip()

        # If summary is close to the persisted limit, use it directly (append edge facts, no LLM call)
        # 内容还不长时直接保留拼接后的 summary，避免不必要的 LLM 调用。超过阈值才需要压缩。
        if summary_with_edges and len(summary_with_edges) <= MAX_SUMMARY_CHARS * 2:
            node.summary = summary_with_edges
            continue

        # Skip if no summary content and no episode to generate from
        if not summary_with_edges and episode is None:
            continue

        # This node needs LLM summarization
        nodes_needing_llm.append(node)

    # If no nodes need LLM summarization, return early
    # 全部节点都能轻量处理时，摘要流程到这里就结束。
    if not nodes_needing_llm:
        return

    # Partition nodes into flights of MAX_NODES
    # 将待总结节点拆成多个 flight，控制单个 prompt 的大小和 LLM 响应复杂度。
    node_flights = [
        nodes_needing_llm[i : i + MAX_NODES] for i in range(0, len(nodes_needing_llm), MAX_NODES)
    ]

    # Process flights in parallel
    await semaphore_gather(
        *[
            _process_summary_flight(
                llm_client,
                flight,
                episode,
                previous_episodes,
                use_episode_prompt=skip_fact_appending,
                entity_types=entity_types,
            )
            for flight in node_flights
        ]
    )


# 处理一个摘要批次。
# 上层把需要 LLM 总结的节点按 MAX_NODES 分批，这里负责构造 prompt、调用 LLM，并把返回 summary 写回对应节点。
async def _process_summary_flight(
    llm_client: LLMClient,
    nodes: list[EntityNode],
    episode: EpisodicNode | list[EpisodicNode] | None,
    previous_episodes: list[EpisodicNode] | None,
    *,
    use_episode_prompt: bool = False,
    entity_types: dict[str, type[BaseModel]] | None = None,
) -> None:
    """Process a single flight of nodes for batch summarization."""
    # Build entity type descriptions from docstrings, stripping GOOD/BAD
    # few-shot examples that are intended for extraction prompts only.
    # 摘要 prompt 可以附带实体类型说明，但会用简化版，避免抽取阶段的 GOOD/BAD 规则干扰总结。
    entity_type_descriptions: dict[str, str] = {}
    if entity_types is not None:
        for type_name, type_model in entity_types.items():
            if type_model.__doc__:
                entity_type_descriptions[type_name] = _truncate_type_description(type_model.__doc__)

    # Build context for batch summarization
    # 每个节点提供 name、旧 summary、类型和属性，LLM 的任务是生成更紧凑的新 summary。
    entities_context = [
        {
            'name': node.name,
            'summary': node.summary,
            'entity_types': node.labels,
            'attributes': node.attributes,
        }
        for node in nodes
    ]

    # episode 可能为空、单个或多个；这里统一整理成 prompt 可读的字符串内容。
    if episode is None:
        episode_content = ''
    elif isinstance(episode, list):
        episode_content = concatenate_episodes(episode)
    else:
        episode_content = episode.content

    batch_context: dict[str, Any] = {
        'entities': entities_context,
        'episode_content': episode_content,
        'previous_episodes': (
            [
                {
                    'content': ep.content,
                    'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
                }
                for ep in previous_episodes
            ]
            if previous_episodes is not None
            else []
        ),
        'entity_type_descriptions': entity_type_descriptions,
    }

    # Get group_id from the first node (all nodes in a batch should have same group_id)
    group_id = nodes[0].group_id if nodes else None

    # 两套 prompt 对应两种摘要模式：基于 episode 重新总结，或基于现有 summary/attributes/facts 批量压缩。
    if use_episode_prompt:
        prompt = prompt_library.extract_nodes.extract_entity_summaries_from_episodes(batch_context)
        prompt_name = 'extract_nodes.extract_entity_summaries_from_episodes'
    else:
        prompt = prompt_library.extract_nodes.extract_summaries_batch(batch_context)
        prompt_name = 'extract_nodes.extract_summaries_batch'

    llm_response = await llm_client.generate_response(
        prompt,
        response_model=SummarizedEntities,
        model_size=ModelSize.small,
        group_id=group_id,
        prompt_name=prompt_name,
    )

    # Build case-insensitive name -> nodes mapping (handles duplicates)
    # LLM 返回是按实体名匹配的，因此先建立大小写不敏感的 name -> nodes 映射；同名节点也要支持。
    name_to_nodes: dict[str, list[EntityNode]] = {}
    for node in nodes:
        key = node.name.lower()
        if key not in name_to_nodes:
            name_to_nodes[key] = []
        name_to_nodes[key].append(node)

    # Apply summaries from LLM response
    # 应用 LLM summary 时仍然做结构化解析；匹配不到节点名的返回会被记录 warning 而不是抛错。
    summaries_response = SummarizedEntities(**llm_response)
    for summarized_entity in summaries_response.summaries:
        matching_nodes = name_to_nodes.get(summarized_entity.name.lower(), [])
        if matching_nodes:
            truncated_summary = truncate_at_sentence(summarized_entity.summary, MAX_SUMMARY_CHARS)
            for node in matching_nodes:
                node.summary = truncated_summary
        else:
            logger.warning(
                'LLM returned summary for unknown entity (first 30 chars): %.30s',
                summarized_entity.name,
            )


# 统一构造“当前 episode + 历史 episode”的上下文格式，供属性抽取和摘要生成复用。
def _build_episode_context(
    node_data: dict[str, Any],
    episode: EpisodicNode | list[EpisodicNode] | None,
    previous_episodes: list[EpisodicNode] | None,
) -> dict[str, Any]:
    if episode is None:
        episode_content = ''
    elif isinstance(episode, list):
        episode_content = concatenate_episodes(episode)
    else:
        episode_content = episode.content

    return {
        'node': node_data,
        'episode_content': episode_content,
        'previous_episodes': (
            [
                {
                    'content': ep.content,
                    'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
                }
                for ep in previous_episodes
            ]
            if previous_episodes is not None
            else []
        ),
    }
