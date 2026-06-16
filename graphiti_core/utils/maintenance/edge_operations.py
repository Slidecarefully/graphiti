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

# 这个文件是 Graphiti 写入链路中“边维护”的核心模块。
# 它主要处理三类边：
# 1. EpisodicEdge：episode 和 entity node 之间的来源归因边，表达“这个 episode 提到了这个实体”。
# 2. CommunityEdge：community 和 entity node 之间的归属边。
# 3. EntityEdge：实体与实体之间的事实关系边，是 Graphiti 记忆图谱中最重要的事实载体。
#
# 阅读主线建议：
# build_episodic_edges / build_community_edges
#   ↓
# extract_edges：从 episode + nodes 中抽取事实边
#   ↓
# resolve_extracted_edges：把新抽取的边和已有图谱边做去重、冲突判断、时间失效处理
#   ↓
# resolve_extracted_edge：单条边级别的 dedupe / invalidation / 属性抽取
#   ↓
# filter_existing_duplicate_of_edges：辅助过滤已经存在的 IS_DUPLICATE_OF 关系
import logging
from datetime import datetime
from time import time

from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import EdgeTimestamps, ExtractedEdges
from graphiti_core.search.search import search
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.attribute_utils import apply_capped_attributes
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.text_utils import concatenate_episodes

logger = logging.getLogger(__name__)


# 构建 episode -> entity 的来源边。
# 这类边不是“事实关系边”，而是 provenance / attribution：
# 它记录某个实体节点是从哪些 episode 中被提到或抽取出来的。
def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str | list[str],
    created_at: datetime,
    node_episode_index_map: dict[str, list[int]] | None = None,
) -> list[EpisodicEdge]:
    """Build episodic (MENTIONED_IN) edges between entity nodes and episodes.

    Parameters
    ----------
    entity_nodes : list[EntityNode]
        Nodes to connect to episodes.
    episode_uuid : str | list[str]
        A single episode UUID or a list of episode UUIDs.
    created_at : datetime
        Timestamp for the edges.
    node_episode_index_map : dict[str, list[int]] | None
        Optional mapping from node UUID to 0-indexed episode positions.
        When provided with a list of episode_uuids, each node is connected
        only to its attributed episodes. When None, every node is connected
        to all episodes.
    """
    # 统一处理单 episode 和多 episode 两种输入。
    # 后面都按 episode_uuids 列表来建立边。
    episode_uuids = [episode_uuid] if isinstance(episode_uuid, str) else episode_uuid

    episodic_edges: list[EpisodicEdge] = []

    # 对每个实体节点，决定它应该连接到哪些 episode。
    # 如果上游 LLM 抽取时提供了 node_episode_index_map，就只连接被归因到的 episode；
    # 否则默认每个节点都连接到所有 episode。
    for node in entity_nodes:
        if node_episode_index_map and node.uuid in node_episode_index_map:
            indices = node_episode_index_map[node.uuid]
        else:
            indices = list(range(len(episode_uuids)))

        # 将 episode index 转换成真实 episode uuid，并创建 EpisodicEdge。
        # 越界 index 会被忽略，避免 LLM 或上游归因异常导致错误边。
        for idx in indices:
            if 0 <= idx < len(episode_uuids):
                episodic_edges.append(
                    EpisodicEdge(
                        source_node_uuid=episode_uuids[idx],
                        target_node_uuid=node.uuid,
                        created_at=created_at,
                        group_id=node.group_id,
                    )
                )

    logger.debug(f'Built {len(episodic_edges)} episodic edges')

    return episodic_edges


# 构建 community -> entity 的社区归属边。
# CommunityNode 表示一组实体的聚类或主题摘要，这里把社区节点连接到它包含的实体节点。
def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
    edges: list[CommunityEdge] = [
        CommunityEdge(
            source_node_uuid=community_node.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=community_node.group_id,
        )
        for node in entity_nodes
    ]

    return edges


# 从 episode 文本中抽取实体之间的事实关系边。
# 注意它依赖上游已经抽取并解析好的 nodes：
# LLM 只能在这些节点之间建立 edge，不能随便创造新的 source/target entity。
async def extract_edges(
    clients: GraphitiClients,
    episode: EpisodicNode | list[EpisodicNode],
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
) -> list[EntityEdge]:
    """Extract edges from one or more episodes.

    Parameters
    ----------
    episode : EpisodicNode | list[EpisodicNode]
        A single episode or a list of episodes to extract edges from.
        When a list is provided, their contents are concatenated for extraction
        and edges are linked to all episode UUIDs.
    """
    # 和 node extraction 一样，这里也统一处理单 episode / 多 episode。
    # primary_episode 用来提供默认 group_id 和兜底 reference_time。
    episodes = episode if isinstance(episode, list) else [episode]
    primary_episode = episodes[0]

    start = time()

    # 边抽取 prompt 可能输出大量事实，因此 max_tokens 给得比较大。
    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    # Build mapping from edge type name to list of valid signatures
    # edge_type_map 的原始形态是：
    #   (source_type, target_type) -> [edge_type_name]
    # 这里反转成：
    #   edge_type_name -> [(source_type, target_type)]
    # 方便放进 prompt，让 LLM 知道某种关系允许出现在什么实体类型组合之间。
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    # 把自定义 edge type 转成 prompt context。
    # 如果没有自定义 edge_types，edge_types_context 为空，模型会走默认关系抽取。
    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signatures': edge_type_signatures_map.get(
                    type_name, [('Entity', 'Entity')]
                ),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Build name-to-node mapping for validation
    # LLM 抽边时返回的是 source_entity_name / target_entity_name。
    # 这里先建立 name -> node 映射，后面用来验证 LLM 返回的实体名确实存在于 nodes 列表里。
    name_to_node: dict[str, EntityNode] = {node.name: node for node in nodes}

    # Build episode attribution instructions for multi-episode extraction
    # 多 episode 合并抽边时，必须让 LLM 标记每条 fact 来源于哪些 episode。
    # 这样 EntityEdge.episodes 可以准确追踪事实来源，而不是粗暴关联到所有 episode。
    episode_attribution = ''
    if len(episodes) > 1:
        episode_attribution = (
            '\n8. **Episode Attribution**: The CURRENT_MESSAGE contains multiple episodes labeled '
            '[Episode 0], [Episode 1], etc. Each episode header includes a timestamp indicating '
            'when that episode occurred. Use the per-episode timestamp to resolve relative time '
            'mentions within each episode rather than relying solely on REFERENCE_TIME. '
            'For each extracted fact, set `episode_indices` '
            'to the 0-based list of episode numbers that the fact was derived from. '
            'A fact sourced from Episodes 0 and 1 should have `episode_indices: [0, 1]`.'
        )

    # Prepare context for LLM
    # Use the latest episode's timestamp as the primary reference time
    # reference_time 用最新 episode 的 valid_at，便于解析 “last week / yesterday” 这类相对时间。
    latest_episode = max(episodes, key=lambda ep: ep.valid_at)
    context = {
        'episode_content': concatenate_episodes(episodes),
        'nodes': [{'name': node.name, 'entity_types': node.labels} for node in nodes],
        'previous_episodes': [
            {
                'content': ep.content,
                'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
            }
            for ep in previous_episodes
        ],
        'reference_time': latest_episode.valid_at,
        'edge_types': edge_types_context,
        'custom_extraction_instructions': (custom_extraction_instructions or '')
        + episode_attribution,
    }

    # 调用边抽取 prompt，要求 LLM 返回 ExtractedEdges 结构。
    # 这里还只是 prompt 层的 ExtractedEdge，不是最终 EntityEdge 对象。
    llm_response = await llm_client.generate_response(
        prompt_library.extract_edges.edge(context),
        response_model=ExtractedEdges,
        max_tokens=extract_edges_max_tokens,
        group_id=group_id or primary_episode.group_id,
        prompt_name='extract_edges.edge',
    )
    all_edges_data = ExtractedEdges(**llm_response).edges

    # Validate entity names
    # LLM 可能返回了不在 nodes 列表里的实体名。
    # Graphiti 在这里做强校验：边的两个端点必须来自已知节点，否则丢弃该边。
    edges_data: list[ExtractedEdge] = []
    for edge_data in all_edges_data:
        source_name = edge_data.source_entity_name
        target_name = edge_data.target_entity_name

        # Validate LLM-returned names exist in the nodes list
        if source_name not in name_to_node:
            logger.warning(
                'Source entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        if target_name not in name_to_node:
            logger.warning(
                'Target entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        # Drop self-edges where source and target resolve to the same node
        # 即使名字不同，经过 node resolve 后也可能指向同一个 uuid。
        # 这种 self-edge 没有事实图谱意义，直接丢弃。
        source_node = name_to_node[source_name]
        target_node = name_to_node[target_name]
        if source_node.uuid == target_node.uuid:
            logger.info(
                'Dropping self-edge for node %s (source and target resolve to same node)',
                source_node.uuid,
            )
            continue

        edges_data.append(edge_data)

    end = time()
    logger.debug(f'Extracted {len(edges_data)} new edges in {(end - start) * 1000:.0f} ms')

    if len(edges_data) == 0:
        return []

    # Convert the extracted data into EntityEdge objects
    # 通过验证后，才把 prompt 层的 ExtractedEdge 转成图谱层的 EntityEdge。
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        # LLM 可能已经在边抽取阶段返回了 valid_at / invalid_at。
        # 这里先尝试解析成 datetime；解析失败不会中断流程，只记录 warning。
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        # Filter out empty edges
        # fact 是 EntityEdge 的核心语义内容，空 fact 没有保存价值。
        if not edge_data.fact.strip():
            continue

        # Names already validated above
        source_node = name_to_node.get(edge_data.source_entity_name)
        target_node = name_to_node.get(edge_data.target_entity_name)

        if source_node is None or target_node is None:
            logger.warning('Could not find source or target node for extracted edge')
            continue

        source_node_uuid = source_node.uuid
        target_node_uuid = target_node.uuid

        # valid_at 表示事实从什么时候开始成立。
        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                logger.warning('Error parsing valid_at date, skipping')

        # invalid_at 表示事实从什么时候开始不成立。
        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')

        # Map episode_indices (0-indexed) to episode UUIDs.
        # Clamp indices to valid range and fall back to all episodes if empty.
        # 把 LLM 返回的 episode_indices 转成真实 episode uuid。
        # 如果 LLM 没给合法 index，就回退为所有 episode，确保 fact 至少有来源。
        edge_episode_uuids = []
        for idx in edge_data.episode_indices:
            if 0 <= idx < len(episodes):
                edge_episode_uuids.append(episodes[idx].uuid)
        if not edge_episode_uuids:
            edge_episode_uuids = [ep.uuid for ep in episodes]

        # 创建最终的 EntityEdge。
        # reference_time 通常使用该 fact 来源 episode 的 valid_at，用于之后时间逻辑和检索排序。
        edge = EntityEdge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=edge_data.relation_type,
            group_id=group_id or primary_episode.group_id,
            fact=edge_data.fact,
            episodes=edge_episode_uuids,
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
            reference_time=(
                episodes[edge_data.episode_indices[0]].valid_at
                if edge_data.episode_indices and 0 <= edge_data.episode_indices[0] < len(episodes)
                else primary_episode.valid_at
            ),
        )
        edges.append(edge)
        logger.debug(
            f'Created new edge {edge.uuid} from {edge.source_node_uuid} to {edge.target_node_uuid}'
        )

    logger.debug(f'Extracted edges: {[e.uuid for e in edges]}')

    return edges


# 批量解析新抽取的 EntityEdge。
# 这一层负责协调：
# 1. 本批 edge 内部的精确去重；
# 2. 给新 edge 生成 embedding；
# 3. 查找同端点的已有边作为 duplicate candidates；
# 4. 查找全图相关边作为 invalidation candidates；
# 5. 调用 resolve_extracted_edge 对每条边做 LLM 去重/冲突判断；
# 6. 返回 resolved_edges、invalidated_edges、new_edges。
async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
    existing_edges_override: list[EntityEdge] | None = None,
) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
    """Resolve extracted edges against existing graph context.

    Returns
    -------
    tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
        A tuple of (resolved_edges, invalidated_edges, new_edges) where:
        - resolved_edges: All edges after resolution (may include existing edges if duplicates found)
        - invalidated_edges: Edges that were invalidated/contradicted by new information
        - new_edges: Only edges that are new to the graph (not duplicates of existing edges)
    """
    # Fast path: deduplicate exact matches within the extracted edges before parallel processing
    # 先在当前批次内部做轻量去重。
    # 同一 source、target、fact 文本完全一致的边，只保留第一条，避免后续重复做 embedding/search/LLM。
    seen: dict[tuple[str, str, str], EntityEdge] = {}
    deduplicated_edges: list[EntityEdge] = []

    for edge in extracted_edges:
        key = (
            edge.source_node_uuid,
            edge.target_node_uuid,
            _normalize_string_exact(edge.fact),
        )
        if key not in seen:
            seen[key] = edge
            deduplicated_edges.append(edge)

    extracted_edges = deduplicated_edges

    driver = clients.driver
    llm_client = clients.llm_client
    embedder = clients.embedder

    # 后续 edge search / similarity 依赖 fact embedding，所以先给所有新抽取边生成 embedding。
    await create_entity_edge_embeddings(embedder, extracted_edges)

    # 对每条新边，先查同一 source-target 之间已有的边。
    # 这些是最直接的 duplicate candidates：端点相同，fact 可能相同或语义重复。
    valid_edges_list: list[list[EntityEdge]] = await semaphore_gather(
        *[
            EntityEdge.get_between_nodes(driver, edge.source_node_uuid, edge.target_node_uuid)
            for edge in extracted_edges
        ]
    )

    # Merge override edges (e.g. from the recent Redis dedup cache) into
    # the per-extracted-edge candidate lists so that recently resolved edges
    # that are not yet visible in the graph-service indexes are still
    # considered during deduplication.
    # existing_edges_override 是补充候选来源。
    # 场景：某些刚解析过的边可能还没有进入图服务索引，但仍应参与去重。
    if existing_edges_override:
        override_by_pair: dict[tuple[str, str], list[EntityEdge]] = {}
        for oe in existing_edges_override:
            key = (oe.source_node_uuid, oe.target_node_uuid)
            override_by_pair.setdefault(key, []).append(oe)

        for i, extracted_edge in enumerate(extracted_edges):
            pair_key = (extracted_edge.source_node_uuid, extracted_edge.target_node_uuid)
            overrides = override_by_pair.get(pair_key, [])
            if overrides:
                existing_uuids = {e.uuid for e in valid_edges_list[i]}
                for oe in overrides:
                    if oe.uuid not in existing_uuids:
                        valid_edges_list[i].append(oe)
                        existing_uuids.add(oe.uuid)

    # 在同端点已有边中，用 hybrid search 找和新 fact 最相关的边。
    # 这些 related_edges_lists 会被送入 dedupe prompt，判断是否是重复事实。
    related_edges_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
            )
            for extracted_edge, valid_edges in zip(extracted_edges, valid_edges_list, strict=True)
        ]
    )

    related_edges_lists: list[list[EntityEdge]] = [result.edges for result in related_edges_results]

    # 再做一次更宽范围的搜索：不限制 source-target，只在同 group 内搜索相关事实。
    # 这批结果用于 invalidation：新事实可能会推翻不同端点或更广范围内的旧事实。
    edge_invalidation_candidate_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(),
            )
            for extracted_edge in extracted_edges
        ]
    )

    # Remove duplicates: if an edge appears in both duplicate candidates and invalidation candidates,
    # keep it only in duplicate candidates
    # 如果同一条边既出现在 duplicate candidates，也出现在 invalidation candidates，
    # 优先把它作为 duplicate candidate，避免 LLM 同时把同一事实当“重复”和“冲突”处理。
    edge_invalidation_candidates: list[list[EntityEdge]] = []
    for related_edges, invalidation_result in zip(
        related_edges_lists, edge_invalidation_candidate_results, strict=True
    ):
        related_uuids = {edge.uuid for edge in related_edges}
        deduplicated = [
            edge for edge in invalidation_result.edges if edge.uuid not in related_uuids
        ]
        edge_invalidation_candidates.append(deduplicated)

    logger.debug(
        f'Related edges: {[e.uuid for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    # 构建 uuid -> EntityNode 映射，用于根据边的端点类型判断可用的自定义 edge schema。
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Collect all node UUIDs referenced by edges that are not in the entities list
    # 某些 edge 的 source/target 可能已经被 resolve 到图中已有节点，不一定出现在当前 entities 列表里。
    # 这里收集缺失节点 uuid，稍后从数据库补齐它们的 labels。
    referenced_node_uuids = set()
    for extracted_edge in extracted_edges:
        if extracted_edge.source_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.source_node_uuid)
        if extracted_edge.target_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.target_node_uuid)

    # Fetch missing nodes from the database
    if referenced_node_uuids:
        # Pass group_id so graph-service implementations can scope the lookup
        edge_group_id = extracted_edges[0].group_id
        missing_nodes = await EntityNode.get_by_uuids(
            driver, list(referenced_node_uuids), group_id=edge_group_id
        )
        for node in missing_nodes:
            uuid_entity_map[node.uuid] = node

    # Determine which edge types are relevant for each edge based on node signatures.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge.
    # 根据 source/target 节点 labels 计算当前边允许使用哪些自定义 edge type。
    # 例如只有 (Person, Company) 才允许 WORKS_AT，那么其他类型组合就不会抽 WORKS_AT 的属性。
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
    for extracted_edge in extracted_edges:
        source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
        target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)
        source_node_labels = (
            source_node.labels + ['Entity'] if source_node is not None else ['Entity']
        )
        target_node_labels = (
            target_node.labels + ['Entity'] if target_node is not None else ['Entity']
        )
        label_tuples = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    # resolve edges with related edges in the graph and find invalidation candidates
    # 对每条 extracted_edge 并发做单边解析。
    # related_edges 用于重复判断，existing_edges 用于冲突/失效判断。
    results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                    episode,
                    extracted_edge_types,
                )
                for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                    extracted_edges,
                    related_edges_lists,
                    edge_invalidation_candidates,
                    edge_types_lst,
                    strict=True,
                )
            ]
        )
    )

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    new_edges: list[EntityEdge] = []

    # 汇总单边解析结果。
    # resolved_edges 包含最终要保存/更新的边；
    # invalidated_edges 是被新事实推翻的旧边；
    # new_edges 只记录真正新加入的边，供后续节点 summary 更新使用。
    for extracted_edge, result in zip(extracted_edges, results, strict=True):
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]
        # result[2] is duplicate_edges list

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_edge_chunk)

        # Track edges that are new (not duplicates of existing edges)
        # An edge is new if the resolved edge UUID matches the extracted edge UUID
        # 如果 resolved_edge.uuid 还是 extracted_edge.uuid，说明没有被解析成已有边，是新事实。
        if resolved_edge.uuid == extracted_edge.uuid:
            new_edges.append(resolved_edge)

    logger.debug(f'Resolved edges: {[e.uuid for e in resolved_edges]}')
    logger.debug(f'New edges (non-duplicates): {[e.uuid for e in new_edges]}')

    # 解析后边内容、属性、时间可能发生变化，所以给 resolved 和 invalidated edges 重新生成 embedding。
    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )

    return resolved_edges, invalidated_edges, new_edges


# 根据新边和候选旧边的时间关系，判断哪些旧边应该被新边失效。
# 这里不负责判断“是否语义矛盾”，语义矛盾由 LLM 在 resolve_extracted_edge 中选出；
# 这里负责在已知矛盾候选中应用 temporal logic。
def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        # 先把时间统一为 UTC，避免 naive datetime / timezone aware datetime 比较出错。
        edge_invalid_at_utc = ensure_utc(edge.invalid_at)
        resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
        edge_valid_at_utc = ensure_utc(edge.valid_at)
        resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at)

        # 如果旧边在新边生效前就已经失效，或者新边在旧边生效前就已失效，
        # 两者时间区间不重叠，不需要互相 invalidation。
        if (
            edge_invalid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_invalid_at_utc <= resolved_edge_valid_at_utc
        ) or (
            edge_valid_at_utc is not None
            and resolved_edge_invalid_at_utc is not None
            and resolved_edge_invalid_at_utc <= edge_valid_at_utc
        ):
            continue

        # New edge invalidates edge
        # 如果旧边更早生效，而新边更晚生效，并且两者被 LLM 判断为矛盾，
        # 则用新边的 valid_at 作为旧边的 invalid_at。
        elif (
            edge_valid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_valid_at_utc < resolved_edge_valid_at_utc
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


# 轻量级时间抽取：只为单条 edge 的 fact 抽 valid_at / invalid_at。
# 这个函数通常作为兜底使用：
# 如果主 edge extraction prompt 没有提供时间字段，就再调用一个小模型 prompt 尝试补齐。
async def _extract_edge_timestamps(
    llm_client: LLMClient,
    edge: EntityEdge,
    episode: EpisodicNode | None,
) -> None:
    """Extract valid_at / invalid_at timestamps for an edge via a lightweight LLM call.

    Modifies the edge in place. Skips if the edge already has timestamps set
    (e.g., from the extraction prompt in the separate-extraction path) or if
    no reference time is available.
    """
    # 如果 edge 已经有 valid_at 或 invalid_at，就不重复抽取，避免覆盖主抽取结果。
    if edge.valid_at is not None or edge.invalid_at is not None:
        return

    # 没有 episode 或 episode.valid_at 时，无法解析相对时间，直接跳过。
    if episode is None or episode.valid_at is None:
        return

    context = {
        'fact': edge.fact,
        'reference_time': episode.valid_at.isoformat(),
    }
    try:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_edges.extract_timestamps(context),
            response_model=EdgeTimestamps,
            model_size=ModelSize.small,
            prompt_name='extract_edges.extract_timestamps',
        )
        timestamps = EdgeTimestamps(**llm_response)

        # 分别解析 valid_at / invalid_at。
        # 解析失败只记录 debug，不影响该边保存；没有时间的边仍然可以作为事实边存在。
        if timestamps.valid_at:
            try:
                edge.valid_at = ensure_utc(
                    datetime.fromisoformat(timestamps.valid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                logger.debug(f'Error parsing valid_at: {timestamps.valid_at}')
        if timestamps.invalid_at:
            try:
                edge.invalid_at = ensure_utc(
                    datetime.fromisoformat(timestamps.invalid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                logger.debug(f'Error parsing invalid_at: {timestamps.invalid_at}')
    except Exception:
        logger.warning('Failed to extract timestamps for edge %s', edge.uuid, exc_info=True)


# 单条 EntityEdge 的解析核心。
# 它回答三个问题：
# 1. 新抽取的 edge 是否是已有 edge 的重复？
# 2. 新 edge 是否和某些旧 edge 矛盾，从而需要 invalidation？
# 3. 这条 edge 是否需要抽取自定义属性和时间字段？
async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    """Resolve an extracted edge against existing graph context.

    Parameters
    ----------
    llm_client : LLMClient
        Client used to invoke the LLM for deduplication and attribute extraction.
    extracted_edge : EntityEdge
        Newly extracted edge whose canonical representation is being resolved.
    related_edges : list[EntityEdge]
        Candidate edges with identical endpoints used for duplicate detection.
    existing_edges : list[EntityEdge]
        Broader set of edges evaluated for contradiction / invalidation.
    episode : EpisodicNode
        Episode providing content context when extracting edge attributes.
    edge_type_candidates : dict[str, type[BaseModel]] | None
        Custom edge types permitted for the current source/target signature.

    Returns
    -------
    tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]
        The resolved edge, any duplicates, and edges to invalidate.
    """
    # 快速路径：如果没有任何相关旧边，也没有任何可能冲突的旧边，
    # 就不需要做 LLM dedupe/contradiction 判断。
    # 但仍要处理自定义属性和时间字段，因为新边自身可能需要结构化补全。
    if len(related_edges) == 0 and len(existing_edges) == 0:
        # Still extract custom attributes and timestamps even when no dedup needed
        edge_model = edge_type_candidates.get(extracted_edge.name) if edge_type_candidates else None
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_context = {
                'fact': extracted_edge.fact,
                'reference_time': episode.valid_at if episode is not None else None,
                'existing_attributes': extracted_edge.attributes,
            }
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
                prompt_name='extract_edges.extract_attributes',
                attribute_extraction=True,
            )

            # edge 属性使用 replace 语义：
            # 这和 node 属性的 overlay 不同。边事实通常更原子，因此属性应该由当前 edge fact 重新决定。
            merged, _ = apply_capped_attributes(
                edge_attributes_response,
                edge_model,
                extracted_edge.attributes,
                merge_mode='replace',
                prompt_name='extract_edges.extract_attributes',
                entity_uuid=extracted_edge.uuid,
                group_id=extracted_edge.group_id,
            )
            extracted_edge.attributes = merged

        # 如果主抽取阶段没有时间字段，在这里尝试补齐 valid_at / invalid_at。
        await _extract_edge_timestamps(llm_client, extracted_edge, episode)

        return extracted_edge, [], []

    # Fast path: if the fact text and endpoints already exist verbatim, reuse the matching edge.
    # 第二个快速路径：端点相同且 fact 文本规范化后完全一致，直接复用已有 edge。
    # 这比 LLM dedupe 更便宜、更稳定。
    normalized_fact = _normalize_string_exact(extracted_edge.fact)
    for edge in related_edges:
        if (
            edge.source_node_uuid == extracted_edge.source_node_uuid
            and edge.target_node_uuid == extracted_edge.target_node_uuid
            and _normalize_string_exact(edge.fact) == normalized_fact
        ):
            resolved = edge

            # 如果这条已有边被新 episode 再次支持，就把 episode uuid 加到 provenance 列表中。
            if episode is not None and episode.uuid not in resolved.episodes:
                resolved.episodes.append(episode.uuid)
            return resolved, [], []

    start = time()

    # Prepare context for LLM with continuous indexing
    # 构造 duplicate candidates 上下文。
    # related_edges 只包括同端点的相关边，索引用 0..N-1。
    related_edges_context = [{'idx': i, 'fact': edge.fact} for i, edge in enumerate(related_edges)]

    # Invalidation candidates start where duplicate candidates end
    # invalidation candidates 索引从 related_edges 之后继续编号。
    # 这样 LLM 返回 contradicted_facts 时，可以在一个连续 index 空间里引用两类候选。
    invalidation_idx_offset = len(related_edges)
    invalidation_edge_candidates_context = [
        {'idx': invalidation_idx_offset + i, 'fact': existing_edge.fact}
        for i, existing_edge in enumerate(existing_edges)
    ]

    context = {
        'existing_edges': related_edges_context,
        'new_edge': extracted_edge.fact,
        'edge_invalidation_candidates': invalidation_edge_candidates_context,
    }

    if related_edges or existing_edges:
        logger.debug(
            'Resolving edge: sent %d EXISTING FACTS%s and %d INVALIDATION CANDIDATES%s',
            len(related_edges),
            f' (idx 0-{len(related_edges) - 1})' if related_edges else '',
            len(existing_edges),
            f' (idx {invalidation_idx_offset}-{invalidation_idx_offset + len(existing_edges) - 1})'
            if existing_edges
            else '',
        )

    # 调用 LLM 做两件事：
    # 1. duplicate_facts：哪些 related_edges 和 new_edge 是重复事实；
    # 2. contradicted_facts：哪些候选事实与 new_edge 矛盾。
    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_edges.resolve_edge(context),
        response_model=EdgeDuplicate,
        model_size=ModelSize.small,
        prompt_name='dedupe_edges.resolve_edge',
    )
    response_object = EdgeDuplicate(**llm_response)
    duplicate_facts = response_object.duplicate_facts

    # Validate duplicate_facts are in valid range for EXISTING FACTS
    # duplicate_facts 只能引用 related_edges，不允许引用 broader invalidation candidates。
    # 这里防御性过滤 LLM 的越界索引。
    invalid_duplicates = [i for i in duplicate_facts if i < 0 or i >= len(related_edges)]
    if invalid_duplicates:
        logger.warning(
            'LLM returned invalid duplicate_facts idx values %s (valid range: 0-%d for EXISTING FACTS)',
            invalid_duplicates,
            len(related_edges) - 1,
        )

    duplicate_fact_ids: list[int] = [i for i in duplicate_facts if 0 <= i < len(related_edges)]

    # 如果 LLM 判断有重复事实，则复用第一条 duplicate edge 作为 resolved_edge。
    # 否则保留 extracted_edge，表示它是新边。
    resolved_edge = extracted_edge
    for duplicate_fact_id in duplicate_fact_ids:
        resolved_edge = related_edges[duplicate_fact_id]
        break

    # 新 episode 也支持这个已有事实，因此追加 episode provenance。
    if duplicate_fact_ids and episode is not None:
        resolved_edge.episodes.append(episode.uuid)

    # Process contradicted facts (continuous indexing across both lists)
    # 处理 LLM 判断为矛盾的事实索引。
    contradicted_facts: list[int] = response_object.contradicted_facts
    invalidation_candidates: list[EntityEdge] = []

    # Only process contradictions if there are edges to check against
    if related_edges or existing_edges:
        max_valid_idx = len(related_edges) + len(existing_edges) - 1
        invalid_contradictions = [i for i in contradicted_facts if i < 0 or i > max_valid_idx]
        if invalid_contradictions:
            logger.warning(
                'LLM returned invalid contradicted_facts idx values %s (valid range: 0-%d)',
                invalid_contradictions,
                max_valid_idx,
            )

        # Split contradicted facts into those from related_edges vs existing_edges based on offset
        # contradicted_facts 可以引用两类候选：
        # - 0..len(related_edges)-1：同端点 duplicate candidates；
        # - offset..max_valid_idx：更广范围 invalidation candidates。
        for idx in contradicted_facts:
            if 0 <= idx < len(related_edges):
                # From EXISTING FACTS (duplicate candidates)
                invalidation_candidates.append(related_edges[idx])
            elif invalidation_idx_offset <= idx <= max_valid_idx:
                # From FACT INVALIDATION CANDIDATES (adjust index by offset)
                invalidation_candidates.append(existing_edges[idx - invalidation_idx_offset])

    # Only extract structured attributes if the edge's relation_type matches an allowed custom type
    # AND the edge model exists for this node pair signature
    # 如果当前边的 relation_type 对应允许的自定义 edge schema，就抽取结构化属性。
    edge_model = edge_type_candidates.get(resolved_edge.name) if edge_type_candidates else None
    if edge_model is not None and len(edge_model.model_fields) != 0:
        edge_attributes_context = {
            'fact': resolved_edge.fact,
            'reference_time': episode.valid_at if episode is not None else None,
            'existing_attributes': resolved_edge.attributes,
        }

        edge_attributes_response = await llm_client.generate_response(
            prompt_library.extract_edges.extract_attributes(edge_attributes_context),
            response_model=edge_model,  # type: ignore
            model_size=ModelSize.small,
            prompt_name='extract_edges.extract_attributes',
            attribute_extraction=True,
        )

        # 边属性采用 replace，不保留不再被当前 fact 支持的旧字段。
        merged, _ = apply_capped_attributes(
            edge_attributes_response,
            edge_model,
            resolved_edge.attributes,
            merge_mode='replace',
            prompt_name='extract_edges.extract_attributes',
            entity_uuid=resolved_edge.uuid,
            group_id=resolved_edge.group_id,
        )
        resolved_edge.attributes = merged
    else:
        # No matching edge schema → no structured attributes apply; clear any stale
        # attributes left from a prior schema. Intentionally not merged.
        # 如果没有匹配 schema，说明这条边不应携带结构化属性。
        # 这里故意清空，防止旧 schema 残留属性污染当前关系。
        resolved_edge.attributes = {}

    # Extract timestamps for new edges (duplicated edges retain their existing timestamps)
    # 只有新边才抽时间；重复边沿用已有时间字段，避免新 episode 覆盖旧事实生命周期。
    if resolved_edge.uuid == extracted_edge.uuid:
        await _extract_edge_timestamps(llm_client, resolved_edge, episode)

    end = time()
    logger.debug(
        f'Resolved Edge: {extracted_edge.uuid} -> {resolved_edge.uuid}, in {(end - start) * 1000} ms'
    )

    now = utc_now()

    # 如果 resolved_edge 自身已经有 invalid_at，但还没有 expired_at，
    # 表示它是一个已知会失效的事实，需要记录系统知道它失效的时间。
    if resolved_edge.invalid_at and not resolved_edge.expired_at:
        resolved_edge.expired_at = now

    # Determine if the new_edge needs to be expired
    # 如果 resolved_edge 还未过期，但存在比它 valid_at 更新的矛盾候选，
    # 说明当前新边其实被更晚的事实覆盖，应让当前边失效。
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at)))
        for candidate in invalidation_candidates:
            candidate_valid_at_utc = ensure_utc(candidate.valid_at)
            resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
            if (
                candidate_valid_at_utc is not None
                and resolved_edge_valid_at_utc is not None
                and candidate_valid_at_utc > resolved_edge_valid_at_utc
            ):
                # Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    # Determine which contradictory edges need to be expired
    # 反过来判断：当前 resolved_edge 是否会让旧的 contradiction candidates 失效。
    invalidated_edges: list[EntityEdge] = resolve_edge_contradictions(
        resolved_edge, invalidation_candidates
    )

    # duplicate_edges 只用于返回记录：这些是被 LLM 判定为重复的相关边。
    duplicate_edges: list[EntityEdge] = [related_edges[idx] for idx in duplicate_fact_ids]

    return resolved_edge, invalidated_edges, duplicate_edges


# 过滤已经存在 IS_DUPLICATE_OF 关系的节点重复对。
# 这不是普通实体去重判断本身，而是避免重复创建 duplicate marker edge。
async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    if not duplicates_node_tuples:
        return []

    # 先把待检查的重复节点对放进 map。
    # key 是 (source_uuid, target_uuid)，value 是原始节点对象对。
    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    # 不同图数据库的关系建模方式不同，所以这里按 provider 生成不同查询。
    if driver.provider == GraphProvider.NEPTUNE:
        query: LiteralString = """
            UNWIND $duplicate_node_uuids AS duplicate_tuple
            MATCH (n:Entity {uuid: duplicate_tuple.source})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple.target})
            RETURN DISTINCT
                n.uuid AS source_uuid,
                m.uuid AS target_uuid
        """

        duplicate_nodes = [
            {'source': source.uuid, 'target': target.uuid}
            for source, target in duplicates_node_tuples
        ]

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_nodes,
            routing_='r',
        )
    else:
        if driver.provider == GraphProvider.KUZU:
            # Kuzu 的 edge 表达方式和 Neo4j 不同，这里通过中间 RelatesToNode_ 匹配关系。
            query = """
                UNWIND $duplicate_node_uuids AS duplicate
                MATCH (n:Entity {uuid: duplicate.src})-[:RELATES_TO]->(e:RelatesToNode_ {name: 'IS_DUPLICATE_OF'})-[:RELATES_TO]->(m:Entity {uuid: duplicate.dst})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = [{'src': src, 'dst': dst} for src, dst in duplicate_nodes_map]
        else:
            # 默认查询形态适用于 Neo4j / 类 Cypher 图数据库。
            query: LiteralString = """
                UNWIND $duplicate_node_uuids AS duplicate_tuple
                MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = list(duplicate_nodes_map.keys())

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_node_uuids,
            routing_='r',
        )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    # 数据库里已经存在 IS_DUPLICATE_OF 的节点对，从待创建列表中移除。
    # 返回值只保留“还没有 duplicate edge、需要后续创建”的重复节点对。
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())
