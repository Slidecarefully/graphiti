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

# 先引入本模块依赖：日志、计时、类型标注、NumPy 向量计算，以及 Graphiti 内部的图查询、节点/边模型和过滤器。
import logging
from collections import defaultdict
from time import time
from typing import Any

import numpy as np
from numpy._typing import NDArray
from typing_extensions import LiteralString

# 下面这些 Graphiti 组件共同构成搜索层：driver 负责后端差异，query helpers 负责生成 Cypher/类 Cypher 片段，record
# 转换函数负责把数据库结果还原成领域对象。
from graphiti_core.driver.driver import (
    GraphDriver,
    GraphProvider,
)
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.graph_queries import (
    get_nodes_query,
    get_relationships_query,
    get_vector_cosine_func_query,
)
from graphiti_core.helpers import (
    lucene_sanitize,
    normalize_l2,
    semaphore_gather,
    validate_group_ids,
)
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    EPISODIC_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodicNode,
    get_community_node_from_record,
    get_entity_node_from_record,
    get_episodic_node_from_record,
)
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

# 模块级 logger 用来记录耗时和调试信息，避免把性能观测逻辑散落在各个搜索函数里。
logger = logging.getLogger(__name__)

# 这些常量是搜索流程的默认阈值：召回数量、向量分数下限、MMR 权重、遍历深度和全文查询长度都在这里集中控制。
RELEVANT_SCHEMA_LIMIT = 10
DEFAULT_MIN_SCORE = 0.6
DEFAULT_MMR_LAMBDA = 0.5
MAX_SEARCH_DEPTH = 3
MAX_QUERY_LENGTH = 128


# 基础工具函数：在后端无法直接做向量相似度时，用 NumPy 本地计算 cosine similarity。
def calculate_cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    """
    Calculates the cosine similarity between two vectors using NumPy.
    """
    # 先计算点积和两个向量的范数，这是 cosine similarity 的三个核心组成部分。
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # 任一向量长度为 0 时没有方向可比，直接返回 0，避免除以 0。
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0  # Handle cases where one or both vectors are zero vectors

    # 非零向量按 cosine 公式归一化点积，结果越接近 1 表示方向越相似。
    return dot_product / (norm_vector1 * norm_vector2)


# 全文查询构造入口：先根据不同图数据库/搜索后端的语法能力生成可执行查询，再把 group_id 约束合并进去。
def fulltext_query(query: str, group_ids: list[str] | None, driver: GraphDriver):
    # 任何查询拼接前都先校验 group_id，避免把不合法的分组标识带进后端查询语法。
    validate_group_ids(group_ids)

    # Kuzu 的全文能力较简单，所以这里不组装复杂 Lucene 语法，只做长度保护后返回原始查询。
    if driver.provider == GraphProvider.KUZU:
        # Kuzu only supports simple queries.
        if len(query.split(' ')) > MAX_QUERY_LENGTH:
            return ''
        return query
    # FalkorDB 有自己的全文查询构造能力，因此把后端差异封装交还给 driver。
    elif driver.provider == GraphProvider.FALKORDB:
        return driver.build_fulltext_query(query, group_ids, MAX_QUERY_LENGTH)
    # 默认路径按 group_id 生成 Lucene 过滤片段，多个 group 会被 OR 连接。
    group_ids_filter_list = (
        [driver.fulltext_syntax + f'group_id:"{g}"' for g in group_ids]
        if group_ids is not None
        else []
    )
    group_ids_filter = ''
    # 逐个拼接 group 过滤条件，保持最终语义为“命中任一指定 group”。
    for f in group_ids_filter_list:
        group_ids_filter += f if not group_ids_filter else f' OR {f}'

    # 如果存在 group 过滤，就在它后面追加 AND，让后续正文查询与分组条件同时生效。
    group_ids_filter += ' AND ' if group_ids_filter else ''

    # 用户查询文本先做 Lucene 转义，防止特殊字符破坏全文检索语法。
    lucene_query = lucene_sanitize(query)
    # If the lucene query is too long return no query
    # 全文查询过长会给后端带来解析或性能问题，因此超过限制时返回空查询，由调用方返回空结果。
    if len(lucene_query.split(' ')) + len(group_ids or '') >= MAX_QUERY_LENGTH:
        return ''

    # 最终查询由 group 过滤和括号包裹的正文 Lucene 查询组成。
    full_query = group_ids_filter + '(' + lucene_query + ')'

    return full_query


# 从关系边反查 episode：边上记录了产生该事实的 episode uuid，这里按出现顺序收集并限制返回数量。
async def get_episodes_by_mentions(
    driver: GraphDriver,
    nodes: list[EntityNode],
    edges: list[EntityEdge],
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    # 先准备一个线性列表保存边引用过的 episode uuid。
    episode_uuids: list[str] = []
    # 每条事实边可能关联多个 episode，这里把这些来源全部展开到同一个列表中。
    for edge in edges:
        episode_uuids.extend(edge.episodes)

    # 只取前 limit 个 episode uuid 批量查询，避免相关来源过多时扩大开销。
    episodes = await EpisodicNode.get_by_uuids(driver, episode_uuids[:limit])

    return episodes


# 从 episode 找被提及实体：优先走 driver 暴露的图操作接口；没有实现时再退回到通用查询。
async def get_mentioned_nodes(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[EntityNode]:
    # 如果后端实现了专用图操作接口，就优先用它；这通常能利用后端特性或更高效的查询。
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.get_mentioned_nodes(driver, episodes)
        # 专用接口声明未实现时不报错，而是继续走下面的通用查询路径。
        except NotImplementedError:
            pass

    # 通用查询只需要 episode uuid，因此先从 episode 对象中抽取主键。
    episode_uuids = [episode.uuid for episode in episodes]

    # 沿 Episodic-[:MENTIONS]->Entity 关系查找被提及实体，并通过 DISTINCT 去重。
    records, _, _ = await driver.execute_query(
        """
        MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity)
        WHERE episode.uuid IN $uuids
        RETURN DISTINCT
        """
        + get_entity_node_return_query(driver.provider),
        uuids=episode_uuids,
        routing_='r',
    )

    # 数据库记录仍是原始 record，这里统一转换为 EntityNode 模型，屏蔽不同 provider 的返回差异。
    nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

    return nodes


# 从实体节点找到所属社区：同样先尝试后端专用接口，再用标准图查询兜底。
async def get_communities_by_nodes(
    driver: GraphDriver, nodes: list[EntityNode]
) -> list[CommunityNode]:
    # 社区查找同样优先走后端专用接口，保持扩展点和默认实现解耦。
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.get_communities_by_nodes(driver, nodes)
        except NotImplementedError:
            pass

    # 通用路径只需要输入实体的 uuid 列表。
    node_uuids = [node.uuid for node in nodes]

    # 通过 Community-[:HAS_MEMBER]->Entity 关系反查包含这些实体的社区。
    records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)
        WHERE m.uuid IN $uuids
        RETURN DISTINCT
        """
        + COMMUNITY_NODE_RETURN,
        uuids=node_uuids,
        routing_='r',
    )

    # 查询结果转换为 CommunityNode，供上层逻辑直接使用领域对象。
    communities = [get_community_node_from_record(record) for record in records]

    return communities


# 边的全文搜索：用文本查询匹配关系名和 fact 内容，最后返回 EntityEdge 对象。
async def edge_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    # 如果 driver 已提供搜索接口，整段默认查询逻辑都被绕过，避免重复实现后端特化能力。
    if driver.search_interface:
        return await driver.search_interface.edge_fulltext_search(
            driver, query, search_filter, group_ids, limit
        )

    # fulltext search over facts
    # 把用户查询转换成当前后端可接受的全文查询；空字符串表示查询不可执行或过长。
    fuzzy_query = fulltext_query(query, group_ids, driver)

    # 全文查询为空时直接返回空列表，避免把无效语句发送到数据库。
    if fuzzy_query == '':
        return []

    # 默认图模型中事实以 RELATES_TO 边保存，全文索引返回关系后需要重新 MATCH 出两端实体。
    match_query = """
    YIELD relationship AS rel, score
    MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)
    """
    # Kuzu 把关系建模为中间节点 RelatesToNode_，因此匹配路径要改成 Entity -> RelatesToNode_ -> Entity。
    if driver.provider == GraphProvider.KUZU:
        match_query = """
        YIELD node, score
        MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: node.uuid})-[:RELATES_TO]->(m:Entity)
        """

    # 先由过滤器构造器生成 provider 兼容的 WHERE 条件和参数，后续再叠加 group 条件。
    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # 指定 group_id 时，只保留同一 group 内的事实边，防止跨租户或跨数据集召回。
    if group_ids is not None:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    # 把所有过滤条件统一拼成 WHERE 子句；没有过滤条件时保持查询片段为空。
    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    # Neptune 路径先用外部 AOSS 索引搜 uuid 和分数，再回到图数据库按 id 取完整边属性。
    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('edge_name_and_fact', query)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # AOSS 命中后保存边 uuid 和搜索分数，用于后续图查询和排序。
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # 第二阶段按 AOSS 返回的 id 匹配实际边，补齐端点 uuid、时间字段、episodes 和属性。
            # Match the edge ids and return the values
            query = (
                """
                                UNWIND $ids as id
                                MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
                                WHERE e.group_id IN $group_ids 
                                AND id(e)=id 
                                """
                + filter_query
                + """
                AND id(e)=id
                WITH e, id.score as score, startNode(e) AS n, endNode(e) AS m
                RETURN
                    e.uuid AS uuid,
                    e.group_id AS group_id,
                    n.uuid AS source_node_uuid,
                    m.uuid AS target_node_uuid,
                    e.created_at AS created_at,
                    e.name AS name,
                    e.fact AS fact,
                    split(e.episodes, ",") AS episodes,
                    e.expired_at AS expired_at,
                    e.valid_at AS valid_at,
                    e.invalid_at AS invalid_at,
                    properties(e) AS attributes
                ORDER BY score DESC LIMIT $limit
                            """
            )

            # 执行 Neptune 的回表查询，filter_params 会继续限制搜索范围。
            records, _, _ = await driver.execute_query(
                query,
                query=fuzzy_query,
                ids=input_ids,
                limit=limit,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    # 非 Neptune 后端直接把全文索引查询、关系匹配、过滤和返回字段拼成一个图查询。
    else:
        query = (
            get_relationships_query('edge_name_and_fact', limit=limit, provider=driver.provider)
            + match_query
            + filter_query
            + """
            WITH e, score, n, m
            RETURN
            """
            + get_entity_edge_return_query(driver.provider)
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        # 执行默认全文搜索查询，query 参数传入的是已经按 provider 处理过的 fuzzy_query。
        records, _, _ = await driver.execute_query(
            query,
            query=fuzzy_query,
            limit=limit,
            routing_='r',
            **filter_params,
        )

    # 最后统一把 record 转为 EntityEdge，调用者不需要关心底层 provider 的字段形态。
    edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

    return edges


# 边的向量相似搜索：用输入 embedding 与边的 fact_embedding 比较，并可按源节点、目标节点和 group 进一步约束。
async def edge_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityEdge]:
    # 有专用搜索接口时优先委托，默认实现只负责通用 fallback。
    if driver.search_interface:
        return await driver.search_interface.edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid,
            target_node_uuid,
            search_filter,
            group_ids,
            limit,
            min_score,
        )

    # 默认关系模型直接匹配实体之间的 RELATES_TO 边，为后续向量打分提供候选集合。
    match_query = """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
    """
    # Kuzu 的关系节点模型需要用两段 RELATES_TO 才能拿到中间的事实节点。
    if driver.provider == GraphProvider.KUZU:
        match_query = """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
        """

    # 边搜索过滤器负责把 SearchFilters 翻译成当前后端支持的条件和参数。
    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # group 过滤存在时，还可以进一步叠加源节点和目标节点限制，把候选边收窄到指定端点。
    if group_ids is not None:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

        if source_node_uuid is not None:
            filter_params['source_uuid'] = source_node_uuid
            filter_queries.append('n.uuid = $source_uuid')

        if target_node_uuid is not None:
            filter_params['target_uuid'] = target_node_uuid
            filter_queries.append('m.uuid = $target_uuid')

    # 过滤条件集中拼接，后面的 Neptune 和非 Neptune 分支都会复用。
    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    # 默认把查询向量作为参数传入；Kuzu 需要显式 CAST 成固定长度 FLOAT 数组。
    search_vector_var = '$search_vector'
    if driver.provider == GraphProvider.KUZU:
        search_vector_var = f'CAST($search_vector AS FLOAT[{len(search_vector)}])'

    # Neptune 分支先取回候选边的 embedding 字符串，再在 Python 中手动计算 cosine。
    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            """
                            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
                            """
            + filter_query
            + """
            RETURN DISTINCT id(e) as id, e.fact_embedding as embedding
            """
        )
        # 第一段 Neptune 查询只返回图内部 id 和 embedding，减少传输字段。
        resp, header, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

        # 有候选时才进入本地相似度计算，否则直接返回空结果。
        if len(resp) > 0:
            # Calculate Cosine similarity then return the edge ids
            # input_ids 同时保存图 id 和相似度分数，后续查询据此排序。
            input_ids = []
            for r in resp:
                # Neptune 中 embedding 可能为空，计算前先跳过缺失向量。
                if r['embedding']:
                    score = calculate_cosine_similarity(
                        search_vector, list(map(float, r['embedding'].split(',')))
                    )
                    # 只有超过 min_score 的边才进入第二阶段回表，降低噪声和查询开销。
                    if score > min_score:
                        input_ids.append({'id': r['id'], 'score': score})

            # 第二阶段按内部 id 找回完整关系属性，并保持相似度降序。
            # Match the edge ides and return the values
            query = """
                UNWIND $ids as i
                MATCH ()-[r]->()
                WHERE id(r) = i.id
                RETURN
                    r.uuid AS uuid,
                    r.group_id AS group_id,
                    startNode(r).uuid AS source_node_uuid,
                    endNode(r).uuid AS target_node_uuid,
                    r.created_at AS created_at,
                    r.name AS name,
                    r.fact AS fact,
                    split(r.episodes, ",") AS episodes,
                    r.expired_at AS expired_at,
                    r.valid_at AS valid_at,
                    r.invalid_at AS invalid_at,
                    properties(r) AS attributes
                ORDER BY i.score DESC
                LIMIT $limit
                    """
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                search_vector=search_vector,
                limit=limit,
                min_score=min_score,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    # 非 Neptune 后端在数据库查询里直接计算向量 cosine，并用 min_score 做阈值过滤。
    else:
        query = (
            match_query
            + filter_query
            + """
            WITH DISTINCT e, n, m, """
            + get_vector_cosine_func_query('e.fact_embedding', search_vector_var, driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + get_entity_edge_return_query(driver.provider)
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        # 执行向量查询时传入 search_vector、limit 和 min_score，过滤参数通过 **filter_params 合并。
        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

    # 所有分支最终都归一化成 EntityEdge 列表。
    edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

    return edges


# 边的 BFS 搜索：从一组起点实体或 episode 出发沿图遍历，收集遍历路径中遇到的事实边。
async def edge_bfs_search(
    driver: GraphDriver,
    bfs_origin_node_uuids: list[str] | None,
    bfs_max_depth: int,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    # 先尝试后端自带 BFS 搜索接口；未实现时继续使用通用图遍历。
    if driver.search_interface:
        try:
            return await driver.search_interface.edge_bfs_search(
                driver, bfs_origin_node_uuids, bfs_max_depth, search_filter, group_ids, limit
            )
        except NotImplementedError:
            pass

    # vector similarity search over embedded facts
    # BFS 没有起点就没有遍历空间，提前返回空结果。
    if bfs_origin_node_uuids is None or len(bfs_origin_node_uuids) == 0:
        return []

    # 构造边过滤条件，确保遍历到的候选边仍满足 SearchFilters。
    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # 如果限定 group，只收集该 group 下的关系边。
    if group_ids is not None:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    # Kuzu 的边是中间节点，因此 BFS 深度要按关系节点模型折算。
    if driver.provider == GraphProvider.KUZU:
        # Kuzu stores entity edges twice with an intermediate node, so we need to match them
        # separately for the correct BFS depth.
        # 实体出发到事实节点的路径长度为普通图边数的两倍减一。
        depth = bfs_max_depth * 2 - 1
        match_queries = [
            f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH path = (origin:Entity {{uuid: origin_uuid}})-[:RELATES_TO*1..{depth}]->(:RelatesToNode_)
            UNWIND nodes(path) AS relNode
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {{uuid: relNode.uuid}})-[:RELATES_TO]->(m:Entity)
            """,
        ]
        # 当搜索深度大于 1 时，episode 起点先通过 MENTIONS 到实体，再继续沿关系扩展。
        if bfs_max_depth > 1:
            depth = (bfs_max_depth - 1) * 2 - 1
            match_queries.append(f"""
                UNWIND $bfs_origin_node_uuids AS origin_uuid
                MATCH path = (origin:Episodic {{uuid: origin_uuid}})-[:MENTIONS]->(:Entity)-[:RELATES_TO*1..{depth}]->(:RelatesToNode_)
                UNWIND nodes(path) AS relNode
                MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {{uuid: relNode.uuid}})-[:RELATES_TO]->(m:Entity)
            """)

        # Kuzu 可能生成多条 match_query，因此用 records 聚合每个子查询的结果。
        records = []
        # 逐条执行 Kuzu 的候选查询，再把结果累加。
        for match_query in match_queries:
            sub_records, _, _ = await driver.execute_query(
                match_query
                + filter_query
                + """
                RETURN DISTINCT
                """
                + get_entity_edge_return_query(driver.provider)
                + """
                LIMIT $limit
                """,
                bfs_origin_node_uuids=bfs_origin_node_uuids,
                limit=limit,
                routing_='r',
                **filter_params,
            )
            records.extend(sub_records)
    # 非 Kuzu 后端可以直接沿 RELATES_TO 或 MENTIONS 做变长路径遍历。
    else:
        # Neptune 的语法和返回字段略有差异，所以单独构造查询。
        if driver.provider == GraphProvider.NEPTUNE:
            query = (
                f"""
                UNWIND $bfs_origin_node_uuids AS origin_uuid
                MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS *1..{bfs_max_depth}]->(n:Entity)
                WHERE origin:Entity OR origin:Episodic
                UNWIND relationships(path) AS rel
                MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]-(m:Entity)
                """
                + filter_query
                + """
                RETURN DISTINCT
                    e.uuid AS uuid,
                    e.group_id AS group_id,
                    startNode(e).uuid AS source_node_uuid,
                    endNode(e).uuid AS target_node_uuid,
                    e.created_at AS created_at,
                    e.name AS name,
                    e.fact AS fact,
                    split(e.episodes, ',') AS episodes,
                    e.expired_at AS expired_at,
                    e.valid_at AS valid_at,
                    e.invalid_at AS invalid_at,
                    properties(e) AS attributes
                LIMIT $limit
                """
            )
        # 其他后端直接从路径中展开 relationships，再用 rel.uuid 找回完整的 RELATES_TO 边。
        else:
            query = (
                f"""
                UNWIND $bfs_origin_node_uuids AS origin_uuid
                MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(:Entity)
                UNWIND relationships(path) AS rel
                MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]-(m:Entity)
                """
                + filter_query
                + """
                RETURN DISTINCT
                """
                + get_entity_edge_return_query(driver.provider)
                + """
                LIMIT $limit
                """
            )

        # 执行 BFS 查询，bfs_origin_node_uuids 控制起点集合，limit 控制最终召回量。
        records, _, _ = await driver.execute_query(
            query,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            depth=bfs_max_depth,
            limit=limit,
            routing_='r',
            **filter_params,
        )

    # 遍历结果仍统一转成 EntityEdge。
    edges = [get_entity_edge_from_record(record, driver.provider) for record in records]

    return edges


# 节点全文搜索：用 BM25/全文索引在实体名称和摘要上召回候选 EntityNode。
async def node_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    # 节点全文搜索也遵循“先委托后 fallback”的模式。
    if driver.search_interface:
        return await driver.search_interface.node_fulltext_search(
            driver, query, search_filter, group_ids, limit
        )

    # BM25 search to get top nodes
    # 复用 fulltext_query 生成 provider 兼容的文本查询。
    fuzzy_query = fulltext_query(query, group_ids, driver)
    # 全文查询不可执行时直接返回空列表。
    if fuzzy_query == '':
        return []

    # 节点过滤器生成 EntityNode 相关的过滤条件。
    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # 指定 group 时把节点限制在目标 group 内。
    if group_ids is not None:
        filter_queries.append('n.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    # 默认全文索引查询使用 YIELD；Kuzu 使用 WITH 来承接索引返回的 node 和 score。
    yield_query = 'YIELD node AS n, score'
    if driver.provider == GraphProvider.KUZU:
        yield_query = 'WITH node AS n, score'

    # Neptune 通过 AOSS 在 node_name_and_summary 索引上先拿 uuid 和分数。
    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('node_name_and_summary', query, limit=limit)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # 保存 AOSS 命中的实体 uuid 与分数，用于回到图中取完整字段。
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # 回表查询按 uuid 匹配 Entity，并沿用 AOSS 分数排序。
            # Match the edge ides and return the values
            query = (
                """
                                UNWIND $ids as i
                                MATCH (n:Entity)
                                WHERE n.uuid=i.id
                                RETURN
                                """
                + get_entity_node_return_query(driver.provider)
                + """
                ORDER BY i.score DESC
                LIMIT $limit
                            """
            )
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                query=fuzzy_query,
                limit=limit,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    # 非 Neptune 分支直接使用节点全文索引，随后拼接 provider 特定的节点返回字段。
    else:
        query = (
            get_nodes_query(
                'node_name_and_summary', '$query', limit=limit, provider=driver.provider
            )
            + yield_query
            + filter_query
            + """
            WITH n, score
            ORDER BY score DESC
            LIMIT $limit
            RETURN
            """
            + get_entity_node_return_query(driver.provider)
        )

        # 执行默认节点全文查询。
        records, _, _ = await driver.execute_query(
            query,
            query=fuzzy_query,
            limit=limit,
            routing_='r',
            **filter_params,
        )

    # 把 record 转换为 EntityNode，隐藏 provider 差异。
    nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

    return nodes


# 节点向量相似搜索：用输入向量与实体节点 name_embedding 比较，得到语义上接近的实体。
async def node_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityNode]:
    # 有专用节点向量搜索接口时直接委托。
    if driver.search_interface:
        return await driver.search_interface.node_similarity_search(
            driver, search_vector, search_filter, group_ids, limit, min_score
        )

    # 节点过滤器会生成对 Entity 节点可用的条件和参数。
    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # group_id 存在时只比较同组实体，避免不同数据域中的同名节点互相干扰。
    if group_ids is not None:
        filter_queries.append('n.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    # Kuzu 需要显式向量类型转换，其他后端直接使用参数。
    search_vector_var = '$search_vector'
    if driver.provider == GraphProvider.KUZU:
        search_vector_var = f'CAST($search_vector AS FLOAT[{len(search_vector)}])'

    # Neptune 路径先读取候选节点 embedding，然后在 Python 中计算相似度。
    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            """
                                                                                                                                    MATCH (n:Entity)
                                                                                                                                    """
            + filter_query
            + """
            RETURN DISTINCT id(n) as id, n.name_embedding as embedding
            """
        )
        # 这里的第一阶段查询只需要节点 id 和 name_embedding。
        resp, header, _ = await driver.execute_query(
            query,
            params=filter_params,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
        )

        # 有结果时逐个计算 cosine，并把超过阈值的候选放入 input_ids。
        if len(resp) > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in resp:
                if r['embedding']:
                    score = calculate_cosine_similarity(
                        search_vector, list(map(float, r['embedding'].split(',')))
                    )
                    if score > min_score:
                        input_ids.append({'id': r['id'], 'score': score})

            # 第二阶段根据内部 id 回表取完整 EntityNode 字段，并按分数排序。
            # Match the edge ides and return the values
            query = (
                """
                                                                                                                                                                UNWIND $ids as i
                                                                                                                                                                MATCH (n:Entity)
                                                                                                                                                                WHERE id(n)=i.id
                                                                                                                                                                RETURN 
                                                                                                                                                                """
                + get_entity_node_return_query(driver.provider)
                + """
                    ORDER BY i.score DESC
                    LIMIT $limit
                """
            )
            records, header, _ = await driver.execute_query(
                query,
                ids=input_ids,
                search_vector=search_vector,
                limit=limit,
                min_score=min_score,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    # 其他后端直接在图查询中计算 n.name_embedding 与 search_vector 的 cosine。
    else:
        query = (
            """
                                                                                                                                    MATCH (n:Entity)
                                                                                                                                    """
            + filter_query
            + """
            WITH n, """
            + get_vector_cosine_func_query('n.name_embedding', search_vector_var, driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + get_entity_node_return_query(driver.provider)
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        # 执行默认向量查询并应用 min_score。
        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

    # 结果统一转为 EntityNode。
    nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

    return nodes


# 节点 BFS 搜索：从起点沿 MENTIONS/RELATES_TO 关系扩展，召回同一 group 内的实体节点。
async def node_bfs_search(
    driver: GraphDriver,
    bfs_origin_node_uuids: list[str] | None,
    search_filter: SearchFilters,
    bfs_max_depth: int,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    # 优先使用专用 node BFS 实现；未实现时用通用查询兜底。
    if driver.search_interface:
        try:
            return await driver.search_interface.node_bfs_search(
                driver, bfs_origin_node_uuids, search_filter, bfs_max_depth, group_ids, limit
            )
        except NotImplementedError:
            pass

    # 没有起点或深度小于 1 时，BFS 没有有效语义，直接返回空列表。
    if bfs_origin_node_uuids is None or len(bfs_origin_node_uuids) == 0 or bfs_max_depth < 1:
        return []

    # 先生成节点过滤条件，后续每个 match_query 都会复用。
    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # group 过滤同时限制目标节点和起点节点，确保遍历在同一数据域内发生。
    if group_ids is not None:
        filter_queries.append('n.group_id IN $group_ids')
        filter_queries.append('origin.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' AND ' + (' AND '.join(filter_queries))

    # 默认路径从起点沿 RELATES_TO 或 MENTIONS 最多扩展 bfs_max_depth 层，最终落到 Entity。
    match_queries = [
        f"""
        UNWIND $bfs_origin_node_uuids AS origin_uuid
        MATCH (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(n:Entity)
        WHERE n.group_id = origin.group_id
        """
    ]

    # Neptune 使用自己的标签/关系匹配形式，因此替换默认 match_query。
    if driver.provider == GraphProvider.NEPTUNE:
        match_queries = [
            f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH (origin {{uuid: origin_uuid}})-[e:RELATES_TO|MENTIONS*1..{bfs_max_depth}]->(n:Entity)
            WHERE origin:Entity OR origin.Episode
            AND n.group_id = origin.group_id
            """
        ]

    # Kuzu 使用关系中间节点，实体到实体的关系需要两跳，所以深度要加倍。
    if driver.provider == GraphProvider.KUZU:
        depth = bfs_max_depth * 2
        match_queries = [
            """
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH (origin:Episodic {uuid: origin_uuid})-[:MENTIONS]->(n:Entity)
            WHERE n.group_id = origin.group_id
            """,
            f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH (origin:Entity {{uuid: origin_uuid}})-[:RELATES_TO*2..{depth}]->(n:Entity)
            WHERE n.group_id = origin.group_id
            """,
        ]
        # 当深度大于 1 时，Kuzu 还补充 episode 起点经过实体后再沿关系扩展的路径。
        if bfs_max_depth > 1:
            depth = (bfs_max_depth - 1) * 2
            match_queries.append(f"""
                UNWIND $bfs_origin_node_uuids AS origin_uuid
                MATCH (origin:Episodic {{uuid: origin_uuid}})-[:MENTIONS]->(:Entity)-[:RELATES_TO*2..{depth}]->(n:Entity)
                WHERE n.group_id = origin.group_id
            """)

    # 所有后端都把一组 match_query 的结果累积到 records 中。
    records = []
    # 逐条执行匹配查询，统一拼接过滤条件和节点返回字段。
    for match_query in match_queries:
        sub_records, _, _ = await driver.execute_query(
            match_query
            + filter_query
            + """
            RETURN
            """
            + get_entity_node_return_query(driver.provider)
            + """
            LIMIT $limit
            """,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            limit=limit,
            routing_='r',
            **filter_params,
        )
        records.extend(sub_records)

    # 把 BFS 命中的节点 record 转换成 EntityNode。
    nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

    return nodes


# Episode 全文搜索：在 episode content 上做文本召回，返回 EpisodicNode。
async def episode_fulltext_search(
    driver: GraphDriver,
    query: str,
    _search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    # Episode 搜索如果由 driver 提供专用接口，就直接交给后端实现。
    if driver.search_interface:
        return await driver.search_interface.episode_fulltext_search(
            driver, query, _search_filter, group_ids, limit
        )

    # BM25 search to get top episodes
    # 正文查询同样先走 fulltext_query，保证长度和 provider 语法一致。
    fuzzy_query = fulltext_query(query, group_ids, driver)
    if fuzzy_query == '':
        return []

    # Episode 这里没有使用完整 SearchFilters，只额外处理 group 过滤。
    filter_params: dict[str, Any] = {}
    group_filter_query: LiteralString = ''
    if group_ids is not None:
        group_filter_query += '\nAND e.group_id IN $group_ids'
        filter_params['group_ids'] = group_ids

    # Neptune 先在 episode_content 索引中做文本召回，再回表到图数据库。
    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('episode_content', query, limit=limit)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # 保存 episode uuid 和分数，后续图查询按分数排序。
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # 回表查询取出 EpisodicNode 所需字段，包括 content、时间、来源和关联边。
            # Match the edge ides and return the values
            query = """
                UNWIND $ids as i
                MATCH (e:Episodic)
                WHERE e.uuid=i.uuid
            RETURN
                    e.content AS content,
                    e.created_at AS created_at,
                    e.valid_at AS valid_at,
                    e.uuid AS uuid,
                    e.name AS name,
                    e.group_id AS group_id,
                    e.source_description AS source_description,
                    e.source AS source,
                    e.entity_edges AS entity_edges
                ORDER BY i.score DESC
                LIMIT $limit
            """
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                query=fuzzy_query,
                limit=limit,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    # 非 Neptune 路径直接使用节点全文索引，再按 episode.uuid 匹配真实 Episodic 节点。
    else:
        query = (
            get_nodes_query('episode_content', '$query', limit=limit, provider=driver.provider)
            + """
            YIELD node AS episode, score
            MATCH (e:Episodic)
            WHERE e.uuid = episode.uuid
            """
            + group_filter_query
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        # 执行 episode 全文查询。
        records, _, _ = await driver.execute_query(
            query, query=fuzzy_query, limit=limit, routing_='r', **filter_params
        )

    # 把查询记录转为 EpisodicNode。
    episodes = [get_episodic_node_from_record(record) for record in records]

    return episodes


# Community 全文搜索：在社区名称上做文本召回，返回 CommunityNode。
async def community_fulltext_search(
    driver: GraphDriver,
    query: str,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[CommunityNode]:
    # 社区全文搜索先尝试专用接口，未实现时继续默认路径。
    if driver.search_interface:
        try:
            return await driver.search_interface.community_fulltext_search(
                driver, query, group_ids, limit
            )
        except NotImplementedError:
            pass

    # BM25 search to get top communities
    # 社区名称查询也复用统一的全文查询构造函数。
    fuzzy_query = fulltext_query(query, group_ids, driver)
    if fuzzy_query == '':
        return []

    # 社区搜索只需要 group 过滤参数，因此这里手工维护 group_filter_query。
    filter_params: dict[str, Any] = {}
    group_filter_query: LiteralString = ''
    if group_ids is not None:
        group_filter_query = 'WHERE c.group_id IN $group_ids'
        filter_params['group_ids'] = group_ids

    # 不同后端承接全文索引返回结果的语法不同，yield_query 在这里提前切换。
    yield_query = 'YIELD node AS c, score'
    if driver.provider == GraphProvider.KUZU:
        yield_query = 'WITH node AS c, score'

    # Neptune 先查询 community_name 索引获取 uuid 和分数。
    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('community_name', query, limit=limit)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            # Calculate Cosine similarity then return the edge ids
            # 保存社区 uuid 与搜索分数，随后按 uuid 找回完整社区字段。
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # 回表查询返回 CommunityNode 需要的字段，并把字符串 embedding 转成浮点数组。
            # Match the edge ides and return the values
            query = """
                UNWIND $ids as i
                MATCH (comm:Community)
                WHERE comm.uuid=i.id
                RETURN
                    comm.uuid AS uuid,
                    comm.group_id AS group_id,
                    comm.name AS name,
                    comm.created_at AS created_at,
                    comm.summary AS summary,
                    [x IN split(comm.name_embedding, ",") | toFloat(x)]AS name_embedding
                ORDER BY i.score DESC
                LIMIT $limit
            """
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                query=fuzzy_query,
                limit=limit,
                routing_='r',
                **filter_params,
            )
        else:
            return []
    # 默认路径直接在社区名称索引上检索，然后拼接统一的 COMMUNITY_NODE_RETURN。
    else:
        query = (
            get_nodes_query('community_name', '$query', limit=limit, provider=driver.provider)
            + yield_query
            + """
            WITH c, score
            """
            + group_filter_query
            + """
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        # 执行社区全文查询。
        records, _, _ = await driver.execute_query(
            query, query=fuzzy_query, limit=limit, routing_='r', **filter_params
        )

    # 把 record 转换成 CommunityNode。
    communities = [get_community_node_from_record(record) for record in records]

    return communities


# Community 向量相似搜索：用社区 name_embedding 做语义召回。
async def community_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
    min_score=DEFAULT_MIN_SCORE,
) -> list[CommunityNode]:
    # 先尝试后端自带的社区向量搜索实现。
    if driver.search_interface:
        try:
            return await driver.search_interface.community_similarity_search(
                driver, search_vector, group_ids, limit, min_score
            )
        except NotImplementedError:
            pass

    # 默认实现走社区 name_embedding 的向量相似度。
    # vector similarity search over entity names
    query_params: dict[str, Any] = {}

    # 如果有 group 限制，就把它合入查询参数和 WHERE 片段。
    group_filter_query: LiteralString = ''
    if group_ids is not None:
        group_filter_query += ' WHERE c.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

    # Neptune 分支先读出候选社区的 embedding 字符串，再在 Python 中算相似度。
    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            """
                                                                                                                                    MATCH (n:Community)
                                                                                                                                    """
            + group_filter_query
            + """
            RETURN DISTINCT id(n) as id, n.name_embedding as embedding
            """
        )
        # 第一阶段只返回内部 id 与 embedding，减少回表前的数据量。
        resp, header, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

        # 有候选后再计算 cosine，并筛出超过 min_score 的社区。
        if len(resp) > 0:
            # Calculate Cosine similarity then return the edge ids
            input_ids = []
            for r in resp:
                if r['embedding']:
                    score = calculate_cosine_similarity(
                        search_vector, list(map(float, r['embedding'].split(',')))
                    )
                    if score > min_score:
                        input_ids.append({'id': r['id'], 'score': score})

            # 第二阶段按内部 id 取回完整社区字段。
            # Match the edge ides and return the values
            query = """
                    UNWIND $ids as i
                    MATCH (comm:Community)
                    WHERE id(comm)=i.id
                    RETURN
                        comm.uuid As uuid,
                        comm.group_id AS group_id,
                        comm.name AS name,
                        comm.created_at AS created_at,
                        comm.summary AS summary,
                        comm.name_embedding AS name_embedding
                    ORDER BY i.score DESC
                    LIMIT $limit
                """
            records, header, _ = await driver.execute_query(
                query,
                ids=input_ids,
                search_vector=search_vector,
                limit=limit,
                min_score=min_score,
                routing_='r',
                **query_params,
            )
        else:
            return []
    # 非 Neptune 分支直接在数据库中计算向量相似度。
    else:
        # Kuzu 仍然需要显式 CAST 查询向量为固定长度 FLOAT 数组。
        search_vector_var = '$search_vector'
        if driver.provider == GraphProvider.KUZU:
            search_vector_var = f'CAST($search_vector AS FLOAT[{len(search_vector)}])'

        query = (
            """
                                                                                                                                    MATCH (c:Community)
                                                                                                                                    """
            + group_filter_query
            + """
            WITH c,
            """
            + get_vector_cosine_func_query('c.name_embedding', search_vector_var, driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            ORDER BY score DESC
            LIMIT $limit
            """
        )

        # 执行社区向量查询。
        records, _, _ = await driver.execute_query(
            query,
            search_vector=search_vector,
            limit=limit,
            min_score=min_score,
            routing_='r',
            **query_params,
        )

    # 所有分支都转换成 CommunityNode 列表返回。
    communities = [get_community_node_from_record(record) for record in records]

    return communities


# 混合节点搜索：把多路全文召回和多路向量召回并发执行，再用 RRF 合并排序。
async def hybrid_node_search(
    queries: list[str],
    embeddings: list[list[float]],
    driver: GraphDriver,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    """
    Perform a hybrid search for nodes using both text queries and embeddings.

    This method combines fulltext search and vector similarity search to find
    relevant nodes in the graph database. It uses a rrf reranker.

    Parameters
    ----------
    queries : list[str]
        A list of text queries to search for.
    embeddings : list[list[float]]
        A list of embedding vectors corresponding to the queries. If empty only fulltext search is performed.
    driver : GraphDriver
        The Neo4j driver instance for database operations.
    group_ids : list[str] | None, optional
        The list of group ids to retrieve nodes from.
    limit : int | None, optional
        The maximum number of results to return per search method. If None, a default limit will be applied.

    Returns
    -------
    list[EntityNode]
        A list of unique EntityNode objects that match the search criteria.

    Notes
    -----
    This method performs the following steps:
    1. Executes fulltext searches for each query.
    2. Executes vector similarity searches for each embedding.
    3. Combines and deduplicates the results from both search types.
    4. Logs the performance metrics of the search operation.

    The search results are deduplicated based on the node UUIDs to ensure
    uniqueness in the returned list. The 'limit' parameter is applied to each
    individual search method before deduplication. If not specified, a default
    limit (defined in the individual search functions) will be used.
    """

    # 从这里开始计时，便于观察混合搜索整体耗时。
    start = time()
    # 全文搜索和向量搜索通过 semaphore_gather 并发执行；每一路先取 2 * limit，为后续合并排序留出候选空间。
    results: list[list[EntityNode]] = list(
        await semaphore_gather(
            *[
                node_fulltext_search(driver, q, search_filter, group_ids, 2 * limit)
                for q in queries
            ],
            *[
                node_similarity_search(driver, e, search_filter, group_ids, 2 * limit)
                for e in embeddings
            ],
        )
    )

    # 把所有候选按 uuid 去重成字典，同一 uuid 多次出现时保留最后一次对象引用。
    node_uuid_map: dict[str, EntityNode] = {
        node.uuid: node for result in results for node in result
    }
    # RRF 只需要每一路结果的 uuid 排名，所以先把对象列表压缩成 uuid 列表。
    result_uuids = [[node.uuid for node in result] for result in results]

    # 用 RRF 合并多路排序，兼顾不同查询方式的一致命中。
    ranked_uuids, _ = rrf(result_uuids)

    # 按 RRF 输出的 uuid 顺序取回 EntityNode 对象，形成最终排序。
    relevant_nodes: list[EntityNode] = [node_uuid_map[uuid] for uuid in ranked_uuids]

    # 记录结束时间并写 debug 日志，方便定位搜索慢点。
    end = time()
    logger.debug(f'Found relevant nodes: {ranked_uuids} in {(end - start) * 1000} ms')
    return relevant_nodes


# 批量查找相关节点：对每个输入节点同时找名称相似和全文相关的实体，并保持每个输入节点对应一组结果。
async def get_relevant_nodes(
    driver: GraphDriver,
    nodes: list[EntityNode],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityNode]]:
    # 空输入直接返回空列表，避免后续访问 nodes[0]。
    if len(nodes) == 0:
        return []

    # 该函数假设输入节点属于同一 group，因此用第一个节点的 group_id 作为批量查询范围。
    group_id = nodes[0].group_id
    # 为每个输入节点构造查询载荷：保留 uuid、名称、embedding，并提前生成名称的全文查询。
    query_nodes = [
        {
            'uuid': node.uuid,
            'name': node.name,
            'name_embedding': node.name_embedding,
            'fulltext_query': fulltext_query(node.name, [node.group_id], driver),
        }
        for node in nodes
    ]

    # 生成节点级过滤条件，稍后会拼入批量查询中。
    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # 批量查询的 filter_query 要紧跟 MATCH，因此这里不带前导空格以外的额外语义。
    filter_query = ''
    if filter_queries:
        filter_query = 'WHERE ' + (' AND '.join(filter_queries))

    # Kuzu 分支先确认 embedding 维度，因为 CAST 需要固定长度。
    if driver.provider == GraphProvider.KUZU:
        embedding_size = len(nodes[0].name_embedding) if nodes[0].name_embedding is not None else 0
        # 如果输入节点没有 embedding，就无法做向量匹配，直接返回空结果。
        if embedding_size == 0:
            return []

        # 保留原来的 FIXME：这里说明 Kuzu 对变量作为全文搜索输入的限制会影响该函数。
        # FIXME: Kuzu currently does not support using variables such as `node.fulltext_query` as an input to FTS, which means `get_relevant_nodes()` won't work with Kuzu as the graph driver.
        # Kuzu 查询先做向量相似召回，再尝试拼接全文召回，并把两类结果合并去重。
        query = (
            """
                                                                                                                                    UNWIND $nodes AS node
                                                                                                                                    MATCH (n:Entity {group_id: $group_id})
                                                                                                                                    """
            + filter_query
            + """
            WITH node, n, """
            + get_vector_cosine_func_query(
                'n.name_embedding',
                f'CAST(node.name_embedding AS FLOAT[{embedding_size}])',
                driver.provider,
            )
            + """ AS score
            WHERE score > $min_score
            WITH node, collect(n)[:$limit] AS top_vector_nodes, collect(n.uuid) AS vector_node_uuids
            """
            + get_nodes_query(
                'node_name_and_summary',
                'node.fulltext_query',
                limit=limit,
                provider=driver.provider,
            )
            + """
            WITH node AS m
            WHERE m.group_id = $group_id AND NOT m.uuid IN vector_node_uuids
            WITH node, top_vector_nodes, collect(m) AS fulltext_nodes

            WITH node, list_concat(top_vector_nodes, fulltext_nodes) AS combined_nodes

            UNWIND combined_nodes AS x
            WITH node, collect(DISTINCT {
                uuid: x.uuid,
                name: x.name,
                name_embedding: x.name_embedding,
                group_id: x.group_id,
                created_at: x.created_at,
                summary: x.summary,
                labels: x.labels,
                attributes: x.attributes
            }) AS matches

            RETURN
            node.uuid AS search_node_uuid, matches
            """
        )
    # 非 Kuzu 查询结构类似，但可以直接把 node.fulltext_query 传给全文索引查询。
    else:
        query = (
            """
                                                                                                                                    UNWIND $nodes AS node
                                                                                                                                    MATCH (n:Entity {group_id: $group_id})
                                                                                                                                    """
            + filter_query
            + """
            WITH node, n, """
            + get_vector_cosine_func_query(
                'n.name_embedding', 'node.name_embedding', driver.provider
            )
            + """ AS score
            WHERE score > $min_score
            WITH node, collect(n)[..$limit] AS top_vector_nodes, collect(n.uuid) AS vector_node_uuids
            """
            + get_nodes_query(
                'node_name_and_summary',
                'node.fulltext_query',
                limit=limit,
                provider=driver.provider,
            )
            + """
            YIELD node AS m
            WHERE m.group_id = $group_id
            WITH node, top_vector_nodes, vector_node_uuids, collect(m) AS fulltext_nodes

            WITH node,
                top_vector_nodes,
                [m IN fulltext_nodes WHERE NOT m.uuid IN vector_node_uuids] AS filtered_fulltext_nodes

            WITH node, top_vector_nodes + filtered_fulltext_nodes AS combined_nodes

            UNWIND combined_nodes AS combined_node
            WITH node, collect(DISTINCT combined_node) AS deduped_nodes

            RETURN
            node.uuid AS search_node_uuid,
            [x IN deduped_nodes | {
                uuid: x.uuid,
                name: x.name,
                name_embedding: x.name_embedding,
                group_id: x.group_id,
                created_at: x.created_at,
                summary: x.summary,
                labels: labels(x),
                attributes: properties(x)
            }] AS matches
            """
        )

    # 一次性执行批量查询，nodes 参数包含每个输入节点的向量和全文查询。
    results, _, _ = await driver.execute_query(
        query,
        nodes=query_nodes,
        group_id=group_id,
        limit=limit,
        min_score=min_score,
        routing_='r',
        **filter_params,
    )

    # 把结果组织成 search_node_uuid -> 匹配节点列表，便于恢复到输入节点的顺序。
    relevant_nodes_dict: dict[str, list[EntityNode]] = {
        result['search_node_uuid']: [
            get_entity_node_from_record(record, driver.provider) for record in result['matches']
        ]
        for result in results
    }

    # 按原始 nodes 顺序取回每个节点的相关节点列表；没有命中时返回空列表。
    relevant_nodes = [relevant_nodes_dict.get(node.uuid, []) for node in nodes]

    return relevant_nodes


# 批量查找相关边：对输入边的 fact_embedding 与数据库中的同端点关系进行相似度匹配。
async def get_relevant_edges(
    driver: GraphDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityEdge]]:
    # 空输入直接返回空列表，避免访问 edges[0] 或构造无意义查询。
    if len(edges) == 0:
        return []

    # 生成边级过滤条件，确保候选边满足调用方指定的 SearchFilters。
    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # 这里的 filter_query 用 WHERE 开头，适配后续 MATCH 后追加条件的查询结构。
    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    # Neptune 分支先找到与输入边端点一致、group 一致的候选关系，并取出 embedding。
    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            """
                                                                                                                                    UNWIND $edges AS edge
                                                                                                                                    MATCH (n:Entity {uuid: edge.source_node_uuid})-[e:RELATES_TO {group_id: edge.group_id}]-(m:Entity {uuid: edge.target_node_uuid})
                                                                                                                                    """
            + filter_query
            + """
            WITH e, edge
            RETURN DISTINCT id(e) as id, e.fact_embedding as source_embedding, edge.uuid as search_edge_uuid,
            edge.fact_embedding as target_embedding
            """
        )
        # 第一阶段返回候选边内部 id、候选 embedding 和输入边 uuid 的对应关系。
        resp, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

        # Neptune 的 embedding 以字符串形式存储，需要在 Python 中转换并计算 cosine。
        # Calculate Cosine similarity then return the edge ids
        input_ids = []
        for r in resp:
            score = calculate_cosine_similarity(
                list(map(float, r['source_embedding'].split(','))), r['target_embedding']
            )
            # 只把超过阈值的候选送入回表查询。
            if score > min_score:
                input_ids.append({'id': r['id'], 'score': score, 'uuid': r['search_edge_uuid']})

        # 回表查询按分数排序，并按每条输入边收集匹配候选。
        # Match the edge ides and return the values
        query = """
        UNWIND $ids AS edge
        MATCH ()-[e]->()
        WHERE id(e) = edge.id
        WITH edge, e
        ORDER BY edge.score DESC
        RETURN edge.uuid AS search_edge_uuid,
            collect({
                uuid: e.uuid,
                source_node_uuid: startNode(e).uuid,
                target_node_uuid: endNode(e).uuid,
                created_at: e.created_at,
                name: e.name,
                group_id: e.group_id,
                fact: e.fact,
                fact_embedding: [x IN split(e.fact_embedding, ",") | toFloat(x)],
                episodes: split(e.episodes, ","),
                expired_at: e.expired_at,
                valid_at: e.valid_at,
                invalid_at: e.invalid_at,
                attributes: properties(e)
            })[..$limit] AS matches
                """

        # 执行 Neptune 回表查询，ids 中同时携带分数和原输入边 uuid。
        results, _, _ = await driver.execute_query(
            query,
            ids=input_ids,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )
    else:
        # 非 Neptune 下，Kuzu 仍需先处理关系节点模型和向量维度。
        if driver.provider == GraphProvider.KUZU:
            embedding_size = (
                len(edges[0].fact_embedding) if edges[0].fact_embedding is not None else 0
            )
            # 没有输入边 embedding 时无法比较相似度，直接返回空。
            if embedding_size == 0:
                return []

            # Kuzu 查询匹配同端点关系节点，再计算 fact_embedding 相似度。
            query = (
                """
                                                                                                                                        UNWIND $edges AS edge
                                                                                                                                        MATCH (n:Entity {uuid: edge.source_node_uuid})-[:RELATES_TO]-(e:RelatesToNode_ {group_id: edge.group_id})-[:RELATES_TO]-(m:Entity {uuid: edge.target_node_uuid})
                                                                                                                                        """
                + filter_query
                + """
                WITH e, edge, n, m, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding',
                    f'CAST(edge.fact_embedding AS FLOAT[{embedding_size}])',
                    driver.provider,
                )
                + """ AS score
                WHERE score > $min_score
                WITH e, edge, n, m, score
                ORDER BY score DESC
                LIMIT $limit
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: n.uuid,
                        target_node_uuid: m.uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: e.attributes
                    }) AS matches
                """
            )
        # 其他后端直接匹配同端点 RELATES_TO 边，并用数据库向量函数打分。
        else:
            query = (
                """
                                                                                                                                        UNWIND $edges AS edge
                                                                                                                                        MATCH (n:Entity {uuid: edge.source_node_uuid})-[e:RELATES_TO {group_id: edge.group_id}]-(m:Entity {uuid: edge.target_node_uuid})
                                                                                                                                        """
                + filter_query
                + """
                WITH e, edge, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding', 'edge.fact_embedding', driver.provider
                )
                + """ AS score
                WHERE score > $min_score
                WITH edge, e, score
                ORDER BY score DESC
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: startNode(e).uuid,
                        target_node_uuid: endNode(e).uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: properties(e)
                    })[..$limit] AS matches
                """
            )

        # 执行非 Neptune 查询。
        results, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

    # 把批量查询结果组织为输入边 uuid 到相关边列表的字典。
    relevant_edges_dict: dict[str, list[EntityEdge]] = {
        result['search_edge_uuid']: [
            get_entity_edge_from_record(record, driver.provider) for record in result['matches']
        ]
        for result in results
    }

    # 恢复输入边顺序，每条边对应一个候选列表。
    relevant_edges = [relevant_edges_dict.get(edge.uuid, []) for edge in edges]

    return relevant_edges


# 边失效候选搜索：在同一 group 且端点相关的关系中找语义相似边，用于判断新事实是否可能覆盖旧事实。
async def get_edge_invalidation_candidates(
    driver: GraphDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityEdge]]:
    # 没有新边时无需寻找失效候选。
    if len(edges) == 0:
        return []

    # 失效候选同样受边过滤条件控制。
    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    # 这里的 filter_query 以 AND 开头，因为主查询已经自带 WHERE 端点约束。
    filter_query = ''
    if filter_queries:
        filter_query = ' AND ' + (' AND '.join(filter_queries))

    # Neptune 先找同 group 且任一端点重合的边，作为可能被新事实影响的候选。
    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            """
                                                                                                                                    UNWIND $edges AS edge
                                                                                                                                    MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)
                                                                                                                                    WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]
                                                                                                                                    """
            + filter_query
            + """
            WITH e, edge
            RETURN DISTINCT id(e) as id, e.fact_embedding as source_embedding,
            edge.fact_embedding as target_embedding,
            edge.uuid as search_edge_uuid
            """
        )
        # 第一阶段只取候选 id、候选 embedding、输入 embedding 和输入边 uuid。
        resp, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

        # 在 Python 中计算候选边与输入边的 fact_embedding 相似度。
        # Calculate Cosine similarity then return the edge ids
        input_ids = []
        for r in resp:
            score = calculate_cosine_similarity(
                list(map(float, r['source_embedding'].split(','))), r['target_embedding']
            )
            # 超过阈值才认为是可能的失效候选。
            if score > min_score:
                input_ids.append({'id': r['id'], 'score': score, 'uuid': r['search_edge_uuid']})

        # 回表取完整边字段，并按相似度排序后聚合到对应输入边下。
        # Match the edge ides and return the values
        query = """
        UNWIND $ids AS edge
        MATCH ()-[e]->()
        WHERE id(e) = edge.id
        WITH edge, e
        ORDER BY edge.score DESC
        RETURN edge.uuid AS search_edge_uuid,
            collect({
                uuid: e.uuid,
                source_node_uuid: startNode(e).uuid,
                target_node_uuid: endNode(e).uuid,
                created_at: e.created_at,
                name: e.name,
                group_id: e.group_id,
                fact: e.fact,
                fact_embedding: [x IN split(e.fact_embedding, ",") | toFloat(x)],
                episodes: split(e.episodes, ","),
                expired_at: e.expired_at,
                valid_at: e.valid_at,
                invalid_at: e.invalid_at,
                attributes: properties(e)
            })[..$limit] AS matches
                """
        # 执行 Neptune 回表查询。
        results, _, _ = await driver.execute_query(
            query,
            ids=input_ids,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )
    else:
        # Kuzu 分支仍需使用 RelatesToNode_ 中间节点，并提前确定 embedding 维度。
        if driver.provider == GraphProvider.KUZU:
            embedding_size = (
                len(edges[0].fact_embedding) if edges[0].fact_embedding is not None else 0
            )
            # 没有 embedding 就无法判断语义冲突或覆盖，直接返回空。
            if embedding_size == 0:
                return []

            # Kuzu 查询把端点任一重合的关系节点作为候选，并计算向量分数。
            query = (
                """
                                                                                                                                        UNWIND $edges AS edge
                                                                                                                                        MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {group_id: edge.group_id})-[:RELATES_TO]->(m:Entity)
                                                                                                                                        WHERE (n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid])
                                                                                                                                        """
                + filter_query
                + """
                WITH edge, e, n, m, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding',
                    f'CAST(edge.fact_embedding AS FLOAT[{embedding_size}])',
                    driver.provider,
                )
                + """ AS score
                WHERE score > $min_score
                WITH edge, e, n, m, score
                ORDER BY score DESC
                LIMIT $limit
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: n.uuid,
                        target_node_uuid: m.uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: e.attributes
                    }) AS matches
                """
            )
        # 其他后端直接匹配端点重合的 RELATES_TO 边。
        else:
            query = (
                """
                                                                                                                                        UNWIND $edges AS edge
                                                                                                                                        MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)
                                                                                                                                        WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]
                                                                                                                                        """
                + filter_query
                + """
                WITH edge, e, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding', 'edge.fact_embedding', driver.provider
                )
                + """ AS score
                WHERE score > $min_score
                WITH edge, e, score
                ORDER BY score DESC
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: startNode(e).uuid,
                        target_node_uuid: endNode(e).uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: properties(e)
                    })[..$limit] AS matches
                """
            )

        # 执行非 Neptune 失效候选查询。
        results, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )
    # 把查询结果按输入边 uuid 分组并转换为 EntityEdge。
    invalidation_edges_dict: dict[str, list[EntityEdge]] = {
        result['search_edge_uuid']: [
            get_entity_edge_from_record(record, driver.provider) for record in result['matches']
        ]
        for result in results
    }

    # 按输入边顺序返回每条边的失效候选列表。
    invalidation_edges = [invalidation_edges_dict.get(edge.uuid, []) for edge in edges]

    return invalidation_edges


# takes in a list of rankings of uuids
# RRF 重排器：把多个已排序 uuid 列表合并成一个综合排序，越靠前的候选获得越高的倒数排名分。
def rrf(
    results: list[list[str]], rank_const=1, min_score: float = 0
) -> tuple[list[str], list[float]]:
    # 用 defaultdict 累加每个 uuid 的综合分数，未出现过的候选默认从 0 开始。
    scores: dict[str, float] = defaultdict(float)
    # 遍历每一路排序结果，位置越靠前贡献越大。
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)

    # 把分数字典转成列表后按分数降序排列。
    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])

    sorted_uuids = [term[0] for term in scored_uuids]

    # 返回 uuid 列表和对应分数，并应用 min_score 过滤。
    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
        scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
    ]


# 基于中心节点距离的重排：优先返回与中心节点直接相连的候选，并把中心节点本身放回列表开头。
async def node_distance_reranker(
    driver: GraphDriver,
    node_uuids: list[str],
    center_node_uuid: str,
    min_score: float = 0,
) -> tuple[list[str], list[float]]:
    # 如果后端有专用距离重排实现，优先使用。
    if driver.search_interface:
        try:
            return await driver.search_interface.node_distance_reranker(
                driver, node_uuids, center_node_uuid, min_score
            )
        except NotImplementedError:
            pass

    # 中心节点需要参与最终排序，但不能参与“到自身距离”的查询，所以先从候选中剔除。
    # filter out node_uuid center node node uuid
    filtered_uuids = list(filter(lambda node_uuid: node_uuid != center_node_uuid, node_uuids))
    scores: dict[str, float] = {center_node_uuid: 0.0}

    # 默认查询只给与中心节点直接相连的候选打分 1。
    query = """
    UNWIND $node_uuids AS node_uuid
    MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
    RETURN 1 AS score, node_uuid AS uuid
    """
    # Kuzu 的直接相连需要经过关系中间节点，因此匹配路径不同。
    if driver.provider == GraphProvider.KUZU:
        query = """
        UNWIND $node_uuids AS node_uuid
        MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(n:Entity {uuid: node_uuid})
        RETURN 1 AS score, node_uuid AS uuid
        """

    # 执行距离查询后，FalkorDB 的返回格式需要手动按 header 转成 dict。
    # Find the shortest path to center node
    results, header, _ = await driver.execute_query(
        query,
        node_uuids=filtered_uuids,
        center_uuid=center_node_uuid,
        routing_='r',
    )
    if driver.provider == GraphProvider.FALKORDB:
        results = [dict(zip(header, row, strict=True)) for row in results]

    # 把查询到的候选距离分数写入 scores。
    for result in results:
        uuid = result['uuid']
        score = result['score']
        scores[uuid] = score

    # 没有直接关系的候选记为无穷远，后续倒数分数会趋近 0。
    for uuid in filtered_uuids:
        if uuid not in scores:
            scores[uuid] = float('inf')

    # 按距离从近到远排序。
    # rerank on shortest distance
    filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    # 如果中心节点原本在候选中，就给它一个较高倒数分数并放在最前。
    # add back in filtered center uuid if it was filtered out
    if center_node_uuid in node_uuids:
        scores[center_node_uuid] = 0.1
        filtered_uuids = [center_node_uuid] + filtered_uuids

    # 最终返回距离倒数分数；距离越近，1 / score 越大。
    return [uuid for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score], [
        1 / scores[uuid] for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score
    ]


# 基于 episode 提及次数的重排：先用 RRF 合并候选，再统计这些实体被 episode 提及的次数作为排序依据。
async def episode_mentions_reranker(
    driver: GraphDriver, node_uuids: list[list[str]], min_score: float = 0
) -> tuple[list[str], list[float]]:
    # 有专用 episode mentions 重排器时先委托。
    if driver.search_interface:
        try:
            return await driver.search_interface.episode_mentions_reranker(
                driver, node_uuids, min_score
            )
        except NotImplementedError:
            pass

    # 多路候选先经过 RRF 合并，得到一个初始 uuid 顺序。
    # use rrf as a preliminary ranker
    sorted_uuids, _ = rrf(node_uuids)
    scores: dict[str, float] = {}

    # 随后统计每个实体被 episode 提及的次数。
    # Find the shortest path to center node
    results, _, _ = await driver.execute_query(
        """
        UNWIND $node_uuids AS node_uuid
        MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
        RETURN count(*) AS score, n.uuid AS uuid
        """,
        node_uuids=sorted_uuids,
        routing_='r',
    )

    # 把提及次数写入 scores。
    for result in results:
        scores[result['uuid']] = result['score']

    # 没有提及记录的实体按当前实现记为无穷大，排序时会落在后续逻辑定义的位置。
    for uuid in sorted_uuids:
        if uuid not in scores:
            scores[uuid] = float('inf')

    # 按 scores 排序后返回，保留原实现的排序方向和阈值逻辑。
    # rerank on shortest distance
    sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
        scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
    ]


# MMR 重排：在相关性和多样性之间做折中，避免返回一批彼此过于相似的候选。
def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: dict[str, list[float]],
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    min_score: float = -2.0,
) -> tuple[list[str], list[float]]:
    # 记录 MMR 计算开始时间，用于 debug 性能日志。
    start = time()
    # 查询向量转为 NumPy 数组，便于后续点积计算。
    query_array = np.array(query_vector)
    # 候选向量先做 L2 归一化，保证相似度比较在同一尺度上。
    candidate_arrays: dict[str, NDArray] = {}
    for uuid, embedding in candidates.items():
        candidate_arrays[uuid] = normalize_l2(embedding)

    # 保留候选 uuid 顺序，用矩阵下标表示两两相似度。
    uuids: list[str] = list(candidate_arrays.keys())

    # 初始化候选之间的相似度矩阵，用于计算多样性惩罚项。
    similarity_matrix = np.zeros((len(uuids), len(uuids)))

    # 只计算矩阵下三角，再镜像到上三角，避免重复计算。
    for i, uuid_1 in enumerate(uuids):
        for j, uuid_2 in enumerate(uuids[:i]):
            u = candidate_arrays[uuid_1]
            v = candidate_arrays[uuid_2]
            similarity = np.dot(u, v)

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # 开始为每个候选计算 MMR 分数。
    mmr_scores: dict[str, float] = {}
    for i, uuid in enumerate(uuids):
        # max_sim 表示该候选与已知候选集合中最相似的程度，用来惩罚冗余。
        max_sim = np.max(similarity_matrix[i, :])
        # MMR = 查询相关性权重 - 候选间相似惩罚；lambda 越大越偏向相关性。
        mmr = mmr_lambda * np.dot(query_array, candidate_arrays[uuid]) + (mmr_lambda - 1) * max_sim
        mmr_scores[uuid] = mmr

    # 按 MMR 分数降序排列候选 uuid。
    uuids.sort(reverse=True, key=lambda c: mmr_scores[c])

    # 记录耗时并写入 debug 日志。
    end = time()
    logger.debug(f'Completed MMR reranking in {(end - start) * 1000} ms')

    # 返回超过 min_score 的候选及其 MMR 分数。
    return [uuid for uuid in uuids if mmr_scores[uuid] >= min_score], [
        mmr_scores[uuid] for uuid in uuids if mmr_scores[uuid] >= min_score
    ]


# 批量加载实体节点 embedding：为后续相似度计算准备 uuid 到 name_embedding 的映射。
async def get_embeddings_for_nodes(
    driver: GraphDriver, nodes: list[EntityNode]
) -> dict[str, list[float]]:
    # 节点 embedding 加载优先使用 graph_operations_interface 的批量接口。
    if driver.graph_operations_interface:
        return await driver.graph_operations_interface.node_load_embeddings_bulk(driver, nodes)
    # Neptune 中 embedding 以逗号分隔字符串存储，因此查询时先 split。
    elif driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (n:Entity)
        WHERE n.uuid IN $node_uuids
        RETURN DISTINCT
            n.uuid AS uuid,
            split(n.name_embedding, ",") AS name_embedding
        """
    # 其他后端通常直接把 name_embedding 作为数组字段返回。
    else:
        query = """
        MATCH (n:Entity)
        WHERE n.uuid IN $node_uuids
        RETURN DISTINCT
            n.uuid AS uuid,
            n.name_embedding AS name_embedding
        """
    # 按输入节点 uuid 批量查询 embedding。
    results, _, _ = await driver.execute_query(
        query,
        node_uuids=[node.uuid for node in nodes],
        routing_='r',
    )

    # 只把 uuid 和 embedding 都存在的记录写入映射，避免下游拿到空向量。
    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('name_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    # 返回 uuid -> embedding 的字典。
    return embeddings_dict


# 批量加载社区 embedding：与节点 embedding 加载逻辑类似，只是目标标签换成 Community。
async def get_embeddings_for_communities(
    driver: GraphDriver, communities: list[CommunityNode]
) -> dict[str, list[float]]:
    # 社区 embedding 加载优先走 search_interface，因为社区搜索相关能力通常放在搜索层接口中。
    if driver.search_interface:
        try:
            return await driver.search_interface.get_embeddings_for_communities(driver, communities)
        except NotImplementedError:
            pass

    # Neptune 的社区 embedding 同样需要从字符串拆分。
    if driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (c:Community)
        WHERE c.uuid IN $community_uuids
        RETURN DISTINCT
            c.uuid AS uuid,
            split(c.name_embedding, ",") AS name_embedding
        """
    # 默认后端直接读取 name_embedding 字段。
    else:
        query = """
        MATCH (c:Community)
        WHERE c.uuid IN $community_uuids
        RETURN DISTINCT
            c.uuid AS uuid,
            c.name_embedding AS name_embedding
        """
    # 按社区 uuid 批量查询。
    results, _, _ = await driver.execute_query(
        query,
        community_uuids=[community.uuid for community in communities],
        routing_='r',
    )

    # 构造 uuid 到 embedding 的映射，跳过字段缺失的记录。
    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('name_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict


# 批量加载边 embedding：根据边 uuid 批量取回 fact_embedding，供边相似度或失效判断使用。
async def get_embeddings_for_edges(
    driver: GraphDriver, edges: list[EntityEdge]
) -> dict[str, list[float]]:
    # 边 embedding 加载优先使用 graph_operations_interface 的批量边加载能力。
    if driver.graph_operations_interface:
        return await driver.graph_operations_interface.edge_load_embeddings_bulk(driver, edges)
    # Neptune 分支直接匹配 RELATES_TO 边，并把字符串 fact_embedding 拆分。
    elif driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
        WHERE e.uuid IN $edge_uuids
        RETURN DISTINCT
            e.uuid AS uuid,
            split(e.fact_embedding, ",") AS fact_embedding
        """
    # 其他后端先确定关系匹配模式。
    else:
        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
        """
        # Kuzu 仍然通过中间关系节点读取事实 embedding。
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity)-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(m:Entity)
            """

        # 拼接通用的 edge_uuid 过滤和 fact_embedding 返回字段。
        query = (
            match_query
            + """
        WHERE e.uuid IN $edge_uuids
        RETURN DISTINCT
            e.uuid AS uuid,
            e.fact_embedding AS fact_embedding
        """
        )
    # 执行批量边 embedding 查询。
    results, _, _ = await driver.execute_query(
        query,
        edge_uuids=[edge.uuid for edge in edges],
        routing_='r',
    )

    # 构造 uuid -> fact_embedding 映射，只保留完整记录。
    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('fact_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict
