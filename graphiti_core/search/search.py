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

# 这个模块是搜索编排层：它不直接拼数据库查询，而是把查询请求拆成边、节点、情节、社区四个搜索域，再把具体检索和重排交给 search_utils 中的底层函数。
import logging
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from time import time
from typing import Any

# 下面这些导入可以按职责分成几类：外部服务客户端、图数据库驱动、领域对象模型、搜索配置、搜索过滤器和搜索工具函数。
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.errors import SearchRerankerError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather, validate_group_ids
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
# search_config 里的枚举和配置对象决定“搜什么、怎么搜、怎么重排、返回多少”；本文件的大量分支都围绕这些配置展开。
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters
# search_utils 提供真正执行检索和重排的原子能力，本文件主要负责组合这些能力，而不是重复实现 BM25、向量检索、BFS 或 MMR。
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_fulltext_search,
    episode_mentions_reranker,
    get_embeddings_for_communities,
    get_embeddings_for_edges,
    get_embeddings_for_nodes,
    maximal_marginal_relevance,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
    rrf,
)
from graphiti_core.tracer import NoOpTracer, Tracer

# 模块级 logger 用来记录整体搜索耗时，细粒度观测则交给 Tracer span。
logger = logging.getLogger(__name__)


# 把枚举值转成可序列化、易读的值，主要用于写入 tracing 属性；如果传入的不是枚举，就原样返回。
def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, 'value') else value


# 搜索逻辑允许没有 tracer：统一在这里补成 NoOpTracer，后续代码就不需要反复判断 tracer 是否为空。
def _resolve_tracer(search_tracer: Tracer | None) -> Tracer:
    return search_tracer if search_tracer is not None else NoOpTracer()


# 这个上下文管理器把“开始 span、记录属性、标记成功/失败、记录异常”封装成一个固定模式，避免每个搜索阶段重复写样板代码。
@contextmanager
def _trace_phase(
    search_tracer: Tracer,
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    # 进入某个搜索阶段时先创建 span；name 表示阶段名，attributes 用来携带本阶段关键上下文，例如候选数量、重排方式、limit。
    with search_tracer.start_span(name) as span:
        # 只有调用方真的提供 attributes 时才写入，避免在 tracer 中记录空属性。
        if attributes:
            span.add_attributes(attributes)
        # yield 把 span 暴露给调用方，调用方可以在阶段内部继续补充动态结果，例如返回数量。
        try:
            yield span
            # 代码块正常结束才把 span 标成 ok，这样 tracing 能区分成功路径和异常路径。
            span.set_status('ok')
        # 出现异常时不吞掉错误，而是先把异常写进 span，再重新抛出，保证业务层仍能感知失败。
        except Exception as e:
            span.set_status('error', str(e))
            span.record_exception(e)
            raise


# search 是本模块的总入口：它负责准备共享资源、必要时生成 query embedding，然后并发启动四个搜索域，最后把结果组装成 SearchResults。
async def search(
    clients: GraphitiClients,
    query: str,
    group_ids: list[str] | None,
    config: SearchConfig,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    query_vector: list[float] | None = None,
    driver: GraphDriver | None = None,
) -> SearchResults:
    # 先记录开始时间，函数末尾用它计算端到端搜索耗时。
    start = time()
    # group_ids 会参与过滤查询；进入主流程前先统一校验，避免非法 group_id 进入底层数据库查询。
    validate_group_ids(group_ids)

    # driver 支持从参数覆盖，便于测试或特殊调用；默认使用 clients 中配置好的图数据库驱动。
    driver = driver or clients.driver
    # embedder 只在向量检索或 MMR 需要 query 向量时才会真正使用。
    embedder = clients.embedder
    # cross_encoder 用于更精细但更昂贵的语义重排，只在配置选择 cross_encoder reranker 时进入对应分支。
    cross_encoder = clients.cross_encoder
    # tracer 从 clients 中取；如果 clients 没有 tracer，就转换成 NoOpTracer，让后续 tracing 代码保持一致。
    search_tracer = _resolve_tracer(getattr(clients, 'tracer', None))

    # 空查询没有可检索的文本语义，直接返回空 SearchResults，避免创建无意义 embedding 或发起数据库请求。
    if query.strip() == '':
        return SearchResults()

    # 下面这个条件是在判断“后续是否真的需要 query 向量”：只要任一搜索域启用了向量相似度，或任一域选择 MMR，就必须准备 search_vector。
    if (
        (
            config.edge_config
            and EdgeSearchMethod.cosine_similarity in config.edge_config.search_methods
        )
        or (config.edge_config and EdgeReranker.mmr == config.edge_config.reranker)
        or (
            config.node_config
            and NodeSearchMethod.cosine_similarity in config.node_config.search_methods
        )
        or (config.node_config and NodeReranker.mmr == config.node_config.reranker)
        or (
            config.community_config
            and CommunitySearchMethod.cosine_similarity in config.community_config.search_methods
        )
        or (config.community_config and CommunityReranker.mmr == config.community_config.reranker)
    ):
        # embedding 生成单独放在 trace 阶段中，便于观察 query 长度、是否复用了外部传入向量，以及最终向量维度。
        with _trace_phase(
            search_tracer,
            'search.embed_query_vector',
            {
                'query.length': len(query),
                'query_vector.provided': query_vector is not None,
            },
        ) as span:
            # 优先使用调用方传入的 query_vector；没有传入时才调用 embedder 生成，避免重复计算 embedding；换行会被替换为空格以减少格式噪声。
            search_vector = (
                query_vector
                if query_vector is not None
                else await embedder.create(input_data=[query.replace('\n', ' ')])
            )
            span.add_attributes({'query_vector.dimension': len(search_vector)})
    # 如果所有配置都不依赖 query 向量，仍提供一个固定维度的零向量，保证后续函数签名统一。
    else:
        search_vector = [0.0] * EMBEDDING_DIM

    # 保留原注释：这里把空列表或只包含空字符串的 group_ids 视为不限制 group。
    # if group_ids is empty, set it to None
    group_ids = group_ids if group_ids and group_ids != [''] else None
    # 四个搜索域会被放到同一个 execute_scopes span 中，这个 span 记录本次搜索打开了哪些范围和全局 limit。
    with _trace_phase(
        search_tracer,
        'search.execute_scopes',
        {
            'group_id.count': len(group_ids or []),
            'scope.edges': config.edge_config is not None,
            'scope.nodes': config.node_config is not None,
            'scope.episodes': config.episode_config is not None,
            'scope.communities': config.community_config is not None,
            'limit': config.limit,
        },
    ) as span:
        # semaphore_gather 会并发执行边、节点、情节、社区四类搜索；返回值按同样顺序解包，每个搜索域都返回“结果列表 + 对应重排分数列表”。
        (
            (edges, edge_reranker_scores),
            (nodes, node_reranker_scores),
            (episodes, episode_reranker_scores),
            (communities, community_reranker_scores),
        ) = await semaphore_gather(
            edge_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.edge_config,
                search_filter,
                center_node_uuid,
                bfs_origin_node_uuids,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            node_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.node_config,
                search_filter,
                center_node_uuid,
                bfs_origin_node_uuids,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            episode_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.episode_config,
                search_filter,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
            community_search(
                driver,
                cross_encoder,
                query,
                search_vector,
                group_ids,
                config.community_config,
                config.limit,
                config.reranker_min_score,
                search_tracer,
            ),
        )
        # 四个搜索域完成后，把结果数量写回 span，用于后续排查某个搜索域是否候选过少或完全为空。
        span.add_attributes(
            {
                'result.edges': len(edges),
                'result.nodes': len(nodes),
                'result.episodes': len(episodes),
                'result.communities': len(communities),
            }
        )

    # 把各搜索域的结果和重排分数打包成统一的 SearchResults，调用方不用关心内部执行了多少种检索方法。
    results = SearchResults(
        edges=edges,
        edge_reranker_scores=edge_reranker_scores,
        nodes=nodes,
        node_reranker_scores=node_reranker_scores,
        episodes=episodes,
        episode_reranker_scores=episode_reranker_scores,
        communities=communities,
        community_reranker_scores=community_reranker_scores,
    )

    # 计算毫秒级耗时，用于 debug 日志；更细粒度的耗时已经由各个 trace span 覆盖。
    latency = (time() - start) * 1000

    # 这里记录的是主入口总耗时，便于和 tracer 的分阶段耗时互相印证。
    logger.debug(f'search returned context in {latency} ms')

    # 返回前不再额外排序不同搜索域之间的结果，因为边、节点、情节、社区是并列的结果槽位。
    return results


# edge_search 处理 EntityEdge：它会根据配置组合 BM25、向量相似度和 BFS，再按指定 reranker 合并候选边。
async def edge_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: EdgeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EntityEdge], list[float]]:
    # 如果调用方没有配置边搜索，边这一栏就明确返回空结果和空分数，主 search 仍可继续处理其他搜索域。
    if config is None:
        return [], []
    # 子搜索函数也使用统一 tracer，避免主 search 和子 search 的可观测性风格不一致。
    search_tracer = _resolve_tracer(search_tracer)

    # edge_search 的外层 span 描述本次边搜索的配置：limit、reranker、启用的方法、BFS 起点和中心节点是否存在。
    with _trace_phase(
        search_tracer,
        'search.edge_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
            'bfs_origin_count': len(bfs_origin_node_uuids or []),
            'center_node_uuid.provided': center_node_uuid is not None,
        },
    ) as span:
        # 保留原注释：下面按配置构建候选生成任务。这里先只组装 coroutine，不立即执行。
        # Build search tasks based on configured search methods
        # search_tasks 是候选召回任务队列；每个元素代表一种召回方法。
        search_tasks = []
        # BM25/fulltext 分支按 fact 或边名称等文本内容召回候选边，适合关键词匹配强的查询。
        if EdgeSearchMethod.bm25 in config.search_methods:
            search_tasks.append(
                edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
            )
        # 向量相似度分支使用 query_vector 与边事实 embedding 做语义匹配，能召回字面不完全相同但语义接近的事实。
        if EdgeSearchMethod.cosine_similarity in config.search_methods:
            search_tasks.append(
                edge_similarity_search(
                    driver,
                    query_vector,
                    None,
                    None,
                    search_filter,
                    group_ids,
                    2 * limit,
                    config.sim_min_score,
                )
            )
        # BFS 分支从给定起点沿图结构扩展，重点利用图邻近性，而不是文本或向量相似度。
        if EdgeSearchMethod.bfs in config.search_methods:
            search_tasks.append(
                edge_bfs_search(
                    driver,
                    bfs_origin_node_uuids,
                    config.bfs_max_depth,
                    search_filter,
                    group_ids,
                    2 * limit,
                )
            )

        # 保留原注释：只有配置中启用的方法才会执行，未启用的方法不会产生额外数据库压力。
        # Execute only the configured search methods
        search_results: list[list[EntityEdge]] = []
        if search_tasks:
            # 把候选生成阶段单独包一层 span，方便区分“召回慢”还是“后续重排慢”。
            with _trace_phase(
                search_tracer,
                'search.edge_search.execute_methods',
                {
                    'method_count': len(search_tasks),
                    'candidate_limit': 2 * limit,
                },
            ) as method_span:
                # semaphore_gather 并发执行多个召回任务；每个任务返回一个列表，整体形成 list[list[EntityEdge]]。
                search_results = list(await semaphore_gather(*search_tasks))
                # 记录结果集数量和非空结果集数量，有助于判断某类召回方法是否长期没有效果。
                method_span.add_attributes(
                    {
                        'result_set_count': len(search_results),
                        'non_empty_result_sets': sum(1 for result in search_results if result),
                    }
                )

        # 如果配置要求 BFS 但调用方没有给起点，就先从已有召回结果里收集 source_node_uuid，作为二次图扩展的起点。
        if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            source_node_uuids = [
                edge.source_node_uuid for result in search_results for edge in result
            ]
            # 这个 expand_bfs 阶段体现了“先用文本/向量找种子，再沿图扩散”的混合搜索逻辑。
            with _trace_phase(
                search_tracer,
                'search.edge_search.expand_bfs',
                {
                    'origin_node_count': len(source_node_uuids),
                    'candidate_limit': 2 * limit,
                },
            ):
                # 二次 BFS 的结果会追加到 search_results 中，后续和其他召回结果一起去重、重排。
                search_results.append(
                    await edge_bfs_search(
                        driver,
                        source_node_uuids,
                        config.bfs_max_depth,
                        search_filter,
                        group_ids,
                        2 * limit,
                    )
                )

        # 把多个召回列表拍平成 uuid 到对象的映射；同一条边被多种方法召回时只保留一个对象，但它在各列表中的排名仍可用于 RRF。
        edge_uuid_map = {edge.uuid: edge for result in search_results for edge in result}

        # reranked_uuids 保存重排后的边 uuid 顺序，edge_scores 保存相同顺序下的重排分数。
        reranked_uuids: list[str] = []
        edge_scores: list[float] = []
        # 重排阶段只关心候选集合和每种召回方法给出的顺序，不再执行新的基础召回。
        with _trace_phase(
            search_tracer,
            'search.edge_search.rerank',
            {
                'candidate_count': len(edge_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            # RRF 和 episode_mentions 都先用各召回列表中的排名做融合；episode_mentions 后面还会按被多少 episode 引用再排一次。
            if (
                config.reranker == EdgeReranker.rrf
                or config.reranker == EdgeReranker.episode_mentions
            ):
                # 把每个召回结果转换成 uuid 排名列表，这是 RRF 所需的输入格式。
                search_result_uuids = [[edge.uuid for edge in result] for result in search_results]

                # RRF 通过多个排名列表投票式融合候选，min_score 用来过滤融合分过低的边。
                reranked_uuids, edge_scores = rrf(search_result_uuids, min_score=reranker_min_score)
            # MMR 需要候选向量：它不仅考虑候选和查询的相似度，也惩罚候选之间的重复性。
            elif config.reranker == EdgeReranker.mmr:
                # 先批量加载候选边的 embedding，避免在 MMR 计算时逐条访问数据库。
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.load_embeddings',
                    {'candidate_count': len(edge_uuid_map)},
                ):
                    search_result_uuids_and_vectors = await get_embeddings_for_edges(
                        driver, list(edge_uuid_map.values())
                    )
                # 拿到候选向量后再计算 MMR；这个阶段是纯重排逻辑，不再改变候选池。
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    reranked_uuids, edge_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            # cross_encoder 分支把候选边的 fact 作为文本对 query 重新打分，通常更准但成本更高，所以只取前 limit 个候选文本。
            elif config.reranker == EdgeReranker.cross_encoder:
                fact_to_uuid_map = {
                    edge.fact: edge.uuid for edge in list(edge_uuid_map.values())[:limit]
                }
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.cross_encoder_rank',
                    {'candidate_count': len(fact_to_uuid_map)},
                ):
                    reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
                # cross encoder 返回的是 fact 文本和分数，借助 fact_to_uuid_map 再映射回边 uuid。
                reranked_uuids = [
                    fact_to_uuid_map[fact]
                    for fact, score in reranked_facts
                    if score >= reranker_min_score
                ]
                edge_scores = [score for _, score in reranked_facts if score >= reranker_min_score]
            # node_distance 重排依赖一个中心节点：它把边按源节点到中心节点的距离来组织，适合“围绕某个实体找相关事实”。
            elif config.reranker == EdgeReranker.node_distance:
                # 没有中心节点时无法计算距离，直接抛出专门的 SearchRerankerError，让调用方知道是配置/参数问题。
                if center_node_uuid is None:
                    raise SearchRerankerError('No center node provided for Node Distance reranker')

                # 在做距离重排前先用 RRF 给候选边一个基础顺序，避免完全丢失多召回方法的初始排序信息。
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    sorted_result_uuids, _ = rrf(
                        [[edge.uuid for edge in result] for result in search_results],
                        min_score=reranker_min_score,
                    )
                # 将排序后的边对象取回，后续按 source_node_uuid 聚合，因为距离重排计算的是节点而不是边本身。
                sorted_results = [edge_uuid_map[uuid] for uuid in sorted_result_uuids]

                # 同一个源节点可能对应多条边；先把源节点映射到边 uuid 列表，节点排好序后再展开回边。
                source_to_edge_uuid_map = defaultdict(list)
                for edge in sorted_results:
                    source_to_edge_uuid_map[edge.source_node_uuid].append(edge.uuid)

                # source_uuids 是实际送入 node_distance_reranker 的节点候选集合。
                source_uuids = [source_node_uuid for source_node_uuid in source_to_edge_uuid_map]

                # node_distance_reranker 根据图上到 center_node_uuid 的距离给源节点排序。
                with _trace_phase(
                    search_tracer,
                    'search.edge_search.node_distance_rank',
                    {
                        'source_node_count': len(source_uuids),
                        'center_node_uuid.provided': center_node_uuid is not None,
                    },
                ):
                    reranked_node_uuids, edge_scores = await node_distance_reranker(
                        driver, source_uuids, center_node_uuid, min_score=reranker_min_score
                    )

                # 把排好序的源节点重新展开为边 uuid，这样边的最终顺序继承了源节点的距离排序。
                for node_uuid in reranked_node_uuids:
                    reranked_uuids.extend(source_to_edge_uuid_map[node_uuid])

        # 根据重排后的 uuid 顺序取回边对象；这里假设所有 reranked_uuids 都来自 edge_uuid_map。
        reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]

        # episode_mentions 作为边重排时，最后按边关联的 episodes 数量降序，让被更多情节支撑的事实更靠前。
        if config.reranker == EdgeReranker.episode_mentions:
            reranked_edges.sort(reverse=True, key=lambda edge: len(edge.episodes))

        # 把候选数、重排后数量和最终返回数量写入外层 span，形成边搜索的完整统计闭环。
        span.add_attributes(
            {
                'candidate_count': len(edge_uuid_map),
                'reranked_count': len(reranked_edges),
                'returned_count': min(len(reranked_edges), limit),
            }
        )

        # 最终只返回 limit 条边及对应的前 limit 个分数；候选阶段用 2 * limit 是为了给重排留余量。
        return reranked_edges[:limit], edge_scores[:limit]


# node_search 处理 EntityNode，整体结构和 edge_search 对称：先多路召回节点，再根据配置选择 RRF、MMR、cross encoder、episode_mentions 或 node_distance。
async def node_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: NodeSearchConfig | None,
    search_filter: SearchFilters,
    center_node_uuid: str | None = None,
    bfs_origin_node_uuids: list[str] | None = None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EntityNode], list[float]]:
    # 没有节点搜索配置时直接返回空结果；这样主 search 可以按配置灵活开启或关闭各搜索域。
    if config is None:
        return [], []
    # 和边搜索一样，节点搜索也会把空 tracer 统一补成 NoOpTracer。
    search_tracer = _resolve_tracer(search_tracer)

    # node_search 外层 span 记录节点搜索的关键配置，用于和边搜索、情节搜索、社区搜索横向对比。
    with _trace_phase(
        search_tracer,
        'search.node_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
            'bfs_origin_count': len(bfs_origin_node_uuids or []),
            'center_node_uuid.provided': center_node_uuid is not None,
        },
    ) as span:
        # 保留原注释：根据节点搜索配置构建候选任务，先组合任务再统一并发执行。
        # Build search tasks based on configured search methods
        # search_tasks 收集启用的节点召回方法，后面一次性并发执行。
        search_tasks = []
        # 节点 BM25 检索通常基于节点名称和 summary，适合精确名称或关键词召回。
        if NodeSearchMethod.bm25 in config.search_methods:
            search_tasks.append(
                node_fulltext_search(driver, query, search_filter, group_ids, 2 * limit)
            )
        # 节点向量检索使用 query_vector 和节点 embedding 计算相似度，补足 BM25 对同义表达不敏感的问题。
        if NodeSearchMethod.cosine_similarity in config.search_methods:
            search_tasks.append(
                node_similarity_search(
                    driver,
                    query_vector,
                    search_filter,
                    group_ids,
                    2 * limit,
                    config.sim_min_score,
                )
            )
        # 节点 BFS 从指定起点沿图找邻近实体，强调结构相关性。
        if NodeSearchMethod.bfs in config.search_methods:
            search_tasks.append(
                node_bfs_search(
                    driver,
                    bfs_origin_node_uuids,
                    search_filter,
                    config.bfs_max_depth,
                    group_ids,
                    2 * limit,
                )
            )

        # 保留原注释：search_tasks 非空时才执行，避免对空任务调用并发工具。
        # Execute only the configured search methods
        search_results: list[list[EntityNode]] = []
        if search_tasks:
            # 节点召回执行阶段与边搜索相同，独立统计候选 limit 和各召回方法返回情况。
            with _trace_phase(
                search_tracer,
                'search.node_search.execute_methods',
                {
                    'method_count': len(search_tasks),
                    'candidate_limit': 2 * limit,
                },
            ) as method_span:
                # 并发结果保持“每个召回方法一个列表”的形态，后续 RRF 需要保留这种来源顺序。
                search_results = list(await semaphore_gather(*search_tasks))
                # 把召回结果集的数量和非空情况写入 span，用于定位具体召回方法是否失效。
                method_span.add_attributes(
                    {
                        'result_set_count': len(search_results),
                        'non_empty_result_sets': sum(1 for result in search_results if result),
                    }
                )

        # 如果启用了 BFS 但调用方没给起点，就把已有召回节点当作图扩展起点，形成“语义/文本召回种子 + 图扩散”的流程。
        if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids is None:
            origin_node_uuids = [node.uuid for result in search_results for node in result]
            with _trace_phase(
                search_tracer,
                'search.node_search.expand_bfs',
                {
                    'origin_node_count': len(origin_node_uuids),
                    'candidate_limit': 2 * limit,
                },
            ):
                # 二次 BFS 的节点结果追加到 search_results，而不是覆盖原始召回结果。
                search_results.append(
                    await node_bfs_search(
                        driver,
                        origin_node_uuids,
                        search_filter,
                        config.bfs_max_depth,
                        group_ids,
                        2 * limit,
                    )
                )

        # search_result_uuids 保留每个召回列表内部的节点顺序，用于 RRF 或 episode_mentions 这类基于排名列表的重排。
        search_result_uuids = [[node.uuid for node in result] for result in search_results]
        # node_uuid_map 用 uuid 去重并保存实体对象；多路召回命中的同一节点只保留一份对象引用。
        node_uuid_map = {node.uuid: node for result in search_results for node in result}

        # 初始化重排输出容器，所有 reranker 最终都要填充 uuid 顺序和分数列表。
        reranked_uuids: list[str] = []
        node_scores: list[float] = []
        # 节点重排阶段根据配置走不同策略，但最终都产出 reranked_uuids 和 node_scores。
        with _trace_phase(
            search_tracer,
            'search.node_search.rerank',
            {
                'candidate_count': len(node_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            # RRF 直接融合多个召回列表的排序，适合兼顾 BM25、向量、BFS 的综合排序。
            if config.reranker == NodeReranker.rrf:
                reranked_uuids, node_scores = rrf(search_result_uuids, min_score=reranker_min_score)
            # MMR 节点重排先加载节点 embedding，再用 query_vector 计算相关性和候选间多样性。
            elif config.reranker == NodeReranker.mmr:
                # 这个 span 覆盖批量加载候选节点向量的耗时。
                with _trace_phase(
                    search_tracer,
                    'search.node_search.load_embeddings',
                    {'candidate_count': len(node_uuid_map)},
                ):
                    # 批量加载候选节点向量，减少数据库往返，也为 MMR 一次性计算做准备。
                    search_result_uuids_and_vectors = await get_embeddings_for_nodes(
                        driver, list(node_uuid_map.values())
                    )

                # 拿到节点向量后才进入 MMR 计算阶段。
                with _trace_phase(
                    search_tracer,
                    'search.node_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    # MMR 输出的是多样化后的 uuid 顺序和分数，可以降低结果中过多相似节点扎堆的问题。
                    reranked_uuids, node_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            # cross_encoder 节点重排使用节点名称和 query 做更细粒度匹配，适合候选数量已经被前序召回压缩后的场景。
            elif config.reranker == NodeReranker.cross_encoder:
                # 先建立 name 到 uuid 的映射，因为 cross_encoder 的输入输出都是文本。
                name_to_uuid_map = {node.name: node.uuid for node in list(node_uuid_map.values())}

                # cross_encoder_rank span 单独记录候选名称数量，方便观察模型精排成本。
                with _trace_phase(
                    search_tracer,
                    'search.node_search.cross_encoder_rank',
                    {'candidate_count': len(name_to_uuid_map)},
                ):
                    reranked_node_names = await cross_encoder.rank(
                        query, list(name_to_uuid_map.keys())
                    )
                # cross_encoder 返回节点名称的排序结果，再通过 name_to_uuid_map 映射回节点 uuid。
                reranked_uuids = [
                    name_to_uuid_map[name]
                    for name, score in reranked_node_names
                    if score >= reranker_min_score
                ]
                node_scores = [
                    score for _, score in reranked_node_names if score >= reranker_min_score
                ]
            # episode_mentions 重排利用情节提及次数作为节点重要性信号：被更多 episodic 记录提到的节点更可能重要。
            elif config.reranker == NodeReranker.episode_mentions:
                # episode_mentions_rank span 包住实际的提及统计和排序过程。
                with _trace_phase(
                    search_tracer,
                    'search.node_search.episode_mentions_rank',
                    {'candidate_count': len(node_uuid_map)},
                ):
                    # 传入的是各召回列表的 uuid 排序，reranker 会结合这些候选再按 episode 提及情况打分。
                    reranked_uuids, node_scores = await episode_mentions_reranker(
                        driver, search_result_uuids, min_score=reranker_min_score
                    )
            # node_distance 节点重排要求 center_node_uuid，用图距离衡量候选节点围绕中心实体的相关程度。
            elif config.reranker == NodeReranker.node_distance:
                # 缺少中心节点时无法进行距离排序，抛出明确错误而不是静默退化成其他排序。
                if center_node_uuid is None:
                    raise SearchRerankerError('No center node provided for Node Distance reranker')
                # 距离重排前先用 RRF 得到候选种子顺序，保留多路召回的初始融合信息。
                with _trace_phase(
                    search_tracer,
                    'search.node_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    seeded_uuids = rrf(search_result_uuids, min_score=reranker_min_score)[0]
                # node_distance_rank span 覆盖实际图距离排序过程。
                with _trace_phase(
                    search_tracer,
                    'search.node_search.node_distance_rank',
                    {
                        'source_node_count': len(seeded_uuids),
                        'center_node_uuid.provided': center_node_uuid is not None,
                    },
                ):
                    # 距离重排返回的是节点 uuid 顺序和对应分数，分数会随结果一起返回给上层。
                    reranked_uuids, node_scores = await node_distance_reranker(
                        driver,
                        seeded_uuids,
                        center_node_uuid,
                        min_score=reranker_min_score,
                    )

        # 把 uuid 顺序转换回节点对象顺序，这是所有重排策略的统一收口点。
        reranked_nodes = [node_uuid_map[uuid] for uuid in reranked_uuids]

        # 写入节点搜索统计：候选数量、重排后数量和最终返回数量。
        span.add_attributes(
            {
                'candidate_count': len(node_uuid_map),
                'reranked_count': len(reranked_nodes),
                'returned_count': min(len(reranked_nodes), limit),
            }
        )

        # 截断到全局 limit，保持节点结果数量和配置一致。
        return reranked_nodes[:limit], node_scores[:limit]


# episode_search 处理 EpisodicNode：目前它只做全文召回，再在 RRF 或 cross_encoder 两种重排方式之间选择。
async def episode_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    _query_vector: list[float],
    group_ids: list[str] | None,
    config: EpisodeSearchConfig | None,
    search_filter: SearchFilters,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[EpisodicNode], list[float]]:
    # 没有情节搜索配置时跳过该搜索域，返回空结果和空分数。
    if config is None:
        return [], []
    # 继续使用统一 tracer 入口，让情节搜索也能和其他搜索域共享观测链路。
    search_tracer = _resolve_tracer(search_tracer)

    # 情节搜索外层 span 记录 limit、reranker 和 search_methods，方便和其他搜索域统一观测。
    with _trace_phase(
        search_tracer,
        'search.episode_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
        },
    ) as span:
        # 情节候选生成目前只调用 episode_fulltext_search；仍然用 semaphore_gather，是为了保持和其他搜索域相同的扩展结构。
        with _trace_phase(
            search_tracer,
            'search.episode_search.execute_methods',
            {'candidate_limit': 2 * limit},
        ):
            # 虽然现在只有一个全文检索任务，结果仍包装成 list[list[EpisodicNode]]，让后续 RRF 的输入格式统一。
            search_results: list[list[EpisodicNode]] = list(
                await semaphore_gather(
                    *[
                        episode_fulltext_search(driver, query, search_filter, group_ids, 2 * limit),
                    ]
                )
            )

        # 保留情节 uuid 的按召回来源排序列表，用于 RRF 或作为 cross_encoder 前的初筛。
        search_result_uuids = [[episode.uuid for episode in result] for result in search_results]
        # episode_uuid_map 负责去重并提供 uuid 到情节对象的快速回查。
        episode_uuid_map = {
            episode.uuid: episode for result in search_results for episode in result
        }

        # 初始化重排输出容器；无论哪种 reranker，最终都会填充 uuid 顺序和分数列表。
        reranked_uuids: list[str] = []
        episode_scores: list[float] = []
        # 情节重排阶段只处理候选排序，不重新查询数据库。
        with _trace_phase(
            search_tracer,
            'search.episode_search.rerank',
            {
                'candidate_count': len(episode_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            # RRF 对目前的单个全文召回列表来说相当于保留原排序，但保留该逻辑方便未来增加更多情节召回方法。
            if config.reranker == EpisodeReranker.rrf:
                reranked_uuids, episode_scores = rrf(
                    search_result_uuids, min_score=reranker_min_score
                )
            # cross_encoder 情节重排先用 RRF 做轻量预筛，再对 episode.content 做更贵的语义打分。
            elif config.reranker == EpisodeReranker.cross_encoder:
                with _trace_phase(
                    search_tracer,
                    'search.episode_search.seed_rrf',
                    {'result_set_count': len(search_results)},
                ):
                    # 这里的 RRF 结果充当 cross_encoder 的候选池，避免直接把所有全文命中都送去精排。
                    rrf_result_uuids, episode_scores = rrf(
                        search_result_uuids, min_score=reranker_min_score
                    )
                # 只取前 limit 个 RRF 结果进入 cross_encoder，控制重排成本。
                rrf_results = [episode_uuid_map[uuid] for uuid in rrf_result_uuids][:limit]

                # cross_encoder 处理文本内容，因此先建立 content 到 uuid 的映射，便于打分后回到对象层。
                content_to_uuid_map = {episode.content: episode.uuid for episode in rrf_results}

                # cross_encoder_rank span 单独记录候选文本数量，便于定位 reranker 成本。
                with _trace_phase(
                    search_tracer,
                    'search.episode_search.cross_encoder_rank',
                    {'candidate_count': len(content_to_uuid_map)},
                ):
                    reranked_contents = await cross_encoder.rank(
                        query, list(content_to_uuid_map.keys())
                    )
                # 只保留分数达到 reranker_min_score 的内容，并转换回情节 uuid。
                reranked_uuids = [
                    content_to_uuid_map[content]
                    for content, score in reranked_contents
                    if score >= reranker_min_score
                ]
                # 分数列表和 reranked_uuids 使用同一个阈值过滤，保持结果与分数一一对应。
                episode_scores = [
                    score for _, score in reranked_contents if score >= reranker_min_score
                ]

        # 按照最终 uuid 顺序取回 EpisodicNode 对象。
        reranked_episodes = [episode_uuid_map[uuid] for uuid in reranked_uuids]
        # 记录情节候选数、重排后数量和实际返回数量。
        span.add_attributes(
            {
                'candidate_count': len(episode_uuid_map),
                'reranked_count': len(reranked_episodes),
                'returned_count': min(len(reranked_episodes), limit),
            }
        )

        # 返回 limit 内的情节和分数；分数列表与情节列表在顺序上保持一致。
        return reranked_episodes[:limit], episode_scores[:limit]


# community_search 处理 CommunityNode：它把社区名称的全文召回和向量召回结合起来，再用 RRF、MMR 或 cross_encoder 重排。
async def community_search(
    driver: GraphDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: list[str] | None,
    config: CommunitySearchConfig | None,
    limit=DEFAULT_SEARCH_LIMIT,
    reranker_min_score: float = 0,
    search_tracer: Tracer | None = None,
) -> tuple[list[CommunityNode], list[float]]:
    # 没有社区搜索配置时跳过该搜索域，返回空结果。
    if config is None:
        return [], []
    # 社区搜索同样把缺省 tracer 解析成 NoOpTracer，保证后续 span 创建逻辑统一。
    search_tracer = _resolve_tracer(search_tracer)

    # 社区搜索外层 span 记录 limit、reranker 和搜索方法，虽然当前实现固定同时执行全文与向量检索。
    with _trace_phase(
        search_tracer,
        'search.community_search',
        {
            'limit': limit,
            'reranker': _enum_value(config.reranker),
            'search_methods': [_enum_value(method) for method in config.search_methods],
        },
    ) as span:
        # 社区候选生成阶段同时跑 fulltext 和 similarity，两种召回互补：一个偏关键词，一个偏语义。
        with _trace_phase(
            search_tracer,
            'search.community_search.execute_methods',
            {'candidate_limit': 2 * limit},
        ):
            # 候选结果保持为多个列表，便于后续 RRF 知道每个候选来自哪一路召回以及在该路中的排序。
            search_results: list[list[CommunityNode]] = list(
                await semaphore_gather(
                    *[
                        community_fulltext_search(driver, query, group_ids, 2 * limit),
                        community_similarity_search(
                            driver, query_vector, group_ids, 2 * limit, config.sim_min_score
                        ),
                    ]
                )
            )

        # 将每个召回列表转换成社区 uuid 列表，为 RRF 或其他重排准备统一输入。
        search_result_uuids = [
            [community.uuid for community in result] for result in search_results
        ]
        # community_uuid_map 用 uuid 去重并保存社区对象，避免同一社区从全文和向量两路重复返回。
        community_uuid_map = {
            community.uuid: community for result in search_results for community in result
        }

        # 初始化社区重排输出，最终所有策略都会汇总到这两个列表。
        reranked_uuids: list[str] = []
        community_scores: list[float] = []
        # 社区重排阶段根据配置选择融合方式或精排方式。
        with _trace_phase(
            search_tracer,
            'search.community_search.rerank',
            {
                'candidate_count': len(community_uuid_map),
                'result_set_count': len(search_results),
                'reranker': _enum_value(config.reranker),
            },
        ):
            # RRF 融合全文和向量两路排序，适合在不额外加载 embedding 或调用模型的情况下综合排序。
            if config.reranker == CommunityReranker.rrf:
                reranked_uuids, community_scores = rrf(
                    search_result_uuids, min_score=reranker_min_score
                )
            # MMR 分支会加载社区 embedding，并在相关性与多样性之间做平衡。
            elif config.reranker == CommunityReranker.mmr:
                # load_embeddings span 用于观察候选社区向量加载耗时。
                with _trace_phase(
                    search_tracer,
                    'search.community_search.load_embeddings',
                    {'candidate_count': len(community_uuid_map)},
                ):
                    # 批量取回候选社区向量，供后面的 maximal_marginal_relevance 使用。
                    search_result_uuids_and_vectors = await get_embeddings_for_communities(
                        driver, list(community_uuid_map.values())
                    )

                # MMR 计算阶段只使用 query_vector 和候选 embedding，不再访问数据库。
                with _trace_phase(
                    search_tracer,
                    'search.community_search.compute_mmr',
                    {'candidate_count': len(search_result_uuids_and_vectors)},
                ):
                    # MMR 返回社区 uuid 顺序和分数，兼顾查询相关性与结果多样性。
                    reranked_uuids, community_scores = maximal_marginal_relevance(
                        query_vector,
                        search_result_uuids_and_vectors,
                        config.mmr_lambda,
                        reranker_min_score,
                    )
            # cross_encoder 社区重排把社区名称交给交叉编码器打分，适合候选已经较少时做更精细语义判断。
            elif config.reranker == CommunityReranker.cross_encoder:
                # 建立名称到 uuid 的映射；cross_encoder 返回名称排序后，再映射回社区对象。
                name_to_uuid_map = {
                    node.name: node.uuid for result in search_results for node in result
                }
                # cross_encoder_rank span 单独记录进入模型精排的社区名称数量。
                with _trace_phase(
                    search_tracer,
                    'search.community_search.cross_encoder_rank',
                    {'candidate_count': len(name_to_uuid_map)},
                ):
                    # cross_encoder.rank 输出按相关性排序的 name-score 列表。
                    reranked_nodes = await cross_encoder.rank(query, list(name_to_uuid_map.keys()))
                # 过滤低于最小重排分的社区，并保留其 uuid 顺序。
                reranked_uuids = [
                    name_to_uuid_map[name]
                    for name, score in reranked_nodes
                    if score >= reranker_min_score
                ]
                # 社区分数列表使用同样阈值过滤，保持和社区 uuid 列表对齐。
                community_scores = [
                    score for _, score in reranked_nodes if score >= reranker_min_score
                ]

        # 根据重排后的 uuid 顺序取回 CommunityNode 对象。
        reranked_communities = [community_uuid_map[uuid] for uuid in reranked_uuids]
        # 记录社区搜索候选、重排和最终返回数量。
        span.add_attributes(
            {
                'candidate_count': len(community_uuid_map),
                'reranked_count': len(reranked_communities),
                'returned_count': min(len(reranked_communities), limit),
            }
        )

        # 最终返回 limit 个社区结果和对应分数。
        return reranked_communities[:limit], community_scores[:limit]
