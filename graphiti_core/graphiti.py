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

# 【注释 01】模块级依赖从通用工具开始：日志、时间、UUID 先提供横切能力，后面的业务流程都会复用。
import logging
from datetime import datetime
from time import time
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import LiteralString

# 【注释 02】随后导入 Graphiti 的核心抽象：driver 负责图数据库，LLM/Embedder/Reranker 负责语义处理。
# 这些依赖在 GraphitiClients 中被打包，方便后续节点、边、搜索等流程统一传递。
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.neo4j_driver import Neo4jDriver
# 【注释 03】边类型按图谱层次组织：实体边描述事实，episode 边描述来源，community/saga 边描述更高层结构。
from graphiti_core.edges import (
    CommunityEdge,
    Edge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import (
    get_default_group_id,
    semaphore_gather,
    validate_excluded_entity_types,
    validate_group_id,
)
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.namespaces import EdgeNamespace, NodeNamespace
# 【注释 04】节点类型与边类型对应：EpisodicNode 是输入来源，EntityNode 是抽取后的知识实体，Saga/Community 是聚合视角。
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    Node,
    SagaNode,
    create_entity_node_embeddings,
)
from graphiti_core.prompts.lib import prompt_library
from graphiti_core.prompts.summarize_sagas import SagaSummary
from graphiti_core.search.search import SearchConfig, search
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT, SearchResults
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    RELEVANT_SCHEMA_LIMIT,
    get_mentioned_nodes,
)
from graphiti_core.telemetry import capture_event
from graphiti_core.tracer import Tracer, create_tracer
# 【注释 05】bulk_utils 中的函数支撑批量写入、去重、指针修正；这是批处理路径能复用单条 episode 逻辑的关键。
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    add_nodes_and_edges_bulk,
    dedupe_edges_bulk,
    dedupe_nodes_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.community_operations import (
    build_communities,
    remove_communities,
    update_community,
)
# 【注释 06】maintenance 模块提供“抽取—解析—构边”的底层操作，Graphiti 类主要负责编排这些步骤。
from graphiti_core.utils.maintenance.edge_operations import (
    build_episodic_edges,
    extract_edges,
    resolve_extracted_edge,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_attributes_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

# 【注释 07】模块初始化只做轻量准备：创建日志器，并加载环境变量，避免把环境读取散落在各个方法里。
logger = logging.getLogger(__name__)

load_dotenv()


# 【注释 08】下面三个 BaseModel 是对外返回值的结构化封装：把一次写图操作产生的节点、边、社区结果统一打包。
class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    episodic_edges: list[EpisodicEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    communities: list[CommunityNode]
    community_edges: list[CommunityEdge]


# 【注释 09】批量添加 episode 的返回结构与单条类似，但 episode 字段变成列表，便于调用方一次性拿到所有新增对象。
class AddBulkEpisodeResults(BaseModel):
    episodes: list[EpisodicNode]
    episodic_edges: list[EpisodicEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    communities: list[CommunityNode]
    community_edges: list[CommunityEdge]


# 【注释 10】Triplet 写入只涉及一条 source-edge-target 事实，因此返回模型只保留实体节点和实体边。
class AddTripletResults(BaseModel):
    nodes: list[EntityNode]
    edges: list[EntityEdge]


# 【注释 11】Graphiti 是门面类：它不直接实现所有算法，而是把数据库、LLM、嵌入、重排、搜索和维护操作串成业务流程。
class Graphiti:
    # 【注释 12】初始化阶段的逻辑顺序是：先确定图数据库连接，再确定 AI 客户端，最后把它们封装成命名空间和 clients。
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        store_raw_episode_content: bool = True,
        graph_driver: GraphDriver | None = None,
        max_coroutines: int | None = None,
        tracer: Tracer | None = None,
        trace_span_prefix: str = 'graphiti',
    ):
        """
        Initialize a Graphiti instance.

        This constructor sets up a connection to a graph database and initializes
        the LLM client for natural language processing tasks.

        Parameters
        ----------
        uri : str
            The URI of the Neo4j database.
        user : str
            The username for authenticating with the Neo4j database.
        password : str
            The password for authenticating with the Neo4j database.
        llm_client : LLMClient | None, optional
            An instance of LLMClient for natural language processing tasks.
            If not provided, a default OpenAIClient will be initialized.
        embedder : EmbedderClient | None, optional
            An instance of EmbedderClient for embedding tasks.
            If not provided, a default OpenAIEmbedder will be initialized.
        cross_encoder : CrossEncoderClient | None, optional
            An instance of CrossEncoderClient for reranking tasks.
            If not provided, a default OpenAIRerankerClient will be initialized.
        store_raw_episode_content : bool, optional
            Whether to store the raw content of episodes. Defaults to True.
        graph_driver : GraphDriver | None, optional
            An instance of GraphDriver for database operations.
            If not provided, a default Neo4jDriver will be initialized.
        max_coroutines : int | None, optional
            The maximum number of concurrent operations allowed. Overrides SEMAPHORE_LIMIT set in the environment.
            If not set, the Graphiti default is used.
        tracer : Tracer | None, optional
            An OpenTelemetry tracer instance for distributed tracing. If not provided, tracing is disabled (no-op).
        trace_span_prefix : str, optional
            Prefix to prepend to all span names. Defaults to 'graphiti'.

        Returns
        -------
        None

        Notes
        -----
        This method establishes a connection to a graph database (Neo4j by default) using the provided
        credentials. It also sets up the LLM client, either using the provided client
        or by creating a default OpenAIClient.

        The default database name is defined during the driver’s construction. If a different database name
        is required, it should be specified in the URI or set separately after
        initialization.

        The OpenAI API key is expected to be set in the environment variables.
        Make sure to set the OPENAI_API_KEY environment variable before initializing
        Graphiti if you're using the default OpenAIClient.
        """

        # 【注释 13】允许外部传入 driver 是为了支持测试或替换后端；否则才根据 URI 创建默认 Neo4jDriver。
        if graph_driver:
            self.driver = graph_driver
        else:
            if uri is None:
                raise ValueError('uri must be provided when graph_driver is None')
            self.driver = Neo4jDriver(uri, user, password)

        # 【注释 14】这些实例配置会影响后续写入行为：是否保存原文、并发上限、以及使用哪些默认客户端。
        self.store_raw_episode_content = store_raw_episode_content
        self.max_coroutines = max_coroutines
        # 【注释 15】LLM、Embedder、CrossEncoder 都采用“外部注入优先，默认 OpenAI 实现兜底”的策略，保持扩展性。
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient()
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = OpenAIEmbedder()
        if cross_encoder:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = OpenAIRerankerClient()

        # 【注释 16】tracer 是横切能力：后续 add_episode / bulk 等长流程会用 span 记录耗时和异常。
        # Initialize tracer
        self.tracer = create_tracer(tracer, trace_span_prefix)

        # Set tracer on clients
        self.llm_client.set_tracer(self.tracer)

        # 【注释 17】GraphitiClients 把多个底层客户端聚合起来，避免每个工具函数都要传一长串依赖。
        self.clients = GraphitiClients(
            driver=self.driver,
            llm_client=self.llm_client,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
            tracer=self.tracer,
        )

        # Initialize namespace API (graphiti.nodes.entity.save(), etc.)
        # 【注释 18】namespace API 是更细粒度的节点/边操作入口，和高层 add_episode/search 流程并行存在。
        self.nodes = NodeNamespace(self.driver, self.embedder)
        self.edges = EdgeNamespace(self.driver, self.embedder)

        # 【注释 19】最后上报初始化遥测；它不参与业务正确性，所以放在资源都准备好之后执行。
        # Capture telemetry event
        self._capture_initialization_telemetry()

    # 【注释 20】初始化遥测只记录“使用了哪些 provider”，帮助统计运行环境，而不改变图谱数据。
    def _capture_initialization_telemetry(self):
        """Capture telemetry event for Graphiti initialization."""
        # 【注释 21】遥测被包在 try 中，说明它是非关键路径：失败时不能影响 Graphiti 实例创建。
        try:
            # Detect provider types from class names
            llm_provider = self._get_provider_type(self.llm_client)
            embedder_provider = self._get_provider_type(self.embedder)
            reranker_provider = self._get_provider_type(self.cross_encoder)
            database_provider = self._get_provider_type(self.driver)

            properties = {
                'llm_provider': llm_provider,
                'embedder_provider': embedder_provider,
                'reranker_provider': reranker_provider,
                'database_provider': database_provider,
            }

            # 【注释 22】真正发送事件前，先把 LLM、嵌入、重排、数据库 provider 统一转换成简单字符串。
            capture_event('graphiti_initialized', properties)
        # 【注释 23】这里吞掉异常是有意设计：遥测故障不应该让用户的图数据库连接或 AI 客户端初始化失败。
        except Exception:
            # Silently handle telemetry errors
            pass

    # 【注释 24】token_tracker 暴露的是 LLM 客户端内部统计，方便外部观察 prompt/token 成本，而不接触 LLM 实现细节。
    @property
    def token_tracker(self):
        """Access the LLM client's token usage tracker.

        Returns the TokenUsageTracker from the LLM client, which can be used to:
        - Get token usage by prompt type: tracker.get_usage()
        - Get total token usage: tracker.get_total_usage()
        - Print a formatted summary: tracker.print_summary()
        - Reset tracking: tracker.reset()
        """
        return self.llm_client.token_tracker

    # 【注释 25】provider 识别采用类名约定，而不是 isinstance；这样新 provider 只要命名规范就能被粗略归类。
    def _get_provider_type(self, client) -> str:
        """Get provider type from client class name."""
        # 【注释 26】None 先处理掉，避免后面对空对象取 class 名。
        if client is None:
            return 'none'

        # 【注释 27】统一转小写后做包含判断，降低不同实现类命名大小写不一致带来的分支复杂度。
        class_name = client.__class__.__name__.lower()

        # 【注释 28】这里按 provider 类型分组判断：先识别 AI 客户端，再识别数据库和嵌入模型。
        # LLM providers
        if 'openai' in class_name:
            return 'openai'
        elif 'azure' in class_name:
            return 'azure'
        elif 'anthropic' in class_name:
            return 'anthropic'
        elif 'crossencoder' in class_name:
            return 'crossencoder'
        elif 'gemini' in class_name:
            return 'gemini'
        elif 'groq' in class_name:
            return 'groq'
        # Database providers
        elif 'neo4j' in class_name:
            return 'neo4j'
        elif 'falkor' in class_name:
            return 'falkordb'
        # Embedder providers
        elif 'voyage' in class_name:
            return 'voyage'
        else:
            return 'unknown'

    # 【注释 29】close 是生命周期收尾：Graphiti 自身不持久化额外状态，关闭 driver 就能释放主要外部资源。
    async def close(self):
        """
        Close the connection to the Neo4j database.

        This method safely closes the driver connection to the Neo4j database.
        It should be called when the Graphiti instance is no longer needed or
        when the application is shutting down.

        Parameters
        ----------
        self

        Returns
        -------
        None

        Notes
        -----
        It's important to close the driver connection to release system resources
        and ensure that all pending transactions are completed or rolled back.
        This method should be called as part of a cleanup process, potentially
        in a context manager or a shutdown hook.

        Example:
            graphiti = Graphiti(uri, user, password)
            try:
                # Use graphiti...
            finally:
                graphiti.close()
        """
        # 【注释 30】异步关闭数据库连接，保证未完成的底层连接清理交给 driver 处理。
        await self.driver.close()

    # 【注释 31】Saga 是跨 episode 的故事线/会话线；这里采用 get-or-create，避免同名同 group 的 saga 重复创建。
    async def _get_or_create_saga(
        self, saga_name: str, group_id: str, created_at: datetime
    ) -> SagaNode:
        """
        Get an existing saga by name or create a new one.

        Parameters
        ----------
        saga_name : str
            The name of the saga.
        group_id : str
            The group id for the saga.
        created_at : datetime
            Timestamp to stamp on a newly created saga. Callers should pass the
            originating episode's reference time (``valid_at``) rather than the
            current wall-clock time so the saga's ``created_at`` reflects the
            episode it was minted from.

        Returns
        -------
        SagaNode
            The existing or newly created saga node.
        """
        # 【注释 32】parse_db_date 局部导入，说明只有数据库返回日期时才需要它，也能减少模块加载时的循环依赖风险。
        from graphiti_core.helpers import parse_db_date

        # 【注释 33】先按 name + group_id 查询，确保 saga 的唯一性限定在同一个图分区内。
        records, _, _ = await self.driver.execute_query(
            """
            MATCH (s:Saga {name: $name, group_id: $group_id})
            RETURN s.uuid AS uuid, s.name AS name, s.group_id AS group_id, s.created_at AS created_at
            """,
            name=saga_name,
            group_id=group_id,
            routing_='r',
        )

        # 【注释 34】查到已有 saga 时直接把数据库记录还原成 SagaNode；这样后续边可以复用同一个 uuid。
        if records:
            record = records[0]
            return SagaNode(
                uuid=record['uuid'],
                name=record['name'],
                group_id=record['group_id'],
                created_at=parse_db_date(record['created_at']),  # type: ignore
            )

        # 【注释 35】没有命中才创建新 SagaNode，并用调用方传入的 created_at 保留事件时间语义。
        saga = SagaNode(name=saga_name, group_id=group_id, created_at=created_at)
        # 【注释 36】所有字段在内存中更新完成后再保存，保证 summary 和水位线同步落库。
        await saga.save(self.driver)
        return saga

    # 【注释 37】查找 saga 中前一个 episode，用来建立 NEXT_EPISODE 链，保持 saga 内 episode 的时间顺序。
    async def _saga_get_previous_episode_uuid(
        self, saga_uuid: str, current_episode_uuid: str
    ) -> str | None:
        """Find the most recent episode UUID in a saga, excluding the current one."""
        # 【注释 38】优先走 driver 的 IoC 接口：不同数据库后端可以提供更优实现；不支持时再回退到通用 Cypher。
        if self.driver.graph_operations_interface:
            try:
                return await self.driver.graph_operations_interface.saga_get_previous_episode_uuid(
                    self.driver, saga_uuid, current_episode_uuid
                )
            except NotImplementedError:
                pass

        # 【注释 39】通用查询按 valid_at、created_at 倒序取最近一条，并排除当前 episode，避免自连。
        records, _, _ = await self.driver.execute_query(
            """
            MATCH (s:Saga {uuid: $saga_uuid})-[:HAS_EPISODE]->(e:Episodic)
            WHERE e.uuid <> $current_episode_uuid
            RETURN e.uuid AS uuid
            ORDER BY e.valid_at DESC, e.created_at DESC
            LIMIT 1
            """,
            saga_uuid=saga_uuid,
            current_episode_uuid=current_episode_uuid,
            routing_='r',
        )
        # 【注释 40】找到前序 episode 就返回 uuid；找不到代表当前 episode 是 saga 的第一条或数据为空。
        if records:
            return records[0]['uuid']
        return None

    # 【注释 41】该方法只尝试通过后端接口取 episode 内容；若后端不支持，返回 None 让 summarize_saga 自己执行回退查询。
    async def _saga_get_episode_contents(
        self,
        saga_uuid: str,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[str, datetime | None]] | None:
        """Retrieve (content, valid_at) per episode for summarization, using IoC if available."""
        if self.driver.graph_operations_interface:
            try:
                return await self.driver.graph_operations_interface.saga_get_episode_contents(
                    self.driver, saga_uuid, since=since, limit=limit
                )
            except NotImplementedError:
                pass
        # 【注释 42】返回 None 与返回空列表含义不同：None 表示“接口不可用”，空列表表示“接口可用但没有新内容”。
        return None

    # 【注释 43】summarize_saga 的核心是增量总结：用上次总结的水位线只拉取新增 episode，再更新 saga 摘要。
    async def summarize_saga(self, saga_id: str) -> SagaNode:
        """Incrementally summarize a saga using only new episodes since the last summary.

        Two watermarks are maintained on the saga node, with deliberately
        different semantics:

        - ``last_summarized_at`` is wall-clock and is the *filter* watermark:
          the next run picks up any episode whose ``created_at`` (ingestion
          time) is greater than this value. Wall-clock is the right semantics
          here because episode ``created_at`` is monotonic with processing
          time, so a backfilled episode added today with ``valid_at`` in the
          past is still picked up next run.
        - ``last_summarized_episode_valid_at`` is the *temporal* watermark:
          the maximum ``valid_at`` (episode reference time) across the
          episodes covered by the current summary. Consumers asking "how
          recent is the content of this summary in event-time?" should use
          this field, not ``last_summarized_at``.

        If the saga has been summarized before, only episodes added after
        ``last_summarized_at`` are fetched. The existing summary is provided
        to the LLM as context so no information is lost.

        On the first call (no prior summary), all episodes are included.

        Parameters
        ----------
        saga_id : str
            The UUID of the saga to summarize.

        Returns
        -------
        SagaNode
            The updated saga node with the new summary.

        Raises
        ------
        NodeNotFoundError
            If the saga with the given UUID does not exist.
        """
        # 【注释 44】先取 saga 本体，因为后续需要已有 summary、水位线以及最终保存的目标节点。
        saga = await SagaNode.get_by_uuid(self.driver, saga_id)

        # Fetch only episodes added since the last summary (or all if never summarized).
        # 【注释 45】限制每次总结的 episode 数量，避免一次 prompt 过大；since 记录增量起点。
        max_episodes = 200
        since = saga.last_summarized_at

        # Try IoC interface first, fall back to raw Cypher
        # 【注释 46】先尝试后端定制接口，让数据库实现有机会优化内容读取。
        episodes_data = await self._saga_get_episode_contents(
            saga_id, since=since, limit=max_episodes
        )
        # 【注释 47】只有接口不可用时才进入 Cypher 回退；如果接口返回空列表，就应该按“无新增内容”处理。
        if episodes_data is None:
            # 【注释 48】有 last_summarized_at 时走增量查询，只读取上次总结后新写入的 episode。
            if since is not None:
                records, _, _ = await self.driver.execute_query(
                    """
                    MATCH (s:Saga {uuid: $saga_uuid})-[:HAS_EPISODE]->(e:Episodic)
                    WHERE e.created_at > $since
                    RETURN e.content AS content, e.valid_at AS valid_at
                    ORDER BY e.valid_at ASC, e.created_at ASC
                    LIMIT $limit
                    """,
                    saga_uuid=saga_id,
                    since=since,
                    limit=max_episodes,
                    routing_='r',
                )
            else:
                records, _, _ = await self.driver.execute_query(
                    """
                    MATCH (s:Saga {uuid: $saga_uuid})-[:HAS_EPISODE]->(e:Episodic)
                    RETURN e.content AS content, e.valid_at AS valid_at
                    ORDER BY e.valid_at DESC, e.created_at DESC
                    LIMIT $limit
                    """,
                    saga_uuid=saga_id,
                    limit=max_episodes,
                    routing_='r',
                )
                # 【注释 49】没有水位线代表首次总结：先取最近 N 条，再反转成时间正序供 LLM 阅读。
                # Reverse to chronological order for the prompt
                records = list(reversed(records))

            # 【注释 50】数据库日期在进入 LLM/水位线逻辑前统一解析，避免后面混用字符串和 datetime。
            from graphiti_core.helpers import parse_db_date

            episodes_data = [
                (r['content'], parse_db_date(r.get('valid_at')))
                for r in records
                if r.get('content')
            ]

        # 【注释 51】没有新增 episode 时直接返回原 saga，避免生成空摘要覆盖已有 summary。
        if not episodes_data:
            logger.info(f'No new episodes found for saga {saga_id}, skipping summary')
            return saga

        episode_contents = [content for content, _ in episodes_data]
        valid_ats = [valid_at for _, valid_at in episodes_data if valid_at is not None]

        # 【注释 52】传给 LLM 的上下文包含 saga 名、旧摘要和新增 episode，目的是在保留历史的基础上吸收新信息。
        context = {
            'saga_name': saga.name,
            'existing_summary': saga.summary or '',
            'episodes': episode_contents,
        }

        # 【注释 53】通过结构化 response_model 约束 LLM 输出，减少后续解析 summary 字段的不确定性。
        llm_response = await self.llm_client.generate_response(
            prompt_library.summarize_sagas.summarize_saga(context),
            response_model=SagaSummary,
            prompt_name='summarize_sagas.summarize_saga',
        )

        summary = llm_response.get('summary', '')
        # 【注释 54】摘要截断是存储层保护：即使 LLM 输出过长，也不会让节点字段无限增长。
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS]

        # 【注释 55】先更新 summary，再分别更新两个水位线：一个服务增量过滤，一个服务事件时间语义。
        saga.summary = summary
        # Wall-clock watermark for the next-run filter: keeps backfilled
        # episodes (valid_at in the past, created_at = now) reachable on
        # subsequent runs.
        saga.last_summarized_at = utc_now()
        # Episode-time watermark for public/temporal consumers: advance only
        # forward to the latest reference time we just summarized. If no
        # episode in this batch carried a valid_at, leave the previous value
        # unchanged so the field never regresses.
        if valid_ats:
            new_episode_watermark = max(valid_ats)
            if (
                saga.last_summarized_episode_valid_at is None
                or new_episode_watermark > saga.last_summarized_episode_valid_at
            ):
                saga.last_summarized_episode_valid_at = new_episode_watermark
        await saga.save(self.driver)

        logger.info(f'Updated summary for saga {saga_id}')

        return saga

    # 【注释 56】索引和约束属于图数据库结构准备，通常在正式写入 episode 前执行一次。
    async def build_indices_and_constraints(self, delete_existing: bool = False):
        """
        Build indices and constraints in the Neo4j database.

        This method sets up the necessary indices and constraints in the Neo4j database
        to optimize query performance and ensure data integrity for the knowledge graph.

        Parameters
        ----------
        self
        delete_existing : bool, optional
            Whether to clear existing indices before creating new ones.


        Returns
        -------
        None

        Notes
        -----
        This method should typically be called once during the initial setup of the
        knowledge graph or when updating the database schema. It uses the
        driver's `build_indices_and_constraints` method to perform
        the actual database operations.

        The specific indices and constraints created depend on the implementation
        of the driver's `build_indices_and_constraints` method. Refer to the specific
        driver documentation for details on the exact database schema modifications.

        Caution: Running this method on a large existing database may take some time
        and could impact database performance during execution.
        """
        # 【注释 57】具体建哪些索引由 driver 决定，Graphiti 只暴露统一入口。
        await self.driver.build_indices_and_constraints(delete_existing)

    # 【注释 58】节点处理被拆成“抽取”和“解析”两步：先从文本中找候选实体，再和已有图谱去重/合并。
    async def _extract_and_resolve_nodes(
        self,
        episode: EpisodicNode | list[EpisodicNode],
        previous_episodes: list[EpisodicNode],
        entity_types: dict[str, type[BaseModel]] | None,
        excluded_entity_types: list[str] | None,
    ) -> tuple[
        list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]], dict[str, list[int]]
    ]:
        """Extract nodes from episode(s) and resolve against existing graph."""
        # 【注释 59】把单条和多条 episode 统一成列表，后续代码只需要处理一种形态。
        episodes = episode if isinstance(episode, list) else [episode]
        # 【注释 60】解析阶段需要一个主 episode 作为上下文锚点；批量传入时默认使用第一条。
        primary_episode = episodes[0]

        # 【注释 61】extract_nodes 只负责产生候选节点，并记录节点来自哪些 episode，用于后面建立 MENTIONS 类边。
        extracted_nodes, node_episode_index_map = await extract_nodes(
            self.clients, episode, previous_episodes, entity_types, excluded_entity_types
        )

        # 【注释 62】resolve_extracted_nodes 会把候选节点映射到已有或新建 uuid，uuid_map 是后续修正边端点的关键。
        nodes, uuid_map, duplicates = await resolve_extracted_nodes(
            self.clients,
            extracted_nodes,
            primary_episode,
            previous_episodes,
            entity_types,
        )

        return nodes, uuid_map, duplicates, node_episode_index_map

    # 【注释 63】边处理依赖节点结果：边的端点必须先通过节点解析得到稳定 uuid，才能进入事实去重和失效判断。
    async def _extract_and_resolve_edges(
        self,
        episode: EpisodicNode | list[EpisodicNode],
        extracted_nodes: list[EntityNode],
        previous_episodes: list[EpisodicNode],
        edge_type_map: dict[tuple[str, str], list[str]],
        group_id: str,
        edge_types: dict[str, type[BaseModel]] | None,
        nodes: list[EntityNode],
        uuid_map: dict[str, str],
        custom_extraction_instructions: str | None = None,
    ) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
        """Extract edges from episode(s) and resolve against existing graph.

        Returns
        -------
        tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
            A tuple of (resolved_edges, invalidated_edges, new_edges) where:
            - resolved_edges: All edges after resolution
            - invalidated_edges: Edges invalidated by new information
            - new_edges: Only edges that are new to the graph (not duplicates)
        """
        # 【注释 64】同样把 episode 统一为列表；primary_episode 用于给边解析提供代表性上下文。
        episodes = episode if isinstance(episode, list) else [episode]
        primary_episode = episodes[0]

        # 【注释 65】先基于 episode 文本、已抽取节点和 edge_type_map 抽取候选事实边。
        extracted_edges = await extract_edges(
            self.clients,
            episode,
            extracted_nodes,
            previous_episodes,
            edge_type_map,
            group_id,
            edge_types,
            custom_extraction_instructions,
        )

        # 【注释 66】候选边可能仍指向抽取阶段的临时节点 uuid，这里用 uuid_map 改写成解析后的真实端点。
        edges = resolve_edge_pointers(extracted_edges, uuid_map)

        # 【注释 67】边解析同时产出三类结果：可保存的边、被新事实否定的旧边、以及真正新增的边。
        resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
            self.clients,
            edges,
            primary_episode,
            nodes,
            edge_types or {},
            edge_type_map,
        )

        return resolved_edges, invalidated_edges, new_edges

    # 【注释 68】该方法负责最终落库：把 episode、MENTIONS 边、实体节点、事实边，以及可选 saga 关系一次性保存。
    async def _process_episode_data(
        self,
        episode: EpisodicNode | list[EpisodicNode],
        nodes: list[EntityNode],
        entity_edges: list[EntityEdge],
        now: datetime,
        group_id: str,
        saga: str | SagaNode | None = None,
        saga_previous_episode_uuid: str | None = None,
        node_episode_index_map: dict[str, list[int]] | None = None,
    ) -> tuple[list[EpisodicEdge], EpisodicNode]:
        """Process and save episode data to the graph.

        Parameters
        ----------
        episode : EpisodicNode | list[EpisodicNode]
            The episode(s) to process.
        nodes : list[EntityNode]
            The entity nodes extracted from the episode(s).
        entity_edges : list[EntityEdge]
            The entity edges extracted from the episode(s).
        now : datetime
            The current timestamp.
        group_id : str
            The group id for the episode.
        saga : str | SagaNode | None
            Optional. Either a saga name (str) or a SagaNode object to associate
            this episode with. If a string is provided, the saga will be looked up
            by name or created if it doesn't exist.
        saga_previous_episode_uuid : str | None
            Optional. UUID of the previous episode in the saga. If provided, skips
            the database query to find the most recent episode. Useful for efficiently
            adding multiple episodes to the same saga in sequence.
        node_episode_index_map : dict[str, list[int]] | None
            Optional mapping from node UUID to 0-indexed episode positions for
            building episodic edges with correct attribution.
        """
        # 【注释 69】保存阶段也统一单条/多条输入，方便 bulk 与单条流程复用同一套写入逻辑。
        episodes = episode if isinstance(episode, list) else [episode]
        # 【注释 70】先收集 episode uuid，后面构建 episodic_edges 时需要知道每个实体来自哪些 episode。
        episode_uuids = [ep.uuid for ep in episodes]

        # 【注释 71】构建 episode 到实体的关联边，让图谱能追溯某个实体/事实来自哪些原始输入。
        episodic_edges = build_episodic_edges(nodes, episode_uuids, now, node_episode_index_map)
        # 【注释 72】每个 episode 记录它关联的实体边 uuid；如果配置不保存原文，则在落库前清空 content。
        for ep in episodes:
            ep.entity_edges = [edge.uuid for edge in entity_edges]
            if not self.store_raw_episode_content:
                ep.content = ''

        # 【注释 73】第一次统一写入 episode、episodic_edges、实体节点和实体边，减少多次数据库往返。
        await add_nodes_and_edges_bulk(
            self.driver,
            episodes,
            episodic_edges,
            nodes,
            entity_edges,
            self.embedder,
        )

        # 【注释 74】后续 saga 关系只绑定主 episode；单条流程中它就是当前 episode。
        primary_episode = episodes[0]

        # 【注释 75】saga 关系是可选增强：不影响基础知识图谱写入，只在调用方传入 saga 时建立故事线。
        # Handle saga association if provided
        if saga is not None:
            # Get or create saga node based on input type
            if isinstance(saga, str):
                # Use the originating episode's reference time (valid_at) for a
                # newly created saga so its created_at matches the episode that
                # minted it, not the wall-clock time of this run.
                saga_created_at = primary_episode.valid_at or now
                saga_node = await self._get_or_create_saga(saga, group_id, saga_created_at)
            else:
                saga_node = saga

            # 【注释 76】如果调用方已经知道前序 episode，就直接使用；否则查询数据库，兼顾批量追加的效率和单次调用的易用性。
            # Use provided previous episode UUID or query for it
            previous_episode_uuid: str | None = saga_previous_episode_uuid
            if previous_episode_uuid is None:
                previous_episode_uuid = await self._saga_get_previous_episode_uuid(
                    saga_node.uuid, primary_episode.uuid
                )

            # 【注释 77】NEXT_EPISODE 表示 saga 内的时间链；只有存在前序 episode 时才需要创建。
            # Create NEXT_EPISODE edge from the previous episode to the new one
            if previous_episode_uuid is not None:
                next_episode_edge = NextEpisodeEdge(
                    source_node_uuid=previous_episode_uuid,
                    target_node_uuid=primary_episode.uuid,
                    group_id=group_id,
                    created_at=now,
                )
                await next_episode_edge.save(self.driver)

            # 【注释 78】HAS_EPISODE 表示 saga 包含该 episode，它和 NEXT_EPISODE 分别表达归属关系和顺序关系。
            # Create HAS_EPISODE edge from saga to the new episode
            has_episode_edge = HasEpisodeEdge(
                source_node_uuid=saga_node.uuid,
                target_node_uuid=primary_episode.uuid,
                group_id=group_id,
                created_at=now,
            )
            await has_episode_edge.save(self.driver)

            # 【注释 79】saga 节点缓存首尾 episode，方便以后快速定位故事线边界，不必每次全量扫描边。
            # Track first and last episode on the saga node
            if saga_node.first_episode_uuid is None:
                saga_node.first_episode_uuid = primary_episode.uuid
            saga_node.last_episode_uuid = primary_episode.uuid
            await saga_node.save(self.driver)

        return episodic_edges, primary_episode

    # 【注释 80】批量路径先对所有 episode 同时抽取，再做内存级去重，减少重复 LLM/数据库工作。
    async def _extract_and_dedupe_nodes_bulk(
        self,
        episode_context: list[tuple[EpisodicNode, list[EpisodicNode]]],
        edge_type_map: dict[tuple[str, str], list[str]],
        edge_types: dict[str, type[BaseModel]] | None,
        entity_types: dict[str, type[BaseModel]] | None,
        excluded_entity_types: list[str] | None,
        custom_extraction_instructions: str | None = None,
    ) -> tuple[
        dict[str, list[EntityNode]],
        dict[str, str],
        list[list[EntityEdge]],
    ]:
        """Extract nodes and edges from all episodes and deduplicate."""
        # 【注释 81】这里一次性拿到 nodes 和 edges 的原始抽取结果；边暂时还可能指向未解析的节点 uuid。
        # Extract all nodes and edges for each episode
        extracted_nodes_bulk, extracted_edges_bulk = await extract_nodes_and_edges_bulk(
            self.clients,
            episode_context,
            edge_type_map=edge_type_map,
            edge_types=edge_types,
            entity_types=entity_types,
            excluded_entity_types=excluded_entity_types,
            custom_extraction_instructions=custom_extraction_instructions,
        )

        # 【注释 82】节点先去重，因为边端点解析依赖最终节点 uuid；这一步越早做，后续边处理越稳定。
        # Dedupe extracted nodes in memory
        nodes_by_episode, uuid_map = await dedupe_nodes_bulk(
            self.clients, extracted_nodes_bulk, episode_context, entity_types
        )

        return nodes_by_episode, uuid_map, extracted_edges_bulk

    # 【注释 83】bulk 解析阶段把“每个 episode 的局部抽取结果”合并成“全批次一致的节点和边”。
    async def _resolve_nodes_and_edges_bulk(
        self,
        nodes_by_episode: dict[str, list[EntityNode]],
        edges_by_episode: dict[str, list[EntityEdge]],
        episode_context: list[tuple[EpisodicNode, list[EpisodicNode]]],
        entity_types: dict[str, type[BaseModel]] | None,
        edge_types: dict[str, type[BaseModel]] | None,
        edge_type_map: dict[tuple[str, str], list[str]],
        episodes: list[EpisodicNode],
    ) -> tuple[list[EntityNode], list[EntityEdge], list[EntityEdge], dict[str, str]]:
        """Resolve nodes and edges against the existing graph."""
        # 【注释 84】先建立 uuid 到节点对象的索引，后面需要频繁根据 uuid 修正节点引用。
        nodes_by_uuid: dict[str, EntityNode] = {
            node.uuid: node for nodes in nodes_by_episode.values() for node in nodes
        }

        # 【注释 85】为每个 episode 保留去重后的节点列表，同时避免同一个节点在全批次中重复解析。
        # Get unique nodes per episode
        nodes_by_episode_unique: dict[str, list[EntityNode]] = {}
        nodes_uuid_set: set[str] = set()
        for episode, _ in episode_context:
            nodes_by_episode_unique[episode.uuid] = []
            nodes = [nodes_by_uuid[node.uuid] for node in nodes_by_episode[episode.uuid]]
            for node in nodes:
                if node.uuid not in nodes_uuid_set:
                    nodes_by_episode_unique[episode.uuid].append(node)
                    nodes_uuid_set.add(node.uuid)

        # 【注释 86】节点解析仍按 episode 上下文并发执行，既保留局部语义，又利用 semaphore_gather 提升吞吐。
        # Resolve nodes
        node_results = await semaphore_gather(
            *[
                resolve_extracted_nodes(
                    self.clients,
                    nodes_by_episode_unique[episode.uuid],
                    episode,
                    previous_episodes,
                    entity_types,
                )
                for episode, previous_episodes in episode_context
            ]
        )

        # 【注释 87】把每个 episode 的解析结果汇总到全局列表，并合并所有临时 uuid 到真实 uuid 的映射。
        resolved_nodes: list[EntityNode] = []
        uuid_map: dict[str, str] = {}
        for result in node_results:
            resolved_nodes.extend(result[0])
            uuid_map.update(result[1])

        # 【注释 88】解析后可能出现新 uuid 或合并到旧节点，因此要刷新索引，保证后续引用拿到最终节点对象。
        # Update nodes_by_uuid with resolved nodes
        for resolved_node in resolved_nodes:
            nodes_by_uuid[resolved_node.uuid] = resolved_node

        # 【注释 89】这里把每个 episode 的节点列表也改写为解析后的节点，避免后面属性抽取使用过期指针。
        # Update nodes_by_episode_unique with resolved pointers
        for episode_uuid, nodes in nodes_by_episode_unique.items():
            updated_nodes: list[EntityNode] = []
            for node in nodes:
                updated_node_uuid = uuid_map.get(node.uuid, node.uuid)
                updated_node = nodes_by_uuid[updated_node_uuid]
                updated_nodes.append(updated_node)
            nodes_by_episode_unique[episode_uuid] = updated_nodes

        # 【注释 90】属性抽取放在节点解析之后，确保属性补充到最终节点，而不是补到会被合并掉的临时节点。
        # Extract attributes for resolved nodes
        hydrated_nodes_results: list[list[EntityNode]] = await semaphore_gather(
            *[
                extract_attributes_from_nodes(
                    self.clients,
                    nodes_by_episode_unique[episode.uuid],
                    episode,
                    previous_episodes,
                    entity_types,
                )
                for episode, previous_episodes in episode_context
            ]
        )

        # 【注释 91】将并发得到的二维节点列表拍平成最终要保存的节点集合。
        final_hydrated_nodes = [node for nodes in hydrated_nodes_results for node in nodes]

        # 【注释 92】边在节点 uuid 稳定后再去重和解析，避免同一事实因为端点 uuid 不同被误判为多条边。
        # Resolve edges with updated pointers
        edges_by_episode_unique: dict[str, list[EntityEdge]] = {}
        edges_uuid_set: set[str] = set()
        for episode_uuid, edges in edges_by_episode.items():
            edges_with_updated_pointers = resolve_edge_pointers(edges, uuid_map)
            edges_by_episode_unique[episode_uuid] = []

            for edge in edges_with_updated_pointers:
                if edge.uuid not in edges_uuid_set:
                    edges_by_episode_unique[episode_uuid].append(edge)
                    edges_uuid_set.add(edge.uuid)

        # 【注释 93】每个 episode 的边解析可并发执行，但共享同一批最终节点作为事实端点参照。
        edge_results = await semaphore_gather(
            *[
                resolve_extracted_edges(
                    self.clients,
                    edges_by_episode_unique[episode.uuid],
                    episode,
                    final_hydrated_nodes,
                    edge_types or {},
                    edge_type_map,
                )
                for episode in episodes
            ]
        )

        # 【注释 94】汇总边解析结果时保留 resolved 和 invalidated 两类，后续一起写入以表达事实更新。
        resolved_edges: list[EntityEdge] = []
        invalidated_edges: list[EntityEdge] = []
        for result in edge_results:
            resolved_edges.extend(result[0])
            invalidated_edges.extend(result[1])
            # result[2] is new_edges - not used in bulk flow since attributes
            # are extracted before edge resolution

        return final_hydrated_nodes, resolved_edges, invalidated_edges, uuid_map

    # 【注释 95】该装饰器让方法可以透明处理多个 group_id；方法内部仍按列表语义传给底层检索函数。
    @handle_multiple_group_ids
    # 【注释 96】retrieve_episodes 是上下文检索入口：给定参考时间，找出最近 episode 作为新抽取的历史背景。
    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
        group_ids: list[str] | None = None,
        source: EpisodeType | None = None,
        driver: GraphDriver | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        """
        Retrieve the last n episodic nodes from the graph.

        This method fetches a specified number of the most recent episodic nodes
        from the graph, relative to the given reference time.

        Parameters
        ----------
        reference_time : datetime
            The reference time to retrieve episodes before.
        last_n : int, optional
            The number of episodes to retrieve. Defaults to EPISODE_WINDOW_LEN.
        group_ids : list[str | None], optional
            The group ids to return data from.
        source : EpisodeType | None, optional
            Filter episodes by source type.
        driver : GraphDriver | None, optional
            The graph driver to use. If not provided, uses the default driver.
        saga : str | None, optional
            If provided, only retrieve episodes that belong to the saga with this name.

        Returns
        -------
        list[EpisodicNode]
            A list of the most recent EpisodicNode objects.

        Notes
        -----
        The actual retrieval is performed by the `retrieve_episodes` function
        from the `graphiti_core.utils` module, unless a saga is specified.
        """
        # 【注释 97】允许调用方传入临时 driver；未传时默认使用 Graphiti 当前 clients 中的 driver。
        if driver is None:
            driver = self.clients.driver

        if driver.graph_operations_interface:
            try:
                return await driver.graph_operations_interface.retrieve_episodes(
                    driver, reference_time, last_n, group_ids, source, saga
                )
            except NotImplementedError:
                pass

        # 【注释 98】通用回退路径把参数原样交给 maintenance 层的 retrieve_episodes 函数。
        return await retrieve_episodes(driver, reference_time, last_n, group_ids, source, saga)

    # 【注释 99】add_episode 是单条写入主流程：校验参数、准备上下文、抽取节点/边、解析去重、落库并可选更新社区。
    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str | None = None,
        uuid: str | None = None,
        update_communities: bool = False,
        entity_types: dict[str, type[BaseModel]] | None = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
        custom_extraction_instructions: str | None = None,
        saga: str | SagaNode | None = None,
        saga_previous_episode_uuid: str | None = None,
    ) -> AddEpisodeResults:
        """
        Process an episode and update the graph.

        This method extracts information from the episode, creates nodes and edges,
        and updates the graph database accordingly.

        Parameters
        ----------
        name : str
            The name of the episode.
        episode_body : str
            The content of the episode.
        source_description : str
            A description of the episode's source.
        reference_time : datetime
            The reference time for the episode.
        source : EpisodeType, optional
            The type of the episode. Defaults to EpisodeType.message.
        group_id : str | None
            An id for the graph partition the episode is a part of.
        uuid : str | None
            Optional uuid of the episode.
        update_communities : bool
            Optional. Whether to update communities with new node information
        entity_types : dict[str, BaseModel] | None
            Optional. Dictionary mapping entity type names to their Pydantic model definitions.
        excluded_entity_types : list[str] | None
            Optional. List of entity type names to exclude from the graph. Entities classified
            into these types will not be added to the graph. Can include 'Entity' to exclude
            the default entity type.
        previous_episode_uuids : list[str] | None
            Optional.  list of episode uuids to use as the previous episodes. If this is not provided,
            the most recent episodes by created_at date will be used.
        custom_extraction_instructions : str | None
            Optional. Custom extraction instructions string to be included in the extract entities and extract edges prompts.
            This allows for additional instructions or context to guide the extraction process.
        saga : str | SagaNode | None
            Optional. Either a saga name (str) or a SagaNode object to associate this episode with.
            If a string is provided and a saga with this name already exists in the group, the episode
            will be added to it. Otherwise, a new saga will be created. Sagas are connected to episodes
            via HAS_EPISODE edges, and consecutive episodes are linked via NEXT_EPISODE edges.
        saga_previous_episode_uuid : str | None
            Optional. UUID of the previous episode in the saga. If provided, skips the database
            query to find the most recent episode. Useful for efficiently adding multiple episodes
            to the same saga in sequence. The returned AddEpisodeResults.episode.uuid can be passed
            as this parameter for the next episode.

        Returns
        -------
        None

        Notes
        -----
        This method performs several steps including node extraction, edge extraction,
        deduplication, and database updates. It also handles embedding generation
        and edge invalidation.

        It is recommended to run this method as a background process, such as in a queue.
        It's important that each episode is added sequentially and awaited before adding
        the next one. For web applications, consider using FastAPI's background tasks
        or a dedicated task queue like Celery for this purpose.

        Example using FastAPI background tasks:
            @app.post("/add_episode")
            async def add_episode_endpoint(episode_data: EpisodeData):
                background_tasks.add_task(graphiti.add_episode, **episode_data.dict())
                return {"message": "Episode processing started"}
        """
        # 【注释 100】先记录耗时和当前时间；reference_time 表示事件时间，now 表示系统处理时间，两者语义不同。
        start = time()
        now = utc_now()

        # 【注释 101】写入前先校验自定义实体类型和排除类型，避免无效 schema 进入 LLM 抽取阶段。
        validate_entity_types(entity_types)
        validate_excluded_entity_types(excluded_entity_types, entity_types)

        # 【注释 102】group_id 决定图分区/数据库；未传时使用 driver provider 的默认分区。
        if group_id is None:
            # if group_id is None, use the default group id by the provider
            # and the preset database name will be used
            group_id = get_default_group_id(self.driver.provider)
        else:
            validate_group_id(group_id)
            if group_id != self.driver._database:
                # if group_id is provided, use it as the database name
                self.driver = self.driver.clone(database=group_id)
                self.clients.driver = self.driver

        # 【注释 103】从这里开始进入可观测的核心流程，异常和耗时都会记录到同一个 span。
        with self.tracer.start_span('add_episode') as span:
            try:
                # 【注释 104】先拿历史 episode，是因为实体/边抽取需要上下文来判断新信息、旧信息和冲突信息。
                # Retrieve previous episodes for context
                previous_episodes = (
                    await self.retrieve_episodes(
                        reference_time,
                        last_n=RELEVANT_SCHEMA_LIMIT,
                        group_ids=[group_id],
                        source=source,
                    )
                    if previous_episode_uuids is None
                    else await EpisodicNode.get_by_uuids(self.driver, previous_episode_uuids)
                )

                # 【注释 105】如果传入 uuid，说明可能是重跑或补处理，优先读取已有 episode；否则按输入创建新节点。
                # Get or create episode
                episode = (
                    await EpisodicNode.get_by_uuid(self.driver, uuid)
                    if uuid is not None
                    else EpisodicNode(
                        name=name,
                        group_id=group_id,
                        labels=[],
                        source=source,
                        content=episode_body,
                        source_description=source_description,
                        created_at=now,
                        valid_at=reference_time,
                    )
                )

                # 【注释 106】edge_type_map 决定哪些实体类型之间允许抽取哪些边；未传时默认允许 Entity-Entity 的通用关系。
                # Create default edge type map
                edge_type_map_default = (
                    {('Entity', 'Entity'): list(edge_types.keys())}
                    if edge_types is not None
                    else {('Entity', 'Entity'): []}
                )

                # 【注释 107】节点先于边处理：只有实体稳定后，事实边的 source/target 才能稳定。
                # Extract and resolve nodes
                extracted_nodes, node_episode_index_map = await extract_nodes(
                    self.clients,
                    episode,
                    previous_episodes,
                    entity_types,
                    excluded_entity_types,
                    custom_extraction_instructions,
                )

                # 【注释 108】这里忽略 duplicates，但保留 uuid_map；后续边端点必须根据这个映射修正。
                nodes, uuid_map, _ = await resolve_extracted_nodes(
                    self.clients,
                    extracted_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                )

                # 【注释 109】这段实际先解析边，再用 new_edges 辅助节点属性摘要，避免旧事实反复写入节点 summary。
                # Extract and resolve edges in parallel with attribute extraction
                (
                    resolved_edges,
                    invalidated_edges,
                    new_edges,
                ) = await self._extract_and_resolve_edges(
                    episode,
                    extracted_nodes,
                    previous_episodes,
                    edge_type_map or edge_type_map_default,
                    group_id,
                    edge_types,
                    nodes,
                    uuid_map,
                    custom_extraction_instructions,
                )

                # 【注释 110】保存时把新解析出的边和被新事实失效的边都带上，让图谱能表达事实随时间变化。
                entity_edges = resolved_edges + invalidated_edges

                # 【注释 111】节点属性补全依赖最终节点集合和新增事实，目的是增强节点内容而不是复制已有事实。
                # Extract node attributes - only pass new edges for summary generation
                # to avoid duplicating facts that already exist in the graph
                hydrated_nodes = await extract_attributes_from_nodes(
                    self.clients,
                    nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                    edges=new_edges,
                )

                # 【注释 112】所有抽取/解析完成后再统一写图，并在同一阶段处理 saga 归属，保证引用的 uuid 都已稳定。
                # Process and save episode data (including saga association if provided)
                episodic_edges, episode = await self._process_episode_data(
                    episode,
                    hydrated_nodes,
                    entity_edges,
                    now,
                    group_id,
                    saga,
                    saga_previous_episode_uuid,
                    node_episode_index_map,
                )

                # 【注释 113】社区更新是可选的高成本操作，只在调用方显式要求时根据本次涉及的节点增量更新。
                # Update communities if requested
                communities = []
                community_edges = []
                if update_communities:
                    communities, community_edges = await semaphore_gather(
                        *[
                            update_community(self.driver, self.llm_client, self.embedder, node)
                            for node in nodes
                        ],
                        max_coroutines=self.max_coroutines,
                    )

                end = time()

                # 【注释 114】核心流程结束后写入观测指标：数量、耗时和配置能帮助排查抽取质量或性能问题。
                # Add span attributes
                span.add_attributes(
                    {
                        'episode.uuid': episode.uuid,
                        'episode.source': source.value,
                        'episode.reference_time': reference_time.isoformat(),
                        'group_id': group_id,
                        'node.count': len(hydrated_nodes),
                        'edge.count': len(entity_edges),
                        'edge.invalidated_count': len(invalidated_edges),
                        'previous_episodes.count': len(previous_episodes),
                        'entity_types.count': len(entity_types) if entity_types else 0,
                        'edge_types.count': len(edge_types) if edge_types else 0,
                        'update_communities': update_communities,
                        'communities.count': len(communities) if update_communities else 0,
                        'duration_ms': (end - start) * 1000,
                    }
                )

                logger.info(f'Completed add_episode in {(end - start) * 1000} ms')

                # 【注释 115】最后返回结构化结果，调用方可以继续使用新 episode.uuid、节点和边做后续处理。
                return AddEpisodeResults(
                    episode=episode,
                    episodic_edges=episodic_edges,
                    nodes=hydrated_nodes,
                    edges=entity_edges,
                    communities=communities,
                    community_edges=community_edges,
                )

            # 【注释 116】异常路径先标记 span，再重新抛出，既不吞错误，也保留可观测性信息。
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise e

    # 【注释 117】add_episode_bulk 是批量写入主流程：尽量把 I/O、抽取和去重批处理化，降低大量 episode 写入的总成本。
    async def add_episode_bulk(
        self,
        bulk_episodes: list[RawEpisode],
        group_id: str | None = None,
        entity_types: dict[str, type[BaseModel]] | None = None,
        excluded_entity_types: list[str] | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
        custom_extraction_instructions: str | None = None,
        saga: str | SagaNode | None = None,
    ) -> AddBulkEpisodeResults:
        """
        Process multiple episodes in bulk and update the graph.

        This method extracts information from multiple episodes, creates nodes and edges,
        and updates the graph database accordingly, all in a single batch operation.

        Parameters
        ----------
        bulk_episodes : list[RawEpisode]
            A list of RawEpisode objects to be processed and added to the graph.
        group_id : str | None
            An id for the graph partition the episode is a part of.
        entity_types : dict[str, type[BaseModel]] | None
            Optional. A dictionary mapping entity type names to Pydantic models.
        excluded_entity_types : list[str] | None
            Optional. A list of entity type names to exclude from extraction.
        edge_types : dict[str, type[BaseModel]] | None
            Optional. A dictionary mapping edge type names to Pydantic models.
        edge_type_map : dict[tuple[str, str], list[str]] | None
            Optional. A mapping of (source_type, target_type) to allowed edge types.
        custom_extraction_instructions : str | None
            Optional. Custom extraction instructions string to be included in the
            extract entities and extract edges prompts. This allows for additional
            instructions or context to guide the extraction process.
        saga : str | SagaNode | None
            Optional. Either a saga name (str) or a SagaNode object to associate all episodes with.
            If a string is provided and a saga with this name already exists in the group, the episodes
            will be added to it. Otherwise, a new saga will be created. Sagas are connected to episodes
            via HAS_EPISODE edges, and consecutive episodes are linked via NEXT_EPISODE edges.

        Returns
        -------
        AddBulkEpisodeResults

        Notes
        -----
        This method performs several steps including:
        - Saving all episodes to the database
        - Retrieving previous episode context for each new episode
        - Extracting nodes and edges from all episodes
        - Generating embeddings for nodes and edges
        - Deduplicating nodes and edges
        - Saving nodes, episodic edges, and entity edges to the knowledge graph

        This bulk operation is designed for efficiency when processing multiple episodes
        at once. However, it's important to ensure that the bulk operation doesn't
        overwhelm system resources. Consider implementing rate limiting or chunking for
        very large batches of episodes.

        Edge invalidation and date extraction (``valid_at`` / ``invalid_at``) are
        performed in the bulk path as well: edges flow through ``extract_edges`` and
        ``resolve_extracted_edges`` just like in ``add_episode``, and any invalidated
        edges are persisted alongside the newly resolved ones.
        """
        # 【注释 118】批处理使用独立 span，并先记录 episode.count，便于和单条写入的性能指标区分。
        with self.tracer.start_span('add_episode_bulk') as bulk_span:
            bulk_span.add_attributes({'episode.count': len(bulk_episodes)})

            try:
                # 【注释 119】批处理同样区分处理时间 now 和每条 episode 自己的 reference_time。
                start = time()
                now = utc_now()

                # if group_id is None, use the default group id by the provider
                # 【注释 120】批量写入要求整批 episode 属于同一 group/database；必要时克隆 driver 指向目标数据库。
                if group_id is None:
                    group_id = get_default_group_id(self.driver.provider)
                else:
                    validate_group_id(group_id)
                    if group_id != self.driver._database:
                        # if group_id is provided, use it as the database name
                        self.driver = self.driver.clone(database=group_id)
                        self.clients.driver = self.driver

                # 【注释 121】批量路径沿用单条路径的边类型默认策略，保证两种入口的 schema 行为一致。
                # Create default edge type map
                edge_type_map_default = (
                    {('Entity', 'Entity'): list(edge_types.keys())}
                    if edge_types is not None
                    else {('Entity', 'Entity'): []}
                )

                # 【注释 122】先把 RawEpisode 转成 EpisodicNode；有 uuid 的走复用，无 uuid 的创建新节点。
                episodes = [
                    await EpisodicNode.get_by_uuid(self.driver, episode.uuid)
                    if episode.uuid is not None
                    else EpisodicNode(
                        name=episode.name,
                        labels=[],
                        source=episode.source,
                        content=episode.content,
                        source_description=episode.source_description,
                        group_id=group_id,
                        created_at=now,
                        valid_at=episode.reference_time,
                    )
                    for episode in bulk_episodes
                ]

                # 【注释 123】先保存 episode 本体，是为了让后续检索上下文和构建来源关系时有稳定的 episode uuid。
                # Save all episodes
                await add_nodes_and_edges_bulk(
                    driver=self.driver,
                    episodic_nodes=episodes,
                    episodic_edges=[],
                    entity_nodes=[],
                    entity_edges=[],
                    embedder=self.embedder,
                )

                # 【注释 124】为每条新 episode 分别取历史上下文，避免批量处理时丢失时间邻近信息。
                # Get previous episode context for each episode
                episode_context = await retrieve_previous_episodes_bulk(self.driver, episodes)

                # 【注释 125】批量抽取后立刻对节点去重，减少后续解析和属性抽取的重复工作。
                # Extract and dedupe nodes and edges
                (
                    nodes_by_episode,
                    uuid_map,
                    extracted_edges_bulk,
                ) = await self._extract_and_dedupe_nodes_bulk(
                    episode_context,
                    edge_type_map or edge_type_map_default,
                    edge_types,
                    entity_types,
                    excluded_entity_types,
                    custom_extraction_instructions,
                )

                # 【注释 126】根据每个 episode 的节点结果生成来源边，保证批量写入也能追溯实体来自哪条输入。
                # Create Episodic Edges
                episodic_edges: list[EpisodicEdge] = []
                for episode_uuid, nodes in nodes_by_episode.items():
                    episodic_edges.extend(build_episodic_edges(nodes, episode_uuid, now))

                # 【注释 127】节点 uuid 修正后再对边去重，避免同一事实因临时端点不同被保留多次。
                # Re-map edge pointers and dedupe edges
                extracted_edges_bulk_updated: list[list[EntityEdge]] = [
                    resolve_edge_pointers(edges, uuid_map) for edges in extracted_edges_bulk
                ]

                edges_by_episode = await dedupe_edges_bulk(
                    self.clients,
                    extracted_edges_bulk_updated,
                    episode_context,
                    [],
                    edge_types or {},
                    edge_type_map or edge_type_map_default,
                )

                # 【注释 128】这一步把本批候选知识与已有图谱合并，同时找出被新信息 invalidated 的旧边。
                # Resolve nodes and edges against the existing graph
                (
                    final_hydrated_nodes,
                    resolved_edges,
                    invalidated_edges,
                    final_uuid_map,
                ) = await self._resolve_nodes_and_edges_bulk(
                    nodes_by_episode,
                    edges_by_episode,
                    episode_context,
                    entity_types,
                    edge_types,
                    edge_type_map or edge_type_map_default,
                    episodes,
                )

                # 【注释 129】节点解析后，episode-to-entity 边也要修正端点，确保来源边指向最终实体节点。
                # Resolved pointers for episodic edges
                resolved_episodic_edges = resolve_edge_pointers(episodic_edges, final_uuid_map)

                # 【注释 130】最终一次性写入 episode、来源边、实体节点和事实边，这是批量路径的主要性能收益。
                # save data to KG
                await add_nodes_and_edges_bulk(
                    self.driver,
                    episodes,
                    resolved_episodic_edges,
                    final_hydrated_nodes,
                    resolved_edges + invalidated_edges,
                    self.embedder,
                )

                # 【注释 131】批量 saga 关联需要额外维护顺序链：不仅要建立归属，还要在整批 episode 内按时间连接。
                # Handle saga association if provided
                if saga is not None:
                    # Get or create saga node based on input type
                    # 【注释 132】传入 saga 名时按需创建；批量创建的 saga created_at 使用最早 episode 的事件时间。
                    if isinstance(saga, str):
                        # Anchor a newly minted saga to the earliest episode
                        # reference time in the bulk, so created_at reflects the
                        # episode window rather than the time this run started.
                        valid_ats = [ep.valid_at for ep in episodes if ep.valid_at is not None]
                        saga_created_at = min(valid_ats) if valid_ats else now
                        saga_node = await self._get_or_create_saga(saga, group_id, saga_created_at)
                    else:
                        saga_node = saga

                    # 【注释 133】排序后再建 NEXT_EPISODE，避免输入列表顺序和真实事件时间不一致导致故事线错乱。
                    # Sort episodes by valid_at to create NEXT_EPISODE chain in correct order
                    sorted_episodes = sorted(episodes, key=lambda e: e.valid_at)

                    # 【注释 134】先找到 saga 中已有的最后一条 episode，才能把本批第一条正确接到旧链尾部。
                    # Find the most recent episode already in the saga
                    previous_episode_uuid = await self._saga_get_previous_episode_uuid(
                        saga_node.uuid, ''
                    )

                    # 【注释 135】循环中每处理一条就更新 previous_episode_uuid，从而把本批 episode 串成连续链。
                    for episode in sorted_episodes:
                        # Create NEXT_EPISODE edge from the previous episode
                        if previous_episode_uuid is not None:
                            next_episode_edge = NextEpisodeEdge(
                                source_node_uuid=previous_episode_uuid,
                                target_node_uuid=episode.uuid,
                                group_id=group_id,
                                created_at=now,
                            )
                            await next_episode_edge.save(self.driver)

                        # Create HAS_EPISODE edge from saga to episode
                        has_episode_edge = HasEpisodeEdge(
                            source_node_uuid=saga_node.uuid,
                            target_node_uuid=episode.uuid,
                            group_id=group_id,
                            created_at=now,
                        )
                        await has_episode_edge.save(self.driver)

                        # Update previous_episode_uuid for the next iteration
                        previous_episode_uuid = episode.uuid

                    # 【注释 136】批量结束后更新 saga 首尾缓存；first 只在为空时设置，last 总是推进到本批最后一条。
                    # Track first and last episode on the saga node
                    if sorted_episodes:
                        if saga_node.first_episode_uuid is None:
                            saga_node.first_episode_uuid = sorted_episodes[0].uuid
                        saga_node.last_episode_uuid = sorted_episodes[-1].uuid
                        await saga_node.save(self.driver)

                end = time()

                # 【注释 137】批量观测指标重点记录 group、节点数、边数和总耗时，用于评估批处理效率。
                # Add span attributes
                bulk_span.add_attributes(
                    {
                        'group_id': group_id,
                        'node.count': len(final_hydrated_nodes),
                        'edge.count': len(resolved_edges + invalidated_edges),
                        'duration_ms': (end - start) * 1000,
                    }
                )

                logger.info(f'Completed add_episode_bulk in {(end - start) * 1000} ms')

                # 【注释 138】批量返回包含所有 episode 和最终写入的节点/边；社区字段保持空，因为该路径没有更新社区。
                return AddBulkEpisodeResults(
                    episodes=episodes,
                    episodic_edges=resolved_episodic_edges,
                    nodes=final_hydrated_nodes,
                    edges=resolved_edges + invalidated_edges,
                    communities=[],
                    community_edges=[],
                )

            # 【注释 139】批量异常同样写入 span 后抛出，调用方可以决定是否重试整批或拆分处理。
            except Exception as e:
                bulk_span.set_status('error', str(e))
                bulk_span.record_exception(e)
                raise e

    # 【注释 140】build_communities 也支持多 group：社区可以按指定分区构建，也可以覆盖整个图。
    @handle_multiple_group_ids
    # 【注释 141】社区构建是全局维护操作：先清理旧社区，再基于当前实体图重新聚类和摘要。
    async def build_communities(
        self, group_ids: list[str] | None = None, driver: GraphDriver | None = None
    ) -> tuple[list[CommunityNode], list[CommunityEdge]]:
        """
        Use a community clustering algorithm to find communities of nodes. Create community nodes summarising
        the content of these communities.
        ----------
        group_ids : list[str] | None
            Optional. Create communities only for the listed group_ids. If blank the entire graph will be used.
        """
        # 【注释 142】可选 driver 让维护任务能针对指定后端运行；默认仍使用当前实例 driver。
        if driver is None:
            driver = self.clients.driver

        # 【注释 143】先删除旧社区，避免新旧聚类结果并存造成搜索或展示时的重复/冲突。
        # Clear existing communities
        await remove_communities(driver)

        # 【注释 144】底层 build_communities 负责真正的聚类与摘要生成，Graphiti 负责后续嵌入和保存。
        community_nodes, community_edges = await build_communities(
            driver, self.llm_client, group_ids
        )

        await semaphore_gather(
            *[node.generate_name_embedding(self.embedder) for node in community_nodes],
            max_coroutines=self.max_coroutines,
        )

        await semaphore_gather(
            *[node.save(driver) for node in community_nodes],
            max_coroutines=self.max_coroutines,
        )
        await semaphore_gather(
            *[edge.save(driver) for edge in community_edges],
            max_coroutines=self.max_coroutines,
        )

        return community_nodes, community_edges

    # 【注释 145】search 同样经过 group_id 装饰器，使同一个查询可以限定在一个或多个图分区。
    @handle_multiple_group_ids
    # 【注释 146】search 是简单事实检索入口：返回 EntityEdge 列表，适合只需要相关事实而不需要完整图对象的场景。
    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results=DEFAULT_SEARCH_LIMIT,
        search_filter: SearchFilters | None = None,
        driver: GraphDriver | None = None,
    ) -> list[EntityEdge]:
        """
        Perform a hybrid search on the knowledge graph.

        This method executes a search query on the graph, combining vector and
        text-based search techniques to retrieve relevant facts, returning the edges as a string.

        This is our basic out-of-the-box search, for more robust results we recommend using our more advanced
        search method graphiti.search_().

        Parameters
        ----------
        query : str
            The search query string.
        center_node_uuid: str, optional
            Facts will be reranked based on proximity to this node
        group_ids : list[str | None] | None, optional
            The graph partitions to return data from.
        num_results : int, optional
            The maximum number of results to return. Defaults to 10.

        Returns
        -------
        list
            A list of EntityEdge objects that are relevant to the search query.

        Notes
        -----
        This method uses a SearchConfig with num_episodes set to 0 and
        num_results set to the provided num_results parameter.

        The search is performed using the current date and time as the reference
        point for temporal relevance.
        """
        # 【注释 147】是否传 center_node_uuid 决定重排策略：无中心节点用 RRF，有中心节点则考虑图距离。
        search_config = (
            EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
        # 【注释 148】这里直接修改 config 的 limit，使外部 num_results 参数影响底层搜索结果数量。
        search_config.limit = num_results

        # 【注释 149】底层 search 返回 SearchResults；这个简化接口只取其中的 edges。
        edges = (
            await search(
                self.clients,
                query,
                group_ids,
                search_config,
                search_filter if search_filter is not None else SearchFilters(),
                driver=driver,
                center_node_uuid=center_node_uuid,
            )
        ).edges

        return edges

    # 【注释 150】_search 是兼容旧接口的薄包装，所有实际逻辑都转发给新的 search_。
    async def _search(
        self,
        query: str,
        config: SearchConfig,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults:
        """DEPRECATED"""
        # 【注释 151】保留该方法可以减少破坏性变更，让旧调用方逐步迁移。
        return await self.search_(
            query, config, group_ids, center_node_uuid, bfs_origin_node_uuids, search_filter
        )

    # 【注释 152】search_ 是高级检索入口，也支持多 group，并保留更多搜索配置能力。
    @handle_multiple_group_ids
    # 【注释 153】search_ 返回完整 SearchResults，调用方可以同时拿到节点、边和更复杂的检索结果。
    async def search_(
        self,
        query: str,
        config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
        driver: GraphDriver | None = None,
    ) -> SearchResults:
        """search_ (replaces _search) is our advanced search method that returns Graph objects (nodes and edges) rather
        than a list of facts. This endpoint allows the end user to utilize more advanced features such as filters and
        different search and reranker methodologies across different layers in the graph.

        For different config recipes refer to search/search_config_recipes.
        """

        return await search(
            self.clients,
            query,
            group_ids,
            config,
            search_filter if search_filter is not None else SearchFilters(),
            center_node_uuid,
            bfs_origin_node_uuids,
            driver=driver,
        )

    # 【注释 154】该方法从 episode 反查它提到的节点和关联的事实边，用于解释某条输入贡献了哪些图谱内容。
    async def get_nodes_and_edges_by_episode(self, episode_uuids: list[str]) -> SearchResults:
        # 【注释 155】先取 episode 节点，因为实体边 uuid 存在 episode.entity_edges 中。
        episodes = await EpisodicNode.get_by_uuids(self.driver, episode_uuids)

        # 【注释 156】每个 episode 的边可以并发读取，最后再拍平成一个边列表。
        edges_list = await semaphore_gather(
            *[EntityEdge.get_by_uuids(self.driver, episode.entity_edges) for episode in episodes],
            max_coroutines=self.max_coroutines,
        )

        edges: list[EntityEdge] = [edge for lst in edges_list for edge in lst]

        # 【注释 157】节点不从 entity_edges 反推，而是通过 MENTIONS 关系获取，能保留 episode 对实体的直接来源关系。
        nodes = await get_mentioned_nodes(self.driver, episodes)

        return SearchResults(edges=edges, nodes=nodes)

    # 【注释 158】add_triplet 是手工事实写入入口：调用方直接给 source、edge、target，系统负责补嵌入、解析冲突并落库。
    async def add_triplet(
        self, source_node: EntityNode, edge: EntityEdge, target_node: EntityNode
    ) -> AddTripletResults:
        # 【注释 159】写入前先确保两个端点和事实边都有嵌入，便于后续检索和去重。
        if source_node.name_embedding is None:
            await source_node.generate_name_embedding(self.embedder)
        if target_node.name_embedding is None:
            await target_node.generate_name_embedding(self.embedder)
        if edge.fact_embedding is None:
            await edge.generate_embedding(self.embedder)

        # 【注释 160】先尝试按 uuid 找已有 source 节点；找不到时再走节点解析，避免无谓创建重复实体。
        try:
            resolved_source = await EntityNode.get_by_uuid(self.driver, source_node.uuid)
        except NodeNotFoundError:
            resolved_source_nodes, _, _ = await resolve_extracted_nodes(
                self.clients,
                [source_node],
            )
            resolved_source = resolved_source_nodes[0]

        # 【注释 161】target 节点使用同样策略，确保手工传入的端点也会与已有图谱对齐。
        try:
            resolved_target = await EntityNode.get_by_uuid(self.driver, target_node.uuid)
        except NodeNotFoundError:
            resolved_target_nodes, _, _ = await resolve_extracted_nodes(
                self.clients,
                [target_node],
            )
            resolved_target = resolved_target_nodes[0]

        # 【注释 162】后续只保存解析后的端点节点，原始节点对象只作为属性补充来源。
        nodes = [resolved_source, resolved_target]

        # 【注释 163】用户传入的 attributes、summary、labels 会合并进解析后的节点，既保留已有实体身份，也吸收新属性。
        # Merge user-provided properties from original nodes into resolved nodes (excluding uuid)
        # Update attributes dictionary (merge rather than replace)
        if source_node.attributes:
            resolved_source.attributes.update(source_node.attributes)
        if target_node.attributes:
            resolved_target.attributes.update(target_node.attributes)

        # Update summary if provided by user (non-empty string)
        if source_node.summary:
            resolved_source.summary = source_node.summary
        if target_node.summary:
            resolved_target.summary = target_node.summary

        # Update labels (merge with existing)
        if source_node.labels:
            resolved_source.labels = list(set(resolved_source.labels) | set(source_node.labels))
        if target_node.labels:
            resolved_target.labels = list(set(resolved_target.labels) | set(target_node.labels))

        # 【注释 164】边端点必须改成解析后的 uuid，否则可能指向临时或重复节点。
        edge.source_node_uuid = resolved_source.uuid
        edge.target_node_uuid = resolved_target.uuid

        # 【注释 165】边 uuid 冲突时要特别处理：同 uuid 但端点不同不能覆盖旧边，只能生成新 uuid。
        # Check if an edge with this UUID already exists with different source/target nodes.
        # If so, generate a new UUID to create a new edge instead of overwriting.
        try:
            existing_edge = await EntityEdge.get_by_uuid(self.driver, edge.uuid)
            # Edge exists - check if source/target nodes match
            if (
                existing_edge.source_node_uuid != edge.source_node_uuid
                or existing_edge.target_node_uuid != edge.target_node_uuid
            ):
                # Source/target mismatch - generate new UUID to create a new edge
                old_uuid = edge.uuid
                edge.uuid = str(uuid4())
                logger.info(
                    f'Edge UUID {old_uuid} already exists with different source/target nodes. '
                    f'Generated new UUID {edge.uuid} to avoid overwriting.'
                )
        except EdgeNotFoundError:
            # Edge doesn't exist yet, proceed normally
            pass

        # 【注释 166】先取两个端点之间已有边，作为判断当前事实是否重复或冲突的局部上下文。
        valid_edges = await EntityEdge.get_between_nodes(
            self.driver, edge.source_node_uuid, edge.target_node_uuid
        )

        # 【注释 167】related_edges 限定在两端点已有边中搜索，用于判断同一对实体之间的相似事实。
        related_edges = (
            await search(
                self.clients,
                edge.fact,
                group_ids=[edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
            )
        ).edges
        # 【注释 168】existing_edges 在整个 group 内搜索，用于发现更广范围内可能重复或冲突的事实。
        existing_edges = (
            await search(
                self.clients,
                edge.fact,
                group_ids=[edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(),
            )
        ).edges

        # 【注释 169】即使是手工 triplet，也复用自动抽取边的解析逻辑，保证 invalidation 规则一致。
        resolved_edge, invalidated_edges, _ = await resolve_extracted_edge(
            self.llm_client,
            edge,
            related_edges,
            existing_edges,
            EpisodicNode(
                name='',
                source=EpisodeType.text,
                source_description='',
                content='',
                valid_at=edge.valid_at or utc_now(),
                entity_edges=[],
                group_id=edge.group_id,
            ),
            None,
        )

        # 【注释 170】最终要写入的不只是新边，还包括因为该事实而失效的旧边。
        edges: list[EntityEdge] = [resolved_edge] + invalidated_edges

        # 【注释 171】解析后的边和节点再统一生成嵌入，确保写入图谱的数据都可被搜索。
        await create_entity_edge_embeddings(self.embedder, edges)
        await create_entity_node_embeddings(self.embedder, nodes)

        # 【注释 172】triplet 没有 episode 来源，因此只写实体节点和实体边，episode/episodic_edges 参数为空。
        await add_nodes_and_edges_bulk(self.driver, [], [], nodes, edges, self.embedder)
        return AddTripletResults(edges=edges, nodes=nodes)

    # 【注释 173】remove_episode 只删除由该 episode 独占产生的内容，避免误删其他 episode 仍在引用的知识。
    async def remove_episode(self, episode_uuid: str):
        # 【注释 174】删除从 episode 本体开始，因为后续需要它记录的 entity_edges 和 MENTIONS 关系。
        # Find the episode to be deleted
        episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)

        # 【注释 175】先读取 episode 关联的事实边，但并不是所有关联边都能删除。
        # Find edges mentioned by the episode
        edges = await EntityEdge.get_by_uuids(self.driver, episode.entity_edges)

        # 【注释 176】只有 edge.episodes 的第一条就是当前 episode 时，才认为该边由它创建，可安全删除。
        # We should only delete edges created by the episode
        edges_to_delete: list[EntityEdge] = []
        for edge in edges:
            if edge.episodes and edge.episodes[0] == episode.uuid:
                edges_to_delete.append(edge)

        # 【注释 177】节点删除更谨慎：先找到该 episode 提到的节点，再确认这些节点没有被其他 episode 提到。
        # Find nodes mentioned by the episode
        nodes = await get_mentioned_nodes(self.driver, [episode])
        # We should delete all nodes that are only mentioned in the deleted episode
        nodes_to_delete: list[EntityNode] = []
        for node in nodes:
            # 【注释 178】这里用计数判断节点是否只有当前 episode 一个来源，防止删除共享实体。
            query: LiteralString = 'MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN count(*) AS episode_count'
            records, _, _ = await self.driver.execute_query(query, uuid=node.uuid, routing_='r')

            for record in records:
                if record['episode_count'] == 1:
                    nodes_to_delete.append(node)

        # 【注释 179】先删边再删节点，符合图数据库中删除节点前应先清理相关边的常见约束。
        await Edge.delete_by_uuids(self.driver, [edge.uuid for edge in edges_to_delete])
        await Node.delete_by_uuids(self.driver, [node.uuid for node in nodes_to_delete])

        # 【注释 180】最后删除 episode 本身；此时它独占的边和节点已经清理完毕。
        await episode.delete(self.driver)
