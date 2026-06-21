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

# 这个文件属于 maintenance 层的图数据操作工具函数。
# 它不直接代表某一种数据库 driver 的实现，而是提供一套“通用回退实现”：
# 当具体 driver 没有提供 graph_operations_interface，或者接口方法没有实现时，
# Graphiti 会退回到这里，用 driver.execute_query/session 去执行 Cypher 查询。
import logging
from datetime import datetime

# LiteralString 用来标注后面拼出来的 query 字符串。
# 它主要是类型层面的辅助，提示这个变量应该是字面量/受控字符串，
# 但这里仍然会通过字符串拼接动态加入过滤条件和 RETURN 子句。
from typing_extensions import LiteralString

# GraphDriver 是抽象的图数据库 driver 类型。
# 这里的函数不关心 driver 具体是 Neo4j、FalkorDB、Kuzu 还是 Neptune，
# 只依赖它暴露出来的统一能力，比如 session()、execute_query()、provider。
from graphiti_core.driver.driver import GraphDriver, GraphProvider

# EPISODIC_NODE_RETURN 是把数据库里的 Episodic 节点字段映射成统一返回结构的 Cypher RETURN 片段。
# Neptune 的返回格式和普通 Cypher provider 有差异，所以单独准备了 EPISODIC_NODE_RETURN_NEPTUNE。
from graphiti_core.models.nodes.node_db_queries import (
    EPISODIC_NODE_RETURN,
    EPISODIC_NODE_RETURN_NEPTUNE,
)

# EpisodeType 用来过滤 episode 的来源类型。
# EpisodicNode 是最终返回的 Python 节点对象。
# get_episodic_node_from_record 负责把数据库 record 转成 EpisodicNode 实例。
from graphiti_core.nodes import EpisodeType, EpisodicNode, get_episodic_node_from_record

# 默认只取最近 3 个 episode。
# 上层 Graphiti.retrieve_episodes 如果没有显式传 last_n，也会使用这个默认窗口。
EPISODE_WINDOW_LEN = 3

logger = logging.getLogger(__name__)


async def clear_data(driver: GraphDriver, group_ids: list[str] | None = None):
    # clear_data 先尝试走 driver 自己挂载的 graph_operations_interface。
    # 这是一个扩展点：如果某个数据库 provider 有更合适、更高效或更特殊的清理实现，
    # 可以在自己的 interface 里重写 clear_data。
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.clear_data(driver, group_ids)
        except NotImplementedError:
            # 如果 interface 存在但 clear_data 没有实现，就不要让整个流程失败。
            # 继续往下走本文件里的通用清理逻辑。
            pass

    # 没有 provider-specific 实现时，使用 driver.session() 打开一个数据库 session。
    # 不同 driver 的 session 具体实现不同，但都需要支持 async context manager。
    async with driver.session() as session:

        async def delete_all(tx):
            # group_ids 为 None 时，表示清空整个图。
            # DETACH DELETE 会删除节点以及与节点相连的关系。
            await tx.run('MATCH (n) DETACH DELETE n')

        async def delete_group_ids(tx):
            # group_ids 不为 None 时，只清理指定 group_id 下的数据。
            # 这里先列出常规业务节点标签：Entity、Episodic、Community。
            labels = ['Entity', 'Episodic', 'Community']

            # Kuzu provider 额外使用 RelatesToNode_ 这样的节点标签来表示关系相关结构，
            # 所以清理 group 数据时也要把它纳入删除范围。
            if driver.provider == GraphProvider.KUZU:
                labels.append('RelatesToNode_')

            # 对每一种标签分别执行删除。
            # 注意这里删除的是 group_id 属于传入 group_ids 的节点，
            # 因此它是“按租户/分组清理”，不是全图清理。
            for label in labels:
                await tx.run(
                    f"""
                    MATCH (n:{label})
                    WHERE n.group_id IN $group_ids
                    DETACH DELETE n
                    """,
                    group_ids=group_ids,
                )

        # 根据是否传入 group_ids 决定清理范围。
        # group_ids is None：清空整个图。
        # group_ids 有值：只清理这些 group 里的节点和关系。
        if group_ids is None:
            await session.execute_write(delete_all)
        else:
            await session.execute_write(delete_group_ids)


async def retrieve_episodes(
    driver: GraphDriver,
    reference_time: datetime,
    last_n: int = EPISODE_WINDOW_LEN,
    group_ids: list[str] | None = None,
    source: EpisodeType | None = None,
    saga: str | None = None,
) -> list[EpisodicNode]:
    """
    Retrieve the last n episodic nodes from the graph.

    Args:
        driver (Driver): The Neo4j driver instance.
        reference_time (datetime): The reference time to filter episodes. Only episodes with a valid_at timestamp
                                   less than or equal to this reference_time will be retrieved. This allows for
                                   querying the graph's state at a specific point in time.
        last_n (int, optional): The number of most recent episodes to retrieve, relative to the reference_time.
        group_ids (list[str], optional): The list of group ids to return data from.
        source (EpisodeType, optional): Filter episodes by source type.
        saga (str, optional): If provided, only retrieve episodes that belong to the saga with this name.

    Returns:
        list[EpisodicNode]: A list of EpisodicNode objects representing the retrieved episodes.
    """
    # 这个函数是 maintenance 层的通用 retrieve_episodes 实现。
    # 但它依然先尊重 driver.graph_operations_interface：
    # 如果当前 driver 提供了专门的 retrieve_episodes 实现，就优先使用专门实现。
    # 这样可以让某些 provider 绕过通用 Cypher，使用更适配自己的查询方式。
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.retrieve_episodes(
                driver, reference_time, last_n, group_ids, source, saga
            )
        except NotImplementedError:
            # interface 只是扩展点，不保证每个方法都被具体实现。
            # 如果 retrieve_episodes 没实现，就继续使用下面的通用查询逻辑。
            pass

    # If saga is provided, retrieve episodes from that saga only
    # 如果传入 saga，就不再从所有 Episodic 节点里直接按 group/source 查最近 episode，
    # 而是先找到指定 Saga 节点，再沿着 (:Saga)-[:HAS_EPISODE]->(:Episodic) 关系取 episode。
    # 这表示“只检索某条 saga/会话/任务链路下的历史 episode”。
    if saga is not None:
        # saga 分支目前只取 group_ids 的第一个 group_id。
        # 因为 Saga 节点匹配条件是 name + group_id，
        # 不是使用 e.group_id IN $group_ids 这种多 group 查询。
        group_id = group_ids[0] if group_ids else None

        # source 是可选过滤条件。
        # 如果 source 不为空，就在 WHERE 子句后面追加 AND e.source = $source；
        # 如果 source 为空，则不限制 episode 来源。
        source_filter = 'AND e.source = $source' if source is not None else ''

        # 这里的查询逻辑是：
        # 1. 匹配名字为 saga_name 且 group_id 相同的 Saga 节点；
        # 2. 沿 HAS_EPISODE 关系找到它包含的 Episodic 节点；
        # 3. 只保留 valid_at <= reference_time 的 episode；
        # 4. 如果传了 source，再按 source 过滤；
        # 5. 按 valid_at 倒序取最近 last_n 条。
        records, _, _ = await driver.execute_query(
            f"""
            MATCH (s:Saga {{name: $saga_name, group_id: $group_id}})-[:HAS_EPISODE]->(e:Episodic)
            WHERE e.valid_at <= $reference_time
            {source_filter}
            RETURN
            """
            # Neptune 的返回字段格式不同，所以这里根据 provider 切换 RETURN 片段。
            # 非 Neptune provider 使用通用 EPISODIC_NODE_RETURN。
            + (
                EPISODIC_NODE_RETURN_NEPTUNE
                if driver.provider == GraphProvider.NEPTUNE
                else EPISODIC_NODE_RETURN
            )
            + """
            ORDER BY e.valid_at DESC
            LIMIT $num_episodes
            """,
            saga_name=saga,
            group_id=group_id,
            reference_time=reference_time,
            source=source.name if source else None,
            num_episodes=last_n,
        )

        # 数据库返回的是 record，业务层需要的是 EpisodicNode 对象。
        # 因此这里逐条转换。
        episodes = [get_episodic_node_from_record(record) for record in records]

        # 查询时为了高效取“最近 N 条”，按 valid_at DESC 倒序取。
        # 但返回给上层时，Graphiti 希望上下文按时间正序排列：
        # oldest first，也就是旧 episode 在前，新 episode 在后。
        return list(reversed(episodes))  # Return in chronological order

    # saga 没有传入时，走普通 episode 检索路径：
    # 直接从所有 Episodic 节点中查 valid_at <= reference_time 的最近 episode，
    # 再根据 group_ids 和 source 追加过滤条件。
    query_params: dict = {}
    query_filter = ''

    # 如果传了 group_ids，就只返回这些 group 下的 episode。
    # 这里使用 IN，是因为普通路径支持多个 group_id。
    if group_ids and len(group_ids) > 0:
        query_filter += '\nAND e.group_id IN $group_ids'
        query_params['group_ids'] = group_ids

    # 如果传了 source，就只返回指定来源类型的 episode。
    # source 是 EpisodeType 枚举，写入查询参数时使用 source.name。
    if source is not None:
        query_filter += '\nAND e.source = $source'
        query_params['source'] = source.name

    # 构造普通 episode 查询。
    # 基础条件永远是 e.valid_at <= reference_time：
    # 这保证检索的是 reference_time 这个时间点之前已经“有效”的 episode，
    # 用来支持基于时间点的上下文回看。
    query: LiteralString = (
        """
                                    MATCH (e:Episodic)
                                    WHERE e.valid_at <= $reference_time
                                    """
        + query_filter
        + """
        RETURN
        """
        # 不同 provider 的 record shape 不一样，
        # 所以 Neptune 和其他 provider 使用不同 RETURN 片段。
        + (
            EPISODIC_NODE_RETURN_NEPTUNE
            if driver.provider == GraphProvider.NEPTUNE
            else EPISODIC_NODE_RETURN
        )
        + """
        ORDER BY e.valid_at DESC
        LIMIT $num_episodes
        """
    )

    # 执行最终拼好的查询。
    # reference_time 和 num_episodes 是必需参数；
    # group_ids/source 则通过 query_params 按需展开传入。
    result, _, _ = await driver.execute_query(
        query,
        reference_time=reference_time,
        num_episodes=last_n,
        **query_params,
    )

    # 把数据库记录转换成业务对象。
    episodes = [get_episodic_node_from_record(record) for record in result]

    # 同 saga 分支一样：
    # 数据库取数时按倒序拿最近 N 条，返回给上层时再反转成时间正序，
    # 这样后续 LLM 抽取实体/边时能按照历史发生顺序阅读上下文。
    return list(reversed(episodes))  # Return in chronological order
