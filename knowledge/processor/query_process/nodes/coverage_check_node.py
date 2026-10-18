"""覆盖度检查节点（Corrective RAG）

在生成答案之前，由LLM将用户问题拆解为子需求并逐条核对当前检索上下文的覆盖情况；
对未覆盖的子需求，从本地RRF剩余队列和网络MCP剩余队列中按语义相关度定向补充chunk；
迭代最多 coverage_max_iterations 轮，仍不完整的子需求记入 state["coverage_missing"]，
由答案生成阶段如实声明缺失。

设计要点：
1. 检查对象是"上下文"而非"答案"，答案只生成一次，避免重复生成成本；
2. 补充是定向的（按missing语义选chunk），不是盲目取下一个；
3. 用户通过 state["enable_coverage_check"] 自行开关，默认关闭。
"""

import json
import re
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from knowledge.utils.llm_client_util import get_llm_client
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model, generate_hybrid_embeddings
from knowledge.prompts.query.query_prompt import COVERAGE_CHECK_TEMPLATE


class CoverageCheckNode(BaseNode):
    name = "coverage_check_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 开关判断：用户未启用则直接透传
        if not state.get("enable_coverage_check"):
            return state

        docs = state.get("reranked_docs") or []
        if not docs:
            return state

        query = state.get("rewritten_query") or state.get("original_query", "")

        # 2. 构建双源补充候选池（本地RRF剩余 + 网络MCP剩余，均排除已在上下文中的chunk）
        local_pool, web_pool = self._build_candidate_pools(state, docs)

        # 3. 迭代检查与定向补充（最多 coverage_max_iterations 轮）
        missing: List[str] = []
        for attempt in range(self.config.coverage_max_iterations):
            # 3.1 LLM检查覆盖度
            check_result = self._check_coverage(query, docs)

            # 3.2 完整或解析失败（fail-open）则结束
            if check_result.get("complete") or not check_result.get("missing"):
                missing = []
                break

            missing = check_result["missing"]
            self.logger.info(f"覆盖度检查第 {attempt + 1} 轮: 未覆盖子需求 {missing}")

            # 3.3 按missing从双源候选池定向补充
            supplemented = self._supplement_by_missing(missing, docs, local_pool, web_pool)
            if not supplemented:
                self.logger.info("双源候选池已无可用补充，结束迭代")
                break

            docs.extend(supplemented)

        # 4. 写回state（仍未覆盖的子需求由生成阶段如实声明）
        state["reranked_docs"] = docs
        state["coverage_missing"] = missing

        return state

    def _build_candidate_pools(
            self, state: QueryGraphState, docs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """构建双源补充候选池。

        Args:
            state: 查询图状态。
            docs: 当前已进入上下文的文档列表。

        Returns:
            (本地候选池, 网络候选池)，均已排除已在上下文中的chunk。
        """
        # 1. 已有内容指纹（前100字符判重）
        existing_fingerprints = {(doc.get("content") or "").strip()[:100] for doc in docs}

        # 2. 本地候选池：RRF融合后的本地队列（已按融合分数降序）
        local_pool = []
        for rrf_doc in (state.get("rrf_chunks") or []):
            if not isinstance(rrf_doc, dict):
                continue
            content = (rrf_doc.get("content") or "").strip()
            if not content or content[:100] in existing_fingerprints:
                continue
            local_pool.append({
                "content": content,
                "title": (rrf_doc.get("title") or "").strip(),
                "chunk_id": rrf_doc.get("chunk_id"),
                "url": "",
                "source": "local",
            })

        # 3. 网络候选池：MCP网络检索剩余结果
        web_pool = []
        for web_doc in (state.get("web_search_docs") or []):
            if not isinstance(web_doc, dict):
                continue
            content = (web_doc.get("content") or web_doc.get("snippet") or "").strip()
            if not content or content[:100] in existing_fingerprints:
                continue
            web_pool.append({
                "content": content,
                "title": (web_doc.get("title") or "").strip(),
                "chunk_id": None,
                "url": (web_doc.get("url") or "").strip(),
                "source": "web",
            })

        self.logger.info(f"覆盖度候选池: 本地 {len(local_pool)} 条, 网络 {len(web_pool)} 条")
        return local_pool, web_pool

    def _check_coverage(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """LLM覆盖度检查。

        将问题拆解为子需求，逐条核对当前上下文是否覆盖，输出结构化JSON。

        Args:
            query: 用户问题（重写后）。
            docs: 当前上下文文档列表。

        Returns:
            {"complete": bool, "missing": [未覆盖子需求]}；调用或解析失败时fail-open返回complete=True。
        """
        fail_open = {"complete": True, "missing": []}

        # 1. 获取LLM客户端（JSON模式）
        llm_client = get_llm_client(response_format=True)
        if llm_client is None:
            self.logger.warning("覆盖度检查: LLM客户端不可用，跳过检查")
            return fail_open

        # 2. 组装带编号的上下文（控制每条长度，避免超出预算）
        context_lines = []
        for idx, doc in enumerate(docs, 1):
            content = (doc.get("content") or "")[:500]
            context_lines.append(f"[{idx}] {content}")
        context_text = "\n\n".join(context_lines)

        try:
            # 3. 调用LLM判定
            prompt = COVERAGE_CHECK_TEMPLATE.format(query=query, context=context_text)
            response = llm_client.invoke([
                SystemMessage(content="你是一个严谨的检索质量检查员，只输出JSON。"),
                HumanMessage(content=prompt),
            ])
            content = (response.content or "").strip()
            if not content:
                return fail_open

            # 4. 清洗并解析JSON
            cleaned = re.sub(r"^```(?:json)?\s*", "", content)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            parsed = json.loads(cleaned)

            complete = bool(parsed.get("complete", True))
            missing = parsed.get("missing") or []
            if not isinstance(missing, list):
                missing = []
            missing = [str(m).strip() for m in missing if str(m).strip()]

            return {"complete": complete, "missing": missing}
        except (JSONDecodeError, Exception) as e:
            # fail-open：检查失败不阻塞主流程
            self.logger.warning(f"覆盖度检查解析失败: {str(e)}，按完整处理")
            return fail_open

    def _supplement_by_missing(
            self,
            missing: List[str],
            docs: List[Dict[str, Any]],
            local_pool: List[Dict[str, Any]],
            web_pool: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """按未覆盖子需求从双源候选池定向补充chunk。

        用BGE-M3对missing文本与候选chunk做语义相似度打分，
        每个来源各取分数最高且超过阈值的 coverage_supplement_per_source 条。

        Args:
            missing: 未覆盖的子需求列表。
            docs: 当前上下文文档（用于补充后判重）。
            local_pool: 本地候选池（会被原地移除已选中的chunk）。
            web_pool: 网络候选池（会被原地移除已选中的chunk）。

        Returns:
            本轮补充的文档列表；无可补充时返回空列表。
        """
        # 1. 获取嵌入模型
        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            self.logger.warning("覆盖度补充: 嵌入模型不可用")
            return []

        # 2. 编码missing文本（合并为一个查询向量）
        missing_text = "；".join(missing)
        missing_embedding = generate_hybrid_embeddings(embedding_model, [missing_text])
        if not missing_embedding or not missing_embedding.get("dense"):
            return []
        missing_vector = missing_embedding["dense"][0]

        supplemented: List[Dict[str, Any]] = []

        # 3. 逐来源定向挑选
        for pool, source_name in ((local_pool, "本地"), (web_pool, "网络")):
            picked = self._pick_top_related(pool, missing_vector, embedding_model)
            for doc in picked:
                doc["supplemented"] = True  # 标记为覆盖度补充，便于下游识别
                supplemented.append(doc)
            if picked:
                self.logger.info(f"覆盖度补充: 从{source_name}候选池补充 {len(picked)} 条")

        return supplemented

    def _pick_top_related(
            self,
            pool: List[Dict[str, Any]],
            missing_vector: List[float],
            embedding_model,
    ) -> List[Dict[str, Any]]:
        """从候选池中挑选与missing语义最相关的chunk（原地移除）。

        Args:
            pool: 候选池。
            missing_vector: missing文本的稠密向量。
            embedding_model: BGE-M3嵌入模型。

        Returns:
            选中的文档列表（不超过 coverage_supplement_per_source 条，且相似度超过阈值）。
        """
        if not pool:
            return []

        # 1. 批量编码候选chunk
        texts = [doc["content"][:500] for doc in pool]
        embedding_result = generate_hybrid_embeddings(embedding_model, texts)
        if not embedding_result or not embedding_result.get("dense"):
            return []

        # 2. 计算余弦相似度并排序
        scored = []
        for index, doc_vector in enumerate(embedding_result["dense"]):
            score = self._cosine_similarity(missing_vector, doc_vector)
            scored.append((score, index))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 3. 取超过阈值的top_k条，并从池中移除
        picked = []
        for score, index in scored[: self.config.coverage_supplement_per_source]:
            if score < self.config.coverage_supplement_min_score:
                continue
            picked.append(pool[index])
            self.logger.info(f"  候选相似度 {score:.4f}: {pool[index]['content'][:50]}...")

        for doc in picked:
            pool.remove(doc)

        return picked

    @staticmethod
    def _cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        """计算两个稠密向量的余弦相似度。"""
        arr_a = np.array(vector_a)
        arr_b = np.array(vector_b)
        denominator = (np.linalg.norm(arr_a) * np.linalg.norm(arr_b)) or 1e-8
        return float(np.dot(arr_a, arr_b) / denominator)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from knowledge.processor.query_process.base import setup_logging

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 覆盖度检查节点 (CoverageCheckNode)")
    print("=" * 60)

    mock_state = {
        "enable_coverage_check": True,
        "original_query": "RS-12数字万用表的操作温度和湿度有什么要求？",
        "reranked_docs": [
            {"content": "RS-12数字万用表操作环境温度为0°C至50°C，储存温度-20°C至60°C。",
             "title": "RS-12说明书", "chunk_id": "c1", "url": "", "source": "web", "score": 6.1},
        ],
        "rrf_chunks": [
            {"content": "RS-12数字万用表操作环境相对湿度要求低于70%，适用于室内使用，海拔不超过2000米。",
             "title": "RS-12说明书", "chunk_id": "c2"},
        ],
        "web_search_docs": [],
    }

    node = CoverageCheckNode()
    result = node.process(mock_state)

    print("\n【检查结果】:")
    print(f"  上下文文档数: {len(result['reranked_docs'])}")
    print(f"  仍未覆盖子需求: {result.get('coverage_missing')}")
    for i, doc in enumerate(result["reranked_docs"], 1):
        tag = "补充" if doc.get("supplemented") else "原始"
        print(f"  [{i}][{tag}][{doc['source']}] {doc['content'][:60]}...")

    print("-" * 60)
    print("测试完成")