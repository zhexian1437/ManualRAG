"""基于 Ragas 框架的 RAG 评估程序（独立可运行，非侵入式）。

功能说明：
    1. 从 CSV 文件读取评估样本（question / ground_truth）；
    2. 通过 import 方式调用项目现有查询流程（query_app），获取最终答案与重排序上下文；
    3. 构造 Ragas 标准字段（user_input / response / retrieved_contexts / reference）；
    4. 使用项目已封装的 LLM 与 Embedding 工具函数运行 Ragas 评估指标；
    5. 输出逐样本明细与指标汇总报告（CSV）。

约束说明：
    - 本脚本仅 import 复用现有流程，不修改任何业务代码、检索链路及生成流程；
    - LLM 使用 knowledge.utils.llm_client_util.get_llm_client；
    - Embedding 使用 knowledge.utils.bge_m3_embedding_util.get_beg_m3_embedding_model；
    - 注意：调用现有流程时，answer_output 节点会照常写入 Mongo 历史记录，
      这是被评估链路的固有副作用，本脚本不做拦截。

运行方式（项目根目录下）：
    python -m knowledge.eval.eval_main
依赖安装：
    pip install ragas datasets pandas
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ==================== 0. 路径与日志初始化 ====================

CURRENT_FILE = Path(__file__).resolve()
# knowledge/eval/eval_main.py -> parents[2] 即项目根目录（保证 `import knowledge.xxx` 可用）
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_eval")

# ==================== 1. 项目内工具函数导入（复用，不自行初始化客户端） ====================

from knowledge.processor.query_process.main_graph import query_app          # 查询主图（编译后的 LangGraph）
from knowledge.utils.llm_client_util import get_llm_client                   # 项目封装的 LLM 客户端
from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model  # 项目封装的 BGE-M3 Embedding

# ==================== 2. Ragas 相关导入 ====================

try:
    import pandas as pd
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        Faithfulness,        # 忠实度：答案声明是否被上下文支持
        AnswerRelevancy,     # 答案相关性：答案是否切题
        ContextPrecision,    # 上下文精度：相关内容是否排在前面
        ContextRecall,       # 上下文召回率：参考答案知识点是否被上下文覆盖
        AnswerCorrectness,   # 答案正确性：与 ground_truth 的事实/语义一致性
        ContextEntityRecall, # 实体召回：参考答案中的实体被上下文覆盖的比例（第 6 项）
    )
except ImportError as e:
    logger.error("缺少评估依赖，请先执行: pip install ragas datasets pandas")
    raise e

# ==================== 3. 全局配置常量 ====================

# 评估数据集路径（含 question 与 ground_truth 两列）
QA_CSV_PATH = r"C:\Users\19394\Downloads\万用表_qa.csv"

# 评估结果输出目录（按时间戳建子目录，避免覆盖历史报告）
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "knowledge" / "eval" / "eval_output"

# Ragas 列映射：显式声明，禁止隐式猜测
# （此处构造的数据集已使用 Ragas 标准字段名，映射为恒等映射，仅为满足显式映射约束）
COLUMN_MAP = {
    "user_input": "user_input",
    "response": "response",
    "retrieved_contexts": "retrieved_contexts",
    "reference": "reference",
}

# Ragas 运行配置：评估 LLM 调用较慢，放宽超时与重试
RUN_CONFIG = RunConfig(timeout=180, max_retries=3, max_wait=60, max_workers=4)


# ... existing code ...
class BgeM3RagasEmbeddings(BaseRagasEmbeddings):
    """BGE-M3 的 Ragas Embedding 适配器。

    Ragas 的 AnswerRelevancy 等指标需要 Embedding 模型。
    本适配器内部复用项目封装的 get_beg_m3_embedding_model()，
    仅取其 dense 稠密向量（默认 1024 维）供 Ragas 计算语义相似度，
    不自行初始化任何第三方 Embedding 客户端。

    注意：ragas 0.4.x 的 BaseRagasEmbeddings（旧版接口）要求实现
    embed_query / embed_documents / aembed_query / aembed_documents
    四个方法（继承自 langchain_core.embeddings.Embeddings 的抽象接口）。
    """

    def __init__(self):
        # 初始化父类（BaseRagasEmbeddings 需要执行其 __init__ 以支持缓存等机制）
        super().__init__()
        # 复用项目单例 Embedding 模型（懒加载，首次调用时才真正加载模型权重）
        self._model = get_beg_m3_embedding_model()
        if self._model is None:
            raise RuntimeError("BGE-M3 Embedding 模型初始化失败，请检查 BGE_M3_PATH 等环境变量")

    def embed_query(self, text: str) -> List[float]:
        """对单条查询文本生成稠密向量（Ragas/langchain 同步接口）。

        Args:
            text: 待向量化的查询文本。

        Returns:
            长度为 1024 的稠密向量。
        """
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """对批量文本生成稠密向量（Ragas/langchain 同步批量接口）。

        Args:
            texts: 待向量化的文本列表。

        Returns:
            与输入等长的稠密向量列表。
        """
        result = self._model.encode_documents(list(texts))
        return [vec.tolist() for vec in result["dense"]]

    async def aembed_query(self, text: str) -> List[float]:
        """异步单条查询接口：直接委托同步实现（评估脚本场景下可接受阻塞）。"""
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量接口：直接委托同步实现。"""
        return self.embed_documents(texts)
# ... existing code ...
def step_1_load_qa_dataset(csv_path: str) -> "pd.DataFrame":
    """step_1 读取评估数据集。

    从 CSV 中读取 question 与 ground_truth 两列，并做基础校验与清洗。

    Args:
        csv_path: 评估数据集 CSV 文件路径。

    Returns:
        仅包含 question / ground_truth 两列、且已去除空行的 DataFrame。

    Raises:
        FileNotFoundError: CSV 文件不存在时抛出。
        ValueError: 缺少必需列时抛出。
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"评估数据集不存在: {csv_path}")

    # 兼容 Excel 导出的 utf-8-sig 与 Windows 常见的 gbk 编码
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            logger.info(f"成功以 {encoding} 编码读取数据集: {csv_path}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法识别的 CSV 编码: {csv_path}")

    # 必需列校验，禁止隐式猜测列名
    required_columns = {"question", "ground_truth"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"数据集缺少必需列: {missing}，实际列: {list(df.columns)}")

    # 仅保留所需列并清洗空值/首尾空白
    df = df[["question", "ground_truth"]].dropna()
    df["question"] = df["question"].astype(str).str.strip()
    df["ground_truth"] = df["ground_truth"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["ground_truth"] != "")].reset_index(drop=True)

    logger.info(f"step_1 完成: 加载有效评估样本 {len(df)} 条")
    return df


def step_2_run_rag_pipeline(df: "pd.DataFrame") -> List[Dict[str, Any]]:
    """step_2 调用现有 RAG 查询流程，逐条生成答案与检索上下文。

    通过 query_app.invoke() 复用项目完整查询链路
    （商品确认 -> 向量检索 -> RRF -> Rerank -> 答案生成），
    以 is_stream=False 的非流式模式获取字符串答案；
    检索上下文取自流程 state 的 reranked_docs（重排序精排后的文档列表）。

    Args:
        df: step_1 输出的评估数据集 DataFrame。

    Returns:
        评估记录列表，每条记录包含:
        question / ground_truth / response / retrieved_contexts。
    """
    records: List[Dict[str, Any]] = []
    total = len(df)

    for index, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        logger.info(f"step_2 处理第 {index + 1}/{total} 条: {question[:50]}...")

        try:
            # 构造查询流程初始状态（与 main_graph 测试用例保持一致的最小字段集）
            init_state = {
                "original_query": question,
                "session_id": f"rag_eval_session_{int(time.time())}",
                "task_id": f"rag_eval_task_{index}",
                "is_stream": False,
            }

            # 非侵入式调用：仅 import 并 invoke，不修改链路任何内部逻辑
            result_state = query_app.invoke(init_state)

            # 最终答案：非流式模式下 answer_output 节点写入的字符串
            response = result_state.get("answer", "") or ""

            # 检索上下文：state.reranked_docs 为重排序后的文档字典列表，取 content 字段
            reranked_docs = result_state.get("reranked_docs") or []
            retrieved_contexts = [
                doc["content"].strip()
                for doc in reranked_docs
                if isinstance(doc, dict) and doc.get("content", "").strip()
            ]
        except Exception as e:
            # 单条失败不中断整体评估，记录空结果（对应指标会偏低，便于暴露问题）
            logger.error(f"第 {index + 1} 条流程执行失败: {e}")
            response = ""
            retrieved_contexts = []

        records.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        })
        logger.info(
            f"  -> 答案长度 {len(response)} 字符, 检索上下文 {len(retrieved_contexts)} 条"
        )

    logger.info(f"step_2 完成: 共生成 {len(records)} 条评估记录")
    return records


def step_3_build_ragas_dataset(records: List[Dict[str, Any]]) -> EvaluationDataset:
    """step_3 构造 Ragas 标准评估数据集。

    字段映射关系（按评估要求显式构造）:
        user_input         <- question（用户问题）
        response           <- 流程输出的最终答案
        retrieved_contexts <- 流程 state 的 reranked_docs 文档内容列表
        reference          <- ground_truth（参考答案）

    Args:
        records: step_2 输出的评估记录列表。

    Returns:
        Ragas EvaluationDataset 对象。
    """
    samples = []
    for record in records:
        sample = SingleTurnSample(
            user_input=record["question"],
            response=record["response"],
            retrieved_contexts=record["retrieved_contexts"],
            reference=record["ground_truth"],
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)
    logger.info(f"step_3 完成: 构造 Ragas 样本 {len(samples)} 条")
    return dataset


def step_4_init_eval_models() -> Tuple[LangchainLLMWrapper, BgeM3RagasEmbeddings]:
    """step_4 初始化评估所用的 LLM 与 Embedding。

    两者均复用项目已封装的工具函数，不自行初始化 OpenAI 等客户端:
        - LLM: get_llm_client() 返回的 ChatOpenAI，经 LangchainLLMWrapper 包装为 Ragas LLM；
        - Embedding: get_beg_m3_embedding_model() 返回的 BGE-M3，经自定义适配器包装。

    Returns:
        (ragas_llm, ragas_embeddings) 元组。

    Raises:
        RuntimeError: 项目 LLM 客户端初始化失败时抛出。
    """
    # 1. LLM：复用项目封装客户端（模型名取环境变量 ITEM_MODEL，默认 qwen-flash）
    llm_client = get_llm_client(temperature=0.0)
    if llm_client is None:
        raise RuntimeError("项目 LLM 客户端初始化失败，请检查 OPENAI_API_KEY 等环境变量")
    ragas_llm = LangchainLLMWrapper(llm_client)

    # 2. Embedding：复用项目封装的 BGE-M3 模型
    ragas_embeddings = BgeM3RagasEmbeddings()

    logger.info("step_4 完成: 评估 LLM 与 Embedding 初始化成功（均复用项目封装）")
    return ragas_llm, ragas_embeddings


def step_5_run_ragas_evaluation(
    dataset: EvaluationDataset,
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: BgeM3RagasEmbeddings,
):
    """step_5 运行 Ragas 指标评估。

    指标集（RAG Triad 及扩展，共 6 项）:
        1. Faithfulness        忠实度：答案声明是否被检索上下文支持
        2. AnswerRelevancy     答案相关性：答案是否直接回应问题
        3. ContextPrecision    上下文精度：相关内容是否排在前面（信噪比）
        4. ContextRecall       上下文召回率：参考答案知识点是否被上下文覆盖
        5. AnswerCorrectness   答案正确性：与 ground_truth 的事实/语义一致性
        6. ContextEntityRecall 实体召回率：参考答案实体被上下文覆盖的比例

    Args:
        dataset: step_3 构造的 Ragas 数据集。
        ragas_llm: step_4 初始化的 Ragas LLM。
        ragas_embeddings: step_4 初始化的 Ragas Embedding。

    Returns:
        Ragas EvaluationResult 对象（含逐样本得分与均分）。
    """
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
        ContextEntityRecall(),
    ]

    logger.info("step_5 开始: 运行 Ragas 评估（LLM 调用较多，请耐心等待）...")

    # 显式传入 column_map；若当前 ragas 版本不支持该参数则降级为默认字段匹配
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RUN_CONFIG,
            column_map=COLUMN_MAP,
        )
    except TypeError:
        logger.warning("当前 ragas 版本不支持 column_map 参数，回退为默认字段匹配")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RUN_CONFIG,
        )

    logger.info("step_5 完成: Ragas 评估结束")
    return result


def step_6_save_eval_report(
    result,
    records: List[Dict[str, Any]],
) -> Path:
    """step_6 保存评估报告。

    输出两个 CSV 文件:
        - eval_scores_*.csv : 逐样本的各项指标得分（Ragas 原生输出）；
        - eval_details_*.csv: 逐样本的问题/参考答案/生成答案/检索上下文明细，
                              便于人工复核低分样本。

    Args:
        result: step_5 的 Ragas EvaluationResult。
        records: step_2 的评估记录列表（用于导出明细）。

    Returns:
        本次报告的输出目录路径。
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = EVAL_OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存逐样本指标得分
    scores_df = result.to_pandas()
    scores_path = output_dir / f"eval_scores_{timestamp}.csv"
    scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")

    # 2. 保存评估明细（上下文列表拼接为换行分隔的文本，便于阅读）
    details_records = []
    for record in records:
        details_records.append({
            "question": record["question"],
            "ground_truth": record["ground_truth"],
            "response": record["response"],
            "retrieved_contexts": "\n---\n".join(record["retrieved_contexts"]),
            "context_count": len(record["retrieved_contexts"]),
        })
    details_df = pd.DataFrame(details_records)
    details_path = output_dir / f"eval_details_{timestamp}.csv"
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    # 3. 控制台打印指标均分汇总
    logger.info("=" * 60)
    logger.info("【评估指标均分汇总】")
    metric_columns = [
        col for col in scores_df.columns
        if col not in ("user_input", "response", "retrieved_contexts", "reference")
    ]
    for col in metric_columns:
        mean_score = pd.to_numeric(scores_df[col], errors="coerce").mean()
        logger.info(f"  {col:25s}: {mean_score:.4f}")
    logger.info("=" * 60)

    logger.info(f"step_6 完成: 报告已保存至 {output_dir}")
    return output_dir


def main():
    """评估主流程：step_1 -> step_6 顺序执行。"""
    logger.info("RAG 评估程序启动")

    # step_1 读取评估数据集
    df = step_1_load_qa_dataset(QA_CSV_PATH)

    # step_2 调用现有 RAG 流程生成答案与上下文
    records = step_2_run_rag_pipeline(df)

    # step_3 构造 Ragas 标准数据集
    dataset = step_3_build_ragas_dataset(records)

    # step_4 初始化评估模型（复用项目封装的 LLM / Embedding）
    ragas_llm, ragas_embeddings = step_4_init_eval_models()

    # step_5 运行 Ragas 评估
    result = step_5_run_ragas_evaluation(dataset, ragas_llm, ragas_embeddings)

    # step_6 保存评估报告
    output_dir = step_6_save_eval_report(result, records)

    logger.info(f"RAG 评估程序结束，报告目录: {output_dir}")


if __name__ == "__main__":
    main()