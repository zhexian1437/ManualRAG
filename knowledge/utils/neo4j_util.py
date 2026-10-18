import os
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

logger = logging.getLogger(__name__)
_neo4j_driver: Optional[Driver] = None

load_dotenv()
def get_neo4j_driver() -> Optional[Driver]:
    """
    获取 Neo4j 驱动实例（单例模式）
    """
    global _neo4j_driver
    try:
        if _neo4j_driver is None:
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")

            logger.info(f"正在初始化 Neo4j 驱动，连接 URI: {uri}")

            _neo4j_driver = GraphDatabase.driver(
                uri=uri,
                auth=(username, password)
            )
            # Neo4j 驱动默认是懒加载，这行代码能确保如果账号密码错误或网络不通，
            # 当场就会抛出异常，而不是等到插入数据时才报错。
            _neo4j_driver.verify_connectivity()

            logger.info("Neo4j 驱动初始化成功并已验证连接！")

        return _neo4j_driver

    except Exception as e:
        # exc_info=True 会在日志中打印出完整的 Error Traceback，
        # 方便排查到底是密码错还是网络不通
        logger.error(f"初始化 Neo4j 驱动失败: {e}", exc_info=True)
        return None


def _print_diagnosis(error: Optional[Exception] = None) -> None:
    """根据错误类型打印常见原因与排查建议"""
    print("\n[诊断建议]")
    err_str = str(error).lower() if error else ""

    if "authentication" in err_str or "unauthorized" in err_str:
        print("  • 认证失败：请检查 NEO4J_USERNAME / NEO4J_PASSWORD 是否正确。")
        print("  • 也请确认宿主机/虚拟机时间差是否过大（认证对时间敏感）。")
    elif "connection refused" in err_str or "unable to connect" in err_str:
        print("  • 连接被拒绝：检查 Neo4j 容器是否运行（docker ps）。")
        print("  • 检查 7687 端口是否映射并可从当前主机访问。")
        print("  • 云服务器需检查防火墙/安全组是否放行 7687。")
    elif "timed out" in err_str or "timeout" in err_str:
        print("  • 连接超时：网络不通或端口未开放，先用 telnet 测试端口。")
    else:
        print("  • 请查看上方日志中的完整 Traceback 定位具体原因。")


def test_neo4j_connection() -> bool:
    """
    Neo4j 连接测试函数

    测试内容：
    1. 环境变量是否配置
    2. 驱动能否创建并验证连通性
    3. 能否执行实际查询
    4. 本机时间 vs Neo4j 服务器时间（排查时间差导致的认证失败）
    5. 给出针对性的错误诊断建议

    Returns:
        bool: 连接测试是否完全通过
    """
    global _neo4j_driver

    print("=" * 60)
    print("开始测试 Neo4j 连接")
    print("=" * 60)

    # ---------- 1. 检查环境变量 ----------
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    print("\n[1] 环境变量检查：")
    print(f"    NEO4J_URI      = {uri}")
    print(f"    NEO4J_USERNAME = {username}")
    print(f"    NEO4J_PASSWORD = {'***' if password else None}")

    if not uri or not username or not password:
        print("    ❌ 环境变量缺失！请检查 .env 或系统环境变量。")
        return False
    print("    ✅ 环境变量齐全")

    # ---------- 2. 获取驱动 ----------
    print("\n[2] 初始化驱动：")
    driver = get_neo4j_driver()
    if driver is None:
        print("    ❌ 驱动初始化失败，请看上面的日志堆栈。")
        _print_diagnosis()
        return False
    print("    ✅ 驱动初始化成功")

    # ---------- 3. 执行实际查询 + 时间对比 ----------
    print("\n[3] 执行查询 & 时间对比：")
    try:
        with driver.session() as session:
            # 3.1 简单连通性查询
            result = session.run("RETURN 1 AS ok")
            record = result.single()
            print(f"    ✅ 查询成功，返回: {record['ok'] if record else None}")

            # 3.2 取 Neo4j 服务器时间
            server_time_record = session.run(
                "RETURN datetime() AS server_time"
            ).single()
            server_time = server_time_record["server_time"]

            # 3.3 取本机时间（对齐时区）
            local_time = (
                datetime.now(server_time.tzinfo)
                if server_time.tzinfo
                else datetime.now()
            )

            delta = abs((local_time - server_time).total_seconds())

            print(f"    Neo4j 服务器时间: {server_time}")
            print(f"    本机时间        : {local_time}")
            print(f"    时间差          : {delta:.2f} 秒")

            if delta > 300:
                print("    ⚠️ 时间差 > 5 分钟！这极可能导致认证/连接失败。")
                print("       建议对虚拟机执行时间同步（open-vm-tools 或 chrony 方案）。")
            elif delta > 60:
                print("    ⚠️ 时间差超过 1 分钟，建议尽快同步时间。")
            else:
                print("    ✅ 时间基本一致")

            # 3.4 列出数据库版本，进一步确认服务正常
            version_record = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            ).single()
            if version_record:
                print(
                    f"    ℹ️ 数据库: {version_record['name']} "
                    f"{version_record['version']} ({version_record['edition']})"
                )

    except Exception as e:
        print(f"    ❌ 查询失败: {e}")
        _print_diagnosis(e)
        return False

    finally:
        # ---------- 4. 关闭驱动并重置单例 ----------
        print("\n[4] 关闭驱动：")
        try:
            if _neo4j_driver is not None:
                _neo4j_driver.close()
                _neo4j_driver = None
                print("    ✅ 驱动已关闭，单例已重置")
        except Exception as e:
            print(f"    ⚠️ 关闭驱动时出现异常: {e}")

    print("\n" + "=" * 60)
    print("✅ Neo4j 连接测试全部通过")
    print("=" * 60)
    return True


# ---------- 直接运行本文件即可测试 ----------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    success = test_neo4j_connection()
    raise SystemExit(0 if success else 1)