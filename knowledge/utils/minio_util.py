import os
import logging

from minio.error import S3Error
from minio import Minio
from dotenv import load_dotenv

load_dotenv()


def get_minio_client():
    # 1. 实例化MinIO客户端
    try:
        client = Minio(os.getenv("MINIO_ENDPOINT"),  # 服务端端口
                       access_key=os.getenv("MINIO_ACCESS_KEY"),  # 访问的key
                       secret_key=os.getenv("MINIO_SECRET_KEY"),  # 访问的秘钥
                       secure=False  # 使用HTTP协议
                       )

        # 2. 判断桶是否存在
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        bucket_exists = client.bucket_exists(bucket_name)
        if not bucket_exists:
            client.make_bucket(bucket_name)
            logging.info(f"桶:{bucket_name} 不存在，已创建")
        else:
            logging.info(f"桶:{bucket_name} 已经存在")

        # 3. 返回MinIO客户端
        return client
    except S3Error as e:
        logging.error(f"MinIO S3错误: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"MinIO客户端创建失败: {e}", exc_info=True)
        return None


# -------------------- main 测试入口 --------------------
if __name__ == '__main__':
    # 配置日志，让 INFO 及以上级别都能在控制台显示
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("开始测试 MinIO 客户端连接...")

    # 可选：打印当前读取到的环境变量，便于确认配置是否加载成功
    logging.info("MINIO_ENDPOINT = %s", os.getenv("MINIO_ENDPOINT"))
    logging.info("MINIO_BUCKET_NAME = %s", os.getenv("MINIO_BUCKET_NAME"))
    logging.info("MINIO_ACCESS_KEY = %s", "已设置" if os.getenv("MINIO_ACCESS_KEY") else "未设置")
    logging.info("MINIO_SECRET_KEY = %s", "已设置" if os.getenv("MINIO_SECRET_KEY") else "未设置")

    client = get_minio_client()

    if client is None:
        logging.error("MinIO 客户端创建失败，请检查上面的错误信息、网络连通性以及 .env 配置。")
    else:
        logging.info("MinIO 客户端创建成功！")
        try:
            # 列出所有桶，验证权限和连接
            buckets = client.list_buckets()
            bucket_names = [b.name for b in buckets]
            logging.info("当前 MinIO 中的桶列表: %s", bucket_names)

            # 再次确认目标桶是否存在
            target_bucket = os.getenv("MINIO_BUCKET_NAME")
            if target_bucket in bucket_names:
                logging.info("目标桶 '%s' 存在，测试通过。", target_bucket)
            else:
                logging.warning("目标桶 '%s' 不在桶列表中，但可能刚创建尚未刷新。", target_bucket)
        except Exception as e:
            logging.error(f"测试 MinIO 操作时发生异常: {e}", exc_info=True)