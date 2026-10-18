from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large", local_dir="F:/model/models")

print(local_dir)