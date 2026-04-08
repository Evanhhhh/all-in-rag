import getpass
import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 交互式输入 API Key，避免在代码中明文保存密钥
api_key = getpass.getpass("请输入 API Key: ").strip()

# 配置全局 LLM：使用 OpenAI 兼容接口接入 AIHubmix 的 glm-4.7-flash-free
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=api_key,
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# 下面是可选的 DeepSeek 配置示例（当前未启用）
# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 配置全局向量化模型：用于将文档与问题编码到同一向量空间
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 读取本地 Markdown 文档作为知识库数据源
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 基于文档构建向量索引
index = VectorStoreIndex.from_documents(docs)

# 将索引包装为查询引擎
query_engine = index.as_query_engine()

# 打印查询时使用的 Prompt 模板，便于调试
print(query_engine.get_prompts())

# 发起问题并输出回答
print(query_engine.query("文中举了哪些例子?"))