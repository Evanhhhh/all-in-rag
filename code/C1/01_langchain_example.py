import getpass
import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块 递归字符分割器（RecursiveCharacterTextSplitter）会根据指定的分割符（如段落、句子等）递归地将文本分割成更小的块，以确保每个块的长度不超过指定的限制。这种方法可以更好地保留文本的结构和上下文信息，适用于处理较长的文本内容。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 索引构建
# 中文嵌入模型
# 首次运行时，脚本会下载BAAI/bge-small-zh-v1.5嵌入模型。
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cuda'},
    # 嵌入归一化
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
# InMemoryVectorStore是一个简单的内存向量存储实现，适用于小规模数据集。它将文档的嵌入向量保存在内存中，并提供相似度搜索功能。对于更大规模的数据集，可以考虑使用更高效的向量数据库，如FAISS、Pinecone等。
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
api_key = getpass.getpass("请输入 API Key: ").strip()
llm = ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=4096,
    api_key=api_key,
    base_url="https://aihubmix.com/v1"
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

print("检索到的上下文：")
print(docs_content)
print("\n" + "=" * 50 + "\n")

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print("回答正文：")
print(answer.content)

print("完整回答：")
print(answer)