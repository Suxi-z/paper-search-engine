import os
import arxiv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 减少内存使用
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ArxivRAGSystem:
    def __init__(self):
        # 初始化嵌入模型 - 使用轻量级配置
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # 初始化OpenAI模型
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.1
        )
        
        self.vector_store = None
        self.qa_chain = None
    
    def search_arxiv_papers(self, query, max_results=5):
        """搜索arXiv论文"""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for paper in client.results(search):
            papers.append({
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'published': paper.published.strftime("%Y-%m-%d"),
                'pdf_url': paper.pdf_url,
                'entry_id': paper.entry_id
            })
        
        return papers
    
    def create_knowledge_base(self, papers):
        """从论文创建知识库"""
        if not papers:
            return None
            
        # 准备文档
        documents = []
        for paper in papers:
            content = f"标题: {paper['title']}\n作者: {', '.join(paper['authors'])}\n摘要: {paper['summary']}"
            documents.append(content)
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.create_documents(documents)
        
        # 创建向量存储
        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embeddings
        )
        
        # 创建检索链
        prompt_template = """使用以下上下文来回答用户的问题。

        上下文:
        {context}

        问题: {question}

        请用中文回答:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 2}),  # 减少检索数量
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return len(chunks)
    
    def ask_question(self, question):
        """提问并获取答案"""
        if not self.qa_chain:
            return {"error": "请先搜索并创建知识库"}
        
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": ["基于搜索的论文内容"]
        }

# 全局RAG系统实例
rag_system = ArxivRAGSystem()
