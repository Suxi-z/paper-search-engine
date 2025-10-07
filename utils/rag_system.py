from langchain_openai import ChatOpenAI
import os
import arxiv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
import json

class ArxivRAGSystem:
    def __init__(self):
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 初始化Ollama模型
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # 或者 "gpt-4" 如果你有访问权限
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
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        # 创建检索链
        prompt_template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道，不要编造答案。
        同时，请提供引用的论文标题。

        上下文:
        {context}

        问题: {question}

        请用中文回答，并注明引用来源:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return len(chunks)
    
    def ask_question(self, question):
        """提问并获取答案"""
        if not self.qa_chain:
            return {"error": "请先搜索并创建知识库"}
        
        result = self.qa_chain({"query": question})
        
        # 提取引用来源
        sources = []
        for doc in result.get('source_documents', []):
            content = doc.page_content
            # 从内容中提取标题
            if "标题:" in content:
                title = content.split("标题:")[1].split("\n")[0].strip()
                sources.append(title)
        
        return {
            "answer": result["result"],
            "sources": list(set(sources))[:3]  # 去重并限制数量
        }

# 全局RAG系统实例
rag_system = ArxivRAGSystem()