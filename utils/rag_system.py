import os
import arxiv
import openai
import requests

class ArxivRAGSystem:
    def __init__(self):
        # 设置OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = os.getenv("OPENAI_BASE_URL")
        self.papers_data = []
    
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
        
        self.papers_data = papers
        return papers
    
    def ask_question(self, question):
        """提问并获取答案"""
        if not self.papers_data:
            return {"error": "请先搜索相关论文"}
        
        # 构建上下文
        context = "以下是一些论文信息：\n"
        for i, paper in enumerate(self.papers_data[:3], 1):
            context += f"{i}. 标题: {paper['title']}\n   摘要: {paper['summary'][:500]}...\n\n"
        
        try:
            # 直接调用OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个学术助手，基于提供的论文信息回答问题。"},
                    {"role": "user", "content": f"{context}\n问题：{question}\n请基于以上论文信息回答："}
                ],
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": [paper['title'] for paper in self.papers_data[:3]]
            }
            
        except Exception as e:
            return {"error": f"AI服务暂时不可用: {str(e)}"}

# 全局RAG系统实例
rag_system = ArxivRAGSystem()
