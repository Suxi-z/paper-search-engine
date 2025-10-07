from flask import Flask, request, jsonify, render_template
from utils.rag_system import rag_system
import os


def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/search', methods=['POST'])
    def search_papers():
        """搜索arXiv论文"""
        try:
            data = request.get_json()
            query = data.get('query', '')
            max_results = data.get('max_results', 5)
            
            if not query:
                return jsonify({'error': '搜索词不能为空'}), 400
            
            papers = rag_system.search_arxiv_papers(query, max_results)
            
            # 创建知识库
            if papers:
                chunk_count = rag_system.create_knowledge_base(papers)
                print(f"创建了 {chunk_count} 个文本块的知识库")
            
            return jsonify({
                'papers': papers,
                'count': len(papers)
            })
            
        except Exception as e:
            return jsonify({'error': f'搜索失败: {str(e)}'}), 500
    
    @app.route('/api/ask', methods=['POST'])
    def ask_question():
        """提问关于已搜索论文的问题"""
        try:
            data = request.get_json()
            question = data.get('question', '')
            
            if not question:
                return jsonify({'error': '问题不能为空'}), 400
            
            if not rag_system.qa_chain:
                return jsonify({'error': '请先搜索相关论文'}), 400
            
            result = rag_system.ask_question(question)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'提问失败: {str(e)}'}), 500
    
    @app.route('/api/health')
    def health_check():
        """健康检查端点"""
        try:
            # 简单测试Ollama连接
            response = rag_system.llm("你好")
            return jsonify({
                'status': 'healthy',
                'ollama_connected': True
            })
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'ollama_connected': False,
                'error': str(e)
            }), 500
    

    return app

