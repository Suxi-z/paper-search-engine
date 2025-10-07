// 前端JavaScript逻辑
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const resultCount = document.getElementById('resultCount');
    const loading = document.getElementById('loading');
    const papersContainer = document.getElementById('papersContainer');
    const papersList = document.getElementById('papersList');
    const qaSection = document.getElementById('qaSection');
    const questionInput = document.getElementById('questionInput');
    const askBtn = document.getElementById('askBtn');
    const answerSection = document.getElementById('answerSection');
    const answerContent = document.getElementById('answerContent');
    const sourcesList = document.getElementById('sourcesList');
    const answerTime = document.getElementById('answerTime');

    // 搜索论文
    searchBtn.addEventListener('click', searchPapers);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchPapers();
        }
    });

    // 提问
    askBtn.addEventListener('click', askQuestion);
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            askQuestion();
        }
    });

    // 搜索论文函数
    async function searchPapers() {
        const query = searchInput.value.trim();
        if (!query) {
            alert('请输入搜索关键词');
            return;
        }

        // 显示加载
        loading.style.display = 'block';
        papersContainer.style.display = 'none';
        qaSection.style.display = 'none';
        answerSection.style.display = 'none';

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    max_results: parseInt(resultCount.value)
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '搜索失败');
            }

            displayPapers(data.papers);
            // 显示问答区域
            qaSection.style.display = 'block';

        } catch (error) {
            console.error('搜索错误:', error);
            alert('搜索失败: ' + error.message);
        } finally {
            loading.style.display = 'none';
        }
    }

    // 显示论文列表
    function displayPapers(papers) {
        if (papers.length === 0) {
            papersList.innerHTML = '<p class="no-results">未找到相关论文</p>';
            papersContainer.style.display = 'block';
            return;
        }

        const papersHTML = papers.map(paper => `
            <div class="paper-card">
                <div class="paper-title">${paper.title}</div>
                <div class="paper-meta">
                    <span class="paper-authors">${paper.authors.join(', ')}</span>
                    <span class="paper-date">${paper.published}</span>
                </div>
                <div class="paper-summary">${paper.summary}</div>
                <div class="paper-actions">
                    <a href="${paper.pdf_url}" class="download-btn" target="_blank">
                        <i class="fas fa-download"></i> 下载PDF
                    </a>
                </div>
            </div>
        `).join('');

        papersList.innerHTML = papersHTML;
        papersContainer.style.display = 'block';
        document.getElementById('resultsCount').textContent = `(${papers.length}篇)`;
    }

    // 提问函数
    async function askQuestion() {
        const question = questionInput.value.trim();
        if (!question) {
            alert('请输入问题');
            return;
        }

        // 显示加载
        askBtn.disabled = true;
        askBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 思考中...';

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '提问失败');
            }

            displayAnswer(data);

        } catch (error) {
            console.error('提问错误:', error);
            alert('提问失败: ' + error.message);
        } finally {
            askBtn.disabled = false;
            askBtn.innerHTML = '<i class="fas fa-paper-plane"></i> 提问';
        }
    }

    // 显示答案
    function displayAnswer(data) {
        answerContent.innerHTML = formatAnswer(data.answer);
        sourcesList.innerHTML = '';

        if (data.sources && data.sources.length > 0) {
            const sourcesHTML = `
                <h5>引用来源</h5>
                <div class="source-tags">
                    ${data.sources.map(source => `<span class="source-tag">${source}</span>`).join('')}
                </div>
            `;
            sourcesList.innerHTML = sourcesHTML;
        }

        answerTime.textContent = new Date().toLocaleString();
        answerSection.style.display = 'block';
    }

    // 格式化答案（简单处理换行）
    function formatAnswer(answer) {
        return answer.replace(/\n/g, '<br>');
    }
});