let __currentJob = null;
let __progressTimer = null;
// persistent client-side like/dislike sets keyed by arxiv_id
window.__likedIds = new Set();
window.__dislikedIds = new Set();

async function recommendPapers() {
    const defaultText = `Enter your workshop CFP; the system will recommend relevant papers...

💡 Example
"Call for Papers: We welcome both original research papers and submissions based on arXiv pre-prints.
Topics of interest include, but are not limited to:
●   Data management systems for generative AI
●   Agentic AI systems and algorithms"`;
    const ta = document.getElementById('workshop-description');
    const raw = ta.value;
    const description = (ta.classList && ta.classList.contains('placeholder-text')) || raw === defaultText ? '' : raw.trim();
    const topK = parseInt(document.getElementById('top-k').value) || 20;
    const dateMinVal = document.getElementById('date-min').value.trim();
    const dateMaxVal = document.getElementById('date-max').value.trim();
    const countries = Array.from(document.querySelectorAll('#country-list input[type="checkbox"]:checked')).map(el => el.value);
    const minSimVal = parseFloat(document.getElementById('min-similarity').value);

    if (!description) {
        alert('Please enter a workshop description');
        return;
    }

    // 显示加载状态
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    try {
        // 记录本次查询参数
        const requestId = Math.random().toString(36).slice(2) + Date.now().toString(36);
        window.__lastQuery = {
            request_id: requestId,
            workshop_description: description,
            top_k: topK,
            date_min: dateMinVal || null,
            date_max: dateMaxVal || null,
            countries: countries,
            categories: Array.from(document.querySelectorAll('#cs-category-list input[type="checkbox"]:checked')).map(el => el.value),
            min_similarity: Number.isNaN(minSimVal) ? null : minSimVal,
        };
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                workshop_description: description,
                top_k: topK,
                date_min: window.__lastQuery.date_min,
                date_max: window.__lastQuery.date_max,
                countries: window.__lastQuery.countries,
                categories: window.__lastQuery.categories,
                min_similarity: window.__lastQuery.min_similarity,
                auto_split: true,
            })
        });

        const start = await response.json();
        if (!response.ok || !start.job_id) {
            displayError((start && start.error) || 'Failed to start job');
            return;
        }
        __currentJob = start.job_id;
        // show progress UI
        document.getElementById('progressBox').style.display = 'block';
        updateProgressUI(0, 'Queued...');
        // start polling
        pollProgress(__currentJob, requestId);
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function updateProgressUI(percent, message){
    const inner = document.getElementById('progressInner');
    const status = document.getElementById('progressStatus');
    const detailsContent = document.getElementById('progressDetailsContent');

    inner.style.width = Math.max(0, Math.min(100, percent)) + '%';

    // 简化状态显示
    if (percent < 5) {
        status.textContent = 'Queued...';
    } else if (percent < 20) {
        status.textContent = 'Computing embeddings and retrieving papers... (click bar to expand for details)';
    } else if (percent < 100) {
        status.textContent = 'Reranking results... (click bar to expand for details)';
    } else {
        status.textContent = 'Completed!';
    }

    // 详细消息存储在隐藏区域
    if (detailsContent) {
        detailsContent.textContent = message || '';
    }
}

function toggleProgressDetails() {
    const details = document.getElementById('progressDetails');
    if (details.style.display === 'none') {
        details.style.display = 'block';
    } else {
        details.style.display = 'none';
    }
}

async function pollProgress(jobId, requestId){
    if (__progressTimer) clearInterval(__progressTimer);
    __progressTimer = setInterval(async () => {
        try {
            const resp = await fetch('/api/progress?job_id=' + encodeURIComponent(jobId));
            const pj = await resp.json();
            if (!resp.ok) {
                updateProgressUI(0, pj.error || 'Error');
                clearInterval(__progressTimer);
                return;
            }
            updateProgressUI(pj.percent || 0, pj.message || 'Processing...');
            if (pj.done) {
                clearInterval(__progressTimer);
                // fetch result
                const rr = await fetch('/api/result?job_id=' + encodeURIComponent(jobId));
                const data = await rr.json();
                if (rr.ok) {
                    if (data.recommendations_grouped) {
                        window.__lastRecommendations = (data.recommendations_grouped || []).flatMap(g => g.recommendations || []);
                        window.__lastQueryResponse = { request_id: requestId, count: data.total_count, search_time: data.search_time };
                    } else {
                        window.__lastRecommendations = data.recommendations || [];
                        window.__lastQueryResponse = { request_id: requestId, count: data.count, search_time: data.search_time };
                    }
                    displayResults(data);
                    document.getElementById('progressBox').style.display = 'none';
                } else {
                    displayError(data.error || 'Failed to load result');
                }
            }
        } catch(e){
            updateProgressUI(0, 'Network error');
            clearInterval(__progressTimer);
        }
    }, 600);
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');

    // Stats
    const statsHtml = `
        <div class="stats">
            <h3>📈 Summary</h3>
            <p>Found <strong>${data.total_count !== undefined ? data.total_count : data.count}</strong> papers | Time: <strong>${data.search_time}</strong> s</p>
            <div style="margin-top:6px;color:#666;">
                <div>Stage 1/2 (Recall): Using allenai/specter2_base to fast compute paper embeddings and retrieve top-200 by cosine similarity.</div>
                <div>Stage 2/2 (Rerank): Using Qwen/Qwen3-Reranker-8B (&lt; 0.01 RMB), processed in 8 batches.</div>
                <div>Results: Globally reranked across all topics and truncated to Top-${Number(document.getElementById('top-k').value)||20}.</div>
                <div style="margin-top:4px;">
                    Filters: ${
                        (() => {
                            const dm = document.getElementById('date-min').value || '';
                            const dx = document.getElementById('date-max').value || '';
                            const cs = Array.from(document.querySelectorAll('#country-list input[type="checkbox"]:checked')).map(el => el.value);
                            const cats = Array.from(document.querySelectorAll('#cs-category-list input[type="checkbox"]:checked')).map(el => el.value);
                            const parts = [];
                            if (dm || dx) parts.push(`Date[${dm||'..'} ~ ${dx||'..'}]`);
                            if (cs.length) parts.push(`Countries[${cs.join(', ')}]`);
                            if (cats.length) parts.push(`Categories[${cats.join(', ')}]`);
                            return parts.join(' ; ')
                        })()
                    }
                </div>
            </div>
        </div>
    `;

    // Results
    let recommendationsHtml = '';
    if (data.recommendations) {
        data.recommendations.forEach((paper, index) => {
            recommendationsHtml += `
                <div class="result-item">
                    <div class="result-header">
                        <div class="result-title">${index + 1}. ${paper.title}</div>
                        <div class="result-similarity">Relevance: ${paper.similarity.toFixed(4)}</div>
                    </div>
                    <div class="result-meta">
                        <span>👥 <strong>Authors:</strong> ${paper.authors}</span>
                        <span>🏷️ <strong>arXiv ID:</strong> ${paper.arxiv_id ? ('<a href=\"https://arxiv.org/abs/' + paper.arxiv_id + '\" target=\"_blank\" rel=\"noopener noreferrer\">' + paper.arxiv_id + '</a>') : ''}</span>
                        <!-- <span>📅 <strong>Year-Month:</strong> ${paper.year_month || paper.year}</span> -->
                        ${paper.date ? ('<span>📆 <strong>Date:</strong> ' + paper.date + '</span>') : ''}
                        <span>🏷️ <strong>Categories:</strong> ${paper.categories}</span>
                        <span>🏫 <strong>Affiliations:</strong> ${paper.affiliation}</span>
                        ${paper.country && String(paper.country).trim().toUpperCase() !== 'N/A' ? ('<span>🌍 <strong>Affiliation Country:</strong> ' + paper.country + '</span>') : ''}
                        ${paper.department && String(paper.department).trim().toUpperCase() !== 'N/A' ? ('<span>🏢 <strong>Department:</strong> ' + paper.department + '</span>') : ''}
                        ${paper.corresponding_email && String(paper.corresponding_email).trim() !== '' && String(paper.corresponding_email).trim().toUpperCase() !== 'N/A' ? ('<span>📧 <strong>Corresponding Email:</strong> <a href="mailto:' + paper.corresponding_email + '">' + paper.corresponding_email + '</a></span>') : ''}
                    </div>
                    <div class="result-abstract">
                        <strong>Abstract:</strong> ${paper.abstract}
                    </div>
                    <div class="button-group" style="margin-top:10px;">
                        <button class="btn btn-muted" data-role="like" data-idx="${index}" onclick="toggleLike(${index})">👍 Like</button>
                        <button class="btn btn-muted" data-role="dislike" data-idx="${index}" onclick="toggleDislike(${index})">👎 Dislike</button>
                    </div>
                </div>
            `;
        });
    }

    resultsDiv.innerHTML = statsHtml + recommendationsHtml;

    // 显示导出按钮
    document.getElementById('exportBtn').style.display = 'inline-block';
    document.getElementById('exportPdfContainer').style.display = 'inline-flex';
    // apply selected styles after render
    applyLikeStyles();
}

function displayError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="error">
            <strong>❌ Error:</strong> ${message}
        </div>
    `;
}

function clearResults() {
    const ta = document.getElementById('workshop-description');
    const defaultText = `Enter your workshop CFP; the system will recommend relevant papers...

💡 Example
"Call for Papers: We welcome both original research papers and submissions based on arXiv pre-prints.
Topics of interest include, but are not limited to:
●   Data management systems for generative AI
●   Agentic AI systems and algorithms"`;
    ta.value = defaultText;
    if (ta.classList) ta.classList.add('placeholder-text');
    document.getElementById('top-k').value = '20';
    document.getElementById('results').innerHTML = '';
    document.getElementById('exportBtn').style.display = 'none';
    document.getElementById('exportPdfContainer').style.display = 'none';
}

async function exportToExcel() {
    try {
        if (!window.__lastRecommendations || window.__lastRecommendations.length === 0) {
            alert('没有推荐结果可导出');
            return;
        }

        const exportData = {
            recommendations: window.__lastRecommendations,
            query_info: window.__lastQuery || {}
        };

        const response = await fetch('/api/export_excel', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(exportData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '导出失败');
        }

        // 创建下载链接
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `CFC_Paper_Recommendations_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

    } catch (error) {
        alert('导出失败: ' + error.message);
    }
}

function togglePdfOptions() {
    const options = document.getElementById('pdfOptions');
    if (options.style.display === 'none') {
        options.style.display = 'block';
    } else {
        options.style.display = 'none';
    }
}

async function exportToPDF() {
    try {
        if (!window.__lastRecommendations || window.__lastRecommendations.length === 0) {
            alert('没有推荐结果可导出');
            return;
        }

        // 获取选择的过滤选项
        const selectedFilter = document.querySelector('input[name="pdfFilter"]:checked').value;

        // 根据过滤选项筛选论文
        let papersToExport = [];
        if (selectedFilter === 'all') {
            papersToExport = window.__lastRecommendations;
        } else if (selectedFilter === 'liked') {
            const liked = window.__likedIds || new Set();
            papersToExport = window.__lastRecommendations.filter(p => liked.has(p.arxiv_id));
        } else if (selectedFilter === 'except_disliked') {
            const disliked = window.__dislikedIds || new Set();
            papersToExport = window.__lastRecommendations.filter(p => !disliked.has(p.arxiv_id));
        }

        if (papersToExport.length === 0) {
            alert('没有符合条件的论文可导出');
            return;
        }

        // 准备导出数据
        const exportData = {
            papers: papersToExport,
            filter: selectedFilter,
            query_info: window.__lastQuery || {}
        };

        const response = await fetch('/api/export_pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(exportData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'PDF导出失败');
        }

        // 创建下载链接
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `CFC_Paper_PDFs_${selectedFilter}_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

    } catch (error) {
        alert('PDF导出失败: ' + error.message);
    }
}

async function sendFeedback(action, idx) {
    try {
        const description = document.getElementById('workshop-description').value.trim();
        const rec = window.__lastRecommendations && window.__lastRecommendations[idx] ? window.__lastRecommendations[idx] : null;
        const payload = {
            action: action,
            workshop_description: description,
            rank: idx + 1,
            similarity: rec && typeof rec.similarity === 'number' ? rec.similarity : null,
            paper: rec,
            arxiv_id: rec ? rec.arxiv_id : null,
            title: rec ? rec.title : null,
            authors: rec ? rec.authors : null,
            recommendation_params: window.__lastQuery || null,
            recommendation_meta: window.__lastQueryResponse || null,
            page_url: window.location.href,
            timestamp: new Date().toISOString(),
        };
        await fetch('/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    } catch (e) {
        console.error('feedback failed', e);
    }
}

// like/dislike state handling and UI
function applyLikeStyles() {
    const cards = document.querySelectorAll('.result-item');
    if (!cards || !window.__lastRecommendations) return;
    cards.forEach((card, i) => {
        const rec = window.__lastRecommendations[i];
        if (!rec) return;
        const likeBtn = card.querySelector('button[data-role="like"]');
        const dislikeBtn = card.querySelector('button[data-role="dislike"]');
        if (!likeBtn || !dislikeBtn) return;
        const id = rec.arxiv_id;
        const liked = window.__likedIds.has(id);
        const disliked = window.__dislikedIds.has(id);
        likeBtn.classList.toggle('btn-liked', liked);
        dislikeBtn.classList.toggle('btn-disliked', disliked);
    });
}

function toggleLike(idx) {
    const rec = window.__lastRecommendations && window.__lastRecommendations[idx];
    if (!rec || !rec.arxiv_id) return;
    const id = rec.arxiv_id;
    if (window.__likedIds.has(id)) {
        window.__likedIds.delete(id);
    } else {
        window.__likedIds.add(id);
        // ensure mutual exclusion
        window.__dislikedIds.delete(id);
    }
    applyLikeStyles();
    // fire feedback event (like or remove-like treated as like again for logging)
    sendFeedback('like', idx);
}

function toggleDislike(idx) {
    const rec = window.__lastRecommendations && window.__lastRecommendations[idx];
    if (!rec || !rec.arxiv_id) return;
    const id = rec.arxiv_id;
    if (window.__dislikedIds.has(id)) {
        window.__dislikedIds.delete(id);
    } else {
        window.__dislikedIds.add(id);
        // ensure mutual exclusion
        window.__likedIds.delete(id);
    }
    applyLikeStyles();
    sendFeedback('dislike', idx);
}

// 保存最近一次推荐结果，便于发送反馈
(function(){
    const _displayResults = displayResults;
    displayResults = function(data){
        window.__lastRecommendations = data.recommendations || [];
        _displayResults(data);
    }
})();

async function discoverTopics() {
    const dateMinVal = document.getElementById('date-min').value.trim();
    const dateMaxVal = document.getElementById('date-max').value.trim();
    const countries = Array.from(document.querySelectorAll('#country-list input[type="checkbox"]:checked')).map(el => el.value);
    const categories = Array.from(document.querySelectorAll('#cs-category-list input[type="checkbox"]:checked')).map(el => el.value);
    const nTopics = parseInt(document.getElementById('n-topics').value) || 10;

    // 显示加载状态
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';

    try {
        const requestId = Math.random().toString(36).slice(2) + Date.now().toString(36);
        window.__lastQuery = {
            request_id: requestId,
            mode: 'discover_topics',
            date_min: dateMinVal || null,
            date_max: dateMaxVal || null,
            countries: countries,
            categories: categories,
            n_topics: nTopics,
        };

        const response = await fetch('/api/discover_topics', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                date_min: window.__lastQuery.date_min,
                date_max: window.__lastQuery.date_max,
                countries: window.__lastQuery.countries,
                categories: window.__lastQuery.categories,
                n_topics: window.__lastQuery.n_topics,
            })
        });

        const start = await response.json();
        if (!response.ok || !start.job_id) {
            displayError((start && start.error) || 'Failed to start topic discovery');
            return;
        }
        __currentJob = start.job_id;
        // show progress UI
        document.getElementById('progressBox').style.display = 'block';
        updateProgressUI(0, 'Queued...');
        // start polling
        pollProgressForTopics(__currentJob, requestId);
    } catch (error) {
        displayError('Network error: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

async function pollProgressForTopics(jobId, requestId){
    if (__progressTimer) clearInterval(__progressTimer);
    __progressTimer = setInterval(async () => {
        try {
            const resp = await fetch('/api/progress?job_id=' + encodeURIComponent(jobId));
            const pj = await resp.json();
            if (!resp.ok) {
                updateProgressUI(0, pj.error || 'Error');
                clearInterval(__progressTimer);
                return;
            }
            updateProgressUI(pj.percent || 0, pj.message || 'Processing...');
            if (pj.done) {
                clearInterval(__progressTimer);
                // fetch result
                const rr = await fetch('/api/result?job_id=' + encodeURIComponent(jobId));
                const data = await rr.json();
                if (rr.ok) {
                    displayTopicsResult(data);
                    document.getElementById('progressBox').style.display = 'none';
                } else {
                    displayError(data.error || 'Failed to load result');
                }
            }
        } catch(e){
            updateProgressUI(0, 'Network error');
            clearInterval(__progressTimer);
        }
    }, 600);
}

function displayTopicsResult(data) {
    const resultsDiv = document.getElementById('results');
    
    // Stats
    const statsHtml = `
        <div class="stats">
            <h3>📈 Topic Discovery Results</h3>
            <p>Discovered <strong>${data.n_topics}</strong> topics from <strong>${data.total_papers}</strong> papers</p>
            <div style="margin-top:6px;color:#666;">
                <div>Used BERTopic clustering with UMAP dimensionality reduction and HDBSCAN clustering</div>
                <div>Filters: ${
                    (() => {
                        const dm = document.getElementById('date-min').value || '';
                        const dx = document.getElementById('date-max').value || '';
                        const cs = Array.from(document.querySelectorAll('#country-list input[type="checkbox"]:checked')).map(el => el.value);
                        const cats = Array.from(document.querySelectorAll('#cs-category-list input[type="checkbox"]:checked')).map(el => el.value);
                        const parts = [];
                        if (dm || dx) parts.push(`Date[${dm||'..'} ~ ${dx||'..'}]`);
                        if (cs.length) parts.push(`Countries[${cs.join(', ')}]`);
                        if (cats.length) parts.push(`Categories[${cats.join(', ')}]`);
                        return parts.length > 0 ? parts.join(' ; ') : 'None';
                    })()
                }
                </div>
            </div>
        </div>
    `;
    
    // Topics
    let topicsHtml = '';
    if (data.topics) {
        data.topics.forEach((topic, index) => {
            topicsHtml += `
                <div class="topic-item">
                    <div class="topic-header">
                        <div class="topic-title">${index + 1}. ${topic.name}</div>
                        <div class="topic-count">${topic.count} papers</div>
                    </div>
                    <div class="topic-keywords">
                        <strong>Keywords:</strong> ${topic.keywords.join(', ')}
                    </div>
                    <div class="topic-papers">
                        <strong>Representative Papers:</strong>
                        ${topic.representative_papers.map(p => `
                            <div class="topic-paper-item">
                                <div class="topic-paper-title">${p.title}</div>
                                <div class="topic-paper-meta">
                                    👥 ${p.authors} | 
                                    🏷️ <a href="https://arxiv.org/abs/${p.arxiv_id}" target="_blank" rel="noopener noreferrer">${p.arxiv_id}</a>
                                </div>
                                <div style="color:#666;font-size:0.9em;margin-top:5px;">${p.abstract}...</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        });
    }
    
    resultsDiv.innerHTML = statsHtml + topicsHtml;
}

// Toggle topic discovery section visibility
function toggleTopicDiscoverySection() {
    const section = document.getElementById('topic-discovery-section');
    if (section) {
        section.style.display = section.style.display === 'none' ? 'block' : 'none';
    }
}

// selectable placeholder behavior + Ctrl+Enter submit
document.addEventListener('DOMContentLoaded', function() {
    const ta = document.getElementById('workshop-description');
    const defaultText = `Enter your workshop CFP; the system will recommend relevant papers...

💡 Example
"Call for Papers: We welcome both original research papers and submissions based on arXiv pre-prints.
Topics of interest include, but are not limited to:
●   Data management systems for generative AI
●   Agentic AI systems and algorithms"`;
    if (ta.value === defaultText) {
        if (ta.classList) ta.classList.add('placeholder-text');
    }
    function isPlaceholderActive() {
        return (ta.classList && ta.classList.contains('placeholder-text')) && ta.value === defaultText;
    }
    ta.addEventListener('blur', function(){
        if (this.value.trim() === '') {
            this.value = defaultText;
            if (this.classList) this.classList.add('placeholder-text');
        }
    });
    ta.addEventListener('keydown', function(e){
        // Ctrl+Enter submission
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            recommendPapers();
            return;
        }
        // When placeholder is shown, clear only when user types real input
        if (isPlaceholderActive()) {
            const navKeys = ['Shift','Control','Alt','Meta','Tab','CapsLock','Escape','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','PageUp','PageDown','Insert','Delete'];
            const isModifier = e.ctrlKey || e.metaKey || e.altKey;
            const isNav = navKeys.includes(e.key);
            const isPrintable = e.key && e.key.length === 1;
            const isEnter = e.key === 'Enter';
            const isBackspace = e.key === 'Backspace';
            const isDelete = e.key === 'Delete';
            if (!isModifier && (isPrintable || isEnter || isBackspace || isDelete)) {
                e.preventDefault();
                this.value = '';
                if (this.classList) this.classList.remove('placeholder-text');
                if (isPrintable) {
                    this.value = e.key;
                } else if (isEnter) {
                    // start with a newline if desired; keep empty by default
                } else if (isBackspace || isDelete) {
                    // keep empty
                }
            }
        }
    });
    ta.addEventListener('paste', function(e){
        if (isPlaceholderActive()) {
            e.preventDefault();
            const text = (e.clipboardData || window.clipboardData).getData('text');
            this.value = text;
            if (this.classList) this.classList.remove('placeholder-text');
        }
    });
    
    // Show topic discovery section by default
    toggleTopicDiscoverySection();
});
