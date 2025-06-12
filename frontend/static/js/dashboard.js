document.addEventListener('DOMContentLoaded', function() {
    loadRecentAnalysis();
    loadQuickStats();
    loadFinancialNews();
});

async function loadRecentAnalysis() {
    const recentAnalysisDiv = document.getElementById('recentAnalysis');
    
    try {
        const response = await axios.get('/api/history/', {
            headers: authManager.getAuthHeaders()
        });
        
        const analyses = response.data.slice(0, 3); // Get last 3
        
        if (analyses.length === 0) {
            recentAnalysisDiv.innerHTML = '<p class="text-muted">No analyses yet</p>';
            return;
        }
        
        let html = '';
        analyses.forEach(analysis => {
            const date = new Date(analysis.created_at).toLocaleDateString();
            html += `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div>
                        <strong>${analysis.filename}</strong><br>
                        <small class="text-muted">${date} - ${analysis.model_used}</small>
                    </div>
                    <span class="badge bg-${analysis.fraud_detected > 0 ? 'danger' : 'success'}">
                        ${analysis.fraud_detected} fraud
                    </span>
                </div>
            `;
        });
        
        recentAnalysisDiv.innerHTML = html;
        
    } catch (error) {
        recentAnalysisDiv.innerHTML = '<p class="text-muted">Error loading analyses</p>';
    }
}

async function loadQuickStats() {
    try {
        const response = await axios.get('/api/history/', {
            headers: authManager.getAuthHeaders()
        });
        
        const analyses = response.data;
        const totalAnalyses = analyses.length;
        const totalFraud = analyses.reduce((sum, analysis) => sum + analysis.fraud_detected, 0);
        
        document.getElementById('totalAnalyses').textContent = totalAnalyses;
        document.getElementById('fraudDetected').textContent = totalFraud;
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function loadFinancialNews() {
    const newsDiv = document.getElementById('financialNews');
    
    // Simulated financial news - in real app, you'd fetch from a news API
    const mockNews = [
        {
            title: "New AI Technologies in Financial Crime Detection",
            summary: "Latest advancements in machine learning for fraud prevention...",
            time: "2 hours ago"
        },
        {
            title: "Global Financial Fraud Losses Reach $5.8 Billion",
            summary: "Annual report shows increasing need for automated detection systems...",
            time: "5 hours ago"
        },
        {
            title: "Regulatory Updates on AML Compliance",
            summary: "New guidelines for anti-money laundering in digital banking...",
            time: "1 day ago"
        }
    ];
    
    let html = '';
    mockNews.forEach(news => {
        html += `
            <div class="border-bottom pb-3 mb-3">
                <h6>${news.title}</h6>
                <p class="text-muted mb-1">${news.summary}</p>
                <small class="text-muted">${news.time}</small>
            </div>
        `;
    });
    
    newsDiv.innerHTML = html;
}