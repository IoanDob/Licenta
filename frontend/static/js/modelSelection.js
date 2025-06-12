export function updateModelDescription() {
    const selectedModel = document.querySelector('input[name="modelType"]:checked');
    if (!selectedModel) return;
    const modelType = selectedModel.value;
    const descriptions = {
        'random_forest': {
            title: 'Random Forest',
            description: 'Advanced ensemble method combining multiple decision trees for superior accuracy.',
            features: [
                'Excellent for complex patterns',
                'Handles missing values automatically',
                'Provides feature importance rankings',
                'Robust against outliers'
            ],
            considerations: [
                'Longer processing time',
                'Higher memory usage',
                'Less interpretable results'
            ],
            icon: 'fa-tree',
            color: 'success'
        },
        'logistic_regression': {
            title: 'Logistic Regression',
            description: 'Fast linear statistical method with highly interpretable results.',
            features: [
                'Lightning-fast processing',
                'Clear coefficient interpretation',
                'Memory efficient',
                'Works well with linear patterns'
            ],
            considerations: [
                'Assumes linear relationships',
                'Sensitive to feature scaling',
                'May miss complex patterns'
            ],
            icon: 'fa-chart-line',
            color: 'primary'
        }
    };
    const desc = descriptions[modelType];
    let descriptionDiv = document.getElementById('modelDescription');
    if (!descriptionDiv) {
        const modelSection = document.querySelector('.mb-4');
        if (modelSection) {
            descriptionDiv = document.createElement('div');
            descriptionDiv.id = 'modelDescription';
            descriptionDiv.className = 'mt-3';
            modelSection.appendChild(descriptionDiv);
        }
    }
    if (descriptionDiv && desc) {
        descriptionDiv.innerHTML = `
            <div class="card border-${desc.color}">
                <div class="card-header bg-${desc.color} text-white">
                    <h6 class="mb-0">
                        <i class="fas ${desc.icon} me-2"></i>${desc.title}
                    </h6>
                </div>
                <div class="card-body">
                    <p class="text-muted mb-3">${desc.description}</p>
                    <div class="row">
                        <div class="col-md-6">
                            <h6 class="text-success"><i class="fas fa-check me-1"></i>Advantages:</h6>
                            <ul class="list-unstyled small">
                                ${desc.features.map(feature => `<li class="mb-1">• ${feature}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h6 class="text-warning"><i class="fas fa-info-circle me-1"></i>Considerations:</h6>
                            <ul class="list-unstyled small">
                                ${desc.considerations.map(consideration => `<li class="mb-1">• ${consideration}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}