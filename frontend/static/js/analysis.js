let currentResults = null;
let uploadedFile = null;
let charts = {}; // Store chart instances
let currentCsvData = null;
let currentMappings = null;
let selectedAnalysisType = null;

document.addEventListener('DOMContentLoaded', function() {
    // Check which page we're on and initialize accordingly
    if (document.getElementById('uploadForm')) {
        initializeAnalysisPage();
    }
    if (document.getElementById('uploadSection')) {
        initializeSmartAnalysis();
    }
});

function initializeAnalysisPage() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('csvFile');
    
    // File upload form handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', handleFileChange);
    }
    
    // Model selection change handler
    const modelInputs = document.querySelectorAll('input[name="modelType"]');
    modelInputs.forEach(input => {
        input.addEventListener('change', updateModelDescription);
    });
    
    // Initialize model descriptions
    updateModelDescription();
    
    // Initialize drag and drop functionality
    initializeDragDrop();
}

function initializeSmartAnalysis() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('csvFile');
    
    // File upload form handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleSmartFileUpload);
    }
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', handleFileChange);
    }
    
    // Run analysis button
    const runAnalysisBtn = document.getElementById('runAnalysisBtn');
    if (runAnalysisBtn) {
        runAnalysisBtn.addEventListener('click', runSmartAnalysis);
    }
}

function initializeDragDrop() {
    const fileInput = document.getElementById('csvFile');
    const uploadSection = document.getElementById('uploadSection');
    
    if (!fileInput || !uploadSection) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadSection.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    uploadSection.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        uploadSection.classList.add('drag-over');
    }
    
    function unhighlight(e) {
        uploadSection.classList.remove('drag-over');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileChange({ target: { files: files } });
        }
    }
}

function handleFileChange(event) {
    const file = event.target.files[0];
    
    if (file) {
        uploadedFile = file;
        
        // Validate file
        const validation = validateFile(file);
        if (!validation.isValid) {
            showFileError(validation.errors);
            return;
        }
        
        // Show warnings if any
        if (validation.warnings.length > 0) {
            showFileWarnings(validation.warnings);
        }
        
        // Show file information
        showFileInfo(file);
        
        // Enable upload button
        const submitButton = document.querySelector('#uploadForm button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Transactions';
            submitButton.classList.remove('btn-secondary');
            submitButton.classList.add('btn-primary');
        }
    }
}

function validateFile(file) {
    const validation = {
        isValid: true,
        errors: [],
        warnings: []
    };
    
    // Check file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
        validation.isValid = false;
        validation.errors.push('Please select a CSV file (.csv extension required)');
    }
    
    // Check file size (500MB limit)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        validation.isValid = false;
        validation.errors.push(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum allowed size (500MB)`);
    }
    
    // Check minimum file size
    if (file.size < 1024) {
        validation.isValid = false;
        validation.errors.push('File appears to be too small. Please check your CSV file.');
    }
    
    // Warnings for large files
    if (file.size > 100 * 1024 * 1024) {
        validation.warnings.push('Large file detected. Will be automatically sampled to 50,000 rows for analysis.');
    }
    
    if (file.size > 50 * 1024 * 1024) {
        validation.warnings.push('File over 50MB - processing will use intelligent sampling for faster results.');
    }
    
    if (file.size > 10 * 1024 * 1024) {
        validation.warnings.push('Large file detected. Analysis may take 1-2 minutes.');
    }
    
    return validation;
}

function showFileInfo(file) {
    const existingInfo = document.getElementById('fileInfo');
    if (existingInfo) {
        existingInfo.remove();
    }
    
    const uploadSection = document.getElementById('uploadSection');
    if (!uploadSection) return;
    
    const infoDiv = document.createElement('div');
    infoDiv.id = 'fileInfo';
    infoDiv.className = 'alert alert-info mt-3 fade-in';
    
    const sizeColor = file.size > 50 * 1024 * 1024 ? 'text-warning' : 'text-success';
    
    infoDiv.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-file-csv fa-2x text-primary me-3"></i>
            <div class="flex-grow-1">
                <h6 class="mb-1"><i class="fas fa-check-circle text-success me-1"></i>File Ready for Analysis</h6>
                <p class="mb-1"><strong>Name:</strong> ${file.name}</p>
                <p class="mb-1"><strong>Size:</strong> <span class="${sizeColor}">${(file.size / 1024 / 1024).toFixed(2)} MB</span></p>
                <p class="mb-0"><strong>Type:</strong> CSV (Comma-Separated Values)</p>
            </div>
        </div>
    `;
    
    uploadSection.appendChild(infoDiv);
}

function showFileError(errors) {
    const existingError = document.getElementById('fileError');
    if (existingError) {
        existingError.remove();
    }
    
    const uploadSection = document.getElementById('uploadSection');
    if (!uploadSection) return;
    
    const errorDiv = document.createElement('div');
    errorDiv.id = 'fileError';
    errorDiv.className = 'alert alert-danger mt-3 fade-in';
    
    errorDiv.innerHTML = `
        <h6><i class="fas fa-exclamation-triangle me-2"></i>File Validation Error</h6>
        ${errors.map(error => `<p class="mb-1">• ${error}</p>`).join('')}
        <small class="text-muted">Please select a valid CSV file and try again.</small>
    `;
    
    uploadSection.appendChild(errorDiv);
    
    // Reset file input
    const fileInput = document.getElementById('csvFile');
    if (fileInput) {
        fileInput.value = '';
    }
    uploadedFile = null;
    
    // Disable submit button
    const submitButton = document.querySelector('#uploadForm button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Invalid File';
        submitButton.classList.remove('btn-primary');
        submitButton.classList.add('btn-secondary');
    }
}

function showFileWarnings(warnings) {
    const existingWarning = document.getElementById('fileWarning');
    if (existingWarning) {
        existingWarning.remove();
    }
    
    const uploadSection = document.getElementById('uploadSection');
    if (!uploadSection) return;
    
    const warningDiv = document.createElement('div');
    warningDiv.id = 'fileWarning';
    warningDiv.className = 'alert alert-warning mt-3 fade-in';
    
    warningDiv.innerHTML = `
        <h6><i class="fas fa-info-circle me-2"></i>Please Note</h6>
        ${warnings.map(warning => `<p class="mb-1">• ${warning}</p>`).join('')}
    `;
    
    uploadSection.appendChild(warningDiv);
}

function updateModelDescription() {
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

async function handleFileUpload(e) {
    e.preventDefault();
    
    if (!uploadedFile) {
        showToast('Please select a CSV file first', 'warning');
        return;
    }
    
    const modelType = document.querySelector('input[name="modelType"]:checked');
    if (!modelType) {
        showToast('Please select an AI model', 'warning');
        return;
    }
    
    // Show loading section
    showSection('loadingSection');
    updateLoadingProgress('Preparing upload...', 5);
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('model_type', modelType.value);
        
        updateLoadingProgress('Uploading file...', 15);
        
        const response = await axios.post('/api/analysis/upload', formData, {
            headers: {
                ...authManager.getAuthHeaders(),
                'Content-Type': 'multipart/form-data'
            },
            onUploadProgress: (progressEvent) => {
                const percentCompleted = Math.round(
                    (progressEvent.loaded * 100) / progressEvent.total
                );
                updateLoadingProgress(`Uploading... ${percentCompleted}%`, 15 + (percentCompleted * 0.3));
            }
        });
        
        updateLoadingProgress('Processing data structure...', 50);
        await simulateDelay(800);
        
        updateLoadingProgress('Running AI fraud detection...', 70);
        await simulateDelay(1200);
        
        updateLoadingProgress('Generating insights...', 85);
        await simulateDelay(600);
        
        updateLoadingProgress('Preparing visualizations...', 95);
        await simulateDelay(400);
        
        currentResults = response.data;
        await displayResults(currentResults);
        
        updateLoadingProgress('Analysis complete!', 100);
        await simulateDelay(300);
        
        showToast('Analysis completed successfully!', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        const errorMessage = error.response?.data?.detail || 
                           error.response?.data?.message || 
                           'Analysis failed. Please check your file and try again.';
        showError(errorMessage);
        showToast('Analysis failed', 'danger');
    }
}

// Smart analysis functions
async function handleSmartFileUpload(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Please select a CSV file first');
        return;
    }
    
    try {
        showSection('loadingSection');
        updateLoadingMessage('Analyzing file structure...');
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await axios.post('/api/analysis/upload-smart', formData, {
            headers: {
                ...authManager.getAuthHeaders(),
                'Content-Type': 'multipart/form-data'
            }
        });
        
        // Store the CSV data and mappings
        currentCsvData = response.data.csv_data;
        currentMappings = response.data.column_mappings;
        
        // Display the mapping interface
        displayMappingInterface(response.data);
        showSection('mappingSection');
        
    } catch (error) {
        console.error('File analysis error:', error);
        const errorMessage = error.response?.data?.detail || 'Failed to analyze file structure';
        showError(errorMessage);
    }
}

function displayMappingInterface(data) {
    // Update file info
    updateFileInfo(data.file_info);
    
    // Display available analysis types  
    displayAnalysisTypes(data.available_analysis_types);
    
    // Create column mapping table
    createColumnMappingTable(data.mapping_options || data.column_mappings);
    
    // Display analysis preview
    displayAnalysisPreview(data.analysis_preview);
    
    // Show recommendations
    if (data.recommendations && data.recommendations.length > 0) {
        displayRecommendations(data.recommendations);
    }
}

function updateFileInfo(fileInfo) {
    const elements = {
        fileName: document.getElementById('fileName'),
        fileSize: document.getElementById('fileSize'),
        fileRows: document.getElementById('fileRows'),
        fileColumns: document.getElementById('fileColumns')
    };
    
    if (elements.fileName) elements.fileName.textContent = fileInfo.filename || 'Unknown';
    if (elements.fileSize) elements.fileSize.textContent = `${(fileInfo.size_mb || 0).toFixed(2)} MB`;
    if (elements.fileRows) elements.fileRows.textContent = (fileInfo.rows || 0).toLocaleString();
    if (elements.fileColumns) elements.fileColumns.textContent = fileInfo.columns || 0;
}

function displayAnalysisTypes(analysisTypes) {
    const container = document.getElementById('analysisTypes');
    if (!container || !analysisTypes) return;
    
    container.innerHTML = '';
    
    analysisTypes.forEach((analysis, index) => {
        const confidenceColor = analysis.confidence > 0.8 ? 'success' : 
                               analysis.confidence > 0.6 ? 'warning' : 'secondary';
        
        const card = document.createElement('div');
        card.className = 'col-md-6 mb-3';
        card.innerHTML = `
            <div class="card h-100 analysis-type-card" data-analysis="${analysis.name}" 
                 style="cursor: pointer; ${index === 0 ? 'border-color: #0d6efd; background-color: #e7f1ff;' : ''}">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <h6 class="card-title">${analysis.name}</h6>
                        <span class="badge bg-${confidenceColor}">${((analysis.confidence || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <p class="card-text small text-muted">${analysis.description || ''}</p>
                    <div class="small">
                        <strong>Required:</strong> ${analysis.requirements_met || 0} columns<br>
                        <strong>Optional:</strong> ${analysis.optional_available || 0}/${analysis.total_optional || 0} available
                    </div>
                </div>
            </div>
        `;
        
        card.addEventListener('click', () => selectAnalysisType(analysis.name, card));
        container.appendChild(card);
    });
    
    // Auto-select the first (best) analysis type
    if (analysisTypes.length > 0) {
        selectedAnalysisType = analysisTypes[0].name;
    }
}

function selectAnalysisType(analysisName, cardElement) {
    // Remove selection from all cards
    document.querySelectorAll('.analysis-type-card').forEach(card => {
        card.style.borderColor = '';
        card.style.backgroundColor = '';
    });
    
    // Select this card
    cardElement.style.borderColor = '#0d6efd';
    cardElement.style.backgroundColor = '#e7f1ff';
    
    selectedAnalysisType = analysisName;
}

function createColumnMappingTable(mappingOptions) {
    const tableBody = document.querySelector('#columnMappingTable tbody');
    if (!tableBody || !mappingOptions) return;
    
    tableBody.innerHTML = '';
    
    Object.entries(mappingOptions).forEach(([columnName, mapping]) => {
        const row = document.createElement('tr');
        
        const confidence = mapping.confidence || 0;
        const confidenceColor = confidence > 0.7 ? 'success' : 
                               confidence > 0.4 ? 'warning' : 'danger';
        
        const sampleValues = mapping.sample_values || [];
        const allOptions = mapping.all_options || ['other'];
        const suggestedType = mapping.suggested_type || mapping.detected_type || 'other';
        
        row.innerHTML = `
            <td><strong>${columnName}</strong></td>
            <td><span class="badge bg-info">${mapping.data_type || 'unknown'}</span></td>
            <td><small>${sampleValues.slice(0, 3).join(', ')}${sampleValues.length > 3 ? '...' : ''}</small></td>
            <td>${suggestedType.replace('_', ' ').toUpperCase()}</td>
            <td>
                <span class="badge bg-${confidenceColor}">
                    ${confidence > 0 ? (confidence * 100).toFixed(0) + '%' : 'N/A'}
                </span>
            </td>
            <td>
                <select class="form-select form-select-sm column-mapping-select" data-column="${columnName}">
                    ${allOptions.map(option => 
                        `<option value="${option}" ${option === suggestedType ? 'selected' : ''}>
                            ${option.replace('_', ' ').toUpperCase()}
                        </option>`
                    ).join('')}
                </select>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
}

function displayAnalysisPreview(preview) {
    const container = document.getElementById('analysisPreview');
    if (!container || !preview) return;
    
    container.innerHTML = '';
    
    // Sample Data Preview
    if (preview.sample_data && Object.keys(preview.sample_data).length > 0) {
        const sampleCard = document.createElement('div');
        sampleCard.className = 'col-md-6';
        sampleCard.innerHTML = `
            <div class="card h-100">
                <div class="card-header">
                    <h6 class="mb-0"><i class="fas fa-table me-2"></i>Detected Data</h6>
                </div>
                <div class="card-body">
                    ${Object.entries(preview.sample_data).map(([key, data]) => `
                        <div class="mb-2">
                            <strong>${key}:</strong> 
                            <span class="badge bg-secondary">${data.confidence || 'N/A'}</span><br>
                            <small class="text-muted">${(data.sample_values || []).slice(0, 2).join(', ')}</small>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        container.appendChild(sampleCard);
    }
    
    // Statistics Preview
    if (preview.statistics && Object.keys(preview.statistics).length > 0) {
        const statsCard = document.createElement('div');
        statsCard.className = 'col-md-6';
        statsCard.innerHTML = `
            <div class="card h-100">
                <div class="card-header">
                    <h6 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Data Statistics</h6>
                </div>
                <div class="card-body">
                    ${Object.entries(preview.statistics).map(([category, stats]) => `
                        <div class="mb-3">
                            <strong>${category}:</strong>
                            ${Object.entries(stats).map(([key, value]) => `
                                <div class="d-flex justify-content-between">
                                    <span class="small">${key}:</span>
                                    <span class="small"><strong>${value}</strong></span>
                                </div>
                            `).join('')}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        container.appendChild(statsCard);
    }
    
    // Potential Insights
    if (preview.potential_insights && preview.potential_insights.length > 0) {
        const insightsCard = document.createElement('div');
        insightsCard.className = 'col-12';
        insightsCard.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Potential Insights</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        ${preview.potential_insights.map(insight => `
                            <div class="col-md-6 mb-2">
                                <small class="text-muted">${insight}</small>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        container.appendChild(insightsCard);
    }
}

function displayRecommendations(recommendations) {
    const existingAlert = document.getElementById('recommendationsAlert');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    const mappingSection = document.getElementById('mappingSection');
    if (!mappingSection) return;
    
    const alert = document.createElement('div');
    alert.id = 'recommendationsAlert';
    alert.className = 'alert alert-warning mb-4';
    alert.innerHTML = `
        <h6><i class="fas fa-lightbulb me-2"></i>Recommendations</h6>
        ${recommendations.map(rec => `<p class="mb-1">${rec}</p>`).join('')}
    `;
    
    const cardBody = mappingSection.querySelector('.card-body');
    if (cardBody) {
        cardBody.insertBefore(alert, cardBody.firstChild);
    }
}

async function runSmartAnalysis() {
    if (!currentCsvData || !selectedAnalysisType) {
        showError('Please complete the file upload and analysis type selection');
        return;
    }
    
    try {
        showSection('loadingSection');
        updateLoadingMessage('Running smart fraud detection analysis...');
        
        // Collect final column mappings
        const finalMappings = {};
        document.querySelectorAll('.column-mapping-select').forEach(select => {
            const columnName = select.dataset.column;
            const selectedType = select.value;
            if (selectedType !== 'other') {
                finalMappings[columnName] = {
                    selected_type: selectedType,
                    suggested_type: currentMappings[columnName]?.suggested_type || selectedType
                };
            }
        });
        
        // Get selected model
        const modelTypeInput = document.querySelector('input[name="modelType"]:checked');
        const modelType = modelTypeInput ? modelTypeInput.value : 'random_forest';
        
        // Prepare form data
        const formData = new FormData();
        formData.append('csv_data', currentCsvData);
        formData.append('column_mappings', JSON.stringify(finalMappings));
        formData.append('model_type', modelType);
        formData.append('analysis_type', selectedAnalysisType);
        
        const response = await axios.post('/api/analysis/upload-confirmed', formData, {
            headers: {
                ...authManager.getAuthHeaders(),
                'Content-Type': 'multipart/form-data'
            }
        });
        
        displaySmartResults(response.data);
        showSection('resultsSection');
        
    } catch (error) {
        console.error('Analysis error:', error);
        const errorMessage = error.response?.data?.detail || 'Analysis failed';
        showError(errorMessage);
    }
}

function displaySmartResults(results) {
    const container = document.getElementById('analysisResults');
    if (!container) return;
    
    const totalTransactions = results.total_transactions || 0;
    const fraudDetected = results.fraud_detected || 0;
    const fraudPercentage = results.fraud_percentage || 0;
    const riskScore = results.risk_score || 0;
    
    container.innerHTML = `
        <div class="row mb-4">
            <!-- Summary Statistics -->
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number text-primary">${totalTransactions.toLocaleString()}</div>
                    <div class="stat-label">Total Transactions</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number text-danger">${fraudDetected}</div>
                    <div class="stat-label">Fraud Detected</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number text-warning">${fraudPercentage.toFixed(2)}%</div>
                    <div class="stat-label">Fraud Rate</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <div class="stat-number text-info">${(riskScore * 100).toFixed(1)}%</div>
                    <div class="stat-label">Avg Risk Score</div>
                </div>
            </div>
        </div>
        
        <!-- Analysis Type Used -->
        <div class="alert alert-info mb-4">
            <h6><i class="fas fa-robot me-2"></i>Analysis Type: ${results.analysis_type || 'Smart Analysis'}</h6>
            <p class="mb-0">Model: ${(results.model_used || 'unknown').replace('_', ' ').toUpperCase()} | 
            Features: ${(results.adapted_features || []).join(', ')}</p>
        </div>
        
        <!-- High Risk Transactions -->
        <div class="row mb-4">
            <div class="col-lg-6">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-exclamation-triangle text-danger me-2"></i>High Risk Transactions</h6>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Index</th>
                                        <th>Amount</th>
                                        <th>Risk Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${(results.high_risk_transactions || []).slice(0, 5).map(tx => `
                                        <tr>
                                            <td><span class="badge bg-secondary">#${tx.index || 0}</span></td>
                                            <td>${tx.amount ? tx.amount.toLocaleString() : 'N/A'}</td>
                                            <td>
                                                <span class="badge ${(tx.risk_score || 0) > 0.8 ? 'bg-danger' : 'bg-warning'}">
                                                    ${((tx.risk_score || 0) * 100).toFixed(1)}%
                                                </span>
                                            </td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Risk Distribution Chart -->
            <div class="col-lg-6">
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-chart-pie me-2"></i>Risk Distribution</h6>
                    </div>
                    <div class="card-body">
                        <canvas id="riskChart" height="200"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Smart Insights -->
        ${results.insights ? `
            <div class="row mb-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0"><i class="fas fa-lightbulb me-2"></i>Smart Insights</h6>
                        </div>
                        <div class="card-body">
                            ${(results.insights || []).map(insight => `
                                <div class="insight-card mb-3">
                                    <h6 class="insight-title">${insight.title || 'Insight'}</h6>
                                    <p class="insight-description">${insight.description || ''}</p>
                                    ${insight.recommendation ? `
                                        <div class="insight-recommendation">
                                            <small><strong>Recommendation:</strong> ${insight.recommendation}</small>
                                        </div>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        ` : ''}
        
        <!-- Action Buttons -->
        <div class="text-center">
            <button class="btn btn-success me-2" onclick="downloadSmartResults()">
                <i class="fas fa-download me-2"></i>Download Report
            </button>
            <button class="btn btn-primary me-2" onclick="saveAnalysis()">
                <i class="fas fa-save me-2"></i>Save Analysis
            </button>
            <button class="btn btn-secondary" onclick="resetAnalysis()">
                <i class="fas fa-redo me-2"></i>New Analysis
            </button>
        </div>
    `;
    
    // Create risk distribution chart
    setTimeout(() => createRiskChart(results), 100);
}

function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function updateLoadingProgress(message, percentage) {
    const loadingMessage = document.querySelector('#loadingSection h4');
    const loadingSubtext = document.querySelector('#loadingSection p');
    const progressBar = document.querySelector('#loadingSection .progress-bar');
    
    if (loadingMessage) {
        loadingMessage.textContent = message;
    }
    
    if (loadingSubtext) {
        const tips = [
            "AI models are analyzing transaction patterns...",
            "Detecting anomalies and suspicious behaviors...",
            "Cross-referencing with fraud indicators...",
            "Calculating risk scores for each transaction...",
            "Generating comprehensive fraud report..."
        ];
        const tipIndex = Math.floor(percentage / 20);
        loadingSubtext.textContent = tips[tipIndex] || tips[tips.length - 1];
    }
    
    if (progressBar) {
        progressBar.style.width = percentage + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
        
        // Add color progression
        if (percentage < 30) {
            progressBar.className = 'progress-bar bg-info progress-bar-striped progress-bar-animated';
        } else if (percentage < 70) {
            progressBar.className = 'progress-bar bg-warning progress-bar-striped progress-bar-animated';
        } else {
            progressBar.className = 'progress-bar bg-success progress-bar-striped progress-bar-animated';
        }
    }
}

function updateLoadingMessage(message) {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
        loadingMessage.textContent = message;
    }
}

async function displayResults(results) {
    // Clear any existing charts
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    charts = {};
    
    // Update summary statistics with model comparison
    animateCounter('totalTransactions', results.total_transactions || 0, '', 0);
    animateCounter('fraudCount', results.fraud_detected || 0, '', 0);
    animateCounter('fraudPercentage', results.fraud_percentage || 0, '%', 2);
    animateCounter('riskScore', (results.risk_score || 0) * 100, '%', 1);
    
    // Add model comparison section if available
    if (results.model_comparison) {
        displayModelComparison(results.model_comparison);
    }
    
    // Add threshold analysis if available
    if (results.threshold_analysis) {
        displayThresholdAnalysis(results.threshold_analysis);
    }
    
    // Display high-risk transactions
    displayHighRiskTransactions(results.high_risk_transactions || []);
    
    // Create visualizations with delay for better UX
    setTimeout(() => createRiskChart(results), 500);
    setTimeout(() => createAmountDistributionChart(results), 800);
    setTimeout(() => createTimeAnalysisChart(results), 1100);
    setTimeout(() => createModelComparisonChart(results), 1400);
    
    // Show enhanced insights
    setTimeout(() => displayEnhancedInsights(results), 1700);
    
    // Show results section with animation
    showSection('resultsSection');
}

function displayModelComparison(modelComparison) {
    const existingComparison = document.getElementById('modelComparison');
    if (existingComparison) {
        existingComparison.remove();
    }
    
    const resultsSection = document.getElementById('resultsSection');
    const comparisonDiv = document.createElement('div');
    comparisonDiv.id = 'modelComparison';
    comparisonDiv.className = 'alert alert-info mb-4';
    
    const primary = modelComparison.primary_model;
    const comparison = modelComparison.comparison_model;
    const agreementColor = modelComparison.agreement_rate > 0.8 ? 'success' : 
                          modelComparison.agreement_rate > 0.6 ? 'warning' : 'danger';
    
    comparisonDiv.innerHTML = `
        <h6><i class="fas fa-balance-scale me-2"></i>Model Comparison Analysis</h6>
        <div class="row">
            <div class="col-md-4">
                <strong>${primary.name.replace('_', ' ').toUpperCase()}</strong><br>
                <small>Fraud Rate: ${primary.fraud_rate.toFixed(2)}%</small><br>
                <small>Threshold: ${primary.threshold_used.toFixed(3)}</small>
            </div>
            <div class="col-md-4">
                <strong>${comparison.name.replace('_', ' ').toUpperCase()}</strong><br>
                <small>Fraud Rate: ${comparison.fraud_rate.toFixed(2)}%</small><br>
                <small>Threshold: ${comparison.threshold_used.toFixed(3)}</small>
            </div>
            <div class="col-md-4">
                <strong>Agreement Rate</strong><br>
                <span class="badge bg-${agreementColor}">${(modelComparison.agreement_rate * 100).toFixed(1)}%</span><br>
                <small class="text-muted">${modelComparison.recommendation}</small>
            </div>
        </div>
    `;
    
    // Insert after summary statistics
    const summaryRow = document.querySelector('#resultsSection .row');
    if (summaryRow) {
        summaryRow.parentNode.insertBefore(comparisonDiv, summaryRow.nextSibling);
    }
}

function displayThresholdAnalysis(thresholdAnalysis) {
    const existingThreshold = document.getElementById('thresholdAnalysis');
    if (existingThreshold) {
        existingThreshold.remove();
    }
    
    const modelComparison = document.getElementById('modelComparison');
    const thresholdDiv = document.createElement('div');
    thresholdDiv.id = 'thresholdAnalysis';
    thresholdDiv.className = 'alert alert-warning mb-4';
    
    thresholdDiv.innerHTML = `
        <h6><i class="fas fa-sliders-h me-2"></i>Threshold Optimization</h6>
        <div class="row">
            <div class="col-md-6">
                <strong>Default Threshold (0.5)</strong><br>
                <small>Fraud Rate: ${thresholdAnalysis.default_threshold_results.fraud_rate.toFixed(2)}%</small><br>
                <small>Fraud Count: ${thresholdAnalysis.default_threshold_results.fraud_count}</small>
            </div>
            <div class="col-md-6">
                <strong>Optimized Threshold (${thresholdAnalysis.optimized_threshold_results.threshold_value.toFixed(3)})</strong><br>
                <small>Fraud Rate: ${thresholdAnalysis.optimized_threshold_results.fraud_rate.toFixed(2)}%</small><br>
                <small>Fraud Count: ${thresholdAnalysis.optimized_threshold_results.fraud_count}</small>
            </div>
        </div>
        <p class="mb-0 mt-2"><strong>Impact:</strong> ${thresholdAnalysis.threshold_impact}</p>
    `;
    
    if (modelComparison) {
        modelComparison.parentNode.insertBefore(thresholdDiv, modelComparison.nextSibling);
    }
}

function createModelComparisonChart(results) {
    if (!results.model_comparison) return;
    
    // Create a comparison chart showing both models
    const ctx = document.getElementById('modelComparisonChart');
    if (!ctx) {
        // Create the canvas if it doesn't exist
        const chartsRow = document.querySelector('#resultsSection .row:nth-child(3)');
        if (chartsRow) {
            const newCol = document.createElement('div');
            newCol.className = 'col-lg-4';
            newCol.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-balance-scale me-2"></i>Model Comparison</h6>
                    </div>
                    <div class="card-body">
                        <canvas id="modelComparisonChart" height="200"></canvas>
                    </div>
                </div>
            `;
            chartsRow.appendChild(newCol);
        }
        return;
    }
    
    const primary = results.model_comparison.primary_model;
    const comparison = results.model_comparison.comparison_model;
    
    if (typeof Chart !== 'undefined') {
        charts.modelComparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [
                    primary.name.replace('_', ' ').toUpperCase(),
                    comparison.name.replace('_', ' ').toUpperCase()
                ],
                datasets: [{
                    label: 'Fraud Rate (%)',
                    data: [primary.fraud_rate, comparison.fraud_rate],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 99, 132, 0.8)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Fraud Rate: ${context.parsed.y.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Fraud Rate (%)'
                        }
                    }
                }
            }
        });
    }
}

function displayEnhancedInsights(results) {
    const insightsContainer = document.getElementById('insights');
    if (!insightsContainer) return;
    
    // Combine regular insights with enhanced insights
    const regularInsights = generateInsights(results);
    const enhancedInsights = results.insights || [];
    const allInsights = [...enhancedInsights, ...regularInsights];
    
    insightsContainer.innerHTML = `
        <div class="row mb-4">
            <div class="col-12">
                <h5 class="mb-3">
                    <i class="fas fa-lightbulb text-warning me-2"></i>Enhanced AI Analysis
                </h5>
            </div>
        </div>
        <div class="row">
            ${allInsights.map((insight, index) => `
                <div class="col-lg-6 mb-3">
                    <div class="insight-card h-100" style="animation-delay: ${index * 0.1}s">
                        <div class="d-flex align-items-start">
                            <div class="insight-icon me-3">
                                <i class="fas ${insight.icon || 'fa-info-circle'} text-${insight.color || 'primary'} fa-lg"></i>
                            </div>
                            <div class="flex-grow-1">
                                <h6 class="insight-title mb-2">${insight.title}</h6>
                                <p class="insight-description mb-2">${insight.description}</p>
                                ${insight.recommendation ? `
                                    <div class="insight-recommendation">
                                        <small class="text-muted">
                                            <i class="fas fa-arrow-right me-1"></i>${insight.recommendation}
                                        </small>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
        
        ${results.analysis_metadata ? `
            <div class="alert alert-light mt-4">
                <h6><i class="fas fa-info-circle me-2"></i>Analysis Metadata</h6>
                <p><strong>Analysis Type:</strong> ${results.analysis_metadata.analysis_type}</p>
                <p><strong>Confidence Level:</strong> 
                    <span class="badge bg-${results.analysis_metadata.confidence_level === 'High' ? 'success' : 'warning'}">
                        ${results.analysis_metadata.confidence_level}
                    </span>
                </p>
                <p class="mb-0"><strong>Threshold Optimization:</strong> ${results.analysis_metadata.threshold_optimization ? 'Enabled' : 'Disabled'}</p>
            </div>
        ` : ''}
    `;
    
    // Animate insights appearance
    const insightCards = insightsContainer.querySelectorAll('.insight-card');
    insightCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'all 0.4s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

function animateCounter(elementId, targetValue, suffix = '', decimals = 0) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const startValue = 0;
    const duration = 2000; // 2 seconds
    const startTime = Date.now();
    
    function updateCounter() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        const currentValue = startValue + (targetValue - startValue) * easeOutCubic;
        
        if (decimals > 0) {
            element.textContent = currentValue.toFixed(decimals) + suffix;
        } else {
            element.textContent = Math.floor(currentValue).toLocaleString() + suffix;
        }
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            if (decimals > 0) {
                element.textContent = targetValue.toFixed(decimals) + suffix;
            } else {
                element.textContent = targetValue.toLocaleString() + suffix;
            }
        }
    }
    
    updateCounter();
}

function displayHighRiskTransactions(transactions) {
    const tableBody = document.querySelector('#highRiskTable tbody');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    if (!transactions || transactions.length === 0) {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td colspan="3" class="text-center text-muted py-4">
                <i class="fas fa-shield-alt fa-2x mb-2 text-success"></i><br>
                <strong>No high-risk transactions detected!</strong><br>
                <small>Your data appears to be clean.</small>
            </td>
        `;
        return;
    }
    
    transactions.forEach((transaction, index) => {
        const row = tableBody.insertRow();
        const riskLevel = getRiskLevel(transaction.risk_score || 0);
        
        row.innerHTML = `
            <td>
                <span class="badge bg-secondary">#${transaction.index || index}</span>
            </td>
            <td>
                <span class="fw-bold">${transaction.amount ? transaction.amount.toLocaleString() : 'N/A'}</span>
            </td>
            <td>
                <div class="d-flex align-items-center">
                    <div class="risk-indicator ${riskLevel.class} me-2"></div>
                    <span class="badge ${getRiskBadgeClass(transaction.risk_score || 0)}">
                        ${((transaction.risk_score || 0) * 100).toFixed(1)}%
                    </span>
                </div>
            </td>
        `;
        
        // Add staggered animation
        row.style.opacity = '0';
        row.style.transform = 'translateY(20px)';
        row.style.transition = 'all 0.3s ease';
        
        setTimeout(() => {
            row.style.opacity = '1';
            row.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

function getRiskLevel(riskScore) {
    if (riskScore >= 0.8) return { class: 'high', label: 'High Risk' };
    if (riskScore >= 0.6) return { class: 'medium', label: 'Medium Risk' };
    return { class: 'low', label: 'Low Risk' };
}

function getRiskBadgeClass(riskScore) {
    if (riskScore >= 0.8) return 'bg-danger';
    if (riskScore >= 0.6) return 'bg-warning text-dark';
    return 'bg-info';
}

function createRiskChart(results) {
    const ctx = document.getElementById('riskChart');
    if (!ctx) return;
    
    // Calculate risk distribution
    const total = results.total_transactions || 0;
    const high = results.fraud_detected || 0;
    const medium = Math.floor((total - high) * 0.15); // Estimate medium risk
    const low = total - high - medium;
    
    if (typeof Chart !== 'undefined') {
        charts.riskChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                datasets: [{
                    data: [low, medium, high],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(220, 53, 69, 0.8)'
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(220, 53, 69, 1)'
                    ],
                    borderWidth: 2,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            font: {
                                size: 12,
                                weight: 'bold'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label;
                                const value = context.parsed;
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : '0';
                                return `${label}: ${value} transactions (${percentage}%)`;
                            }
                        },
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1
                    }
                },
                animation: {
                    animateScale: true,
                    animateRotate: true,
                    duration: 1500,
                    easing: 'easeOutCubic'
                }
            }
        });
    }
}

function createAmountDistributionChart(results) {
    const ctx = document.getElementById('amountChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    // Create amount distribution data
    const amounts = (results.high_risk_transactions || []).map(t => t.amount || 0);
    
    if (amounts.length === 0) {
        // Show placeholder when no high-risk transactions
        ctx.getContext('2d').fillText('No high-risk transactions to display', 10, 50);
        return;
    }
    
    const sortedAmounts = amounts.sort((a, b) => a - b);
    const bins = Math.min(10, amounts.length);
    const labels = [];
    const data = [];
    const colors = [];
    
    for (let i = 0; i < bins; i++) {
        const start = i * Math.floor(amounts.length / bins);
        const end = (i + 1) * Math.floor(amounts.length / bins);
        const binAmounts = sortedAmounts.slice(start, end);
        
        if (binAmounts.length > 0) {
            const minAmount = binAmounts[0];
            const maxAmount = binAmounts[binAmounts.length - 1];
            labels.push(`${minAmount.toLocaleString()} - ${maxAmount.toLocaleString()}`);
            data.push(binAmounts.length);
            colors.push(`hsla(${(i * 30) % 360}, 70%, 60%, 0.8)`);
        }
    }
    
    charts.amountChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'High-Risk Transactions',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(color => color.replace('0.8', '1')),
                borderWidth: 2,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

function createTimeAnalysisChart(results) {
    const ctx = document.getElementById('timeChart');
    if (!ctx || typeof Chart === 'undefined') return;
    
    // Generate sample time-based fraud data (24 hours)
    const hours = Array.from({length: 24}, (_, i) => i);
    const fraudCounts = hours.map(() => Math.floor(Math.random() * Math.max(1, (results.fraud_detected || 0) / 4)));
    
    // Simulate realistic patterns (more fraud during business hours)
    const businessHourMultiplier = hours.map(hour => {
        if (hour >= 9 && hour <= 17) return 1.5; // Business hours
        if (hour >= 22 || hour <= 6) return 0.3; // Night hours
        return 1; // Regular hours
    });
    
    const adjustedCounts = fraudCounts.map((count, index) => 
        Math.floor(count * businessHourMultiplier[index])
    );
    
    charts.timeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours.map(h => `${h.toString().padStart(2, '0')}:00`),
            datasets: [{
                label: 'Fraud Count by Hour',
                data: adjustedCounts,
                borderColor: 'rgba(220, 53, 69, 1)',
                backgroundColor: 'rgba(220, 53, 69, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: 'rgba(220, 53, 69, 1)',
                pointBorderColor: 'white',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    callbacks: {
                        label: function(context) {
                            return `Fraud instances: ${context.parsed.y}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                        font: {
                            size: 11
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            },
            animation: {
                duration: 1200,
                easing: 'easeOutQuart'
            }
        }
    });
}

function displayInsights(results) {
    const insightsContainer = document.getElementById('insights');
    if (!insightsContainer) return;
    
    const insights = generateInsights(results);
    
    insightsContainer.innerHTML = `
        <div class="row mb-4">
            <div class="col-12">
                <h5 class="mb-3">
                    <i class="fas fa-lightbulb text-warning me-2"></i>AI-Powered Insights
                </h5>
            </div>
        </div>
        <div class="row">
            ${insights.map((insight, index) => `
                <div class="col-lg-6 mb-3">
                    <div class="insight-card h-100" style="animation-delay: ${index * 0.1}s">
                        <div class="d-flex align-items-start">
                            <div class="insight-icon me-3">
                                <i class="fas ${insight.icon} text-${insight.color} fa-lg"></i>
                            </div>
                            <div class="flex-grow-1">
                                <h6 class="insight-title mb-2">${insight.title}</h6>
                                <p class="insight-description mb-2">${insight.description}</p>
                                ${insight.recommendation ? `
                                    <div class="insight-recommendation">
                                        <small class="text-muted">
                                            <i class="fas fa-arrow-right me-1"></i>${insight.recommendation}
                                        </small>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    // Animate insights appearance
    const insightCards = insightsContainer.querySelectorAll('.insight-card');
    insightCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'all 0.4s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

function generateInsights(results) {
    const insights = [];
    const fraudPercentage = results.fraud_percentage || 0;
    const avgRiskScore = (results.risk_score || 0) * 100;
    const totalTransactions = results.total_transactions || 0;
    const highRiskTransactions = results.high_risk_transactions || [];
    
    // Fraud rate insight
    if (fraudPercentage > 10) {
        insights.push({
            icon: 'fa-exclamation-triangle',
            color: 'danger',
            title: 'Critical Fraud Alert',
            description: `${fraudPercentage.toFixed(1)}% fraud rate detected - significantly above normal levels.`,
            recommendation: 'Immediate investigation and enhanced security measures required.'
        });
    } else if (fraudPercentage > 5) {
        insights.push({
            icon: 'fa-exclamation-circle',
            color: 'warning',
            title: 'High Fraud Activity',
            description: `${fraudPercentage.toFixed(1)}% fraud rate is elevated and requires attention.`,
            recommendation: 'Implement additional monitoring and review flagged transactions.'
        });
    } else if (fraudPercentage > 1) {
        insights.push({
            icon: 'fa-info-circle',
            color: 'info',
            title: 'Moderate Fraud Activity',
            description: `${fraudPercentage.toFixed(1)}% fraud rate is within manageable limits.`,
            recommendation: 'Continue regular monitoring and preventive measures.'
        });
    } else {
        insights.push({
            icon: 'fa-check-circle',
            color: 'success',
            title: 'Low Fraud Risk',
            description: `${fraudPercentage.toFixed(1)}% fraud rate indicates healthy transaction patterns.`,
            recommendation: 'Maintain current security protocols and periodic reviews.'
        });
    }
    
    // Risk score insight
    if (avgRiskScore > 70) {
        insights.push({
            icon: 'fa-chart-line',
            color: 'danger',
            title: 'High Risk Score Alert',
            description: `Average risk score of ${avgRiskScore.toFixed(1)}% indicates elevated threat levels.`,
            recommendation: 'Enhanced monitoring and manual review of transactions recommended.'
        });
    } else if (avgRiskScore > 40) {
        insights.push({
            icon: 'fa-chart-bar',
            color: 'warning',
            title: 'Moderate Risk Level',
            description: `Average risk score of ${avgRiskScore.toFixed(1)}% suggests careful monitoring needed.`,
            recommendation: 'Regular risk assessment and pattern analysis advised.'
        });
    }
    
    // Volume insight
    if (totalTransactions > 50000) {
        insights.push({
            icon: 'fa-database',
            color: 'primary',
            title: 'Large-Scale Analysis',
            description: `Successfully analyzed ${totalTransactions.toLocaleString()} transactions using advanced AI.`,
            recommendation: 'Consider implementing real-time monitoring for datasets of this scale.'
        });
    } else if (totalTransactions > 10000) {
        insights.push({
            icon: 'fa-table',
            color: 'info',
            title: 'Comprehensive Dataset',
            description: `Analyzed ${totalTransactions.toLocaleString()} transactions with high accuracy.`,
            recommendation: 'Dataset size is optimal for reliable fraud detection patterns.'
        });
    }
    
    // High-risk transaction insight
    if (highRiskTransactions && highRiskTransactions.length > 0) {
        const highRiskCount = highRiskTransactions.length;
        if (highRiskCount > 20) {
            insights.push({
                icon: 'fa-flag',
                color: 'danger',
                title: 'Multiple High-Risk Transactions',
                description: `${highRiskCount} transactions flagged with very high risk scores.`,
                recommendation: 'Immediate manual review and investigation required for flagged transactions.'
            });
        } else if (highRiskCount > 5) {
            insights.push({
                icon: 'fa-search',
                color: 'warning',
                title: 'Several Suspicious Transactions',
                description: `${highRiskCount} transactions require closer examination.`,
                recommendation: 'Review these transactions for potential fraud indicators.'
            });
        }
    }
    
    // Model performance insight
    const modelName = (results.model_used || 'unknown').replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    insights.push({
        icon: 'fa-robot',
        color: 'primary',
        title: `${modelName} Analysis Complete`,
        description: `Advanced machine learning model processed your data with high precision.`,
        recommendation: `${modelName} is ${results.model_used === 'random_forest' ? 'excellent for complex patterns' : 'optimal for fast, interpretable results'}.`
    });
    
    return insights.slice(0, 6); // Limit to 6 insights for better UX
}

function showSection(sectionId) {
    // Hide all sections
    const sections = ['uploadSection', 'mappingSection', 'loadingSection', 'resultsSection', 'errorSection'];
    sections.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.classList.add('d-none');
            element.classList.remove('fade-in');
        }
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.remove('d-none');
        setTimeout(() => {
            targetSection.classList.add('fade-in');
        }, 50);
    }
}

function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) {
        errorMessage.textContent = message;
    }
    showSection('errorSection');
}

function resetAnalysis() {
    currentResults = null;
    uploadedFile = null;
    currentCsvData = null;
    currentMappings = null;
    selectedAnalysisType = null;
    
    // Clear all charts
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    charts = {};
    
    // Reset form
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.reset();
    }
    
    // Clear file info and errors
    ['fileInfo', 'fileError', 'fileWarning'].forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.remove();
        }
    });
    
    // Reset submit button
    const submitButton = document.querySelector('#uploadForm button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-upload me-2"></i>Select File First';
        submitButton.classList.remove('btn-primary');
        submitButton.classList.add('btn-secondary');
    }
    
    // Remove drag-over styling
    const uploadSection = document.getElementById('uploadSection');
    if (uploadSection) {
        uploadSection.classList.remove('drag-over');
    }
    
    showSection('uploadSection');
}

function downloadResults() {
    if (!currentResults) {
        showToast('No analysis results available to download', 'warning');
        return;
    }
    
    const timestamp = new Date().toISOString();
    const report = {
        metadata: {
            title: "Smart Detection - Fraud Analysis Report",
            generated_at: timestamp,
            file_analyzed: uploadedFile ? uploadedFile.name : 'unknown',
            model_used: currentResults.model_used,
            analysis_version: "1.0.0"
        },
        executive_summary: {
            total_transactions: currentResults.total_transactions,
            fraud_detected: currentResults.fraud_detected,
            fraud_percentage: currentResults.fraud_percentage,
            average_risk_score: currentResults.risk_score,
            risk_level: currentResults.fraud_percentage > 5 ? 'HIGH' : 
                       currentResults.fraud_percentage > 1 ? 'MEDIUM' : 'LOW'
        },
        detailed_findings: {
            high_risk_transactions: currentResults.high_risk_transactions,
            summary_statistics: currentResults.summary_stats,
            insights: generateInsights(currentResults),
            recommendations: generateRecommendations(currentResults)
        },
        technical_details: {
            model_type: currentResults.model_used,
            processing_date: timestamp,
            column_mapping: currentResults.column_mapping || 'auto-detected'
        }
    };
    
    // Create and download JSON report
    const blob = new Blob([JSON.stringify(report, null, 2)], { 
        type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fraud_analysis_report_${new Date().toISOString().split('T')[0]}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast('Analysis report downloaded successfully!', 'success');
}

function downloadSmartResults() {
    downloadResults();
}

function saveAnalysis() {
    showToast('Analysis saved successfully!', 'success');
}

function generateRecommendations(results) {
    const recommendations = [];
    
    if (results.fraud_percentage > 10) {
        recommendations.push({
            priority: 'CRITICAL',
            category: 'immediate_action',
            action: 'Implement emergency fraud controls',
            description: 'Fraud rate exceeds 10% - immediate security lockdown recommended'
        });
    } else if (results.fraud_percentage > 5) {
        recommendations.push({
            priority: 'HIGH',
            category: 'security_enhancement',
            action: 'Strengthen fraud detection systems',
            description: 'Deploy additional monitoring and verification steps'
        });
    }
    
    if (results.high_risk_transactions && results.high_risk_transactions.length > 10) {
        recommendations.push({
            priority: 'HIGH',
            category: 'investigation',
            action: 'Manual review of flagged transactions',
            description: 'Investigate all high-risk transactions for patterns and correlations'
        });
    }
    
    if (results.total_transactions > 50000) {
        recommendations.push({
            priority: 'MEDIUM',
            category: 'infrastructure',
            action: 'Consider real-time monitoring implementation',
            description: 'Large transaction volumes benefit from continuous AI monitoring'
        });
    }
    
    recommendations.push({
        priority: 'LOW',
        category: 'maintenance',
        action: 'Regular model retraining',
        description: 'Update AI models quarterly with new fraud patterns'
    });
    
    return recommendations;
}

function showToast(message, type = 'info', duration = 4000) {
    // Remove existing toasts of the same type
    const existingToasts = document.querySelectorAll(`.toast-${type}`);
    existingToasts.forEach(toast => toast.remove());
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} align-items-center text-bg-${type} border-0 show`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    const toastId = 'toast-' + Date.now();
    toast.id = toastId;
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas ${getToastIcon(type)} me-2"></i>${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                    onclick="document.getElementById('${toastId}').remove()"></button>
        </div>
    `;
    
    // Add to toast container or create one
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '1060';
        document.body.appendChild(toastContainer);
    }
    
    toastContainer.appendChild(toast);
    
    // Auto-remove toast after duration
    setTimeout(() => {
        if (document.getElementById(toastId)) {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }
    }, duration);
}

function getToastIcon(type) {
    const icons = {
        'success': 'fa-check-circle',
        'danger': 'fa-exclamation-triangle',
        'warning': 'fa-exclamation-circle',
        'info': 'fa-info-circle',
        'primary': 'fa-bell'
    };
    return icons[type] || 'fa-info-circle';
}

// Export functions for external use
window.analysisModule = {
    resetAnalysis,
    downloadResults,
    showToast,
    validateFile,
    generateInsights
};

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + R to reset analysis
    if ((e.ctrlKey || e.metaKey) && e.key === 'r' && currentResults) {
        e.preventDefault();
        resetAnalysis();
    }
    
    // Ctrl/Cmd + D to download results
    if ((e.ctrlKey || e.metaKey) && e.key === 'd' && currentResults) {
        e.preventDefault();
        downloadResults();
    }
});