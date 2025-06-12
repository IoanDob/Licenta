import { showToast, simulateDelay, showSection, updateLoadingProgress, updateLoadingMessage, showError } from './utils.js';

export let uploadedFile = null;
export let currentResults = null;

export function validateFile(file) {
    const validation = {
        isValid: true,
        errors: [],
        warnings: []
    };
    if (!file.name.toLowerCase().endsWith('.csv')) {
        validation.isValid = false;
        validation.errors.push('Please select a CSV file (.csv extension required)');
    }
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        validation.isValid = false;
        validation.errors.push(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum allowed size (500MB)`);
    }
    if (file.size < 1024) {
        validation.isValid = false;
        validation.errors.push('File appears to be too small. Please check your CSV file.');
    }
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

export function showFileInfo(file) {
    const existingInfo = document.getElementById('fileInfo');
    if (existingInfo) existingInfo.remove();
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

export function showFileError(errors) {
    const existingError = document.getElementById('fileError');
    if (existingError) existingError.remove();
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
    const fileInput = document.getElementById('csvFile');
    if (fileInput) fileInput.value = '';
    uploadedFile = null;
    const submitButton = document.querySelector('#uploadForm button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Invalid File';
        submitButton.classList.remove('btn-primary');
        submitButton.classList.add('btn-secondary');
    }
}

export function showFileWarnings(warnings) {
    const existingWarning = document.getElementById('fileWarning');
    if (existingWarning) existingWarning.remove();
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

export function handleFileChange(event) {
    const file = event.target.files[0];
    if (file) {
        uploadedFile = file;
        const validation = validateFile(file);
        if (!validation.isValid) {
            showFileError(validation.errors);
            return;
        }
        if (validation.warnings.length > 0) {
            showFileWarnings(validation.warnings);
        }
        showFileInfo(file);
        const submitButton = document.querySelector('#uploadForm button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Transactions';
            submitButton.classList.remove('btn-secondary');
            submitButton.classList.add('btn-primary');
        }
    }
}

export async function handleFileUpload(e) {
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
        // displayResults(currentResults); // You can import and use displayResults if needed
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