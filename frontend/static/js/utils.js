// Utility functions: toasts, delays, section management, error display

export function showToast(message, type = 'info') {
    // Implement your toast logic here (or use a library)
    alert(`[${type.toUpperCase()}] ${message}`);
}

export function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(sec => sec.style.display = 'none');
    const section = document.getElementById(sectionId);
    if (section) section.style.display = '';
}

export function updateLoadingProgress(message, percent) {
    const loadingMsg = document.getElementById('loadingMessage');
    const loadingBar = document.getElementById('loadingBar');
    if (loadingMsg) loadingMsg.textContent = message;
    if (loadingBar) loadingBar.style.width = `${percent}%`;
}

export function updateLoadingMessage(message) {
    const loadingMsg = document.getElementById('loadingMessage');
    if (loadingMsg) loadingMsg.textContent = message;
}

export function showError(errorMessage) {
    // Implement your error UI logic here
    alert(`Error: ${errorMessage}`);
}