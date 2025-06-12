// Authentication utilities
class AuthManager {
    constructor() {
        this.token = localStorage.getItem('accessToken');
        this.initNavigation();
    }
    
    setToken(token) {
        this.token = token;
        localStorage.setItem('accessToken', token);
        this.initNavigation();
    }
    
    removeToken() {
        this.token = null;
        localStorage.removeItem('accessToken');
        this.initNavigation();
    }
    
    isAuthenticated() {
        return this.token !== null;
    }
    
    getAuthHeaders() {
        return this.token ? { 'Authorization': `Bearer ${this.token}` } : {};
    }
    
    initNavigation() {
        const navbarLinks = document.getElementById('navbar-links');
        if (!navbarLinks) return;
        
        if (this.isAuthenticated()) {
            navbarLinks.innerHTML = `
                <a class="nav-link" href="/dashboard">Dashboard</a>
                <a class="nav-link" href="/analysis">Analysis</a>
                <a class="nav-link" href="/history">History</a>
                <button class="btn btn-outline-light btn-sm" onclick="logout()">Logout</button>
            `;
        } else {
            navbarLinks.innerHTML = `
                <a class="nav-link" href="/">Login</a>
                <a class="nav-link" href="/register">Register</a>
            `;
        }
    }
    
    redirectIfNotAuthenticated() {
        if (!this.isAuthenticated()) {
            window.location.href = '/';
        }
    }
}

// Global auth manager instance
const authManager = new AuthManager();

// Global logout function
function logout() {
    authManager.removeToken();
    window.location.href = '/';
}

// Check authentication on protected pages
document.addEventListener('DOMContentLoaded', function() {
    const protectedPages = ['/dashboard', '/analysis', '/history'];
    const currentPath = window.location.pathname;
    
    if (protectedPages.includes(currentPath)) {
        authManager.redirectIfNotAuthenticated();
    }
});