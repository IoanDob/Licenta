document.addEventListener('DOMContentLoaded', function() {
    // Redirect if already authenticated
    if (authManager.isAuthenticated()) {
        window.location.href = '/dashboard';
    }
    
    const loginForm = document.getElementById('loginForm');
    const loginError = document.getElementById('loginError');
    
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        
        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);
            
            const response = await axios.post('/api/auth/login', formData);
            
            authManager.setToken(response.data.access_token);
            window.location.href = '/dashboard';
            
        } catch (error) {
            loginError.textContent = error.response?.data?.detail || 'Login failed';
            loginError.classList.remove('d-none');
        }
    });
});