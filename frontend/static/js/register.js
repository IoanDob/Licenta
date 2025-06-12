document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.getElementById('registerForm');
    const registerError = document.getElementById('registerError');
    const registerSuccess = document.getElementById('registerSuccess');
    
    registerForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const username = document.getElementById('username').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;
        
        // Hide previous messages
        registerError.classList.add('d-none');
        registerSuccess.classList.add('d-none');
        
        // Validate passwords match
        if (password !== confirmPassword) {
            registerError.textContent = 'Passwords do not match';
            registerError.classList.remove('d-none');
            return;
        }
        
        try {
            const response = await axios.post('/api/auth/register', {
                username: username,
                email: email,
                password: password
            });
            
            registerSuccess.textContent = 'Registration successful! Please login.';
            registerSuccess.classList.remove('d-none');
            registerForm.reset();
            
            // Redirect to login after 2 seconds
            setTimeout(() => {
                window.location.href = '/';
            }, 2000);
            
        } catch (error) {
            registerError.textContent = error.response?.data?.detail || 'Registration failed';
            registerError.classList.remove('d-none');
        }
    });
});