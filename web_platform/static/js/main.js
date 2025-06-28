/**
 * CardioPredict Web Platform - Interactive Features
 * Professional JavaScript for enhanced user experience
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // ===== NAVIGATION ENHANCEMENT =====
    initializeNavigation();
    
    // ===== SMOOTH SCROLLING =====
    initializeSmoothScrolling();
    
    // ===== ANIMATION ON SCROLL =====
    initializeScrollAnimations();
    
    // ===== STATISTICS COUNTER =====
    initializeCounters();
    
    // ===== FORM ENHANCEMENTS =====
    initializeFormHandlers();
    
    // ===== TOOLTIPS AND POPOVERS =====
    initializeTooltips();
    
    console.log('✓ CardioPredict platform initialized');
});

/**
 * Navigation enhancements
 */
function initializeNavigation() {
    const navbar = document.querySelector('.navbar');
    
    // Navbar scroll effect
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
    
    // Mobile menu handling
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
        navbarToggler.addEventListener('click', function() {
            navbarCollapse.classList.toggle('show');
        });
        
        // Close mobile menu when clicking on links
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth < 992) {
                    navbarCollapse.classList.remove('show');
                }
            });
        });
    }
}

/**
 * Smooth scrolling for anchor links
 */
function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                e.preventDefault();
                
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Intersection Observer for scroll animations
 */
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements that should animate
    const animatedElements = document.querySelectorAll('.feature-card, .stat-item, .research-highlight');
    animatedElements.forEach(el => observer.observe(el));
}

/**
 * Animated counters for statistics
 */
function initializeCounters() {
    const counters = document.querySelectorAll('.stat-number[data-count]');
    
    const observerOptions = {
        threshold: 0.5,
        rootMargin: '0px'
    };
    
    const counterObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                counterObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    counters.forEach(counter => counterObserver.observe(counter));
}

/**
 * Animate a single counter
 */
function animateCounter(element) {
    const target = parseInt(element.getAttribute('data-count'));
    const duration = 2000; // 2 seconds
    const start = performance.now();
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - start;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(easeOut * target);
        
        element.textContent = current.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            element.textContent = target.toLocaleString();
        }
    }
    
    requestAnimationFrame(updateCounter);
}

/**
 * Form handling and validation
 */
function initializeFormHandlers() {
    const forms = document.querySelectorAll('form[data-ajax]');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            handleFormSubmission(this);
        });
    });
    
    // Real-time validation for inputs
    const inputs = document.querySelectorAll('input[required], select[required]');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateInput(this);
        });
        
        input.addEventListener('input', function() {
            if (this.classList.contains('is-invalid')) {
                validateInput(this);
            }
        });
    });
}

/**
 * Handle AJAX form submissions
 */
function handleFormSubmission(form) {
    const formData = new FormData(form);
    const submitButton = form.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    
    // Show loading state
    submitButton.disabled = true;
    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    
    // Get form action and method
    const action = form.getAttribute('action') || '/api/predict';
    const method = form.getAttribute('method') || 'POST';
    
    fetch(action, {
        method: method,
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
    .then(response => response.json())
    .then(data => {
        handleFormResponse(form, data);
    })
    .catch(error => {
        console.error('Form submission error:', error);
        showNotification('An error occurred. Please try again.', 'error');
    })
    .finally(() => {
        // Restore button state
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    });
}

/**
 * Handle form response
 */
function handleFormResponse(form, data) {
    if (data.success) {
        showNotification(data.message || 'Request processed successfully!', 'success');
        
        // Display results if available
        if (data.results) {
            displayPredictionResults(data.results);
        }
        
        // Reset form if specified
        if (data.reset_form) {
            form.reset();
        }
    } else {
        showNotification(data.message || 'An error occurred.', 'error');
        
        // Display field errors
        if (data.errors) {
            displayFormErrors(form, data.errors);
        }
    }
}

/**
 * Validate individual input
 */
function validateInput(input) {
    const value = input.value.trim();
    const type = input.type;
    const required = input.hasAttribute('required');
    
    let isValid = true;
    let errorMessage = '';
    
    // Required field validation
    if (required && !value) {
        isValid = false;
        errorMessage = 'This field is required.';
    }
    
    // Type-specific validation
    if (value && type === 'email') {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(value)) {
            isValid = false;
            errorMessage = 'Please enter a valid email address.';
        }
    }
    
    if (value && type === 'number') {
        const min = parseFloat(input.getAttribute('min'));
        const max = parseFloat(input.getAttribute('max'));
        const numValue = parseFloat(value);
        
        if (isNaN(numValue)) {
            isValid = false;
            errorMessage = 'Please enter a valid number.';
        } else if (!isNaN(min) && numValue < min) {
            isValid = false;
            errorMessage = `Value must be at least ${min}.`;
        } else if (!isNaN(max) && numValue > max) {
            isValid = false;
            errorMessage = `Value must be at most ${max}.`;
        }
    }
    
    // Update input appearance
    updateInputValidation(input, isValid, errorMessage);
    
    return isValid;
}

/**
 * Update input validation appearance
 */
function updateInputValidation(input, isValid, errorMessage) {
    const feedbackElement = input.parentNode.querySelector('.invalid-feedback');
    
    if (isValid) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
        if (feedbackElement) {
            feedbackElement.style.display = 'none';
        }
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
        
        if (feedbackElement) {
            feedbackElement.textContent = errorMessage;
            feedbackElement.style.display = 'block';
        } else {
            // Create feedback element if it doesn't exist
            const feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            feedback.textContent = errorMessage;
            input.parentNode.appendChild(feedback);
        }
    }
}

/**
 * Display form errors
 */
function displayFormErrors(form, errors) {
    Object.keys(errors).forEach(fieldName => {
        const input = form.querySelector(`[name="${fieldName}"]`);
        if (input) {
            updateInputValidation(input, false, errors[fieldName]);
        }
    });
}

/**
 * Display prediction results
 */
function displayPredictionResults(results) {
    const resultsContainer = document.getElementById('prediction-results');
    if (!resultsContainer) return;
    
    const riskScore = results.risk_score || 0;
    const riskLevel = results.risk_level || 'Unknown';
    const confidence = results.confidence || 0;
    
    resultsContainer.innerHTML = `
        <div class="card shadow-custom">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Prediction Results</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-4 text-center">
                        <div class="risk-score-display">
                            <div class="risk-score-circle ${getRiskColorClass(riskScore)}">
                                <span class="risk-score-value">${riskScore.toFixed(1)}</span>
                                <span class="risk-score-label">Risk Score</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <h6>Risk Assessment</h6>
                        <p class="risk-level ${getRiskColorClass(riskScore)}">
                            <strong>${riskLevel}</strong>
                        </p>
                        <div class="mb-3">
                            <label class="form-label">Confidence Level</label>
                            <div class="progress">
                                <div class="progress-bar" style="width: ${confidence}%">
                                    ${confidence.toFixed(1)}%
                                </div>
                            </div>
                        </div>
                        ${results.recommendations ? `
                            <h6>Recommendations</h6>
                            <ul class="list-unstyled">
                                ${results.recommendations.map(rec => `<li><i class="fas fa-check-circle text-success me-2"></i>${rec}</li>`).join('')}
                            </ul>
                        ` : ''}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/**
 * Get risk color class based on score
 */
function getRiskColorClass(score) {
    if (score < 30) return 'text-success';
    if (score < 70) return 'text-warning';
    return 'text-danger';
}

/**
 * Initialize tooltips and popovers
 */
function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Utility function to debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Loading overlay utility
 */
function showLoadingOverlay(message = 'Loading...') {
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.7);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 10000;
        color: white;
        font-size: 1.2rem;
    `;
    
    overlay.innerHTML = `
        <div class="text-center">
            <div class="spinner-border mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <div>${message}</div>
        </div>
    `;
    
    document.body.appendChild(overlay);
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

// Make utility functions globally available
window.CardioPredict = {
    showNotification,
    showLoadingOverlay,
    hideLoadingOverlay,
    debounce
};
