class SkinLesionAnalyzer {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.selectedImage = null;
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.uploadContent = document.getElementById('uploadContent');
        this.imageInput = document.getElementById('imageInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.changeImageBtn = document.getElementById('changeImage');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.predictionValue = document.getElementById('predictionValue');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.confidenceText = document.getElementById('confidenceText');
        
        // Feedback elements
        this.feedbackSection = document.getElementById('feedbackSection');
        this.helpfulBtn = document.getElementById('helpfulBtn');
        this.notHelpfulBtn = document.getElementById('notHelpfulBtn');
        this.detailedFeedback = document.getElementById('detailedFeedback');
        this.feedbackComment = document.getElementById('feedbackComment');
        this.feedbackCategory = document.getElementById('feedbackCategory');
        this.submitFeedback = document.getElementById('submitFeedback');
        this.skipFeedback = document.getElementById('skipFeedback');
        this.feedbackThanks = document.getElementById('feedbackThanks');
    }

    attachEventListeners() {
        // File input change
        this.imageInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            if (!this.selectedImage) {
                this.imageInput.click();
            }
        });

        // Drag and drop functionality
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Button events
        this.changeImageBtn.addEventListener('click', () => this.resetUpload());
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        
        // Feedback events
        this.helpfulBtn.addEventListener('click', () => this.handleFeedbackRating('helpful'));
        this.notHelpfulBtn.addEventListener('click', () => this.handleFeedbackRating('not-helpful'));
        this.submitFeedback.addEventListener('click', () => this.submitUserFeedback());
        this.skipFeedback.addEventListener('click', () => this.skipUserFeedback());

        // Prevent default drag behaviors on document
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => e.preventDefault());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/heic'];
        if (!validTypes.includes(file.type)) {
            this.showError('Please select a valid image file (JPG, JPEG, PNG, or HEIC)');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showError('File size must be less than 10MB');
            return;
        }

        this.selectedImage = file;
        this.displayImagePreview(file);
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.uploadContent.style.display = 'none';
            this.imagePreview.style.display = 'block';
            this.hideResults();
        };
        reader.readAsDataURL(file);
    }

    resetUpload() {
        this.selectedImage = null;
        this.imageInput.value = '';
        this.uploadContent.style.display = 'block';
        this.imagePreview.style.display = 'none';
        this.hideResults();
    }

    async analyzeImage() {
        if (!this.selectedImage) {
            this.showError('Please select an image first');
            return;
        }

        this.showLoading();

        try {
            // Convert image to base64
            const base64Image = await this.fileToBase64(this.selectedImage);
            
            // Remove the data URL prefix
            const base64Data = base64Image.split(',')[1];

            // Send to Flask API
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Data
                })
            });

            const result = await response.json();
            
            if (!response.ok || result.error) {
                // Handle invalid image errors specially
                if (result.type === 'invalid_image') {
                    this.showInvalidImageError(result.error, result.message);
                    return;
                } else {
                    throw new Error(result.error || `HTTP error! status: ${response.status}`);
                }
            }

            this.displayResults(result);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    displayResults(result) {
        // Update prediction
        this.predictionValue.textContent = result.prediction || result.class_name;

        // Update confidence
        const confidence = Math.round((result.confidence || 0) * 100);
        this.confidenceText.textContent = `${confidence}%`;
        this.confidenceFill.style.width = `${confidence}%`;

        // Set confidence bar color based on value
        if (confidence >= 80) {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #10b981 0%, #059669 100%)';
        } else if (confidence >= 60) {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)';
        } else {
            this.confidenceFill.style.background = 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)';
        }

        // Show results with animation
        this.resultsSection.style.display = 'block';
        this.resultsSection.classList.add('show');
        
        
        // Scroll to results
        this.resultsSection.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
        this.resultsSection.classList.remove('show');
        this.resetFeedback();
    }
    
    resetFeedback() {
        // Reset feedback form
        this.helpfulBtn.classList.remove('selected');
        this.notHelpfulBtn.classList.remove('selected');
        this.detailedFeedback.style.display = 'none';
        this.feedbackThanks.style.display = 'none';
        this.feedbackComment.value = '';
        this.feedbackCategory.value = '';
        
        // Reset rating buttons visibility
        document.getElementById('ratingButtons').style.display = 'flex';
    }
    
    handleFeedbackRating(rating) {
        // Update button states
        this.helpfulBtn.classList.remove('selected');
        this.notHelpfulBtn.classList.remove('selected');
        
        if (rating === 'helpful') {
            this.helpfulBtn.classList.add('selected');
        } else {
            this.notHelpfulBtn.classList.add('selected');
        }
        
        // Store the rating
        this.currentFeedbackRating = rating;
        
        // Show detailed feedback form
        this.detailedFeedback.style.display = 'block';
    }
    
    async submitUserFeedback() {
        const feedbackData = {
            rating: this.currentFeedbackRating,
            comment: this.feedbackComment.value.trim(),
            category: this.feedbackCategory.value,
            prediction: this.predictionValue.textContent,
            confidence: this.confidenceText.textContent,
            timestamp: new Date().toISOString(),
            session_id: this.generateSessionId()
        };
        
        try {
            const response = await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(feedbackData)
            });
            
            if (response.ok) {
                this.showFeedbackThanks();
            } else {
                console.error('Failed to submit feedback');
                this.showFeedbackThanks(); // Still show thanks to avoid user confusion
            }
        } catch (error) {
            console.error('Error submitting feedback:', error);
            this.showFeedbackThanks(); // Still show thanks to avoid user confusion
        }
    }
    
    skipUserFeedback() {
        this.showFeedbackThanks();
    }
    
    showFeedbackThanks() {
        // Hide rating buttons and detailed feedback
        document.getElementById('ratingButtons').style.display = 'none';
        this.detailedFeedback.style.display = 'none';
        
        // Show thanks message
        this.feedbackThanks.style.display = 'block';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            this.feedbackThanks.style.display = 'none';
            document.getElementById('ratingButtons').style.display = 'flex';
            this.resetFeedback();
        }, 3000);
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        this.analyzeBtn.disabled = true;
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
        this.analyzeBtn.disabled = false;
    }

    showInvalidImageError(title, message) {
        // Create invalid image notification with different styling
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-image-notification';
        errorDiv.innerHTML = `
            <div class="invalid-image-content">
                <i class="fas fa-image"></i>
                <div class="invalid-image-text">
                    <h4>${title}</h4>
                    <p>${message}</p>
                    <small>Please upload a clear photo of a skin lesion for analysis.</small>
                </div>
                <button class="invalid-image-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add invalid image styles if not already present
        if (!document.querySelector('.invalid-image-notification-styles')) {
            const style = document.createElement('style');
            style.className = 'invalid-image-notification-styles';
            style.textContent = `
                .invalid-image-notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #fef3c7;
                    border: 1px solid #f59e0b;
                    border-radius: 8px;
                    padding: 1.25rem;
                    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
                    z-index: 1001;
                    animation: slideInRight 0.3s ease-out;
                    max-width: 450px;
                }
                
                .invalid-image-content {
                    display: flex;
                    align-items: flex-start;
                    gap: 1rem;
                    color: #92400e;
                }
                
                .invalid-image-content i:first-child {
                    color: #f59e0b;
                    flex-shrink: 0;
                    font-size: 1.25rem;
                    margin-top: 0.125rem;
                }
                
                .invalid-image-text h4 {
                    margin: 0 0 0.5rem 0;
                    font-weight: 600;
                    color: #92400e;
                }
                
                .invalid-image-text p {
                    margin: 0 0 0.5rem 0;
                    color: #92400e;
                }
                
                .invalid-image-text small {
                    color: #a16207;
                    font-style: italic;
                }
                
                .invalid-image-close {
                    background: none;
                    border: none;
                    color: #92400e;
                    cursor: pointer;
                    padding: 0.25rem;
                    border-radius: 4px;
                    margin-left: auto;
                    flex-shrink: 0;
                }
                
                .invalid-image-close:hover {
                    background: #fbbf24;
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(errorDiv);

        // Auto remove after 8 seconds (longer for invalid image)
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 8000);
    }

    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.innerHTML = `
            <div class="error-content">
                <i class="fas fa-exclamation-circle"></i>
                <span>${message}</span>
                <button class="error-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        // Add error styles if not already present
        if (!document.querySelector('.error-notification-styles')) {
            const style = document.createElement('style');
            style.className = 'error-notification-styles';
            style.textContent = `
                .error-notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #fef2f2;
                    border: 1px solid #fecaca;
                    border-radius: 8px;
                    padding: 1rem;
                    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
                    z-index: 1001;
                    animation: slideInRight 0.3s ease-out;
                    max-width: 400px;
                }
                
                .error-content {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    color: #991b1b;
                }
                
                .error-content i:first-child {
                    color: #dc2626;
                    flex-shrink: 0;
                }
                
                .error-close {
                    background: none;
                    border: none;
                    color: #991b1b;
                    cursor: pointer;
                    padding: 0.25rem;
                    border-radius: 4px;
                    margin-left: auto;
                }
                
                .error-close:hover {
                    background: #fecaca;
                }
                
                @keyframes slideInRight {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(errorDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 5000);
    }
}

// Mobile device detection and optimization
function detectMobileDevice() {
    const userAgent = navigator.userAgent;
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent);
    const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    const screenWidth = window.innerWidth;
    
    return {
        isMobile: isMobile || screenWidth <= 768,
        isTouch: isTouch,
        isIOS: /iPad|iPhone|iPod/.test(userAgent),
        isAndroid: /Android/.test(userAgent),
        screenWidth: screenWidth,
        screenHeight: window.innerHeight,
        devicePixelRatio: window.devicePixelRatio || 1
    };
}

function optimizeForMobile() {
    const device = detectMobileDevice();
    const body = document.body;
    
    if (device.isMobile) {
        body.classList.add('mobile-optimized');
        
        // Add iOS specific optimizations
        if (device.isIOS) {
            body.classList.add('ios-device');
            // Prevent zoom on input focus
            const inputs = document.querySelectorAll('input, select, textarea');
            inputs.forEach(input => {
                input.addEventListener('focus', () => {
                    document.querySelector('meta[name=viewport]').setAttribute(
                        'content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'
                    );
                });
                input.addEventListener('blur', () => {
                    document.querySelector('meta[name=viewport]').setAttribute(
                        'content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'
                    );
                });
            });
        }
        
        // Add Android specific optimizations
        if (device.isAndroid) {
            body.classList.add('android-device');
        }
        
        // Optimize upload area for mobile
        const uploadArea = document.getElementById('uploadArea');
        if (uploadArea && device.isTouch) {
            uploadArea.style.cursor = 'pointer';
            // Add haptic feedback for touch devices
            uploadArea.addEventListener('touchstart', () => {
                if (navigator.vibrate) {
                    navigator.vibrate(50);
                }
            });
        }
        
        // Optimize scroll behavior for mobile
        if (device.screenWidth <= 480) {
            // Smooth scroll polyfill for older browsers
            const smoothScroll = (target) => {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                    inline: 'nearest'
                });
            };
            
            // Override default scroll behavior
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        smoothScroll(target);
                    }
                });
            });
        }
        
        // Add mobile-specific upload hints
        const uploadContent = document.getElementById('uploadContent');
        if (uploadContent && device.isTouch) {
            const originalHTML = uploadContent.innerHTML;
            uploadContent.innerHTML = originalHTML.replace(
                'Drag and drop your image here or click to browse',
                'Tap to select an image from your device'
            );
        }
    }
    
    console.log('Device info:', device);
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Optimize for mobile first
    optimizeForMobile();
    
    // Initialize the main application
    new SkinLesionAnalyzer();
});

// Add smooth scrolling for all anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add intersection observer for animation on scroll
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for scroll animations
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.info-card, .type-card, .progress-card');
    animateElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(el);
    });
    
    // Initialize FAQ functionality
    initializeFAQ();
});

// FAQ functionality
function initializeFAQ() {
    const faqItems = document.querySelectorAll('.faq-item');
    
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        
        question.addEventListener('click', () => {
            const isActive = item.classList.contains('active');
            
            // Close all other FAQ items
            faqItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.classList.remove('active');
                }
            });
            
            // Toggle current item
            if (isActive) {
                item.classList.remove('active');
            } else {
                item.classList.add('active');
            }
        });
    });
}