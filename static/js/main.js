document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropZoneContent = document.getElementById('drop-zone-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const classifyBtn = document.getElementById('classify-btn');
    
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsContainer = document.getElementById('results-container');
    
    const varietyName = document.getElementById('variety-name');
    const confidenceBadge = document.getElementById('confidence-badge');
    const predictionsList = document.getElementById('predictions-list');
    const inferenceTime = document.getElementById('inference-time');
    
    let currentFile = null;
    let toastTimeout = null;

    // Toast Notification System
    const toastContainer = document.getElementById('toast-container');
    const toastMessage = document.getElementById('toast-message');

    function showErrorToast(message) {
        toastMessage.textContent = message;
        toastContainer.classList.add('visible');
        if(toastTimeout) clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            toastContainer.classList.remove('visible');
        }, 4000);
    }

    // Drag and Drop Handling
    dropZone.addEventListener('click', (e) => {
        if(e.target === removeBtn || e.target.closest('#remove-btn')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            showErrorToast("Please select a valid image file (JPG, PNG, WEBP).");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.add('active');
            dropZoneContent.classList.add('hidden');
            classifyBtn.disabled = false;
        };
        reader.readAsDataURL(file);
        showState('empty');
    }

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.remove('active');
        setTimeout(() => {
            imagePreview.src = '';
            dropZoneContent.classList.remove('hidden');
        }, 300); // Wait for fade out
        classifyBtn.disabled = true;
        showState('empty');
    }

    // API Interaction
    classifyBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        showState('loading');
        classifyBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Prediction failed');
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            showErrorToast(error.message || "An unexpected error occurred.");
            showState('empty');
            classifyBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Hero Result
        varietyName.textContent = data.top_class;
        const confPercent = (data.confidence * 100).toFixed(1);
        
        // Chip Styling
        confidenceBadge.className = 'confidence-chip';
        let iconHtml = '';
        
        if (data.confidence >= 0.9) {
            confidenceBadge.classList.add('high');
            iconHtml = '<i class="ph-fill ph-check-circle"></i>';
            confidenceBadge.innerHTML = `${iconHtml} <span>${confPercent}% Match</span>`;
        } else if (data.confidence >= data.threshold) {
            confidenceBadge.classList.add('medium');
            iconHtml = '<i class="ph-fill ph-warning-circle"></i>';
            confidenceBadge.innerHTML = `${iconHtml} <span>${confPercent}% Moderate</span>`;
        } else {
            confidenceBadge.classList.add('low');
            iconHtml = '<i class="ph-fill ph-x-circle"></i>';
            confidenceBadge.innerHTML = `${iconHtml} <span>${confPercent}% Low</span>`;
        }

        inferenceTime.textContent = `Inference: ${data.inference_time_ms}ms`;

        // Render Animated Progress Bars
        predictionsList.innerHTML = '';
        
        data.top_predictions.forEach((pred, index) => {
            const percentage = (pred.score * 100).toFixed(1);
            
            const item = document.createElement('div');
            item.className = 'progress-item';
            
            item.innerHTML = `
                <div class="progress-header">
                    <span class="progress-name">${pred.class_name}</span>
                    <span class="progress-value">${percentage}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            `;
            predictionsList.appendChild(item);
            
            // Trigger animation after brief delay
            setTimeout(() => {
                item.querySelector('.progress-fill').style.width = `${percentage}%`;
            }, 50 + (index * 100)); // Stagger animations
        });

        showState('results');
        classifyBtn.disabled = false;
    }

    function showState(stateName) {
        emptyState.classList.add('hidden');
        loadingState.classList.add('hidden');
        resultsContainer.classList.add('hidden');

        if (stateName === 'empty') emptyState.classList.remove('hidden');
        if (stateName === 'loading') loadingState.classList.remove('hidden');
        if (stateName === 'results') resultsContainer.classList.remove('hidden');
    }
});
