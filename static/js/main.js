document.addEventListener('DOMContentLoaded', () => {
    // Elements
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
    
    const mainResult = document.getElementById('main-result');
    const varietyName = document.getElementById('variety-name');
    const scoreValue = document.getElementById('score-value');
    const confidenceBadge = document.getElementById('confidence-badge');
    const inferenceTime = document.getElementById('inference-time');
    const warningAlert = document.getElementById('warning-alert');

    let currentFile = null;
    let predictionsChart = null;

    // --- Event Listeners for file upload ---

    // Click to browse
    dropZone.addEventListener('click', (e) => {
        if(e.target === removeBtn || e.target.closest('#remove-btn')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // --- File Handling ---

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file (JPG, PNG, WEBP).');
            return;
        }

        currentFile = file;
        
        // Show Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZoneContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            classifyBtn.classList.remove('hidden');
            classifyBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        // Reset analysis side
        showState('empty');
    }

    function resetUpload() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropZoneContent.classList.remove('hidden');
        classifyBtn.classList.add('hidden');
        classifyBtn.disabled = true;
        showState('empty');
    }

    // --- API Interaction ---

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

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert(`Error: ${error.message}`);
            showState('empty');
            classifyBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Update main result
        varietyName.textContent = data.top_class;
        const confidencePercentage = (data.confidence * 100).toFixed(1);
        scoreValue.textContent = `${confidencePercentage}%`;
        inferenceTime.textContent = `${data.inference_time_ms}ms`;

        // Style the card based on confidence
        mainResult.className = 'main-result'; // reset
        if (data.confidence >= 0.9) {
            mainResult.classList.add('success');
            confidenceBadge.textContent = 'High Confidence';
            mainResult.style.borderLeftColor = 'var(--accent-glow)';
        } else if (data.confidence >= data.threshold) {
            mainResult.classList.add('warning');
            confidenceBadge.textContent = 'Moderate Confidence';
            mainResult.style.borderLeftColor = 'var(--accent-yellow)';
        } else {
            mainResult.classList.add('danger');
            confidenceBadge.textContent = 'Low Confidence';
            mainResult.style.borderLeftColor = 'var(--accent-red)';
        }

        if (data.is_high_confidence) {
            warningAlert.classList.add('hidden');
        } else {
            warningAlert.classList.remove('hidden');
        }

        // --- Render Chart.js Graph ---
        const labels = data.top_predictions.map(pred => pred.class_name);
        const values = data.top_predictions.map(pred => (pred.score * 100).toFixed(1));
        
        // Colors for chart
        const mainColor = data.confidence >= 0.9 ? '#00ff88' : (data.confidence >= data.threshold ? '#ffcc00' : '#ff3b30');
        const bgColors = values.map((_, i) => i === 0 ? mainColor : 'rgba(255, 255, 255, 0.2)');
        const borderColors = values.map((_, i) => i === 0 ? mainColor : 'rgba(255, 255, 255, 0.5)');

        const ctx = document.getElementById('predictionsChart').getContext('2d');
        
        if (predictionsChart) {
            predictionsChart.destroy();
        }

        predictionsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Confidence (%)',
                    data: values,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 6,
                    barThickness: 'flex',
                    maxBarThickness: 40
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y', // horizontal bar chart
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(25, 25, 28, 0.9)',
                        titleColor: '#86868b',
                        bodyColor: '#fff',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1,
                        padding: 10,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return context.parsed.x + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#86868b'
                        }
                    },
                    y: {
                        grid: {
                            display: false,
                            drawBorder: false
                        },
                        ticks: {
                            color: '#fff',
                            font: { family: "'Inter', sans-serif" }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
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
