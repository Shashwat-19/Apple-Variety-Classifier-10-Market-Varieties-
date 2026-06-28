/* =============================================
   AppleVision AI — Prediction Page Logic
   Drag-drop, upload, API calls, results rendering
   ============================================= */

(function () {
  'use strict';

  // ——— DOM Elements ———
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const uploadSection = document.getElementById('upload-section');
  const previewSection = document.getElementById('preview-section');
  const previewImage = document.getElementById('preview-image');
  const removeImageBtn = document.getElementById('remove-image-btn');
  const analyzeBtn = document.getElementById('analyze-btn');
  const analyzeBtnText = document.getElementById('analyze-btn-text');
  const analyzeBtnSpinner = document.getElementById('analyze-btn-spinner');
  const progressContainer = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const resultsPanel = document.getElementById('results-panel');
  const resultsContent = document.getElementById('results-content');
  const emptyState = document.getElementById('results-empty');
  const reuploadBtn = document.getElementById('reupload-btn');
  const downloadBtn = document.getElementById('download-btn');

  let selectedFile = null;
  let resultData = null;

  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

  // ——— Upload Zone Events ———
  if (uploadZone) {
    // Click to upload
    uploadZone.addEventListener('click', () => fileInput && fileInput.click());

    // Drag events
    ['dragenter', 'dragover'].forEach(evt => {
      uploadZone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(evt => {
      uploadZone.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('drag-over');
      });
    });

    uploadZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) handleFile(files[0]);
    });
  }

  // File input change
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });
  }

  // Remove image
  if (removeImageBtn) {
    removeImageBtn.addEventListener('click', resetUpload);
  }

  // Analyze button
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', analyzeImage);
  }

  // Re-upload button
  if (reuploadBtn) {
    reuploadBtn.addEventListener('click', resetUpload);
  }

  // Download JSON
  if (downloadBtn) {
    downloadBtn.addEventListener('click', downloadResult);
  }

  // ——— File Handling ———
  function handleFile(file) {
    // Validate type
    if (!ALLOWED_TYPES.includes(file.type)) {
      Toast.error('Please upload a JPG, PNG, or WEBP image.');
      return;
    }

    // Validate size
    if (file.size > MAX_FILE_SIZE) {
      Toast.error('File size must be under 10MB.');
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImage) previewImage.src = e.target.result;
      showPreview();
    };
    reader.readAsDataURL(file);

    Toast.info(`Selected: ${file.name}`);
  }

  function showPreview() {
    if (uploadSection) uploadSection.classList.add('hidden');
    if (previewSection) previewSection.classList.remove('hidden');
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
    // Hide results
    hideResults();
  }

  function resetUpload() {
    selectedFile = null;
    resultData = null;

    if (fileInput) fileInput.value = '';
    if (previewImage) previewImage.src = '';
    if (uploadSection) uploadSection.classList.remove('hidden');
    if (previewSection) previewSection.classList.add('hidden');
    if (progressContainer) progressContainer.classList.add('hidden');
    if (progressFill) progressFill.style.width = '0%';

    hideResults();
    resetAnalyzeBtn();
  }

  function hideResults() {
    if (resultsContent) resultsContent.classList.add('hidden');
    if (emptyState) emptyState.classList.remove('hidden');
    if (downloadBtn) downloadBtn.classList.add('hidden');
    if (reuploadBtn) reuploadBtn.classList.add('hidden');
  }

  function resetAnalyzeBtn() {
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    }
    if (analyzeBtnText) analyzeBtnText.textContent = 'Analyze Specimen';
    if (analyzeBtnSpinner) analyzeBtnSpinner.classList.add('hidden');
  }

  // ——— API Call ———
  async function analyzeImage() {
    if (!selectedFile) {
      Toast.error('Please select an image first.');
      return;
    }

    // Disable button, show loading
    if (analyzeBtn) {
      analyzeBtn.disabled = true;
      analyzeBtn.classList.add('opacity-50', 'cursor-not-allowed');
    }
    if (analyzeBtnText) analyzeBtnText.textContent = 'Analyzing...';
    if (analyzeBtnSpinner) analyzeBtnSpinner.classList.remove('hidden');

    // Show progress
    if (progressContainer) progressContainer.classList.remove('hidden');
    animateProgress();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      // Complete progress
      if (progressFill) progressFill.style.width = '100%';

      setTimeout(() => {
        resultData = data;
        renderResults(data);
        Toast.success('Analysis complete!');

        /* Notify the analytics module so it can refresh live data.
           This is a no-op if the analytics page is not currently open. */
        window.dispatchEvent(new CustomEvent('predictionSaved', { detail: data }));
      }, 300);

    } catch (error) {
      Toast.error(error.message || 'Something went wrong. Please try again.');
      if (progressContainer) progressContainer.classList.add('hidden');
      if (progressFill) progressFill.style.width = '0%';
    } finally {
      resetAnalyzeBtn();
    }
  }

  function animateProgress() {
    if (!progressFill) return;
    progressFill.style.width = '0%';

    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 15 + 5;
      if (progress >= 90) {
        clearInterval(interval);
        progressFill.style.width = '90%';
        return;
      }
      progressFill.style.width = progress + '%';
    }, 200);

    // Store interval for cleanup
    progressFill._interval = interval;
  }

  // ——— Render Results ———
  function renderResults(data) {
    if (!resultsContent || !emptyState) return;

    emptyState.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    resultsContent.classList.add('result-reveal');

    // Main prediction
    const mainClass = document.getElementById('result-class');
    const mainConfidence = document.getElementById('result-confidence');
    const confidenceBadge = document.getElementById('confidence-badge');
    const inferenceTime = document.getElementById('inference-time');

    if (mainClass) mainClass.textContent = data.top_class || 'Unknown';

    const confPercent = ((data.confidence || 0) * 100).toFixed(1);
    if (mainConfidence) mainConfidence.textContent = confPercent + '%';

    // Confidence badge styling
    if (confidenceBadge) {
      confidenceBadge.className = 'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-medium ';
      if (data.confidence >= 0.9) {
        confidenceBadge.classList.add('badge-high');
        confidenceBadge.innerHTML = '<i class="ph ph-check-circle"></i> High Confidence';
      } else if (data.confidence >= 0.75) {
        confidenceBadge.classList.add('badge-medium');
        confidenceBadge.innerHTML = '<i class="ph ph-warning"></i> Medium Confidence';
      } else {
        confidenceBadge.classList.add('badge-low');
        confidenceBadge.innerHTML = '<i class="ph ph-x-circle"></i> Low Confidence';
      }
    }

    if (inferenceTime) {
      inferenceTime.textContent = (data.inference_time_ms || 0).toFixed(0) + 'ms';
    }

    // Top predictions
    const topList = document.getElementById('top-predictions-list');
    if (topList && data.top_predictions) {
      topList.innerHTML = '';

      data.top_predictions.forEach((pred, index) => {
        const pct = (pred.score * 100).toFixed(1);
        const barColor = index === 0
          ? 'bg-gradient-to-r from-indigo-500 to-violet-500'
          : 'bg-gradient-to-r from-slate-600 to-slate-500';

        const item = document.createElement('div');
        item.className = 'mb-3';
        item.innerHTML = `
          <div class="flex justify-between items-center mb-1">
            <span class="text-sm font-medium" style="color:var(--text-primary)">${pred.class_name}</span>
            <span class="text-sm font-semibold" style="color:var(--text-secondary)">${pct}%</span>
          </div>
          <div class="progress-bar-track">
            <div class="progress-bar-fill ${barColor}" style="width: 0%"></div>
          </div>
        `;
        topList.appendChild(item);

        // Animate bar
        requestAnimationFrame(() => {
          setTimeout(() => {
            const fill = item.querySelector('.progress-bar-fill');
            if (fill) fill.style.width = pct + '%';
          }, 100 + index * 120);
        });
      });
    }

    // Show buttons
    if (downloadBtn) downloadBtn.classList.remove('hidden');
    if (reuploadBtn) reuploadBtn.classList.remove('hidden');

    // Hide progress
    setTimeout(() => {
      if (progressContainer) progressContainer.classList.add('hidden');
    }, 500);
  }

  // ——— Download Result ———
  function downloadResult() {
    if (!resultData) return;

    const blob = new Blob([JSON.stringify(resultData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `applevision-result-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    Toast.success('Result downloaded!');
  }

})();
