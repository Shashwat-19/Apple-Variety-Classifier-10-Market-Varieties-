/* =============================================================================
   AppleVision AI — Analytics Dashboard
   Fetches live data from /api/analytics and renders all charts + stat cards.

   Reusable functions:
     loadAnalytics()   — master fetch + render
     updateCards(data)  — fill KPI stat cards
     updateBarChart(data)  — bar chart: predictions by variety
     updatePieChart(data)  — donut chart: variety distribution
     updateTrendChart(data) — line chart: 30-day trend

   Charts are created once, then updated in-place via chart.update() on
   subsequent refreshes to avoid memory leaks and flicker.
   ============================================================================= */

(function (global) {
  'use strict';

  /* ── Cached Chart.js instances (created once, updated via chart.update()) ─ */
  let barChart    = null;
  let donutChart  = null;
  let lineChart   = null;

  /* ── Auto-refresh state ───────────────────────────────────────────────── */
  let autoRefreshInterval = null;

  /* ── Brand colour palette — vibrant on dark backgrounds ─────────────── */
  const CHART_COLORS = [
    '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e',
    '#f59e0b', '#10b981', '#06b6d4', '#3b82f6',
    '#a855f7', '#14b8a6'
  ];

  /* =========================================================================
     loadAnalytics — master entry point
     Fetches /api/analytics, then delegates to the four update helpers.
     Exported on window.Analytics so predict.js can call it after a save.
     ========================================================================= */
  async function loadAnalytics() {
    showLoadingState();

    let data;
    try {
      const res = await fetch('/api/analytics');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (!json.success) throw new Error(json.error || 'Analytics API error');
      data = json;
    } catch (err) {
      console.error('[Analytics] Fetch failed:', err.message);
      hideLoadingState();
      showFetchError();
      return;
    }

    /* Debug logs — confirm that the frontend receives valid data */
    console.log('Analytics API:', data);

    /* Update every section with the fresh payload */
    updateCards(data);
    updateBarChart(data);
    updatePieChart(data);
    updateTrendChart(data);
    hideLoadingState();
  }

  /* =========================================================================
     updateCards — fills the four KPI stat cards from live data
     ========================================================================= */
  function updateCards(data) {
    const totalEl = document.getElementById('stat-total');
    const mostEl  = document.getElementById('stat-most');
    const todayEl = document.getElementById('stat-today');
    const avgEl   = document.getElementById('stat-avg');

    if (totalEl) animateCounter(totalEl, data.total_predictions || 0);
    if (mostEl)  mostEl.textContent = data.most_predicted || '—';
    if (todayEl) animateCounter(todayEl, data.today_predictions || 0);
    if (avgEl) {
      /* average_confidence is already a percentage (e.g. 96.4) */
      const pct = parseFloat(data.average_confidence) || 0;
      animateCounter(avgEl, pct, 1);   /* 1 decimal place */
    }
  }

  /* =========================================================================
     updateBarChart — Predictions by Variety (bar chart)
     Creates the chart on first call; updates in-place on subsequent calls.
     ========================================================================= */
  function updateBarChart(data) {
    const canvas = document.getElementById('bar-chart');
    if (!canvas) return;

    const dist   = data.variety_distribution || {};
    const labels = Object.keys(dist);
    const values = Object.values(dist).map(Number);

    /* Debug logs */
    console.log('Bar Chart — Labels:', labels);
    console.log('Bar Chart — Values:', values);

    /* Empty-state guard: do not create an empty chart */
    if (labels.length === 0) {
      if (barChart) { barChart.destroy(); barChart = null; }
      renderNoData(canvas, 'bar-chart-nodata');
      return;
    }
    removeNoData('bar-chart-nodata');

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';

    /* If the chart already exists, update it in-place */
    if (barChart) {
      barChart.data.labels = labels;
      barChart.data.datasets[0].data = values;
      barChart.data.datasets[0].backgroundColor = CHART_COLORS.slice(0, labels.length).map(c => c + '99');
      barChart.data.datasets[0].borderColor = CHART_COLORS.slice(0, labels.length);
      barChart.options.scales.y.grid.color = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
      barChart.update();
      return;
    }

    /* First render — create the chart instance */
    barChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Predictions',
          data: values,
          backgroundColor: CHART_COLORS.slice(0, labels.length).map(c => c + '99'),
          borderColor:     CHART_COLORS.slice(0, labels.length),
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
          barPercentage: 0.7,
          categoryPercentage: 0.8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (tipCtx) => `${tipCtx.parsed.y} prediction${tipCtx.parsed.y !== 1 ? 's' : ''}`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { maxRotation: 45, font: { size: 11 } }
          },
          y: {
            beginAtZero: true,
            grid: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' },
            ticks: { precision: 0 }
          }
        },
        animation: { duration: 1200, easing: 'easeOutQuart' }
      }
    });
  }

  /* =========================================================================
     updatePieChart — Variety Distribution (donut chart)
     Creates the chart on first call; updates in-place on subsequent calls.
     ========================================================================= */
  function updatePieChart(data) {
    const canvas = document.getElementById('donut-chart');
    if (!canvas) return;

    const dist   = data.variety_distribution || {};
    const labels = Object.keys(dist);
    const values = Object.values(dist).map(Number);

    /* Debug logs */
    console.log('Pie Chart — Labels:', labels);
    console.log('Pie Chart — Values:', values);

    /* Empty-state guard */
    if (labels.length === 0) {
      if (donutChart) { donutChart.destroy(); donutChart = null; }
      renderNoData(canvas, 'donut-chart-nodata');
      return;
    }
    removeNoData('donut-chart-nodata');

    /* If the chart already exists, update it in-place */
    if (donutChart) {
      donutChart.data.labels = labels;
      donutChart.data.datasets[0].data = values;
      donutChart.data.datasets[0].backgroundColor = CHART_COLORS.slice(0, labels.length);
      donutChart.update();
      return;
    }

    /* First render */
    donutChart = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: CHART_COLORS.slice(0, labels.length),
          borderColor: 'transparent',
          borderWidth: 0,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: {
            position: 'right',
            labels: {
              padding: 16,
              boxWidth: 12, boxHeight: 12,
              useBorderRadius: true, borderRadius: 3,
              font: { size: 11 }
            }
          },
          tooltip: {
            callbacks: {
              label: (tipCtx) => {
                const total = tipCtx.dataset.data.reduce((a, b) => a + b, 0);
                const pct   = total ? ((tipCtx.parsed / total) * 100).toFixed(1) : '0.0';
                return `${tipCtx.label}: ${tipCtx.parsed} (${pct}%)`;
              }
            }
          }
        },
        animation: { animateRotate: true, duration: 1200, easing: 'easeOutQuart' }
      }
    });
  }

  /* =========================================================================
     updateTrendChart — 30-day prediction trend (line chart)
     Creates the chart on first call; updates in-place on subsequent calls.
     ========================================================================= */
  function updateTrendChart(data) {
    const canvas = document.getElementById('line-chart');
    if (!canvas) return;

    const trend  = data.daily_predictions || [];
    const labels = trend.map(t => {
      const d = new Date(t.date + 'T00:00:00');    /* avoid UTC offset shift */
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    const values = trend.map(t => Number(t.count));

    /* Debug logs */
    console.log('Trend Chart — Labels:', labels);
    console.log('Trend Chart — Values:', values);

    /* If every day is 0, show empty-state instead of a flat line */
    const hasAnyData = values.some(v => v > 0);
    if (!hasAnyData) {
      if (lineChart) { lineChart.destroy(); lineChart = null; }
      renderNoData(canvas, 'line-chart-nodata');
      return;
    }
    removeNoData('line-chart-nodata');

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';

    /* If the chart already exists, update it in-place */
    if (lineChart) {
      lineChart.data.labels = labels;
      lineChart.data.datasets[0].data = values;
      lineChart.options.scales.y.grid.color = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
      lineChart.update();
      return;
    }

    /* First render — create chart with gradient fill */
    const gradCtx = canvas.getContext('2d');
    const grad    = gradCtx.createLinearGradient(0, 0, 0, 300);
    grad.addColorStop(0, 'rgba(99,102,241,0.25)');
    grad.addColorStop(1, 'rgba(99,102,241,0)');

    lineChart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Predictions',
          data: values,
          borderColor: '#6366f1',
          backgroundColor: grad,
          borderWidth: 2.5,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: '#6366f1',
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            intersect: false,
            mode: 'index',
            callbacks: {
              label: (tipCtx) => `${tipCtx.parsed.y} prediction${tipCtx.parsed.y !== 1 ? 's' : ''}`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { maxTicksLimit: 8, font: { size: 11 } }
          },
          y: {
            beginAtZero: true,
            grid: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' },
            ticks: { precision: 0 }
          }
        },
        interaction: { intersect: false, mode: 'index' },
        animation: { duration: 1500, easing: 'easeOutQuart' }
      }
    });
  }

  /* =========================================================================
     Loading skeleton helpers
     ========================================================================= */
  function showLoadingState() {
    document.querySelectorAll('.stat-skeleton').forEach(el => el.classList.remove('hidden'));
    document.querySelectorAll('.stat-value').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.chart-skeleton').forEach(el => el.classList.remove('hidden'));
  }

  function hideLoadingState() {
    document.querySelectorAll('.stat-skeleton').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.stat-value').forEach(el => el.classList.remove('hidden'));
    document.querySelectorAll('.chart-skeleton').forEach(el => el.classList.add('hidden'));
  }

  /* =========================================================================
     "No prediction data available" overlay helpers
     Inserts a styled message over a canvas when there is no data.
     Does not create an empty chart. Does not throw errors.
     ========================================================================= */
  function renderNoData(canvas, overlayId) {
    /* Hide the canvas itself */
    canvas.style.display = 'none';

    /* Avoid duplicating the overlay */
    if (document.getElementById(overlayId)) return;

    const wrapper = canvas.parentElement;
    const div     = document.createElement('div');
    div.id = overlayId;
    div.style.cssText = [
      'display:flex', 'flex-direction:column', 'align-items:center',
      'justify-content:center', 'height:100%', 'min-height:200px',
      'gap:10px', 'color:var(--text-tertiary,#9ca3af)',
    ].join(';');
    div.innerHTML = `
      <i class="ph ph-chart-bar" style="font-size:2.5rem;opacity:0.35"></i>
      <span style="font-size:0.85rem;font-weight:500">No prediction data available</span>
    `;
    wrapper.appendChild(div);
  }

  function removeNoData(overlayId) {
    const overlay = document.getElementById(overlayId);
    if (overlay) overlay.remove();

    /* Restore canvas visibility */
    const canvasId = overlayId.replace('-nodata', '');
    const canvas   = document.getElementById(canvasId);
    if (canvas) canvas.style.display = '';
  }

  /* =========================================================================
     Fetch-error fallback — shown when the API returns an error
     ========================================================================= */
  function showFetchError() {
    ['stat-total', 'stat-today', 'stat-avg'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.textContent = '—';
    });
    const mostEl = document.getElementById('stat-most');
    if (mostEl) mostEl.textContent = '—';

    ['bar-chart', 'donut-chart', 'line-chart'].forEach(id => {
      const canvas = document.getElementById(id);
      if (canvas) renderNoData(canvas, id + '-nodata');
    });

    hideLoadingState();
  }

  /* =========================================================================
     animateCounter — smoothly counts up to a target number
     ========================================================================= */
  function animateCounter(el, target, decimals) {
    decimals = decimals || 0;
    const start    = 0;
    const duration = 1200;
    const startTs  = performance.now();

    function step(ts) {
      const elapsed  = ts - startTs;
      const progress = Math.min(elapsed / duration, 1);
      /* ease-out cubic */
      const eased    = 1 - Math.pow(1 - progress, 3);
      const value    = start + (target - start) * eased;
      el.textContent = decimals > 0 ? value.toFixed(decimals) : Math.floor(value).toString();
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  }

  /* =========================================================================
     Auto-refresh toggle binding
     ========================================================================= */
  function bindAutoRefresh() {
    const toggle = document.getElementById('auto-refresh-toggle');
    if (!toggle) return;

    toggle.addEventListener('change', function () {
      if (toggle.checked) {
        autoRefreshInterval = setInterval(loadAnalytics, 30000); /* 30s */
        if (typeof Toast !== 'undefined') Toast.info('Auto-refresh enabled (30s)');
      } else {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        if (typeof Toast !== 'undefined') Toast.info('Auto-refresh disabled');
      }
    });
  }

  /* =========================================================================
     Theme-change listener — need to destroy and recreate charts so grid
     colours adapt. This is the only case where charts are rebuilt.
     ========================================================================= */
  function bindThemeChange() {
    window.addEventListener('themeChanged', function () {
      /* Destroy cached instances so they're recreated with new theme colours */
      if (barChart)   { barChart.destroy();   barChart   = null; }
      if (donutChart) { donutChart.destroy(); donutChart = null; }
      if (lineChart)  { lineChart.destroy();  lineChart  = null; }
      loadAnalytics();
    });
  }

  /* =========================================================================
     Live-refresh hook — called from predict.js after a successful prediction
     Listens for the custom 'predictionSaved' event so the dashboard
     updates automatically without requiring a page reload.
     ========================================================================= */
  function bindPredictionSavedEvent() {
    window.addEventListener('predictionSaved', loadAnalytics);
  }

  /* =========================================================================
     Init
     ========================================================================= */
  document.addEventListener('DOMContentLoaded', function () {
    loadAnalytics();
    bindAutoRefresh();
    bindThemeChange();
    bindPredictionSavedEvent();
  });

  /* ── Expose public API ────────────────────────────────────────────────── */
  global.Analytics = {
    loadAnalytics:   loadAnalytics,
    updateCards:      updateCards,
    updateBarChart:   updateBarChart,
    updatePieChart:   updatePieChart,
    updateTrendChart: updateTrendChart
  };

})(window);
