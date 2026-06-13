/* =============================================
   AppleVision AI — Analytics Dashboard Charts
   Chart.js initialization, data fetching, rendering
   ============================================= */

(function () {
  'use strict';

  let barChart = null;
  let donutChart = null;
  let lineChart = null;
  let autoRefreshInterval = null;

  const CHART_COLORS = [
    '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e',
    '#f59e0b', '#10b981', '#06b6d4', '#3b82f6',
    '#a855f7', '#14b8a6'
  ];

  const APPLE_VARIETIES = [
    'Braeburn', 'Crimson Snow', 'Golden 1', 'Golden 2', 'Golden 3',
    'Granny Smith', 'Pink Lady', 'Red 1', 'Red 2', 'Red 3'
  ];

  // ——— Init ———
  document.addEventListener('DOMContentLoaded', () => {
    fetchAndRender();
    bindAutoRefresh();

    // Re-render on theme change
    window.addEventListener('themeChanged', () => {
      if (barChart) barChart.destroy();
      if (donutChart) donutChart.destroy();
      if (lineChart) lineChart.destroy();
      fetchAndRender();
    });
  });

  // ——— Fetch Data ———
  async function fetchAndRender() {
    showSkeletons();

    try {
      const res = await fetch('/api/stats');
      if (!res.ok) throw new Error('Failed to fetch stats');

      const data = await res.json();
      renderStatCards(data);
      renderBarChart(data);
      renderDonutChart(data);
      renderLineChart(data);
      hideSkeletons();

    } catch (err) {
      console.warn('Analytics fetch failed, using demo data:', err.message);
      const demoData = generateDemoData();
      renderStatCards(demoData);
      renderBarChart(demoData);
      renderDonutChart(demoData);
      renderLineChart(demoData);
      hideSkeletons();
    }
  }

  // ——— Demo Data Generator ———
  function generateDemoData() {
    const varietyCounts = {};
    APPLE_VARIETIES.forEach(v => {
      varietyCounts[v] = Math.floor(Math.random() * 150) + 20;
    });

    const dailyTrend = [];
    const today = new Date();
    for (let i = 29; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      dailyTrend.push({
        date: d.toISOString().split('T')[0],
        count: Math.floor(Math.random() * 40) + 5
      });
    }

    const totalPredictions = Object.values(varietyCounts).reduce((a, b) => a + b, 0);
    const mostPredicted = Object.entries(varietyCounts).sort((a, b) => b[1] - a[1])[0];

    return {
      total_predictions: totalPredictions,
      most_predicted: mostPredicted[0],
      today_count: dailyTrend[dailyTrend.length - 1].count,
      avg_confidence: (Math.random() * 10 + 88).toFixed(1),
      variety_counts: varietyCounts,
      daily_trend: dailyTrend
    };
  }

  // ——— Stat Cards ———
  function renderStatCards(data) {
    const totalEl = document.getElementById('stat-total');
    const mostEl = document.getElementById('stat-most');
    const todayEl = document.getElementById('stat-today');
    const avgEl = document.getElementById('stat-avg');

    if (totalEl) animateCounter(totalEl, data.total_predictions || 0);
    if (mostEl) mostEl.textContent = data.most_predicted || '—';
    if (todayEl) animateCounter(todayEl, data.today_count || 0);
    if (avgEl) {
      const target = parseFloat(data.avg_confidence) || 0;
      animateCounter(avgEl, target);
      avgEl.dataset.suffix = '%';
    }
  }

  // ——— Bar Chart ———
  function renderBarChart(data) {
    const ctx = document.getElementById('bar-chart');
    if (!ctx) return;

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const varieties = data.variety_counts || {};
    const labels = Object.keys(varieties);
    const values = Object.values(varieties);

    barChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Predictions',
          data: values,
          backgroundColor: CHART_COLORS.slice(0, labels.length).map(c => c + '99'),
          borderColor: CHART_COLORS.slice(0, labels.length),
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
          barPercentage: 0.7,
          categoryPercentage: 0.8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.parsed.y} predictions`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              maxRotation: 45,
              font: { size: 11 }
            }
          },
          y: {
            beginAtZero: true,
            grid: {
              color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
            },
            ticks: {
              precision: 0
            }
          }
        },
        animation: {
          duration: 1200,
          easing: 'easeOutQuart'
        }
      }
    });
  }

  // ——— Donut Chart ———
  function renderDonutChart(data) {
    const ctx = document.getElementById('donut-chart');
    if (!ctx) return;

    const varieties = data.variety_counts || {};
    const labels = Object.keys(varieties);
    const values = Object.values(varieties);

    donutChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: CHART_COLORS.slice(0, labels.length),
          borderColor: 'transparent',
          borderWidth: 0,
          hoverOffset: 8
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
              boxWidth: 12,
              boxHeight: 12,
              useBorderRadius: true,
              borderRadius: 3,
              font: { size: 11 }
            }
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                const pct = ((ctx.parsed / total) * 100).toFixed(1);
                return `${ctx.label}: ${ctx.parsed} (${pct}%)`;
              }
            }
          }
        },
        animation: {
          animateRotate: true,
          duration: 1200,
          easing: 'easeOutQuart'
        }
      }
    });
  }

  // ——— Line Chart ———
  function renderLineChart(data) {
    const ctx = document.getElementById('line-chart');
    if (!ctx) return;

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const trend = data.daily_trend || [];
    const labels = trend.map(t => {
      const d = new Date(t.date);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    const values = trend.map(t => t.count);

    const gradient = ctx.getContext('2d');
    const gradientFill = gradient.createLinearGradient(0, 0, 0, 300);
    gradientFill.addColorStop(0, 'rgba(99, 102, 241, 0.25)');
    gradientFill.addColorStop(1, 'rgba(99, 102, 241, 0)');

    lineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Predictions',
          data: values,
          borderColor: '#6366f1',
          backgroundColor: gradientFill,
          borderWidth: 2.5,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: '#6366f1',
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2
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
              label: (ctx) => `${ctx.parsed.y} predictions`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              maxTicksLimit: 8,
              font: { size: 11 }
            }
          },
          y: {
            beginAtZero: true,
            grid: {
              color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'
            },
            ticks: {
              precision: 0
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        animation: {
          duration: 1500,
          easing: 'easeOutQuart'
        }
      }
    });
  }

  // ——— Skeletons ———
  function showSkeletons() {
    document.querySelectorAll('.stat-skeleton').forEach(el => el.classList.remove('hidden'));
    document.querySelectorAll('.stat-value').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.chart-skeleton').forEach(el => el.classList.remove('hidden'));
  }

  function hideSkeletons() {
    document.querySelectorAll('.stat-skeleton').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.stat-value').forEach(el => el.classList.remove('hidden'));
    document.querySelectorAll('.chart-skeleton').forEach(el => el.classList.add('hidden'));
  }

  // ——— Auto Refresh ———
  function bindAutoRefresh() {
    const toggle = document.getElementById('auto-refresh-toggle');
    if (!toggle) return;

    toggle.addEventListener('change', () => {
      if (toggle.checked) {
        autoRefreshInterval = setInterval(fetchAndRender, 30000); // 30s
        Toast.info('Auto-refresh enabled (30s)');
      } else {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        Toast.info('Auto-refresh disabled');
      }
    });
  }

})();
