/* =============================================
   AppleVision AI — Core App Logic
   Theme toggle, navigation, scroll animations,
   toast system, and shared utilities.
   ============================================= */

(function () {
  'use strict';

  // ——— Theme Management ———
  const ThemeManager = {
    STORAGE_KEY: 'applevision-theme',

    init() {
      const saved = localStorage.getItem(this.STORAGE_KEY);
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      const theme = saved || (prefersDark ? 'dark' : 'dark'); // default dark
      this.apply(theme);
      this.bindToggle();
    },

    apply(theme) {
      document.documentElement.setAttribute('data-theme', theme);
      localStorage.setItem(this.STORAGE_KEY, theme);
      this.updateIcon(theme);
    },

    toggle() {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      this.apply(next);
    },

    updateIcon(theme) {
      const btn = document.getElementById('theme-toggle-btn');
      if (!btn) return;
      const icon = btn.querySelector('i');
      if (icon) {
        icon.className = theme === 'dark' ? 'ph ph-sun' : 'ph ph-moon';
      }
    },

    bindToggle() {
      const btn = document.getElementById('theme-toggle-btn');
      if (btn) {
        btn.addEventListener('click', () => this.toggle());
      }
      const mobileBtn = document.getElementById('theme-toggle-mobile');
      if (mobileBtn) {
        mobileBtn.addEventListener('click', () => this.toggle());
      }
    }
  };

  // ——— Navigation ———
  const Navigation = {
    init() {
      this.setActiveLink();
      this.bindMobileMenu();
      this.bindScrollBehavior();
    },

    setActiveLink() {
      const path = window.location.pathname;
      document.querySelectorAll('.nav-link').forEach(link => {
        const href = link.getAttribute('href');
        if (href === path || (href !== '/' && path.startsWith(href))) {
          link.classList.add('active');
        }
      });
    },

    bindMobileMenu() {
      const hamburger = document.getElementById('hamburger-btn');
      const mobileMenu = document.getElementById('mobile-menu');
      const overlay = document.getElementById('mobile-menu-overlay');

      if (!hamburger || !mobileMenu) return;

      const toggle = () => {
        hamburger.classList.toggle('active');
        mobileMenu.classList.toggle('active');
        if (overlay) overlay.classList.toggle('active');
        document.body.style.overflow = mobileMenu.classList.contains('active') ? 'hidden' : '';
      };

      hamburger.addEventListener('click', toggle);
      if (overlay) overlay.addEventListener('click', toggle);

      mobileMenu.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', toggle);
      });
    },

    bindScrollBehavior() {
      const nav = document.getElementById('main-nav');
      if (!nav) return;

      let lastScroll = 0;
      window.addEventListener('scroll', () => {
        const current = window.scrollY;
        if (current > 50) {
          nav.classList.add('shadow-lg');
        } else {
          nav.classList.remove('shadow-lg');
        }
        lastScroll = current;
      }, { passive: true });
    }
  };

  // ——— Scroll Reveal ———
  const ScrollReveal = {
    init() {
      const els = document.querySelectorAll('.reveal');
      if (!els.length) return;

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            observer.unobserve(entry.target);
          }
        });
      }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
      });

      els.forEach(el => observer.observe(el));
    }
  };

  // ——— Toast Notifications ———
  window.Toast = {
    container: null,

    getContainer() {
      if (!this.container) {
        this.container = document.getElementById('toast-container');
        if (!this.container) {
          this.container = document.createElement('div');
          this.container.id = 'toast-container';
          this.container.className = 'toast-container';
          document.body.appendChild(this.container);
        }
      }
      return this.container;
    },

    show(message, type = 'info', duration = 4000) {
      const container = this.getContainer();
      const toast = document.createElement('div');
      toast.className = `toast toast-${type}`;

      const icons = {
        success: 'ph ph-check-circle',
        error: 'ph ph-x-circle',
        info: 'ph ph-info'
      };

      toast.innerHTML = `
        <i class="${icons[type] || icons.info}" style="font-size:1.25rem"></i>
        <span>${message}</span>
      `;

      container.appendChild(toast);

      setTimeout(() => {
        toast.classList.add('toast-out');
        toast.addEventListener('animationend', () => toast.remove());
      }, duration);
    },

    success(msg, dur) { this.show(msg, 'success', dur); },
    error(msg, dur)   { this.show(msg, 'error', dur); },
    info(msg, dur)    { this.show(msg, 'info', dur); }
  };

  // ——— Counter Animation ———
  window.animateCounter = function (el, target, duration = 1500) {
    const start = 0;
    const startTime = performance.now();
    const isFloat = String(target).includes('.');

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = start + (target - start) * eased;

      if (isFloat) {
        el.textContent = current.toFixed(1);
      } else {
        el.textContent = Math.round(current).toLocaleString();
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  };

  // ——— Copy to Clipboard ———
  window.copyToClipboard = function (text, btnEl) {
    navigator.clipboard.writeText(text).then(() => {
      const originalText = btnEl.textContent;
      btnEl.textContent = 'Copied!';
      btnEl.style.color = 'var(--accent-emerald)';
      setTimeout(() => {
        btnEl.textContent = originalText;
        btnEl.style.color = '';
      }, 2000);
    }).catch(() => {
      Toast.error('Failed to copy');
    });
  };

  // ——— Utility: Format Number ———
  window.formatNumber = function (n) {
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
    return n.toString();
  };

  // ——— Chart.js Global Defaults ———
  function configureChartDefaults() {
    if (typeof Chart === 'undefined') return;

    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const textColor = isDark ? '#94a3b8' : '#64748b';
    const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';

    Chart.defaults.color = textColor;
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.padding = 20;
    Chart.defaults.plugins.tooltip.backgroundColor = isDark ? '#1e1e2e' : '#ffffff';
    Chart.defaults.plugins.tooltip.titleColor = isDark ? '#f8fafc' : '#0f172a';
    Chart.defaults.plugins.tooltip.bodyColor = isDark ? '#94a3b8' : '#64748b';
    Chart.defaults.plugins.tooltip.borderColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 10;
    Chart.defaults.plugins.tooltip.displayColors = true;
    Chart.defaults.plugins.tooltip.boxPadding = 4;
    Chart.defaults.scale.grid = { color: gridColor };
  }

  // ——— Theme Observer for Charts ———
  const themeObserver = new MutationObserver(() => {
    configureChartDefaults();
    // Dispatch event for pages to re-render charts
    window.dispatchEvent(new CustomEvent('themeChanged'));
  });

  // ——— Init ———
  document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    Navigation.init();
    ScrollReveal.init();
    configureChartDefaults();

    // Watch for theme changes
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    // Page enter animation
    const main = document.querySelector('main');
    if (main) main.classList.add('page-enter');

    // Code copy buttons
    document.querySelectorAll('.code-copy-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const code = btn.closest('.code-block').querySelector('code');
        if (code) copyToClipboard(code.textContent, btn);
      });
    });
  });
})();
