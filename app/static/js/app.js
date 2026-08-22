// app.js — shared utilities for colab_listening_b_web

// Jinja2 date filter helper (used in runs.html)
// Since Jinja2 doesn't have a native datestr filter, we handle it here
document.addEventListener('DOMContentLoaded', function() {
    // Convert Unix timestamps to readable dates if needed
    document.querySelectorAll('[data-timestamp]').forEach(el => {
        const ts = parseInt(el.getAttribute('data-timestamp'));
        if (ts) {
            const d = new Date(ts * 1000);
            el.textContent = d.toLocaleDateString('zh-CN', {month: 'short', day: 'numeric'});
        }
    });
});
