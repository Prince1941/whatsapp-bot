// Tab Switching Logic
function switchTab(tabId) {
  // Update tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });

  // Find matching button and activate
  const buttons = document.querySelectorAll('.tab-btn');
  buttons.forEach(btn => {
    if (btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabId)) {
      btn.classList.add('active');
    }
  });

  // Update panels
  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.remove('active');
  });

  const targetPanel = document.getElementById(tabId);
  if (targetPanel) {
    targetPanel.classList.add('active');
  }
}

// File Selection Label
function handleFileSelected(input) {
  const label = document.getElementById('selected-filename');
  if (input.files && input.files.length > 0) {
    const file = input.files[0];
    label.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  } else {
    label.textContent = '';
  }
}

// Drag & Drop Support
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('chat-file-input');

  if (dropZone && fileInput) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
      }, false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        fileInput.files = files;
        handleFileSelected(fileInput);
      }
    });
  }
});

// Toast notification
function showToast(message) {
  const toast = document.getElementById('toast');
  if (!toast) return;

  toast.textContent = message;
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
  }, 2800);
}

// Copy draft text to clipboard
async function copyDraftText(textareaId) {
  const textarea = document.getElementById(textareaId);
  if (!textarea) return;

  const text = textarea.value;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      // Fallback
      textarea.select();
      document.execCommand('copy');
    }
    showToast('✓ Draft copied to clipboard! Remember to review before sending.');
  } catch (err) {
    console.error('Failed to copy text: ', err);
    showToast('Failed to copy text to clipboard.');
  }
}

// Confirm Data Purge
async function confirmPurge() {
  const confirmed = confirm(
    'PERMANENT DELETE CONFIRMATION:\n\n' +
    'Are you sure you want to delete ALL style profiles, custom exclusion rules, and imported chats from this machine?\n\n' +
    'This cannot be undone.'
  );

  if (!confirmed) return;

  try {
    const resp = await fetch('/api/purge', { method: 'POST' });
    const html = await resp.text();
    const alertArea = document.getElementById('purge-alert-area');
    if (alertArea) {
      alertArea.innerHTML = html;
    }
    showToast('All local data has been purged.');

    // Reload profile container
    if (window.htmx) {
      htmx.ajax('GET', '/api/profile/view', { target: '#profile-container' });
    }
  } catch (err) {
    alert('Error purging data: ' + err.message);
  }
}

// One-time client migration: clear any stale cached references to llama3.2
try {
  for (let i = localStorage.length - 1; i >= 0; i--) {
    const key = localStorage.key(i);
    const val = localStorage.getItem(key);
    if (val && typeof val === 'string' && val.toLowerCase().includes('llama3.2')) {
      localStorage.removeItem(key);
    }
  }
} catch (e) {}
