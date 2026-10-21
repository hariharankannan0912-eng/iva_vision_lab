const tabs = [...document.querySelectorAll('.tab')];
const panels = [...document.querySelectorAll('.panel')];
tabs.forEach(tab => tab.addEventListener('click', () => {
  tabs.forEach(t => t.classList.toggle('active', t === tab));
  panels.forEach(p => p.classList.toggle('active', p.id === tab.dataset.target));
}));
document.querySelectorAll('input[type="file"]').forEach(input => input.addEventListener('change', () => {
  const label = input.closest('.upload-card');
  const name = label.querySelector('.file-name');
  name.textContent = input.files[0] ? input.files[0].name : 'Choose an image';
}));
const resultSection = document.getElementById('result');
const resultContent = document.getElementById('result-content');
const toast = document.getElementById('toast');
function notify(message) { toast.textContent = message; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2400); }
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
document.querySelectorAll('.tool-form').forEach(form => form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = form.querySelector('.run-btn');
  if (!form.reportValidity()) return;
  const original = button.innerHTML;
  button.disabled = true; button.innerHTML = 'Processing image… <span>◌</span>';
  resultSection.hidden = false;
  resultContent.innerHTML = '<p class="result-message">Running analysis on the Python backend. Larger images or first-time model loads can take a little while.</p>';
  resultSection.scrollIntoView({behavior:'smooth', block:'start'});
  try {
    const response = await fetch(form.dataset.endpoint, {method:'POST', body:new FormData(form)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The analysis failed.');
    let metrics = '';
    if (form.dataset.kind === 'template') {
      metrics = `<div class="metric"><small>Best match score</small><strong>${(data.score*100).toFixed(1)}%</strong></div><div class="metric"><small>Top-left location</small><strong style="font-size:18px">x ${data.location.x}, y ${data.location.y}</strong></div><div class="metric"><small>Template size</small><strong style="font-size:18px">${data.template_size.width} × ${data.template_size.height}</strong></div>`;
    } else if (form.dataset.kind === 'single') {
      metrics = `<div class="metric"><small>Faces detected</small><strong>${data.count}</strong></div><div class="metric"><small>Method</small><strong style="font-size:17px">Haar cascade</strong></div>`;
    } else if (form.dataset.endpoint.includes('deepface')) {
      metrics = `<div class="metric"><small>Verification</small><strong style="font-size:22px">${data.verified ? 'Likely match' : 'Not a match'}</strong></div><div class="metric"><small>Distance</small><strong>${data.distance}</strong></div><div class="metric"><small>Model</small><strong style="font-size:15px">${esc(data.model || 'Facenet512')}</strong></div>`;
    } else {
      metrics = `<div class="metric"><small>Cosine similarity</small><strong>${data.similarity}</strong></div><div class="metric"><small>Interpretation</small><strong style="font-size:15px">Compare cautiously</strong></div>`;
    }
    resultContent.innerHTML = `<div class="result-body">${data.image ? `<img src="${data.image}" alt="Processed result">` : '<div class="metric"><small>Analysis complete</small><strong style="font-size:18px">✓ Done</strong></div>'}<div class="metrics">${metrics}<p class="result-message">${esc(data.message || '')}</p></div></div>`;
    notify('Analysis completed');
  } catch (err) {
    resultContent.innerHTML = `<div class="error-message">${esc(err.message)}<br><br>Check the server terminal for details. If this is an optional model, follow the install command shown above.</div>`;
    notify('Analysis needs attention');
  } finally {
    button.disabled = false; button.innerHTML = original;
  }
}));
document.getElementById('clear-result').addEventListener('click', () => { resultSection.hidden = true; resultContent.innerHTML = ''; });
