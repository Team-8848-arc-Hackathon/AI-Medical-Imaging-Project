// Drawer
const drawer = document.getElementById('drawer');
const scrim = document.getElementById('scrim');
const btnMenu = document.getElementById('btnMenu');
const btnClose = document.getElementById('btnClose');

function openDrawer(){drawer.classList.add('open'); scrim.classList.add('show');}
function closeDrawer(){drawer.classList.remove('open'); scrim.classList.remove('show');}
btnMenu?.addEventListener('click', openDrawer);
btnClose?.addEventListener('click', closeDrawer);
scrim?.addEventListener('click', closeDrawer);

// Dark mode & Nepali font persistence
const toggleDark = document.getElementById('toggleDark');
const toggleNepaliFont = document.getElementById('toggleNepaliFont');

function applyTheme(){
  const isDark = localStorage.getItem('theme') === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'dark':'light');
  if (toggleDark) toggleDark.checked = isDark;
}
function applyFont(){
  const np = localStorage.getItem('np-font') === '1';
  document.body.classList.toggle('lang-np', np);
  if (toggleNepaliFont) toggleNepaliFont.checked = np;
}
applyTheme();
applyFont();

toggleDark?.addEventListener('change', () => {
  localStorage.setItem('theme', toggleDark.checked ? 'dark':'light');
  applyTheme();
});
toggleNepaliFont?.addEventListener('change', () => {
  localStorage.setItem('np-font', toggleNepaliFont.checked ? '1':'0');
  applyFont();
});

// Language selection mirrored into settings radios
const langSelect = document.getElementById('langSelect');
const radios = document.querySelectorAll('input[name="lang"]');
radios.forEach(r => r.addEventListener('change', () => {
  langSelect.value = document.querySelector('input[name="lang"]:checked')?.value || 'en';
}));
langSelect?.addEventListener('change', () => {
  const val = langSelect.value;
  document.querySelectorAll('input[name="lang"]').forEach(r => r.checked = (r.value === val));
});

// Loading overlay
const form = document.getElementById('analyzeForm');
const loading = document.getElementById('loading');
form?.addEventListener('submit', () => { loading.classList.remove('hidden'); });

// HARD RESET: go to clean GET route to avoid 405 and fully clear results
form?.addEventListener('reset', (e) => {
  e.preventDefault();
  loading?.classList.add('hidden');
  const home = document.body?.dataset?.home || "/";
  window.location.assign(home);
});

// Feedback modal
const btnReport = document.getElementById('btnReport');
const dlg = document.getElementById('dlgFeedback');
const fbSend = document.getElementById('fbSend');
const fbText = document.getElementById('fbText');
btnReport?.addEventListener('click', () => { dlg.showModal(); });
fbSend?.addEventListener('click', async (e) => {
  e.preventDefault();
  const message = fbText.value.trim();
  if(!message) { dlg.close(); return; }
  try{
    await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message})});
  }catch(e){}
  fbText.value='';
  dlg.close();
});
