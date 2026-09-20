/* Progressive enhancement; all content and links work without JavaScript. */
(() => {
  let observer;
  let copyReset;
  const setup = () => {
    observer?.disconnect();
    const home = document.querySelector('.eg-home');
    if (!home) return;
    if ('IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
      home.classList.add('eg-motion');
      observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add('eg-visible');
            observer.unobserve(entry.target);
          }
        }
      }, { threshold: 0.12 });
      home.querySelectorAll('.eg-reveal, .eg-trace').forEach(el => observer.observe(el));
    } else {
      home.querySelectorAll('.eg-reveal, .eg-trace').forEach(el => el.classList.add('eg-visible'));
    }
    home.querySelectorAll('[data-eg-copy]').forEach(button => {
      if (button.dataset.egReady) return;
      button.dataset.egReady = 'true';
      button.addEventListener('click', async () => {
        const command = document.getElementById(button.dataset.egCopy);
        const status = home.querySelector('.eg-copy-status');
        const label = button.querySelector('[data-eg-copy-label]');
        if (!command || !status || !label) return;
        try {
          if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
          await navigator.clipboard.writeText(command.textContent.trim());
          label.textContent = 'copied';
          status.textContent = 'installation command copied.';
          clearTimeout(copyReset);
          copyReset = setTimeout(() => { label.textContent = 'copy'; status.textContent = ''; }, 4000);
        } catch {
          const selection = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(command);
          selection.removeAllRanges();
          selection.addRange(range);
          status.textContent = 'command selected. press ⌘C or Ctrl+C to copy.';
        }
      });
    });
  };
  // Material's observable fires on initial and instant navigation.
  if (typeof document$ !== 'undefined') document$.subscribe(setup);
  else if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setup, { once: true });
  else setup();
})();
