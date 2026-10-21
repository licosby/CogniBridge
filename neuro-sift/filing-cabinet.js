/**
 * CogniBridge Hub — Document Filing Cabinet Drawer
 * Archives, searches, and deconstructs passive corporate emails and memos.
 */

export class FilingCabinetModule {
  constructor(store, logAuditCallback) {
    this.store = store;
    this.logAudit = logAuditCallback;

    this.container = document.getElementById('cabinet-list-container');
    this.searchInput = document.getElementById('cabinet-search');
    this.countBadge = document.getElementById('cabinet-badge-count');

    this.searchInput.addEventListener('input', () => this.render());
    this.render();
  }

  render() {
    const query = this.searchInput.value.toLowerCase().trim();
    this.container.innerHTML = "";

    const items = this.store.cabinetItems.filter(item => {
      if (!query) return true;
      return item.title.toLowerCase().includes(query) || item.tlDr.toLowerCase().includes(query);
    });

    this.countBadge.textContent = this.store.cabinetItems.length;

    if (items.length === 0) {
      this.container.innerHTML = `
        <div style="text-align:center; padding:2rem; color:var(--text-muted); font-size:0.82rem;">
          Filing Cabinet empty. Ingest corporate memos or passive emails to file them away here.
        </div>
      `;
      return;
    }

    items.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'task-card';
      card.style.borderLeft = "4px solid var(--slate-blue)";

      const takeawaysHtml = item.takeaways.map(t => `<li style="font-size:0.8rem; margin-bottom:0.25rem;">${t}</li>`).join('');

      card.innerHTML = `
        <div class="task-card-header">
          <div>
            <strong style="font-size:0.95rem; color:var(--slate-blue);">${item.title}</strong>
            <small style="display:block; color:var(--text-muted); font-size:0.72rem;">Filed on ${item.timestamp}</small>
          </div>
          <button class="btn-clear" style="color:var(--terracotta); font-size:0.74rem;" data-action="delete" data-idx="${idx}">Archive / Delete</button>
        </div>
        <p style="font-size:0.85rem; color:var(--text-main); margin:0.35rem 0; line-height:1.4;">
          ${item.tlDr}
        </p>
        <div style="background:var(--bg-subtle); padding:0.6rem; border-radius:8px; border:1px solid var(--border-line);">
          <strong style="font-size:0.74rem; color:var(--text-muted); display:block; margin-bottom:0.25rem;">Plain-English Takeaways:</strong>
          <ul style="padding-left:1rem;">${takeawaysHtml}</ul>
        </div>
        <div style="margin-top:0.4rem; display:flex; justify-content:space-between; align-items:center;">
          <small style="font-size:0.72rem; color:var(--text-muted);">Suggested Reply Drafted</small>
          <button class="btn btn-outline btn-sm" data-action="copy" data-reply="${encodeURIComponent(item.replyDraft)}">Copy Reply Draft</button>
        </div>
      `;

      card.querySelector('[data-action="delete"]').addEventListener('click', () => {
        this.store.cabinetItems.splice(idx, 1);
        this.store.save();
        this.render();
        this.logAudit("Document removed from Filing Cabinet.");
      });

      card.querySelector('[data-action="copy"]').addEventListener('click', (e) => {
        const reply = decodeURIComponent(e.target.dataset.reply);
        navigator.clipboard.writeText(reply);
        e.target.textContent = "Copied to Clipboard!";
        setTimeout(() => e.target.textContent = "Copy Reply Draft", 2000);
        this.logAudit("Copied polite email reply draft to clipboard.");
      });

      this.container.appendChild(card);
    });
  }
}