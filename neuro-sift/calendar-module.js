/**
 * CogniBridge Hub — Reverse Accomplishment Calendar ("Done Log")
 * Tracks and visualizes what you actually finished on each day.
 */

export class CalendarModule {
  constructor(store, onSelectDateCallback) {
    this.store = store;
    this.onSelectDateCallback = onSelectDateCallback;
    this.currentViewDate = new Date();
    this.selectedDateKey = new Date().toISOString().split('T')[0]; // Default: today

    this.monthYearLabel = document.getElementById('cal-month-year');
    this.daysGrid = document.getElementById('calendar-days-grid');
    this.itemsList = document.getElementById('accomplishment-items-list');
    this.selectedBadge = document.getElementById('cal-selected-label');
    this.doneCountBadge = document.getElementById('done-count-badge');

    document.getElementById('cal-prev-month').addEventListener('click', () => this.changeMonth(-1));
    document.getElementById('cal-next-month').addEventListener('click', () => this.changeMonth(1));

    this.render();
  }

  changeMonth(offset) {
    this.currentViewDate.setMonth(this.currentViewDate.getMonth() + offset);
    this.render();
  }

  render() {
    const year = this.currentViewDate.getFullYear();
    const month = this.currentViewDate.getMonth();

    const monthNames = [
      "January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"
    ];
    this.monthYearLabel.textContent = `${monthNames[month]} ${year}`;

    this.daysGrid.innerHTML = "";

    // Day headers
    const days = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
    days.forEach(d => {
      const el = document.createElement('div');
      el.className = 'cal-header-day';
      el.textContent = d;
      this.daysGrid.appendChild(el);
    });

    const firstDayIndex = new Date(year, month, 1).getDay();
    const totalDays = new Date(year, month + 1, 0).getDate();
    const activeDatesSet = this.store.getAllActiveDatesWithAccomplishments();
    const todayKey = new Date().toISOString().split('T')[0];

    // Padding cells
    for (let i = 0; i < firstDayIndex; i++) {
      const blank = document.createElement('div');
      blank.className = 'cal-cell';
      blank.style.opacity = '0.2';
      this.daysGrid.appendChild(blank);
    }

    // Date cells
    for (let day = 1; day <= totalDays; day++) {
      const cell = document.createElement('div');
      const dateKey = `${year}-${String(month + 1).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
      
      cell.className = 'cal-cell';
      cell.textContent = day;

      if (dateKey === todayKey) cell.classList.add('today');
      if (dateKey === this.selectedDateKey) cell.classList.add('selected');
      if (activeDatesSet.has(dateKey)) cell.classList.add('has-activity');

      cell.addEventListener('click', () => {
        this.selectedDateKey = dateKey;
        this.render();
        this.renderAccomplishmentDocket(dateKey);
      });

      this.daysGrid.appendChild(cell);
    }

    this.renderAccomplishmentDocket(this.selectedDateKey);
  }

  renderAccomplishmentDocket(dateKey) {
    const items = this.store.getAccomplishmentsForDate(dateKey);
    this.itemsList.innerHTML = "";
    this.selectedBadge.textContent = dateKey;
    this.doneCountBadge.textContent = `${items.length} Done`;

    if (items.length === 0) {
      this.itemsList.innerHTML = `
        <li style="color:var(--text-muted); font-size:0.78rem; list-style:none;">
          No tasks recorded as completed on ${dateKey}.
        </li>
      `;
      return;
    }

    items.forEach(item => {
      const li = document.createElement('li');
      li.className = 'accomplishment-item';
      li.innerHTML = `
        <span>
          <strong style="color:var(--sage-dark);">✓</strong> ${item.title}
          <small style="color:var(--text-muted); margin-left:0.35rem;">(${item.domain})</small>
        </span>
        <span class="time-chip" style="font-size:0.68rem;">${item.timeStr}</span>
      `;
      this.itemsList.appendChild(li);
    });
  }
}