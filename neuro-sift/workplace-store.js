/**
 * CogniBridge Hub — Persistent Workspace & Accomplishment Store
 * Manages task dossiers, filing cabinet emails, and calendar accomplishment logs.
 */

export class WorkspaceStore {
  constructor() {
    this.STORAGE_KEY_DOSSIERS = 'cognibridge_dossiers_v2';
    this.STORAGE_KEY_CABINET = 'cognibridge_cabinet_v2';
    this.STORAGE_KEY_ACCOMPLISHMENTS = 'cognibridge_done_log_v2';

    this.dossiers = [];
    this.cabinetItems = [];
    this.accomplishments = []; // Array of { id, text, domain, completedDateStr, completedTimeStr }

    this.load();
  }

  load() {
    try {
      this.dossiers = JSON.parse(localStorage.getItem(this.STORAGE_KEY_DOSSIERS)) || [];
      this.cabinetItems = JSON.parse(localStorage.getItem(this.STORAGE_KEY_CABINET)) || [];
      this.accomplishments = JSON.parse(localStorage.getItem(this.STORAGE_KEY_ACCOMPLISHMENTS)) || [];
    } catch (e) {
      this.dossiers = [];
      this.cabinetItems = [];
      this.accomplishments = [];
    }
  }

  save() {
    try {
      localStorage.setItem(this.STORAGE_KEY_DOSSIERS, JSON.stringify(this.dossiers));
      localStorage.setItem(this.STORAGE_KEY_CABINET, JSON.stringify(this.cabinetItems));
      localStorage.setItem(this.STORAGE_KEY_ACCOMPLISHMENTS, JSON.stringify(this.accomplishments));
    } catch (e) {}
  }

  addDossier(dossier) {
    this.dossiers.unshift(dossier);
    this.save();
  }

  addCabinetItem(item) {
    this.cabinetItems.unshift(item);
    this.save();
  }

  recordAccomplishment(title, domain) {
    const now = new Date();
    const dateKey = now.toISOString().split('T')[0]; // "YYYY-MM-DD"
    const timeStr = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}`;

    this.accomplishments.unshift({
      id: Date.now() + Math.random(),
      title: title,
      domain: domain || 'work',
      dateKey: dateKey,
      timeStr: timeStr
    });
    this.save();
  }

  getAccomplishmentsForDate(dateKey) {
    return this.accomplishments.filter(a => a.dateKey === dateKey);
  }

  getAllActiveDatesWithAccomplishments() {
    const dates = new Set();
    this.accomplishments.forEach(a => dates.add(a.dateKey));
    return dates;
  }

  clearAll() {
    this.dossiers = [];
    this.cabinetItems = [];
    this.accomplishments = [];
    this.save();
  }
}