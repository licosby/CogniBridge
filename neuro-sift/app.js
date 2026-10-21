/**
 * CogniBridge Hub — Complete Application & Online Machine Learning Controller
 * - Removed all email reply drafting elements from Filing Cabinet
 * - Removed 'Expand Day Page' text link; calendar date numbers are the direct CTA buttons
 */

(function () {
  'use strict';

  // --- 1. JARGON GLOSSARY ---
  const dynamicGlossary = {
    apprised: "Informed or updated about something.",
    pursuant: "Following or in accordance with an official rule or decision.",
    synergy: "Working together so the combined result is slightly better than working separately.",
    initiatives: "New projects or corporate campaigns.",
    paradigms: "Established habits, standard methods, or ways of doing work.",
    transition: "Gradually moving or changing from an old way to a new way.",
    decentralized: "Distributed across multiple teams instead of controlled by one central boss.",
    workflows: "The step-by-step sequence of tasks needed to finish a project.",
    pivot: "A rapid change in business direction or product strategy.",
    necessitates: "Requires or makes something necessary.",
    holistic: "Looking at the whole picture rather than just one small part.",
    stakeholder: "Anyone with an interest, investment, or say in a project's outcome.",
    cohorts: "Specific groups of colleagues, departments, or users.",
    roi: "Return On Investment: calculating if the money or time spent was worth the payoff.",
    leverage: "Taking advantage of an existing tool or asset to get a bigger result.",
    deprecating: "Phasing out, retiring, or discontinuing an old feature or service.",
    incumbent: "A mandatory obligation or required duty resting on you.",
    juncture: "This exact point in time.",
    synchronous: "Happening at the exact same time (like a live meeting instead of an email).",
    synchronization: "Making multiple databases match without losing any changes.",
    protocol: "A formal set of rules computers follow to talk to each other safely.",
    atomic: "All-or-nothing: either the entire save succeeds, or nothing changes at all.",
    concurrency: "Multiple people or servers trying to modify the exact same item at once.",
    optimistic: "Letting everyone edit without locking files, then gently checking before saving.",
    deadlocks: "Two tasks stuck waiting on each other, freezing the whole system.",
    exponential: "Waiting progressively longer between retry attempts (e.g. 1s, 2s, 4s, 8s).",
    verification: "Running a double-check to confirm no records were corrupted or missed.",
    asynchronous: "Tasks running in the background without freezing your screen.",
    reschedule: "Moving a booked appointment or meeting to a different date or time.",
    investors: "Venture capitalists or financial backers providing capital for equity.",
    appointment: "A scheduled clinical, medical, or professional consultation.",
    css: "Cascading Style Sheets: code controlling layout, fonts, and colors of web pages.",
    contract: "A legally binding agreement outlining duties, deliverables, and terms.",
    microservices: "Breaking a big software program into smaller, independent mini-programs.",
    laundry: "Washing, drying, folding, and putting away clothing garments.",
    groceries: "Purchasing essential food items and household goods.",
    dentist: "A healthcare appointment focused on oral health and hygiene."
  };

  function sanitize(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // --- 2. MULTINOMIAL NAIVE BAYES ML CLASSIFIER ---
  const NaiveBayesClassifier = {
    corpus: {
      work: [
        "meet with investors", "pitch valuation", "clean up css file", "database synchronization",
        "atomic verification", "deploy microservices", "code review", "pull request", "q3 revenue",
        "client call", "stakeholder sync", "sprint planning", "refactor code", "contract audit",
        "host team meeting", "organize department workshop", "plan annual conference", "lead company townhall"
      ],
      personal: [
        "doctors appointment", "dentist checkup", "pick up laundry", "wash the dishes",
        "buy groceries", "call mom and dad", "family dinner", "clean bedroom", "vet appointment",
        "pharmacy prescription", "gym workout", "renew passport", "oil change", "birthday party planning"
      ]
    },

    wordCounts: { work: {}, personal: {} },
    totalWords: { work: 0, personal: 0 },
    docCounts: { work: 0, personal: 0 },
    vocab: new Set(),

    init: function () {
      try {
        const savedTraining = localStorage.getItem('cognibridge_nb_weights_clean');
        if (savedTraining) {
          const parsed = JSON.parse(savedTraining);
          if (parsed && parsed.work) this.corpus.work = parsed.work;
          if (parsed && parsed.personal) this.corpus.personal = parsed.personal;
        }
      } catch (e) {}
      this.train();
    },

    tokenize: function (text) {
      if (!text) return [];
      return text.toLowerCase().match(/[a-z0-9]+/g) || [];
    },

    train: function () {
      this.wordCounts = { work: {}, personal: {} };
      this.totalWords = { work: 0, personal: 0 };
      this.docCounts = { work: this.corpus.work.length, personal: this.corpus.personal.length };
      this.vocab = new Set();

      const categories = ['work', 'personal'];
      for (let c = 0; c < categories.length; c++) {
        const cat = categories[c];
        const phrases = this.corpus[cat];
        for (let p = 0; p < phrases.length; p++) {
          const tokens = this.tokenize(phrases[p]);
          for (let t = 0; t < tokens.length; t++) {
            const w = tokens[t];
            this.wordCounts[cat][w] = (this.wordCounts[cat][w] || 0) + 1;
            this.totalWords[cat]++;
            this.vocab.add(w);
          }
        }
      }
    },

    predict: function (text) {
      const tokens = this.tokenize(text);
      if (tokens.length === 0) return { category: 'work', confidence: 50 };

      const totalDocs = this.docCounts.work + this.docCounts.personal;
      const alpha = 1.0;
      const vocabSize = this.vocab.size || 1;

      let logProbWork = Math.log((this.docCounts.work || 1) / totalDocs);
      let logProbPersonal = Math.log((this.docCounts.personal || 1) / totalDocs);

      for (let i = 0; i < tokens.length; i++) {
        const w = tokens[i];
        const countWork = (this.wordCounts.work[w] || 0) + alpha;
        const countPersonal = (this.wordCounts.personal[w] || 0) + alpha;

        logProbWork += Math.log(countWork / (this.totalWords.work + alpha * vocabSize));
        logProbPersonal += Math.log(countPersonal / (this.totalWords.personal + alpha * vocabSize));
      }

      const maxLog = Math.max(logProbWork, logProbPersonal);
      const expWork = Math.exp(logProbWork - maxLog);
      const expPersonal = Math.exp(logProbPersonal - maxLog);
      const totalExp = expWork + expPersonal;

      const probWork = Math.round((expWork / totalExp) * 100);
      const probPersonal = 100 - probWork;

      if (probWork >= probPersonal) {
        return { category: 'work', confidence: probWork };
      } else {
        return { category: 'personal', confidence: probPersonal };
      }
    },

    teach: function (phrase, correctedCategory) {
      if (this.corpus[correctedCategory]) {
        this.corpus[correctedCategory].push(phrase);
        this.train();
        try {
          localStorage.setItem('cognibridge_nb_weights_clean', JSON.stringify(this.corpus));
        } catch (e) {}
      }
    }
  };

  // --- 3. NLP & SEMANTIC PARSING ---
  function isPassiveInformational(text) {
    if (!text) return false;
    const lower = text.toLowerCase();
    const isMemo = /\b(pursuant|synergy|paradigm|apprised|stakeholder|restructuring|pivot|townhall|memorandum|newsletter)\b/i.test(text);
    const hasImmediateAction = /\b(call|fix|clean up|commit|deploy|buy|reschedule|host|organize|plan|submit by)\b/i.test(text);
    return isMemo && (!hasImmediateAction || lower.indexOf('no immediate deliverables') !== -1 || lower.indexOf('for your information') !== -1);
  }

  function deJargonizeEmail(text) {
    const lower = (text || '').toLowerCase();
    let tlDr = "This email provides a high-level organizational or departmental update. No immediate action or deadline is required of you.";

    if (lower.indexOf('restructuring') !== -1 || lower.indexOf('synergy') !== -1) {
      tlDr = "The company is reorganizing team structures and aligning goals. You do not need to deliver anything today; attend upcoming meetings for context.";
    } else if (lower.indexOf('policy') !== -1 || lower.indexOf('compliance') !== -1) {
      tlDr = "An internal policy has been updated. Review the documentation at your convenience to stay informed.";
    }

    const keyTakeaways = [];
    if (lower.indexOf('no immediate deliverables') !== -1 || lower.indexOf('not incumbent') !== -1) {
      keyTakeaways.push("✅ No work, code, or deliverables are due from you today.");
    } else {
      keyTakeaways.push("ℹ️ Informational memo: Read to remain aligned with leadership.");
    }

    if (lower.indexOf('townhall') !== -1 || lower.indexOf('sync') !== -1 || lower.indexOf('meeting') !== -1) {
      keyTakeaways.push("📅 Keep an eye out for a calendar invite for an upcoming team/townhall session.");
    }

    let count = 0;
    const keys = Object.keys(dynamicGlossary);
    for (let k = 0; k < keys.length; k++) {
      if (lower.indexOf(keys[k]) !== -1) count++;
    }
    keyTakeaways.push("🔍 We detected and indexed " + count + " corporate jargon terms in this message.");

    return {
      tlDr: tlDr,
      takeaways: keyTakeaways
    };
  }

  function parseTasksFromText(text, userSelectedDomain) {
    if (!text) return [];
    const lines = text.split(/[\r\n;]+|\.(?!\d)/g);
    const cleanLines = [];
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (trimmed.length > 3) cleanLines.push(trimmed);
    }

    const parsed = [];
    for (let i = 0; i < cleanLines.length; i++) {
      const line = cleanLines[i];
      const timeRegex = /\b(?:(?:by|at|on|for|before|after)\s+)?(?:tomorrow|today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)?(?:\s*(?:at|by)\s*)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)\b/i;
      const match = line.match(timeRegex);
      const statusBadge = match ? ("Target: " + match[0].trim()) : "Active Step";

      const prediction = NaiveBayesClassifier.predict(line);
      let finalDomain = prediction.category;
      let mlExplanation = "ML: " + prediction.category.toUpperCase() + " (" + prediction.confidence + "%)";

      if (userSelectedDomain === 'work' || userSelectedDomain === 'personal') {
        finalDomain = userSelectedDomain;
        mlExplanation = "User Overridden: " + userSelectedDomain.toUpperCase();
      }

      const steps = generateStepsForLine(line, finalDomain);
      const stepObjs = [];
      for (let s = 0; s < steps.length; s++) {
        stepObjs.push({ label: steps[s], done: false });
      }

      parsed.push({
        id: Date.now() + i,
        title: line.charAt(0).toUpperCase() + line.slice(1),
        domain: finalDomain,
        mlLabel: mlExplanation,
        statusBadge: statusBadge,
        collapsed: false,
        isEditingContainer: false,
        steps: stepObjs,
        done: false
      });
    }
    return parsed;
  }

  function generateStepsForLine(clause, domain) {
    const lower = (clause || '').toLowerCase();
    const steps = [];

    // Meeting Host
    if (lower.indexOf('host') !== -1 && (lower.indexOf('meeting') !== -1 || lower.indexOf('workshop') !== -1 || lower.indexOf('sync') !== -1 || lower.indexOf('team') !== -1)) {
      steps.push("Define meeting objective and draft a concise 3-bullet agenda");
      steps.push("Send calendar invite with agenda and video conference link at least 24h prior");
      steps.push("Prepare presentation slides or open shared workspace doc for collaborative notes");
      steps.push("Log on 5 minutes early to test microphone, camera, and screen sharing");
      steps.push("Kick off meeting by reviewing agenda and appointing a designated note-taker");
      steps.push("Guide discussion, maintain time discipline for each agenda item, and prevent tangents");
      steps.push("Conclude by summarizing key decisions and explicit action item owners");
      steps.push("Send follow-up recap email with action items and recording link within 2 hours");
      return steps;
    }

    // Big Event / Conference
    if (lower.indexOf('event') !== -1 || lower.indexOf('conference') !== -1 || lower.indexOf('summit') !== -1 || lower.indexOf('retreat') !== -1) {
      steps.push("Establish event scope: date, target attendance, total budget, and core theme");
      steps.push("Finalize venue (or virtual platform credentials) and review contract/AV capabilities");
      steps.push("Confirm speaker lineup, keynote topics, and run-of-show schedule");
      steps.push("Arrange catering, dietary accommodations, and attendee check-in badges");
      steps.push("Deploy registration page and send promotional invitations to attendee list");
      steps.push("Conduct a full technical rehearsal 48 hours prior to test microphones, lighting, and stage");
      steps.push("Assign staff/facilitator roles for day-of reception, timekeeping, and emergency support");
      steps.push("Execute day-of run-of-show checklist and manage schedule transitions");
      steps.push("Send thank-you notes, attendee feedback survey, and reconcile final vendor invoices");
      return steps;
    }

    // Medical / Doctor
    if (lower.indexOf('doctor') !== -1 || lower.indexOf('appointment') !== -1 || lower.indexOf('dentist') !== -1) {
      steps.push("Put photo ID, insurance card, and co-pay in wallet now");
      steps.push("Check commute transit time and set departure alarm 15 minutes early");
      steps.push("Write down any specific symptoms, medications, or questions you want to ask");
      steps.push("Arrive with 10 minutes to spare to check in at front desk");
      return steps;
    }

    // Chores
    if (lower.indexOf('laundry') !== -1 || lower.indexOf('dishes') !== -1 || lower.indexOf('clean') !== -1) {
      steps.push("Walk to the area and gather all items into one central staging spot");
      steps.push("Complete the active cleaning or washing cycle without stopping for distractions");
      steps.push("Put clean items away immediately so the living space remains uncluttered");
      return steps;
    }

    // Database Architecture
    if (lower.indexOf('database') !== -1 || lower.indexOf('atomic') !== -1 || lower.indexOf('lock') !== -1 || lower.indexOf('protocol') !== -1) {
      steps.push("Open project repository and locate database configuration module");
      steps.push("Check current lock-timeout settings across isolated cloud regions");
      steps.push("Replace pessimistic database locks with optimistic version check columns");
      steps.push("Implement exponential backoff retry logic with jitter buffer");
      steps.push("Run automated test suite simulating 500 simultaneous writes");
      steps.push("Verify deadlocks drop to zero during automated failover");
      return steps;
    }

    // Investor Pitch
    if (lower.indexOf('investor') !== -1 || lower.indexOf('pitch') !== -1 || lower.indexOf('meet with') !== -1) {
      steps.push("Review pitch deck slides and verify ARR and valuation ask");
      steps.push("Test microphone, camera, and screen sharing 15 minutes early");
      steps.push("Have running live prototype ready at localhost in your browser");
      steps.push("Join the video meeting room 2 minutes prior");
      return steps;
    }

    // Code Refactor
    if (lower.indexOf('css') !== -1 || lower.indexOf('code') !== -1 || lower.indexOf('refactor') !== -1) {
      steps.push("Open style.css in code editor");
      steps.push("Prune unused classes and organize colors into design variables");
      steps.push("Hard-refresh browser (Cmd+Shift+R) and test responsive breakpoints");
      return steps;
    }

    steps.push("Identify the main workspace or file needed for: " + clause.slice(0, 32) + "...");
    steps.push("Execute the primary milestone without context-switching");
    steps.push("Verify output meets standard and mark completed");
    return steps;
  }

  function computeMetrics(text) {
    if (!text) return { entropy: 0, friction: 0, tokens: [] };
    const tokens = text.match(/[A-Za-z0-9_-]+/g) || [];
    if (tokens.length === 0) return { entropy: 0, friction: 0, tokens: [] };

    const freq = {};
    for (let i = 0; i < tokens.length; i++) {
      const lower = tokens[i].toLowerCase();
      freq[lower] = (freq[lower] || 0) + 1;
    }

    let entropy = 0;
    const total = tokens.length;
    const words = Object.keys(freq);
    for (let i = 0; i < words.length; i++) {
      const p = freq[words[i]] / total;
      entropy -= p * Math.log2(p);
    }

    let complex = 0;
    for (let i = 0; i < tokens.length; i++) {
      const vowels = tokens[i].match(/[aeiouy]{1,2}/gi);
      if (vowels && vowels.length >= 3) complex++;
    }
    const polyRatio = (complex / total) * 100;
    const friction = Math.min(Math.round((entropy * 11) + (polyRatio * 0.75)), 100);

    return {
      entropy: parseFloat(entropy.toFixed(2)),
      friction: friction,
      tokens: tokens
    };
  }

  // --- 4. STATE STORE ---
  const Store = {
    dossiers: [],
    cabinetItems: [],
    accomplishments: [],
    dayMemories: {},

    load: function () {
      try {
        const storedD = localStorage.getItem('cognibridge_dossiers_v9');
        const storedC = localStorage.getItem('cognibridge_cabinet_v9');
        const storedA = localStorage.getItem('cognibridge_done_log_v9');
        const storedM = localStorage.getItem('cognibridge_day_memories_v9');

        this.dossiers = storedD ? JSON.parse(storedD) : [];
        this.cabinetItems = storedC ? JSON.parse(storedC) : [];
        this.accomplishments = storedA ? JSON.parse(storedA) : [];
        this.dayMemories = storedM ? JSON.parse(storedM) : {};

        for (let i = 0; i < this.dossiers.length; i++) {
          const d = this.dossiers[i];
          if (!Array.isArray(d.tasks)) d.tasks = [];
          for (let t = 0; t < d.tasks.length; t++) {
            if (!Array.isArray(d.tasks[t].steps)) d.tasks[t].steps = [];
          }
        }
      } catch (e) {
        this.dossiers = [];
        this.cabinetItems = [];
        this.accomplishments = [];
        this.dayMemories = {};
      }
    },

    save: function () {
      try {
        localStorage.setItem('cognibridge_dossiers_v9', JSON.stringify(this.dossiers));
        localStorage.setItem('cognibridge_cabinet_v9', JSON.stringify(this.cabinetItems));
        localStorage.setItem('cognibridge_done_log_v9', JSON.stringify(this.accomplishments));
        localStorage.setItem('cognibridge_day_memories_v9', JSON.stringify(this.dayMemories));
      } catch (e) {}
    },

    recordAccomplishment: function (title, domain) {
      const now = new Date();
      const dateKey = now.toISOString().split('T')[0];
      const timeStr = String(now.getHours()).padStart(2, '0') + ":" + String(now.getMinutes()).padStart(2, '0');

      this.accomplishments.unshift({
        id: Date.now() + Math.random(),
        title: title,
        domain: domain || 'work',
        dateKey: dateKey,
        timeStr: timeStr
      });
      this.save();
    },

    getDayMemory: function (dateKey) {
      if (!this.dayMemories[dateKey]) {
        this.dayMemories[dateKey] = { notes: '', artifacts: [] };
      }
      return this.dayMemories[dateKey];
    },

    saveDayNotes: function (dateKey, notes) {
      const mem = this.getDayMemory(dateKey);
      mem.notes = notes;
      this.save();
    },

    addDayArtifact: function (dateKey, artifact) {
      const mem = this.getDayMemory(dateKey);
      if (!Array.isArray(mem.artifacts)) mem.artifacts = [];
      mem.artifacts.unshift(artifact);
      this.save();
    },

    removeDayArtifact: function (dateKey, artifactId) {
      const mem = this.getDayMemory(dateKey);
      if (Array.isArray(mem.artifacts)) {
        mem.artifacts = mem.artifacts.filter(function (a) { return a.id !== artifactId; });
      }
      this.save();
    },

    clearAll: function () {
      this.dossiers = [];
      this.cabinetItems = [];
      this.accomplishments = [];
      this.dayMemories = {};
      this.save();
    }
  };

  // --- 5. REVERSE ACCOMPLISHMENT CALENDAR ---
  const CalendarModule = {
    currentDate: new Date(),
    selectedDateKey: new Date().toISOString().split('T')[0],

    init: function () {
      const prevBtn = document.getElementById('cal-prev-month');
      const nextBtn = document.getElementById('cal-next-month');
      if (prevBtn) {
        prevBtn.addEventListener('click', () => {
          this.currentDate.setMonth(this.currentDate.getMonth() - 1);
          this.render();
        });
      }
      if (nextBtn) {
        nextBtn.addEventListener('click', () => {
          this.currentDate.setMonth(this.currentDate.getMonth() + 1);
          this.render();
        });
      }

      // Close modal button
      const closeModalBtn = document.getElementById('btn-close-day-modal');
      const modalOverlay = document.getElementById('day-modal-overlay');
      if (closeModalBtn && modalOverlay) {
        closeModalBtn.addEventListener('click', () => modalOverlay.classList.remove('active'));
        modalOverlay.addEventListener('click', (e) => {
          if (e.target === modalOverlay) modalOverlay.classList.remove('active');
        });
      }

      // Save Notes inside Day Dossier Modal
      const saveNotesBtn = document.getElementById('btn-save-day-notes');
      if (saveNotesBtn) {
        saveNotesBtn.addEventListener('click', () => {
          const notesText = (document.getElementById('modal-day-notes') || {}).value || '';
          Store.saveDayNotes(this.selectedDateKey, notesText);
          logAudit("Saved memory notes for " + this.selectedDateKey + ".");
          alert("Day memory notes saved successfully!");
        });
      }

      // File Dropzone handler
      const dropzone = document.getElementById('modal-file-dropzone');
      const fileInput = document.getElementById('modal-file-input');
      if (dropzone && fileInput) {
        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
          const files = e.target.files;
          if (files && files[0]) this.handleFileUpload(files[0]);
        });

        dropzone.addEventListener('dragover', (e) => {
          e.preventDefault();
          dropzone.style.borderColor = 'var(--sage-primary)';
        });
        dropzone.addEventListener('dragleave', () => {
          dropzone.style.borderColor = 'var(--border-line)';
        });
        dropzone.addEventListener('drop', (e) => {
          e.preventDefault();
          dropzone.style.borderColor = 'var(--border-line)';
          if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
            this.handleFileUpload(e.dataTransfer.files[0]);
          }
        });
      }

      this.render();
    },

    handleFileUpload: function (file) {
      const self = this;
      const reader = new FileReader();
      reader.onload = function (e) {
        const artifact = {
          id: Date.now() + Math.random(),
          name: file.name,
          type: file.type,
          dataUrl: e.target.result
        };
        Store.addDayArtifact(self.selectedDateKey, artifact);
        self.renderModalArtifacts(self.selectedDateKey);
        self.render();
        logAudit("Archived artifact: " + file.name);
      };
      reader.readAsDataURL(file);
    },

    render: function () {
      const year = this.currentDate.getFullYear();
      const month = this.currentDate.getMonth();
      const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
      const label = document.getElementById('cal-month-year');
      if (label) label.textContent = monthNames[month] + " " + year;

      const grid = document.getElementById('calendar-days-grid');
      if (!grid) return;
      grid.innerHTML = "";

      const dayHeaders = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];
      for (let d = 0; d < dayHeaders.length; d++) {
        const th = document.createElement('div');
        th.className = 'cal-header-day';
        th.textContent = dayHeaders[d];
        grid.appendChild(th);
      }

      const firstDay = new Date(year, month, 1).getDay();
      const totalDays = new Date(year, month + 1, 0).getDate();
      const todayKey = new Date().toISOString().split('T')[0];

      const activeDates = new Set();
      for (let a = 0; a < (Store.accomplishments || []).length; a++) {
        activeDates.add(Store.accomplishments[a].dateKey);
      }
      const memoryKeys = Object.keys(Store.dayMemories || {});
      for (let m = 0; m < memoryKeys.length; m++) {
        const mem = Store.dayMemories[memoryKeys[m]];
        if (mem.artifacts && mem.artifacts.length > 0) {
          activeDates.add(memoryKeys[m]);
        }
      }

      for (let i = 0; i < firstDay; i++) {
        const blank = document.createElement('div');
        blank.className = 'cal-cell';
        blank.style.opacity = '0.2';
        grid.appendChild(blank);
      }

      const self = this;
      for (let day = 1; day <= totalDays; day++) {
        (function (dNum) {
          const cell = document.createElement('div');
          const dateKey = year + "-" + String(month + 1).padStart(2, '0') + "-" + String(dNum).padStart(2, '0');
          cell.className = 'cal-cell';
          cell.textContent = String(dNum);
          cell.setAttribute('role', 'button');
          cell.setAttribute('tabindex', '0');
          cell.setAttribute('title', "Open Day Page for " + dateKey);

          if (dateKey === todayKey) cell.classList.add('today');
          if (dateKey === self.selectedDateKey) cell.classList.add('selected');
          if (activeDates.has(dateKey)) cell.classList.add('has-activity');

          // THE CALENDAR DATE NUMBER IS THE BUTTON TO OPEN THE FULL DAY PAGE DIRECTLY
          cell.addEventListener('click', function () {
            self.selectedDateKey = dateKey;
            self.render();
            self.openDayDossierModal(dateKey);
          });

          cell.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              self.selectedDateKey = dateKey;
              self.render();
              self.openDayDossierModal(dateKey);
            }
          });

          grid.appendChild(cell);
        })(day);
      }

      this.renderDocket(this.selectedDateKey);
    },

    renderDocket: function (dateKey) {
      const items = (Store.accomplishments || []).filter(function (a) { return a.dateKey === dateKey; });
      const list = document.getElementById('accomplishment-items-list');
      if (!list) return;
      list.innerHTML = "";

      const selBadge = document.getElementById('cal-selected-label');
      const countBadge = document.getElementById('done-count-badge');
      if (selBadge) selBadge.textContent = dateKey;
      if (countBadge) countBadge.textContent = items.length + " Done";

      if (items.length === 0) {
        list.innerHTML = '<li style="color:var(--text-muted); font-size:0.78rem; list-style:none;">Click any calendar date number above to view accomplishments or open the Day Page.</li>';
        return;
      }

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const li = document.createElement('li');
        li.className = 'accomplishment-item';
        li.innerHTML = '<span><strong style="color:var(--sage-dark);">✓</strong> ' + sanitize(item.title) + ' <small style="color:var(--text-muted);">(' + sanitize(item.domain) + ')</small></span><span class="time-chip" style="font-size:0.68rem;">' + sanitize(item.timeStr) + '</span>';
        list.appendChild(li);
      }
    },

    openDayDossierModal: function (dateKey) {
      const modalOverlay = document.getElementById('day-modal-overlay');
      if (!modalOverlay) return;

      const titleEl = document.getElementById('modal-day-title');
      if (titleEl) titleEl.textContent = "Day Dossier: " + dateKey;
      const mem = Store.getDayMemory(dateKey);

      const notesInput = document.getElementById('modal-day-notes');
      if (notesInput) notesInput.value = mem.notes || '';

      const items = (Store.accomplishments || []).filter(function (a) { return a.dateKey === dateKey; });
      const list = document.getElementById('modal-accomplishment-list');
      const doneBadge = document.getElementById('modal-done-count');
      if (doneBadge) doneBadge.textContent = items.length + " Completed";
      if (list) {
        list.innerHTML = "";
        if (items.length === 0) {
          list.innerHTML = '<li style="color:var(--text-muted); font-size:0.80rem;">No tasks checked off on this date yet.</li>';
        } else {
          for (let i = 0; i < items.length; i++) {
            const item = items[i];
            const li = document.createElement('li');
            li.style.fontSize = '0.82rem';
            li.style.borderBottom = '1px dotted var(--border-line)';
            li.style.paddingBottom = '0.25rem';
            li.innerHTML = '<strong>✓</strong> ' + sanitize(item.title) + ' <small style="color:var(--text-muted);">(' + sanitize(item.domain) + ' • ' + sanitize(item.timeStr) + ')</small>';
            list.appendChild(li);
          }
        }
      }

      this.renderModalArtifacts(dateKey);
      modalOverlay.classList.add('active');
      logAudit("Opened Day Dossier Page for " + dateKey);
    },

    renderModalArtifacts: function (dateKey) {
      const mem = Store.getDayMemory(dateKey);
      const grid = document.getElementById('modal-gallery-grid');
      const countEl = document.getElementById('modal-artifact-count');
      if (!grid) return;
      grid.innerHTML = "";

      const artifacts = mem.artifacts || [];
      if (countEl) countEl.textContent = artifacts.length + " Saved";

      if (artifacts.length === 0) {
        grid.innerHTML = '<div style="grid-column:1/-1; color:var(--text-muted); font-size:0.75rem; text-align:center; padding:1rem;">No scanned cards, photos, or documents saved for this date yet.</div>';
        return;
      }

      for (let i = 0; i < artifacts.length; i++) {
        (function (art) {
          const item = document.createElement('div');
          item.className = 'gallery-item';

          if (art.type && art.type.indexOf('image/') !== -1) {
            const img = document.createElement('img');
            img.src = art.dataUrl;
            img.alt = art.name;
            img.title = "Click to view full image";
            img.addEventListener('click', function () {
              const w = window.open('');
              if (w) {
                const fullImg = w.document.createElement('img');
                fullImg.src = art.dataUrl;
                fullImg.style.maxWidth = '100%';
                w.document.body.appendChild(fullImg);
              }
            });
            item.appendChild(img);
          } else {
            const docFallback = document.createElement('div');
            docFallback.className = 'doc-fallback';
            docFallback.innerHTML = '📄<br><strong>' + sanitize(art.name) + '</strong>';
            docFallback.addEventListener('click', function () {
              const w = window.open('');
              if (w) {
                const frame = w.document.createElement('iframe');
                frame.src = art.dataUrl;
                frame.style.width = '100%';
                frame.style.height = '100%';
                frame.style.border = 'none';
                w.document.body.appendChild(frame);
              }
            });
            item.appendChild(docFallback);
          }

          const delBtn = document.createElement('button');
          delBtn.className = 'btn-del-photo';
          delBtn.textContent = '✕';
          delBtn.title = 'Delete from date';
          delBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            if (confirm('Remove this document/card from this date?')) {
              Store.removeDayArtifact(dateKey, art.id);
              CalendarModule.renderModalArtifacts(dateKey);
              CalendarModule.render();
              logAudit("Removed artifact from " + dateKey);
            }
          });
          item.appendChild(delBtn);

          grid.appendChild(item);
        })(artifacts[i]);
      }
    }
  };

  // --- 6. FILING CABINET MODULE (READ-ONLY ARCHIVE: NO REPLY DRAFT) ---
  const FilingCabinet = {
    render: function () {
      const container = document.getElementById('cabinet-list-container');
      if (!container) return;

      const searchInput = document.getElementById('cabinet-search');
      const query = (searchInput ? searchInput.value : '').toLowerCase().trim();
      container.innerHTML = "";

      const items = (Store.cabinetItems || []).filter(function (item) {
        if (!query) return true;
        const titleMatch = (item.title || '').toLowerCase().indexOf(query) !== -1;
        const textMatch = (item.tlDr || '').toLowerCase().indexOf(query) !== -1;
        return titleMatch || textMatch;
      });

      const badge = document.getElementById('cabinet-badge-count');
      if (badge) badge.textContent = (Store.cabinetItems || []).length;

      if (items.length === 0) {
        container.innerHTML = '<div style="text-align:center; padding:2rem; color:var(--text-muted); font-size:0.82rem;">Filing Cabinet empty. Ingest corporate memos to file them away here.</div>';
        return;
      }

      for (let i = 0; i < items.length; i++) {
        (function (idx) {
          const item = items[idx];
          const card = document.createElement('div');
          card.className = 'task-card';
          card.style.borderLeft = "4px solid var(--slate-blue)";

          let takeawaysHtml = '';
          const list = item.takeaways || [];
          for (let t = 0; t < list.length; t++) {
            takeawaysHtml += '<li style="font-size:0.8rem; margin-bottom:0.25rem;">' + sanitize(list[t]) + '</li>';
          }

          // CLEAN CARD: ZERO EMAIL REPLY LINKS OR DRAFT BUTTONS
          card.innerHTML = 
            '<div class="task-card-header">' +
              '<div>' +
                '<strong style="font-size:0.95rem; color:var(--slate-blue);">' + sanitize(item.title) + '</strong>' +
                '<small style="display:block; color:var(--text-muted); font-size:0.72rem;">Filed on ' + sanitize(item.timestamp) + '</small>' +
              '</div>' +
              '<button class="btn-clear" style="color:var(--terracotta); font-size:0.74rem;" data-idx="' + idx + '">Archive / Delete</button>' +
            '</div>' +
            '<p style="font-size:0.85rem; color:var(--text-main); margin:0.35rem 0; line-height:1.4;">' + sanitize(item.tlDr) + '</p>' +
            '<div style="background:var(--bg-subtle); padding:0.6rem; border-radius:8px; border:1px solid var(--border-line);">' +
              '<strong style="font-size:0.74rem; color:var(--text-muted); display:block; margin-bottom:0.25rem;">Plain-English Takeaways:</strong>' +
              '<ul style="padding-left:1rem;">' + takeawaysHtml + '</ul>' +
            '</div>';

          const delBtn = card.querySelector('[data-idx]');
          if (delBtn) {
            delBtn.addEventListener('click', function () {
              Store.cabinetItems.splice(idx, 1);
              Store.save();
              FilingCabinet.render();
              logAudit("Document removed from Filing Cabinet.");
            });
          }

          container.appendChild(card);
        })(i);
      }
    }
  };

  // --- 7. MAIN CONTROLLER ---
  let currentIngestDomainMode = 'auto';
  let taskFilter = 'all';
  let audioCtx = null;
  let allCollapsed = false;

  function logAudit(msg) {
    const log = document.getElementById('audit-log');
    if (!log) return;
    const now = new Date();
    const t = String(now.getHours()).padStart(2, '0') + ":" + String(now.getMinutes()).padStart(2, '0') + ":" + String(now.getSeconds()).padStart(2, '0');
    const div = document.createElement('div');
    div.className = 'ledger-row';
    div.textContent = "[" + t + "] " + msg;
    log.prepend(div);
  }

  function renderTasks() {
    const container = document.getElementById('dossier-feed-container');
    if (!container) return;
    container.innerHTML = "";

    const dossiers = Store.dossiers || [];
    for (let d = 0; d < dossiers.length; d++) {
      (function (dossier) {
        const visibleTasks = (dossier.tasks || []).filter(function (t) {
          if (taskFilter === 'all') return true;
          return t.domain === taskFilter;
        });

        if (visibleTasks.length === 0 && taskFilter !== 'all') return;

        const card = document.createElement('div');
        card.className = 'task-card';
        card.style.borderLeft = "4px solid var(--sage-primary)";

        const header = document.createElement('div');
        header.className = 'task-card-header';
        header.innerHTML = 
          '<div style="cursor:pointer;" data-action="toggle-dossier">' +
            '<strong style="font-size:0.95rem;">' + (dossier.collapsed ? '▶' : '▼') + ' ' + sanitize(dossier.title) + '</strong>' +
            '<small style="color:var(--text-muted); font-size:0.72rem; display:block;">' + sanitize(dossier.timestamp) + ' • ' + visibleTasks.length + ' visible tasks</small>' +
          '</div>' +
          '<div style="display:flex; align-items:center; gap:0.5rem;">' +
            '<button class="btn-clear" style="color:var(--terracotta); font-size:0.74rem;" data-action="delete-dossier">Remove</button>' +
          '</div>';

        const toggleDossierBtn = header.querySelector('[data-action="toggle-dossier"]');
        if (toggleDossierBtn) {
          toggleDossierBtn.addEventListener('click', function () {
            dossier.collapsed = !dossier.collapsed;
            Store.save();
            renderTasks();
          });
        }

        const delDossierBtn = header.querySelector('[data-action="delete-dossier"]');
        if (delDossierBtn) {
          delDossierBtn.addEventListener('click', function () {
            if (confirm("Remove this entire task group?")) {
              Store.dossiers = Store.dossiers.filter(function (item) { return item.id !== dossier.id; });
              Store.save();
              renderTasks();
              updateBodyDouble();
              updateMomentum();
            }
          });
        }

        card.appendChild(header);

        if (!dossier.collapsed) {
          const body = document.createElement('div');
          body.style.marginTop = '0.4rem';
          body.style.borderTop = '1px solid var(--border-line)';
          body.style.paddingTop = '0.4rem';

          for (let t = 0; t < visibleTasks.length; t++) {
            (function (task) {
              const wrap = document.createElement('div');
              wrap.style.background = 'var(--bg-surface)';
              wrap.style.border = '1px solid var(--border-line)';
              wrap.style.borderRadius = '8px';
              wrap.style.padding = '0.75rem';
              wrap.style.marginTop = '0.5rem';

              const isWork = task.domain === 'work';
              const domainBadgeHtml = 
                '<button class="btn-clear" style="font-size:0.70rem; padding:2px 7px; border-radius:4px; font-weight:700; cursor:pointer; border:1px solid ' + (isWork ? 'var(--sage-primary)' : 'var(--slate-blue)') + '; background:' + (isWork ? 'var(--sage-soft)' : 'var(--slate-soft)') + '; color:' + (isWork ? 'var(--sage-dark)' : 'var(--slate-blue)') + ';" data-action="toggle-task-domain" title="Click to switch domain (Trains Naive Bayes ML)">' +
                  (isWork ? '💼 WORK' : '🏠 PERSONAL') + ' (Click to switch)' +
                '</button>';

              // INLINE EDIT FORM
              if (task.isEditingContainer) {
                const stepsLines = [];
                for (let s = 0; s < (task.steps || []).length; s++) {
                  stepsLines.push(task.steps[s].label);
                }

                wrap.innerHTML = 
                  '<div style="display:flex; flex-direction:column; gap:0.5rem;">' +
                    '<label style="font-size:0.75rem; font-weight:700; color:var(--sage-dark);">Edit Task Title:</label>' +
                    '<input type="text" value="' + sanitize(task.title) + '" data-edit-task-title style="padding:0.35rem 0.5rem; font-size:0.85rem; font-family:inherit;">' +
                    '<label style="font-size:0.75rem; font-weight:700; color:var(--sage-dark);">Edit Micro-Steps (one per line):</label>' +
                    '<textarea rows="5" data-edit-task-steps style="padding:0.35rem 0.5rem; font-size:0.80rem; font-family:inherit; line-height:1.4;">' + sanitize(stepsLines.join('\n')) + '</textarea>' +
                    '<div style="display:flex; justify-content:flex-end; gap:0.4rem; margin-top:0.25rem;">' +
                      '<button class="btn btn-sage btn-sm" data-action="save-container-edit">Save</button>' +
                      '<button class="btn btn-outline btn-sm" data-action="cancel-container-edit">Cancel</button>' +
                    '</div>' +
                  '</div>';

                const saveBtn = wrap.querySelector('[data-action="save-container-edit"]');
                if (saveBtn) {
                  saveBtn.addEventListener('click', function () {
                    const titleInput = wrap.querySelector('[data-edit-task-title]');
                    const stepsInput = wrap.querySelector('[data-edit-task-steps]');
                    const newTitle = titleInput ? titleInput.value.trim() : '';
                    const newStepsRaw = stepsInput ? stepsInput.value.trim() : '';

                    if (newTitle) task.title = newTitle;
                    if (newStepsRaw) {
                      const stepLines = newStepsRaw.split('\n');
                      task.steps = [];
                      for (let sl = 0; sl < stepLines.length; sl++) {
                        const lineTrim = stepLines[sl].trim();
                        if (lineTrim.length > 0) task.steps.push({ label: lineTrim, done: false });
                      }
                    }
                    task.isEditingContainer = false;
                    Store.save();
                    renderTasks();
                    updateBodyDouble();
                    updateMomentum();
                    logAudit("Updated task: " + task.title);
                  });
                }

                const cancelBtn = wrap.querySelector('[data-action="cancel-container-edit"]');
                if (cancelBtn) {
                  cancelBtn.addEventListener('click', function () {
                    task.isEditingContainer = false;
                    renderTasks();
                  });
                }

                body.appendChild(wrap);
                return;
              }

              // NORMAL TASK CARD VIEW
              const taskHeader = 
                '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem;">' +
                  '<div style="display:flex; align-items:center; gap:0.35rem; flex:1; min-width:240px;">' +
                    '<button class="btn-clear" style="font-size:0.85rem; font-weight:bold; color:var(--sage-primary); padding:0 0.35rem 0 0.1rem;" data-action="toggle-arrow">' +
                      (task.collapsed ? '▶' : '▼') +
                    '</button>' +
                    '<label class="subtask-row" style="flex:1;">' +
                      '<input type="checkbox" ' + (task.done ? 'checked' : '') + ' data-action="check-task">' +
                      '<strong style="' + (task.done ? 'text-decoration:line-through; opacity:0.5;' : '') + '">' + sanitize(task.title) + '</strong>' +
                    '</label>' +
                    '<button class="btn-clear" style="font-size:0.85rem; color:var(--slate-blue); padding:0 4px; cursor:pointer;" data-action="open-container-edit" title="Edit">✏️</button>' +
                  '</div>' +
                  '<div style="display:flex; align-items:center; gap:0.4rem;">' +
                    domainBadgeHtml +
                    '<span class="time-chip" style="font-size:0.7rem;">' + sanitize(task.statusBadge) + '</span>' +
                  '</div>' +
                '</div>';

              let stepsHtml = "";
              if (task.collapsed) {
                let doneSteps = 0;
                for (let s = 0; s < (task.steps || []).length; s++) {
                  if (task.steps[s].done) doneSteps++;
                }
                stepsHtml = 
                  '<div style="padding:0.35rem 0.6rem; margin-top:0.35rem; background:var(--bg-subtle); border-radius:6px; font-size:0.74rem; color:var(--text-muted); display:flex; justify-content:space-between; align-items:center; cursor:pointer;" data-action="toggle-arrow">' +
                    '<span>' + doneSteps + '/' + (task.steps || []).length + ' steps complete • <small style="color:var(--sage-primary);">' + sanitize(task.mlLabel || '') + '</small></span>' +
                    '<span style="color:var(--sage-primary); font-weight:600;">▶</span>' +
                  '</div>';
              } else {
                let stepRows = '';
                for (let s = 0; s < (task.steps || []).length; s++) {
                  const st = task.steps[s];
                  stepRows += 
                    '<div style="display:flex; justify-content:space-between; align-items:center; font-size:0.80rem; gap:0.4rem; margin-bottom:0.3rem;">' +
                      '<label style="display:flex; align-items:center; gap:0.45rem; flex:1; cursor:pointer;">' +
                        '<input type="checkbox" ' + (st.done ? 'checked' : '') + ' style="accent-color:var(--sage-primary);" data-action="check-step" data-sidx="' + s + '">' +
                        '<span style="' + (st.done ? 'text-decoration:line-through; opacity:0.55;' : '') + '">' + sanitize(st.label) + '</span>' +
                      '</label>' +
                      '<button class="btn-clear" style="color:var(--text-muted); font-size:0.75rem; padding:0 3px;" data-action="delete-step" data-sidx="' + s + '" title="Delete step">×</button>' +
                    '</div>';
                }

                stepsHtml = 
                  '<div class="subtask-box" style="margin-top:0.4rem; padding-left:0.5rem; border-left:2px solid var(--border-line);">' +
                    '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">' +
                      '<small style="color:var(--text-muted); font-size:0.7rem;">' + (task.steps || []).length + ' steps • <span style="color:var(--sage-primary);">' + sanitize(task.mlLabel || '') + '</span></small>' +
                      '<button class="btn-clear" style="font-size:0.7rem; color:var(--sage-primary);" data-action="add-step">+ Step</button>' +
                    '</div>' +
                    stepRows +
                  '</div>';
              }

              wrap.innerHTML = taskHeader + stepsHtml;

              const editBtn = wrap.querySelector('[data-action="open-container-edit"]');
              if (editBtn) {
                editBtn.addEventListener('click', function (e) {
                  e.stopPropagation();
                  task.isEditingContainer = true;
                  renderTasks();
                });
              }

              const domainBtn = wrap.querySelector('[data-action="toggle-task-domain"]');
              if (domainBtn) {
                domainBtn.addEventListener('click', function () {
                  const newDomain = task.domain === 'work' ? 'personal' : 'work';
                  task.domain = newDomain;
                  task.mlLabel = "User Corrected: " + newDomain.toUpperCase();
                  NaiveBayesClassifier.teach(task.title, newDomain);
                  Store.save();
                  renderTasks();
                  logAudit("User corrected task to " + newDomain.toUpperCase());
                });
              }

              const toggleArrowBtn = wrap.querySelector('[data-action="toggle-arrow"]');
              if (toggleArrowBtn) {
                toggleArrowBtn.addEventListener('click', function () {
                  task.collapsed = !task.collapsed;
                  Store.save();
                  renderTasks();
                });
              }

              const checkTaskInput = wrap.querySelector('[data-action="check-task"]');
              if (checkTaskInput) {
                checkTaskInput.addEventListener('change', function (e) {
                  const isDone = e.target.checked;
                  task.done = isDone;
                  for (let s = 0; s < (task.steps || []).length; s++) {
                    task.steps[s].done = isDone;
                  }
                  if (isDone) {
                    Store.recordAccomplishment(task.title, task.domain);
                    CalendarModule.render();
                    logAudit("Completed task: " + task.title);
                  }
                  Store.save();
                  renderTasks();
                  updateBodyDouble();
                  updateMomentum();
                });
              }

              const checkStepInputs = wrap.querySelectorAll('[data-action="check-step"]');
              for (let cs = 0; cs < checkStepInputs.length; cs++) {
                checkStepInputs[cs].addEventListener('change', function (e) {
                  const sIdx = parseInt(e.target.dataset.sidx, 10);
                  if (task.steps && task.steps[sIdx]) {
                    task.steps[sIdx].done = e.target.checked;
                    Store.save();
                    updateBodyDouble();
                    updateMomentum();
                  }
                });
              }

              const delStepBtns = wrap.querySelectorAll('[data-action="delete-step"]');
              for (let ds = 0; ds < delStepBtns.length; ds++) {
                delStepBtns[ds].addEventListener('click', function (e) {
                  const sIdx = parseInt(e.target.dataset.sidx, 10);
                  if (task.steps && task.steps[sIdx]) {
                    task.steps.splice(sIdx, 1);
                    Store.save();
                    renderTasks();
                    updateMomentum();
                  }
                });
              }

              const addBtn = wrap.querySelector('[data-action="add-step"]');
              if (addBtn) {
                addBtn.addEventListener('click', function () {
                  const val = prompt("Add micro-step:");
                  if (val && val.trim().length > 0) {
                    task.steps.push({ label: val.trim(), done: false });
                    Store.save();
                    renderTasks();
                    updateMomentum();
                  }
                });
              }

              body.appendChild(wrap);
            })(visibleTasks[t]);
          }

          card.appendChild(body);
        }

        container.appendChild(card);
      })(dossiers[d]);
    }
  }

  function updateBodyDouble() {
    let firstPending = null;
    const dossiers = Store.dossiers || [];
    for (let d = 0; d < dossiers.length; d++) {
      const tasks = dossiers[d].tasks || [];
      for (let t = 0; t < tasks.length; t++) {
        const task = tasks[t];
        if (!task.done) {
          const steps = task.steps || [];
          for (let s = 0; s < steps.length; s++) {
            if (!steps[s].done) {
              firstPending = { title: task.title, step: steps[s].label, badge: task.statusBadge };
              break;
            }
          }
        }
        if (firstPending) break;
      }
      if (firstPending) break;
    }

    const textEl = document.getElementById('bd-task-text');
    const badgeEl = document.getElementById('bd-status-tag');

    if (firstPending) {
      if (textEl) textEl.innerHTML = '<strong>' + sanitize(firstPending.title) + '</strong><div style="font-size:0.8rem; color:var(--text-muted); margin-top:0.2rem;">Next: ' + sanitize(firstPending.step) + '</div>';
      if (badgeEl) badgeEl.textContent = firstPending.badge;
    } else {
      if (textEl) textEl.textContent = "All pending tasks completed! Great work.";
      if (badgeEl) badgeEl.textContent = "All Done";
    }
  }

  function updateMomentum() {
    let total = 0;
    let completed = 0;
    const dossiers = Store.dossiers || [];
    for (let d = 0; d < dossiers.length; d++) {
      const tasks = dossiers[d].tasks || [];
      for (let t = 0; t < tasks.length; t++) {
        const steps = tasks[t].steps || [];
        for (let s = 0; s < steps.length; s++) {
          total++;
          if (steps[s].done) completed++;
        }
      }
    }

    const bar = document.getElementById('momentum-bar');
    const label = document.getElementById('momentum-label');

    if (total === 0) {
      if (bar) bar.style.width = '0%';
      if (label) label.textContent = 'No tasks active';
      return;
    }

    const pct = Math.round((completed / total) * 100);
    if (bar) bar.style.width = pct + '%';
    if (label) label.textContent = pct + "% Complete (" + completed + "/" + total + " steps)";
  }

  function renderAttentionMap(tokens) {
    const container = document.getElementById('attention-map');
    if (!container) return;
    container.innerHTML = "";

    for (let i = 0; i < tokens.length; i++) {
      (function (tok) {
        const clean = tok.toLowerCase();
        const span = document.createElement('span');
        const vowels = tok.match(/[aeiouy]{1,2}/gi);
        const syllables = vowels ? vowels.length : 1;

        if (dynamicGlossary[clean]) {
          span.className = "token-high";
          span.addEventListener('click', function () {
            const jargonEl = document.getElementById('jargon-text');
            if (jargonEl) jargonEl.innerHTML = '<strong>' + sanitize(tok) + ':</strong> ' + sanitize(dynamicGlossary[clean]);
          });
        } else if (syllables >= 4) {
          span.className = "token-med";
          span.addEventListener('click', function () {
            const jargonEl = document.getElementById('jargon-text');
            if (jargonEl) jargonEl.innerHTML = '<strong>' + sanitize(tok) + ':</strong> Dense polysyllabic term.';
          });
        }
        span.textContent = tok + " ";
        container.appendChild(span);
      })(tokens[i]);
    }
  }

  function handleIngest() {
    const inputEl = document.getElementById('raw-input');
    const raw = inputEl ? inputEl.value.trim() : '';
    if (!raw) return;

    logAudit("Analyzing incoming text with Naive Bayes...");
    const metrics = computeMetrics(raw);
    const fricEl = document.getElementById('val-friction');
    if (fricEl) fricEl.textContent = metrics.friction + " / 100";
    renderAttentionMap(metrics.tokens);

    const intentBadge = document.getElementById('detected-intent-badge');

    if (isPassiveInformational(raw)) {
      const summary = deJargonizeEmail(raw);
      const title = raw.split("\n")[0].slice(0, 40).replace(/^[#\s\-–]+/, '') || "Informational Memo";
      const now = new Date();
      const timeStr = String(now.getHours()).padStart(2, '0') + ":" + String(now.getMinutes()).padStart(2, '0');

      Store.cabinetItems.unshift({
        id: Date.now(),
        title: title,
        timestamp: "Today at " + timeStr,
        tlDr: summary.tlDr,
        takeaways: summary.takeaways
      });
      Store.save();
      FilingCabinet.render();
      if (intentBadge) intentBadge.textContent = "Archived to Filing Cabinet";
      logAudit("Document filed in Filing Cabinet drawer.");
      alert("This document is informational with no pending tasks. It has been filed cleanly in your Filing Cabinet drawer!");
    } else {
      const tasks = parseTasksFromText(raw, currentIngestDomainMode);
      const title = raw.split("\n")[0].slice(0, 40).replace(/^[#\s\-–]+/, '') || "Action Deliverable";
      const now = new Date();
      const timeStr = String(now.getHours()).padStart(2, '0') + ":" + String(now.getMinutes()).padStart(2, '0');

      Store.dossiers.unshift({
        id: Date.now(),
        title: title,
        timestamp: "Today at " + timeStr,
        collapsed: false,
        tasks: tasks
      });
      Store.save();
      renderTasks();
      updateBodyDouble();
      updateMomentum();
      if (intentBadge) intentBadge.textContent = "Added to Active Tasks";
      logAudit("Decompiled " + tasks.length + " actionable items into feed.");
    }

    if (inputEl) inputEl.value = "";
  }

  function handleAudio(type) {
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    if (type === 'off') {
      logAudit("Focus soundscape stopped.");
      return;
    }

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const bufferSize = audioCtx.sampleRate * 2;
    const noiseBuffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
    const output = noiseBuffer.getChannelData(0);

    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      output[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.04;
      b6 = white * 0.115926;
    }

    const whiteNoise = audioCtx.createBufferSource();
    whiteNoise.buffer = noiseBuffer;
    whiteNoise.loop = true;
    whiteNoise.connect(audioCtx.destination);
    whiteNoise.start(0);
    logAudit("Focus soundscape engaged: " + type.toUpperCase());
  }

  // --- 8. DOM BINDINGS & INIT ---
  window.addEventListener('DOMContentLoaded', function () {
    NaiveBayesClassifier.init();
    Store.load();

    if (Store.dossiers.length === 0 && Store.cabinetItems.length === 0) {
      const sampleTasks = parseTasksFromText("Tomorrow at 3pm meet with investors. Clean up the css file by 12 pm Thursday. Doctors appointment at 8am Friday.", "auto");
      Store.dossiers.push({
        id: Date.now() - 50000,
        title: "Executive Schedule & Code Cleanup",
        timestamp: "Earlier Today",
        collapsed: false,
        tasks: sampleTasks
      });

      const emailSummary = deJargonizeEmail("Dear Team, please be apprised that pursuant to our Q4 corporate restructuring and synergy initiatives, our enterprise operational paradigms will transition toward decentralized workflows. No immediate deliverables are incumbent upon contributors.");
      Store.cabinetItems.push({
        id: Date.now() - 20000,
        title: "Q4 Corporate Restructuring Memo",
        timestamp: "Earlier Today",
        tlDr: emailSummary.tlDr,
        takeaways: emailSummary.takeaways
      });

      Store.recordAccomplishment("Finalize Q3 revenue report", "work");
      Store.save();
    }

    CalendarModule.init();
    FilingCabinet.render();
    renderTasks();
    updateBodyDouble();
    updateMomentum();

    const domAuto = document.getElementById('domain-mode-auto');
    const domWork = document.getElementById('domain-mode-work');
    const domPersonal = document.getElementById('domain-mode-personal');

    if (domAuto) {
      domAuto.addEventListener('click', function () {
        currentIngestDomainMode = 'auto';
        domAuto.classList.add('active');
        if (domWork) domWork.classList.remove('active');
        if (domPersonal) domPersonal.classList.remove('active');
        logAudit("Domain Authority set to: [ML Suggest]");
      });
    }

    if (domWork) {
      domWork.addEventListener('click', function () {
        currentIngestDomainMode = 'work';
        domWork.classList.add('active');
        if (domAuto) domAuto.classList.remove('active');
        if (domPersonal) domPersonal.classList.remove('active');
        logAudit("Domain Authority set to: [Force Work]");
      });
    }

    if (domPersonal) {
      domPersonal.addEventListener('click', function () {
        currentIngestDomainMode = 'personal';
        domPersonal.classList.add('active');
        if (domAuto) domAuto.classList.remove('active');
        if (domWork) domWork.classList.remove('active');
        logAudit("Domain Authority set to: [Force Personal]");
      });
    }

    const ingestBtn = document.getElementById('btn-ingest-doc');
    if (ingestBtn) ingestBtn.addEventListener('click', handleIngest);

    const clearInputBtn = document.getElementById('btn-clear-input');
    if (clearInputBtn) {
      clearInputBtn.addEventListener('click', function () {
        const inp = document.getElementById('raw-input');
        if (inp) inp.value = '';
      });
    }

    const sampleTechBtn = document.getElementById('btn-sample-tech');
    if (sampleTechBtn) {
      sampleTechBtn.addEventListener('click', function () {
        const inp = document.getElementById('raw-input');
        if (inp) inp.value = "Host the Q4 department strategy workshop on Friday at 10am. Plan the annual regional developer conference in November.";
      });
    }

    const sampleEmailBtn = document.getElementById('btn-sample-email');
    if (sampleEmailBtn) {
      sampleEmailBtn.addEventListener('click', function () {
        const inp = document.getElementById('raw-input');
        if (inp) inp.value = "Dear Team,\n\nPlease be apprised that pursuant to our Q4 corporate restructuring and synergy initiatives, our enterprise operational paradigms will transition toward cross-functional decentralized agile workflows. This pivot necessitates holistic alignment across stakeholder cohorts to maximize ROI and leverage low-hanging fruit while deprecating legacy non-core product surface areas.\n\nNo immediate deliverables are incumbent upon individual contributors at this juncture; however, strict compliance with upcoming synchronous stakeholder townhalls is strongly encouraged.\n\nBest regards,\nCorporate Strategy & Operational Excellence Division";
      });
    }

    const clearWorkspaceBtn = document.getElementById('btn-clear-workspace');
    if (clearWorkspaceBtn) {
      clearWorkspaceBtn.addEventListener('click', function () {
        if (confirm("Clear all active dossiers, filing cabinet emails, calendar accomplishments, and saved memories?")) {
          Store.clearAll();
          renderTasks();
          FilingCabinet.render();
          CalendarModule.render();
          updateBodyDouble();
          updateMomentum();
          logAudit("Entire workspace reset.");
        }
      });
    }

    const clearLogBtn = document.getElementById('btn-clear-log');
    if (clearLogBtn) {
      clearLogBtn.addEventListener('click', function () {
        const auditLog = document.getElementById('audit-log');
        if (auditLog) auditLog.innerHTML = '<div class="ledger-row">[Ledger cleared].</div>';
      });
    }

    const tabTasks = document.getElementById('tab-view-tasks');
    const tabCabinet = document.getElementById('tab-view-cabinet');
    const tasksView = document.getElementById('view-tasks-container');
    const cabinetView = document.getElementById('view-cabinet-container');

    if (tabTasks && tabCabinet && tasksView && cabinetView) {
      tabTasks.addEventListener('click', function () {
        tabTasks.classList.add('active');
        tabCabinet.classList.remove('active');
        tasksView.style.display = 'flex';
        cabinetView.style.display = 'none';
      });

      tabCabinet.addEventListener('click', function () {
        tabCabinet.classList.add('active');
        tabTasks.classList.remove('active');
        cabinetView.style.display = 'flex';
        tasksView.style.display = 'none';
        FilingCabinet.render();
      });
    }

    const filterAllBtn = document.getElementById('filter-task-all');
    const filterWorkBtn = document.getElementById('filter-task-work');
    const filterPersonalBtn = document.getElementById('filter-task-personal');

    if (filterAllBtn) {
      filterAllBtn.addEventListener('click', function () {
        taskFilter = 'all';
        filterAllBtn.classList.add('active');
        if (filterWorkBtn) filterWorkBtn.classList.remove('active');
        if (filterPersonalBtn) filterPersonalBtn.classList.remove('active');
        renderTasks();
      });
    }

    if (filterWorkBtn) {
      filterWorkBtn.addEventListener('click', function () {
        taskFilter = 'work';
        filterWorkBtn.classList.add('active');
        if (filterAllBtn) filterAllBtn.classList.remove('active');
        if (filterPersonalBtn) filterPersonalBtn.classList.remove('active');
        renderTasks();
      });
    }

    if (filterPersonalBtn) {
      filterPersonalBtn.addEventListener('click', function () {
        taskFilter = 'personal';
        filterPersonalBtn.classList.add('active');
        if (filterAllBtn) filterAllBtn.classList.remove('active');
        if (filterWorkBtn) filterWorkBtn.classList.remove('active');
        renderTasks();
      });
    }

    const collapseAllBtn = document.getElementById('btn-collapse-all-dossiers');
    if (collapseAllBtn) {
      collapseAllBtn.addEventListener('click', function () {
        allCollapsed = !allCollapsed;
        const dossiers = Store.dossiers || [];
        for (let d = 0; d < dossiers.length; d++) {
          dossiers[d].collapsed = allCollapsed;
          const tasks = dossiers[d].tasks || [];
          for (let t = 0; t < tasks.length; t++) {
            tasks[t].collapsed = allCollapsed;
          }
        }
        Store.save();
        renderTasks();
      });
    }

    const cabSearch = document.getElementById('cabinet-search');
    if (cabSearch) {
      cabSearch.addEventListener('input', function () {
        FilingCabinet.render();
      });
    }

    const audioSelect = document.getElementById('audio-preset');
    if (audioSelect) {
      audioSelect.addEventListener('change', function (e) {
        handleAudio(e.target.value);
      });
    }

    handleIngest();
  });
})();