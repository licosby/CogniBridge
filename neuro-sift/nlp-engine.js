/**
 * CogniBridge Hub — NLP Semantic Role & Classification Engine
 * Extracts clauses, detects work vs personal domain, and translates corporate jargon.
 */

export class NLPEngine {
  static glossary = {
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
    groceries: "Purchasing essential food items and household household goods.",
    dentist: "A healthcare appointment focused on oral health and hygiene."
  };

  /**
   * Detects whether a document/clause is Work/Professional or Personal.
   */
  static classifyDomain(text) {
    const lower = text.toLowerCase();
    const personalKeywords = [
      'laundry', 'groceries', 'dentist', 'doctor', 'mom', 'dad', 'family', 'clean room',
      'dishes', 'dog', 'cat', 'vet', 'gym', 'workout', 'dinner', 'prescription', 'pharmacy'
    ];
    for (const kw of personalKeywords) {
      if (lower.includes(kw)) return 'personal';
    }
    return 'work';
  }

  /**
   * Detects whether incoming text is a Passive Informational Email / Memo.
   */
  static isPassiveInformational(text) {
    const lower = text.toLowerCase();
    const isMemo = /\b(pursuant|synergy|paradigm|apprised|stakeholder|restructuring|pivot|townhall|memorandum|newsletter)\b/i.test(text);
    const hasImmediateAction = /\b(call|fix|clean up|commit|deploy|buy|reschedule|submit by)\b/i.test(text);
    return isMemo && (!hasImmediateAction || lower.includes('no immediate deliverables') || lower.includes('for your information'));
  }

  /**
   * Translates a passive corporate email into plain English.
   */
  static deJargonizeEmail(text) {
    const lower = text.toLowerCase();
    let tlDr = "This email provides a high-level organizational or departmental update. No immediate action or deadline is required of you.";

    if (lower.includes('restructuring') || lower.includes('synergy')) {
      tlDr = "The company is reorganizing team structures and aligning goals. You do not need to deliver anything today; attend upcoming meetings for context.";
    } else if (lower.includes('policy') || lower.includes('compliance')) {
      tlDr = "An internal policy has been updated. Review the documentation at your convenience to stay informed.";
    }

    const keyTakeaways = [];
    if (lower.includes('no immediate deliverables') || lower.includes('not incumbent')) {
      keyTakeaways.push("✅ No work, code, or deliverables are due from you today.");
    } else {
      keyTakeaways.push("ℹ️ Informational memo: Read to remain aligned with leadership.");
    }

    if (lower.includes('townhall') || lower.includes('sync') || lower.includes('meeting')) {
      keyTakeaways.push("📅 Keep an eye out for a calendar invite for an upcoming team/townhall session.");
    }

    let jargonFound = 0;
    Object.keys(this.glossary).forEach(term => {
      if (lower.includes(term)) jargonFound++;
    });
    keyTakeaways.push(`🔍 We detected and indexed ${jargonFound} corporate jargon terms in this message.`);

    return {
      tlDr,
      takeaways: keyTakeaways,
      replyDraft: "Thank you for the update! Acknowledged on my end; I'll look out for the upcoming team session."
    };
  }

  /**
   * Breaks text into domain-aware, correlated task items with unbounded steps.
   */
  static parseTasks(text) {
    const rawLines = text
      .split(/(?:\r?\n|;|\.(?!\d))/g)
      .map(s => s.trim())
      .filter(s => s.length > 3);

    return rawLines.map(line => {
      const timeRegex = /\b(?:(?:by|at|on|for|before|after)\s+)?(?:tomorrow|today|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)?(?:\s*(?:at|by)\s*)?(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|noon|midnight)\b/i;
      const timeMatch = line.match(timeRegex);
      
      let statusBadge = "Active Step";
      if (timeMatch) {
        let matchedStr = timeMatch[0].trim();
        statusBadge = "Target: " + matchedStr.charAt(0).toUpperCase() + matchedStr.slice(1);
      }

      const domain = this.classifyDomain(line);
      const cleanTitle = line.charAt(0).toUpperCase() + line.slice(1);
      const steps = this.generateStepsForLine(line, domain);

      return {
        id: Date.now() + Math.random(),
        title: cleanTitle,
        domain: domain, // 'work' | 'personal'
        statusBadge: statusBadge,
        collapsed: false,
        steps: steps.map(s => ({ label: s, done: false })),
        done: false,
        completedAt: null
      };
    });
  }

  static generateStepsForLine(clause, domain) {
    const lower = clause.toLowerCase();
    const steps = [];

    // Personal: Doctor / Dentist / Health
    if (lower.includes('doctor') || lower.includes('appointment') || lower.includes('dentist')) {
      steps.push("Put photo ID, insurance card, and co-pay in wallet now");
      steps.push("Check commute transit time and set departure alarm 15 minutes early");
      steps.push("Arrive with 10 minutes to spare to check in at front desk");
      return steps;
    }

    // Personal: Chores / Household
    if (lower.includes('laundry') || lower.includes('dishes') || lower.includes('clean')) {
      steps.push("Walk to the area and gather all items into one central spot");
      steps.push("Complete the active cleaning or washing cycle without distraction");
      steps.push("Put clean items away immediately so the space remains clear");
      return steps;
    }

    // Work: Tech / Database Spec
    if (lower.includes('database') || lower.includes('atomic') || lower.includes('lock') || lower.includes('protocol')) {
      steps.push("Open project repository and locate database configuration module");
      steps.push("Check current lock-timeout settings across isolated cloud regions");
      steps.push("Replace pessimistic database locks with optimistic version check columns");
      steps.push("Implement exponential backoff retry logic with jitter buffer");
      steps.push("Run automated test suite simulating 500 simultaneous writes");
      steps.push("Verify deadlocks drop to zero during automated failover");
      return steps;
    }

    // Work: Investor Meeting
    if (lower.includes('investor') || lower.includes('pitch') || lower.includes('meet with')) {
      steps.push("Review pitch deck slides and verify ARR and valuation ask");
      steps.push("Test microphone, camera, and screen sharing 15 minutes early");
      steps.push("Have running live prototype ready at localhost in your browser");
      steps.push("Join the video meeting room 2 minutes prior");
      return steps;
    }

    // Work: Code / CSS Refactor
    if (lower.includes('css') || lower.includes('code') || lower.includes('refactor')) {
      steps.push("Open style.css in code editor");
      steps.push("Prune unused classes and organize colors into design variables");
      steps.push("Hard-refresh browser (Cmd+Shift+R) and test responsive breakpoints");
      return steps;
    }

    // Default Dynamic Steps
    steps.push(`Identify the main workspace or file needed for: "${clause.slice(0, 32)}..."`);
    steps.push("Execute the primary milestone without context-switching");
    steps.push("Verify output meets standard and mark completed");
    return steps;
  }

  /**
   * Computes Shannon Entropy and Lexical Polysyllable metrics.
   */
  static computeMetrics(text) {
    const tokens = text.match(/\b[A-Za-z0-9_-]+\b/g) || [];
    if (tokens.length === 0) return { entropy: 0, friction: 0, tokens: [] };

    const freq = {};
    tokens.forEach(t => {
      const lower = t.toLowerCase();
      freq[lower] = (freq[lower] || 0) + 1;
    });

    let entropy = 0;
    const total = tokens.length;
    Object.keys(freq).forEach(w => {
      const p = freq[w] / total;
      entropy -= p * Math.log2(p);
    });

    let complex = 0;
    tokens.forEach(w => {
      const vowels = w.match(/[aeiouy]{1,2}/gi);
      if (vowels && vowels.length >= 3) complex++;
    });
    const polyRatio = (complex / total) * 100;
    const friction = Math.min(Math.round((entropy * 11) + (polyRatio * 0.75)), 100);

    return {
      entropy: parseFloat(entropy.toFixed(2)),
      friction,
      tokens
    };
  }
}