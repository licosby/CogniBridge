/**
 * NeuroSift AI — Explainable AI (XAI) & Mathematical Entropy Engine
 * Computes Shannon Entropy, Polysyllabic Density, and Lexical Salience in O(N).
 */

export class XAIEngine {
  /**
   * Calculates Shannon Entropy: H = -Σ (p(x) * log2(p(x)))
   * Measures uncertainty and contextual density of the incoming text block.
   */
  static computeEntropy(tokens) {
    if (!tokens || tokens.length === 0) return 0;
    
    const freq = {};
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i].toLowerCase();
      freq[t] = (freq[t] || 0) + 1;
    }

    let entropy = 0;
    const total = tokens.length;
    for (const word in freq) {
      const p = freq[word] / total;
      entropy -= p * Math.log2(p);
    }
    return parseFloat(entropy.toFixed(3));
  }

  /**
   * Counts approximate syllables to detect dense polysyllabic words (>3 syllables)
   */
  static countSyllables(word) {
    word = word.toLowerCase().trim();
    if (word.length <= 3) return 1;
    word = word.replace(/(?:[^laeiouy]|ed|es|e)$/, '');
    word = word.replace(/^y/, '');
    const matches = word.match(/[aeiouy]{1,2}/g);
    return matches ? matches.length : 1;
  }

  /**
   * Generates a composite Cognitive Friction Score (0 to 100)
   */
  static analyzeComplexity(rawText) {
    const tokens = rawText.match(/\b[A-Za-z0-9_-]+\b/g) || [];
    if (tokens.length === 0) {
      return {
        tokens: [],
        entropy: 0,
        polysyllabicRatio: 0,
        frictionScore: 0,
        riskLevel: 'Optimal'
      };
    }

    const entropy = this.computeEntropy(tokens);
    let polysyllables = 0;
    
    const weightedTokens = tokens.map(token => {
      const syllables = this.countSyllables(token);
      if (syllables >= 3) polysyllables++;
      return {
        word: token,
        syllables: syllables,
        isDense: syllables >= 4 || token.length > 10,
        isModerate: syllables === 3
      };
    });

    const polysyllabicRatio = parseFloat(((polysyllables / tokens.length) * 100).toFixed(1));
    const frictionScore = Math.min(Math.round((entropy * 11) + (polysyllabicRatio * 0.75)), 100);

    let riskLevel = 'Optimal';
    if (frictionScore > 65) {
      riskLevel = 'Critical (Fatigue)';
    } else if (frictionScore > 35) {
      riskLevel = 'Moderate Friction';
    }

    return {
      tokens: weightedTokens,
      entropy,
      polysyllabicRatio,
      frictionScore,
      riskLevel
    };
  }
}