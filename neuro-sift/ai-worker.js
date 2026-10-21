/**
 * NeuroSift AI — Background Inference Worker
 * Runs Hugging Face Transformers.js via ONNX Runtime WebAssembly/WebGPU.
 * Zero main-thread blocking. Zero egress cost.
 */

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

// Enforce local caching in browser CacheStorage
env.allowLocalModels = false;
env.useBrowserCache = true;

let classifierPipeline = null;

// Initialize ML Pipeline
async function initPipeline() {
  try {
    postMessage({ type: 'STATUS', status: 'Downloading Quantized Transformer (Cached)...' });
    // Zero-shot/Sentiment Classifier for high-friction cognitive load classification
    classifierPipeline = await pipeline('text-classification', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
    postMessage({ type: 'READY' });
  } catch (error) {
    postMessage({ type: 'ERROR', error: error.message });
  }
}

initPipeline();

// Listen for job dispatches from the main thread
self.onmessage = async (e) => {
  const { type, text } = e.data;

  if (type === 'RUN_INFERENCE') {
    try {
      postMessage({ type: 'INFERENCE_START' });

      // Split text into semantic sentences
      const sentences = text.split(/(?<=[.?!])\s+/).filter(s => s.trim().length > 0);

      // Perform transformer classification on sentences
      const tasks = [];
      for (let i = 0; i < sentences.length; i++) {
        const sentence = sentences[i];
        let sentimentScore = 0.5;
        
        if (classifierPipeline) {
          const res = await classifierPipeline(sentence.slice(0, 250));
          if (res && res[0]) {
            sentimentScore = res[0].score;
          }
        }

        tasks.push({
          id: Date.now() + i,
          title: `Milestone ${i + 1}: ${sentence.slice(0, 48)}...`,
          rawSentence: sentence,
          confidence: (sentimentScore * 100).toFixed(1),
          substeps: [
            { text: `Parse direct deliverable dependencies`, completed: false },
            { text: `Execute focused single-pass implementation`, completed: false },
            { text: `Audit against quality standard`, completed: false }
          ]
        });
      }

      postMessage({ type: 'INFERENCE_SUCCESS', tasks });
    } catch (err) {
      postMessage({ type: 'ERROR', error: err.message });
    }
  }
};