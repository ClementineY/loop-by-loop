export const SITE = {
  name: 'Loop by Loop',
  tagline: 'From Python fluency to a real PyTorch training loop.',
  description:
    'The visual bridge from knowing Python to reading and writing a real PyTorch training loop.',
  repository: 'https://github.com/ClementineY/loop-by-loop',
  colabRepository: 'ClementineY/loop-by-loop',
} as const;

export const moduleDescriptions: Record<string, string> = {
  'M1 · Tensors': 'The shapes, values, and operations that carry every model.',
  'M2 · Autograd': 'How computation graphs turn local derivatives into learning.',
  'M3 · Optimization': 'How loss becomes a direction and parameters improve.',
  'M4 · Neural networks': 'Build a useful network from linear layers and activations.',
  'M5 · Data': 'Feed models reliably with datasets, batches, and shuffling.',
  'M6 · Convolution': 'Learn how small filters discover structure in images.',
  'M7 · Training well': 'Diagnose overfitting and make experiments reproducible.',
  'M8 · Ship a model': 'Save, load, evaluate, and complete an end-to-end project.',
};
