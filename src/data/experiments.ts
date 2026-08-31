export interface Experiment {
  id: string;
  module: string;
  title: string;
  summary: string;
  question: string;
  notebook: string;
  duration: number;
  level: 'Starter' | 'Core' | 'Capstone';
  skills: string[];
}

export const experiments: Experiment[] = [
  {
    id: 'tensor-shape-lab',
    module: 'M1 · Tensors',
    title: 'Shape, reshape, and broadcast',
    summary: 'Create tensors, trace their metadata, normalize a batch, and test which shape operations are legal.',
    question: 'Can you predict every result shape before PyTorch prints it?',
    notebook: 'm1_tensors.ipynb',
    duration: 15,
    level: 'Starter',
    skills: ['shape', 'dtype', 'reshape', 'broadcasting'],
  },
  {
    id: 'gradient-trace',
    module: 'M2 · Autograd',
    title: 'Trace a gradient by hand and by code',
    summary: 'Build a scalar graph, run backward, inspect leaf gradients, and observe accumulation across calls.',
    question: 'Can you derive the gradients before autograd reveals them?',
    notebook: 'm2_autograd.ipynb',
    duration: 18,
    level: 'Core',
    skills: ['requires_grad', 'backward', 'grad', 'zeroing'],
  },
  {
    id: 'fit-a-line',
    module: 'M3 · Optimization',
    title: 'Fit a noisy line from scratch',
    summary: 'Connect prediction, loss, backward, and optimizer step in a complete, inspectable training loop.',
    question: 'How does the learning rate change convergence?',
    notebook: 'm3_optimization.ipynb',
    duration: 20,
    level: 'Core',
    skills: ['MSE', 'SGD', 'training loop', 'parameters'],
  },
  {
    id: 'inspect-a-module',
    module: 'M4 · Neural networks',
    title: 'Build and inspect an nn.Module',
    summary: 'Define a two-layer classifier, count its parameters, and verify the shape at every boundary.',
    question: 'Where does each trainable value live?',
    notebook: 'm4_networks.ipynb',
    duration: 20,
    level: 'Core',
    skills: ['nn.Module', 'Linear', 'activation', 'parameter count'],
  },
  {
    id: 'batch-pipeline',
    module: 'M5 · Data',
    title: 'Turn examples into shuffled batches',
    summary: 'Create a dataset, configure a DataLoader, and inspect labels, shapes, and the final smaller batch.',
    question: 'What changes when shuffle or batch size changes?',
    notebook: 'm5_data.ipynb',
    duration: 15,
    level: 'Starter',
    skills: ['Dataset', 'DataLoader', 'batching', 'shuffle'],
  },
  {
    id: 'cnn-shape-story',
    module: 'M6 · Convolution',
    title: 'Follow an image through a CNN',
    summary: 'Assemble a small convolutional network and confirm how channels and spatial dimensions evolve.',
    question: 'Can you write the complete shape story without running the cell?',
    notebook: 'm6_convolution.ipynb',
    duration: 22,
    level: 'Core',
    skills: ['Conv2d', 'pooling', 'channels', 'logits'],
  },
  {
    id: 'reproducible-run',
    module: 'M7 · Training well',
    title: 'Design a reproducible experiment',
    summary: 'Control random streams and record the evidence needed to compare two training decisions honestly.',
    question: 'What must be fixed before two runs are comparable?',
    notebook: 'm7_training_well.ipynb',
    duration: 18,
    level: 'Core',
    skills: ['seeds', 'validation', 'configuration', 'experiment record'],
  },
  {
    id: 'checkpoint-capstone',
    module: 'M8 · Ship a model',
    title: 'Checkpoint and verify a classifier',
    summary: 'Save model state, restore it into a fresh instance, verify identical outputs, and plan the CIFAR-10 capstone.',
    question: 'Can another learner reproduce your evaluation from the saved artifact?',
    notebook: 'm8_ship.ipynb',
    duration: 30,
    level: 'Capstone',
    skills: ['state_dict', 'checkpoint', 'evaluation', 'model card'],
  },
];
