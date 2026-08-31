/** Small, fast, seedable PRNG. Deterministic given a seed — important so the
 *  training widget can be reset to the exact same starting point. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Dataset {
  X: [number, number][];
  y: number[];
}

/** Two interleaving half-circles ("moons"), a classic non-linearly-separable
 *  2D toy dataset — perfect for showing a decision boundary form. */
export function makeMoons(n: number, noise = 0.1, seed = 1): Dataset {
  if (!Number.isInteger(n) || n < 2) throw new RangeError('n must be an integer of at least 2');
  if (!Number.isFinite(noise) || noise < 0) throw new RangeError('noise must be a non-negative number');
  const rnd = mulberry32(seed);
  const X: [number, number][] = [];
  const y: number[] = [];
  const half = Math.floor(n / 2);
  const lowerCount = n - half;
  for (let i = 0; i < n; i++) {
    const top = i < half;
    const classIndex = top ? i : i - half;
    const classCount = top ? half : lowerCount;
    const t = (Math.PI * classIndex) / Math.max(1, classCount - 1);
    const nx = (rnd() - 0.5) * 2 * noise;
    const ny = (rnd() - 0.5) * 2 * noise;
    if (top) {
      X.push([Math.cos(t) + nx, Math.sin(t) + ny]);
      y.push(0);
    } else {
      X.push([1 - Math.cos(t) + nx, 0.5 - Math.sin(t) + ny]);
      y.push(1);
    }
  }
  return { X, y };
}
