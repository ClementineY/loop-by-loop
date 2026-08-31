export type Point = [number, number];

export interface PathState {
  point: Point;
  gradient: Point;
  velocity: Point;
  loss: number;
}

export function bowlLoss([x, y]: Point, condition = 12): number {
  return 0.5 * (x * x + condition * y * y);
}

export function bowlGradient([x, y]: Point, condition = 12): Point {
  return [x, condition * y];
}

export function optimizerPath(
  start: Point,
  steps: number,
  lr: number,
  momentum = 0,
  condition = 12,
): PathState[] {
  let point: Point = [...start];
  let velocity: Point = [0, 0];
  const path: PathState[] = [];
  for (let step = 0; step <= steps; step++) {
    const gradient = bowlGradient(point, condition);
    path.push({ point: [...point], gradient, velocity: [...velocity], loss: bowlLoss(point, condition) });
    velocity = [momentum * velocity[0] + gradient[0], momentum * velocity[1] + gradient[1]];
    point = [point[0] - lr * velocity[0], point[1] - lr * velocity[1]];
  }
  return path;
}

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normal(random: () => number): number {
  const u = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/** Loss changes for fixed-length random moves on L(w)=1/2||w||² at ||w||=1. */
export function randomDirectionDeltas(dimension: number, trials = 120, step = 0.1, seed = 7): number[] {
  if (!Number.isInteger(dimension) || dimension < 1) throw new RangeError('dimension must be a positive integer');
  const random = seeded(seed);
  const w = 1 / Math.sqrt(dimension);
  return Array.from({ length: trials }, () => {
    const direction = Array.from({ length: dimension }, () => normal(random));
    const norm = Math.hypot(...direction);
    const alignment = direction.reduce((sum, value) => sum + w * value / norm, 0);
    return step * alignment + 0.5 * step * step;
  });
}

export function gradientDirectionDelta(step = 0.1): number {
  return 0.5 * (1 - step) ** 2 - 0.5;
}

export function mean(values: number[]): number {
  if (!values.length) throw new RangeError('cannot average an empty list');
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
