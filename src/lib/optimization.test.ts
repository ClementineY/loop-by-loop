import { describe, expect, it } from 'vitest';
import { bowlGradient, bowlLoss, gradientDirectionDelta, mean, optimizerPath, randomDirectionDeltas } from './optimization';

describe('optimization teaching models', () => {
  it('computes the conditioned bowl and its gradient', () => {
    expect(bowlLoss([2, 1], 12)).toBe(8);
    expect(bowlGradient([2, 1], 12)).toEqual([2, 12]);
  });

  it('plain gradient descent matches its first numerical update', () => {
    const path = optimizerPath([4, 2], 1, 0.05, 0, 12);
    expect(path[1].point[0]).toBeCloseTo(3.8);
    expect(path[1].point[1]).toBeCloseTo(0.8);
    expect(path[1].loss).toBeCloseTo(11.06);
  });

  it('momentum carries the previous gradient into the next update', () => {
    const path = optimizerPath([1, 0], 2, 0.1, 0.9, 1);
    expect(path[1].point[0]).toBeCloseTo(0.9);
    expect(path[2].point[0]).toBeCloseTo(0.72);
    expect(path[2].velocity[0]).toBeCloseTo(1.8);
  });

  it('the same-length steepest descent move has the exact quadratic change', () => {
    expect(gradientDirectionDelta(0.1)).toBeCloseTo(-0.095);
  });

  it('random-search samples are deterministic and become poorly aligned in high dimensions', () => {
    const low = randomDirectionDeltas(2, 120, 0.1, 7);
    const high = randomDirectionDeltas(1000, 120, 0.1, 7);
    expect(low[0]).toBeCloseTo(randomDirectionDeltas(2, 1, 0.1, 7)[0]);
    expect(Math.min(...low)).toBeLessThan(-0.08);
    expect(Math.min(...low)).toBeGreaterThanOrEqual(gradientDirectionDelta(0.1));
    expect(Math.min(...high)).toBeGreaterThan(-0.02);
  });

  it('averages minibatch gradients', () => {
    expect(mean([-6, -4, -2, -1, 0, 1, 3, 4, 5, 6, 8, 10])).toBe(2);
  });
});
