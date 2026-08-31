import {describe,expect,it} from 'vitest';
import {bestValidationEpoch,conv2dGeometry,examplesPerSecond,generalizationLoss,linearParameterCount,neuronForward,quantizeScalar,reduceShape,softmax2,splitCounts,tensorStorageMB} from './curriculum';

describe('course visual calculations',()=>{
  it('matches the default neuron arithmetic',()=>{
    const result=neuronForward(1.5,-1,.8,-.4,.2,'relu');
    expect(result.contribution1).toBeCloseTo(1.2);
    expect(result.contribution2).toBeCloseTo(.4);
    expect(result.z).toBeCloseTo(1.8);
    expect(result.output).toBeCloseTo(1.8);
  });
  it('counts weights and biases for both linear layers',()=>{
    expect(linearParameterCount(2,6)).toBe(18);
    expect(linearParameterCount(6,3)).toBe(21);
  });
  it('softmax ignores a shared logit offset',()=>{
    const base=softmax2(1.2,-.3),shifted=softmax2(101.2,99.7);
    expect(base[0]).toBeCloseTo(.817574);
    expect(shifted[0]).toBeCloseTo(base[0]);
    expect(base[0]+base[1]).toBeCloseTo(1);
  });
  it('removes exactly the selected axis',()=>expect(reduceShape([2,3,4],1)).toEqual([2,4]));
  it('keeps split counts exhaustive',()=>expect(splitCounts(100,70,15)).toEqual({train:70,validation:15,test:15}));
  it('matches Conv2d geometry and parameter count',()=>expect(conv2dGeometry(32,3,16,3,1,1)).toEqual({outputSize:32,parameters:448}));
  it('places the overfitting validation minimum before the final epoch',()=>{
    const best=bestValidationEpoch('overfit');
    expect(best).toBeGreaterThan(15);
    expect(best).toBeLessThan(40);
    expect(generalizationLoss('overfit',best)[1]).toBeLessThan(generalizationLoss('overfit',40)[1]);
  });
  it('computes exact storage and measured throughput examples',()=>{
    expect(tensorStorageMB(64*128*768,32)).toBeCloseTo(25.165824);
    expect(tensorStorageMB(64*128*768,16)).toBeCloseTo(12.582912);
    expect(examplesPerSecond(64,80)).toBe(800);
  });
  it('shows signed scalar quantization error',()=>{
    expect(quantizeScalar(.63,.25)).toEqual({quantized:.75,error:.12});
  });
});
