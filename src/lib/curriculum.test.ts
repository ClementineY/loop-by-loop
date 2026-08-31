import {describe,expect,it} from 'vitest';
import {bestValidationEpoch,binaryTrainingStep,conv2dGeometry,examplesPerSecond,generalizationLoss,linearParameterCount,multiclassTrainingStep,neuronForward,parseTuneValues,quantizeScalar,reconstructionTrainingStep,reduceShape,regressionTrainingStep,softmax2,splitCounts,tensorStorageMB} from './curriculum';

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
  it('matches one editable regression step',()=>{
    const step=regressionTrainingStep(2,5,1,.5,.05);
    expect(step.prediction).toBe(2.5);
    expect(step.loss).toBe(6.25);
    expect(step.gradientWeight).toBe(-10);
    expect(step.nextPrediction).toBe(3.75);
  });
  it('moves binary probability toward a positive target',()=>{
    const step=binaryTrainingStep(1,1,-.4,0,.5);
    expect(step.probability).toBeCloseTo(.401312);
    expect(step.loss).toBeCloseTo(.913015);
    expect(step.nextProbability).toBeGreaterThan(step.probability);
  });
  it('raises the target probability in multiclass training',()=>{
    const step=multiclassTrainingStep([1.1,.4,-.2],1,.4);
    expect(step.probabilities.reduce((a,b)=>a+b,0)).toBeCloseTo(1);
    expect(step.nextProbabilities[1]).toBeGreaterThan(step.probabilities[1]);
  });
  it('moves every reconstruction value toward its target',()=>{
    const step=reconstructionTrainingStep([.2,.8,.3,.7],[0,1,1,0],.5);
    step.nextPrediction.forEach((value,index)=>expect(Math.abs(value-step.target[index])).toBeLessThan(Math.abs(step.prediction[index]-step.target[index])));
  });
  it('reads editable scalar and vector values from marked PyTorch code',()=>{
    expect(parseTuneValues('lr = 0.05 # tune:lr','lr',[1])).toEqual([.05]);
    expect(parseTuneValues('x = torch.tensor([0.2, 0.8, -0.3]) # tune:x','x',[])).toEqual([.2,.8,-.3]);
  });
});
