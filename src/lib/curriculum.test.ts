import {describe,expect,it} from 'vitest';
import {bestValidationEpoch,binaryTrainingStep,compileCornerCasePlan,controlFlowCompilePlan,conv2dGeometry,estimateTrainingMemory,estimateTrainingScale,examplesPerSecond,generalizationLoss,gradientScalingTrace,linearParameterCount,matrix2x2Trace,multiclassTrainingStep,neuronForward,parseTuneValues,quantizeScalar,reconstructionTrainingStep,reduceShape,regressionTrainingStep,roundToBFloat16,roundToFloat16,softmax2,splitCounts,tensorStorageMB} from './curriculum';

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
  it('estimates per-device Adam memory under DDP',()=>{
    const memory=estimateTrainingMemory({parametersMillions:100,microBatch:8,savedActivationMillionsPerSample:20,activationBits:16,optimizer:'adam',devices:4,strategy:'ddp',reservePercent:15});
    expect(memory.weights).toBeCloseTo(.372529);
    expect(memory.gradients).toBeCloseTo(.372529);
    expect(memory.optimizer).toBeCloseTo(.745058);
    expect(memory.activations).toBeCloseTo(.298023);
    expect(memory.total).toBeCloseTo(2.05636,4);
  });
  it('shards gradients and optimizer state under ZeRO stage 2',()=>{
    const memory=estimateTrainingMemory({parametersMillions:100,microBatch:8,savedActivationMillionsPerSample:20,activationBits:16,optimizer:'adam',devices:4,strategy:'zero2',reservePercent:15});
    expect(memory.weights).toBeCloseTo(.372529);
    expect(memory.gradients).toBeCloseTo(.093132);
    expect(memory.optimizer).toBeCloseTo(.186265);
    expect(memory.activations).toBeCloseTo(.298023);
  });
  it('connects micro-batch, accumulation, throughput, and epoch time',()=>{
    const scale=estimateTrainingScale({datasetExamples:100000,microBatch:8,devices:4,accumulationSteps:2,microStepMilliseconds:100});
    expect(scale.globalBatch).toBe(64);
    expect(scale.optimizerStepsPerEpoch).toBe(1563);
    expect(scale.throughput).toBe(320);
    expect(scale.epochSeconds).toBeCloseTo(312.6);
  });
  it('rounds values to real FP16 and BF16 representations',()=>{
    expect(roundToFloat16(1.337)).toBeCloseTo(1.3369140625,10);
    expect(roundToBFloat16(1.337)).toBeCloseTo(1.3359375,10);
    expect(roundToFloat16(1e-8)).toBe(0);
  });
  it('traces matrix rounding without changing its shape',()=>{
    const trace=matrix2x2Trace([1.2345,-.3333,.0001,2.7183],[.7071,1.4142,-1.1111,.0625],'fp16');
    expect(trace.output).toHaveLength(4);
    expect(trace.bytesPerElement).toBe(2);
    expect(trace.maxAbsoluteError).toBeGreaterThan(0);
  });
  it('shows how scaling rescues a tiny FP16 gradient',()=>{
    const trace=gradientScalingTrace(1e-8,4096);
    expect(trace.withoutScaling).toBe(0);
    expect(trace.storedScaledGradient).not.toBe(0);
    expect(trace.restoredGradient).toBeCloseTo(1e-8,9);
    expect(trace.finite).toBe(true);
  });
  it('distinguishes eager paths, guarded branches, graph breaks, and torch.cond',()=>{
    expect(controlFlowCompilePlan('eager',4,true).cacheKey).toBeNull();
    expect(controlFlowCompilePlan('python-flag',4,true).guard).toContain('flag is true');
    expect(controlFlowCompilePlan('python-flag',4,false).cacheKey).not.toBe(controlFlowCompilePlan('python-flag',4,true).cacheKey);
    expect(controlFlowCompilePlan('tensor-if',4,false).graphBreak).toBe(true);
    expect(controlFlowCompilePlan('torch-cond',4,true).cacheKey).toBe(controlFlowCompilePlan('torch-cond',4,false).cacheKey);
    expect(controlFlowCompilePlan('torch-cond',4,true).capturesBothBranches).toBe(true);
  });
  it('models compile corner cases without pretending every control flow form is equivalent',()=>{
    expect(compileCornerCasePlan('fixed-loop',4).nodeCount).toBe(8);
    expect(compileCornerCasePlan('tensor-loop',false).graphBreak).toBe(true);
    expect(compileCornerCasePlan('tensor-loop',true).outcome).toBe('inference only');
    expect(compileCornerCasePlan('scalar-item',false).graphBreak).toBe(true);
    expect(compileCornerCasePlan('scalar-item',true).graphBreak).toBe(false);
    expect(compileCornerCasePlan('shape-branch',16).guard).toBe('batch <= 16');
    expect(compileCornerCasePlan('shape-branch',64).guard).toBe('batch > 16');
    expect(compileCornerCasePlan('cond-contract',false).outcome).toBe('capture error');
  });
});
