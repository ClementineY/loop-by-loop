export function neuronForward(x1:number,x2:number,w1:number,w2:number,bias:number,activation:'relu'|'tanh') {
  const contribution1=x1*w1, contribution2=x2*w2, z=contribution1+contribution2+bias;
  return { contribution1, contribution2, z, output:activation==='relu'?Math.max(0,z):Math.tanh(z) };
}

export const linearParameterCount=(inputs:number,outputs:number)=>inputs*outputs+outputs;

export function softmax2(a:number,b:number):[number,number] {
  const max=Math.max(a,b),ea=Math.exp(a-max),eb=Math.exp(b-max),sum=ea+eb;
  return [ea/sum,eb/sum];
}

export function reduceShape(shape:number[],axis:number):number[] {
  if(axis<0||axis>=shape.length)throw new RangeError('axis is outside shape');
  return shape.filter((_,index)=>index!==axis);
}

export function splitCounts(total:number,trainPercent:number,validationPercent:number) {
  const train=Math.round(total*trainPercent/100),validation=Math.round(total*validationPercent/100);
  return {train,validation,test:total-train-validation};
}

export function conv2dGeometry(size:number,inChannels:number,outChannels:number,kernel:number,stride:number,padding:number) {
  const outputSize=Math.floor((size+2*padding-kernel)/stride+1);
  const parameters=outChannels*(inChannels*kernel*kernel+1);
  return {outputSize,parameters};
}

export type FitScenario='underfit'|'learning'|'overfit';
export function generalizationLoss(scenario:FitScenario,epoch:number):[number,number] {
  if(scenario==='underfit')return [1.15-.012*epoch,1.22-.01*epoch];
  if(scenario==='learning')return [1.25*Math.exp(-.08*epoch)+.16,1.2*Math.exp(-.072*epoch)+.22];
  return [1.3*Math.exp(-.09*epoch)+.08,1.15*Math.exp(-.075*epoch)+.24+Math.max(0,epoch-15)**2*.0018];
}

export function bestValidationEpoch(scenario:FitScenario,maxEpoch=40):number {
  let best=0;
  for(let epoch=1;epoch<=maxEpoch;epoch++)if(generalizationLoss(scenario,epoch)[1]<generalizationLoss(scenario,best)[1])best=epoch;
  return best;
}

export const tensorStorageMB=(elements:number,bits:number)=>elements*bits/8/1e6;
export const examplesPerSecond=(batch:number,stepMilliseconds:number)=>batch/(stepMilliseconds/1000);
export function quantizeScalar(value:number,step:number) {
  if(step<=0)throw new RangeError('quantization step must be positive');
  const quantized=Math.round(value/step)*step;
  return {quantized,error:quantized-value};
}

export function regressionTrainingStep(x:number,target:number,weight:number,bias:number,lr:number) {
  const prediction=weight*x+bias,error=prediction-target,loss=error**2;
  const gradientWeight=2*error*x,gradientBias=2*error;
  const nextWeight=weight-lr*gradientWeight,nextBias=bias-lr*gradientBias;
  return {x,weight,bias,prediction,target,loss,gradientWeight,gradientBias,nextWeight,nextBias,nextPrediction:nextWeight*x+nextBias};
}

export function binaryTrainingStep(x:number,target:number,weight:number,bias:number,lr:number) {
  const logit=weight*x+bias,probability=1/(1+Math.exp(-logit));
  const loss=-(target*Math.log(probability)+(1-target)*Math.log(1-probability));
  const gradientLogit=probability-target,gradientWeight=gradientLogit*x,gradientBias=gradientLogit;
  const nextWeight=weight-lr*gradientWeight,nextBias=bias-lr*gradientBias;
  const nextLogit=nextWeight*x+nextBias,nextProbability=1/(1+Math.exp(-nextLogit));
  return {logit,probability,target,loss,gradientLogit,gradientWeight,gradientBias,nextWeight,nextBias,nextLogit,nextProbability};
}

export function multiclassTrainingStep(logits:number[],target:number,lr:number) {
  const max=Math.max(...logits),exp=logits.map(value=>Math.exp(value-max)),sum=exp.reduce((a,b)=>a+b,0),probabilities=exp.map(value=>value/sum);
  const loss=-Math.log(probabilities[target]);
  const gradients=probabilities.map((value,index)=>value-(index===target?1:0));
  const nextLogits=logits.map((value,index)=>value-lr*gradients[index]);
  const nextMax=Math.max(...nextLogits),nextExp=nextLogits.map(value=>Math.exp(value-nextMax)),nextSum=nextExp.reduce((a,b)=>a+b,0),nextProbabilities=nextExp.map(value=>value/nextSum);
  return {logits,target,probabilities,loss,gradients,nextLogits,nextProbabilities};
}

export function reconstructionTrainingStep(prediction:number[],target:number[],lr:number) {
  if(prediction.length!==target.length||!prediction.length)throw new RangeError('prediction and target need matching non-empty shapes');
  const errors=prediction.map((value,index)=>value-target[index]);
  const loss=errors.reduce((sum,value)=>sum+value**2,0)/prediction.length;
  const gradients=errors.map(value=>2*value/prediction.length);
  const nextPrediction=prediction.map((value,index)=>value-lr*gradients[index]);
  return {prediction,target,loss,gradients,nextPrediction};
}

export function parseTuneValues(source:string,key:string,fallback:number[]) {
  const line=source.split('\n').find(item=>item.includes(`# tune:${key}`));
  if(!line)return fallback;
  const values=(line.split('#')[0].match(/-?\d+(?:\.\d+)?/g)||[]).map(Number);
  return values.length?values:fallback;
}

export type OptimizerMemory='sgd'|'momentum'|'adam';
export type ShardingStrategy='ddp'|'zero1'|'zero2'|'zero3';
export function estimateTrainingMemory(options:{parametersMillions:number;microBatch:number;savedActivationMillionsPerSample:number;activationBits:16|32;optimizer:OptimizerMemory;devices:number;strategy:ShardingStrategy;reservePercent:number}) {
  const parameters=options.parametersMillions*1e6,devices=Math.max(1,options.devices),gib=1024**3;
  const shardParameters=options.strategy==='zero3'?devices:1;
  const shardGradients=options.strategy==='zero2'||options.strategy==='zero3'?devices:1;
  const shardOptimizer=options.strategy==='ddp'?1:devices;
  const optimizerBytesPerParameter=options.optimizer==='adam'?8:options.optimizer==='momentum'?4:0;
  const weights=parameters*4/shardParameters/gib;
  const gradients=parameters*4/shardGradients/gib;
  const optimizer=parameters*optimizerBytesPerParameter/shardOptimizer/gib;
  const activations=options.microBatch*options.savedActivationMillionsPerSample*1e6*(options.activationBits/8)/gib;
  const subtotal=weights+gradients+optimizer+activations;
  const reserve=subtotal*options.reservePercent/100;
  return {weights,gradients,optimizer,activations,reserve,total:subtotal+reserve,optimizerBytesPerParameter};
}

export function estimateTrainingScale(options:{datasetExamples:number;microBatch:number;devices:number;accumulationSteps:number;microStepMilliseconds:number}) {
  const globalBatch=options.microBatch*options.devices*options.accumulationSteps;
  const optimizerStepsPerEpoch=Math.ceil(options.datasetExamples/globalBatch);
  const updateMilliseconds=options.microStepMilliseconds*options.accumulationSteps;
  const throughput=globalBatch/(updateMilliseconds/1000);
  const epochSeconds=optimizerStepsPerEpoch*updateMilliseconds/1000;
  return {globalBatch,optimizerStepsPerEpoch,updateMilliseconds,throughput,epochSeconds};
}

export type FloatingPrecision='fp32'|'fp16'|'bf16';
export function roundToFloat16(value:number) {
  const float=new Float32Array(1),bits=new Uint32Array(float.buffer);float[0]=value;
  const word=bits[0],sign=(word>>>16)&0x8000,exponent=(word>>>23)&0xff,mantissa=word&0x7fffff;
  let half:number;
  if(exponent===0xff)half=sign|0x7c00|(mantissa?0x200:0);
  else {
    const halfExponent=exponent-127+15;
    if(halfExponent>=31)half=sign|0x7c00;
    else if(halfExponent<=0){
      if(halfExponent<-10)half=sign;
      else {const fullMantissa=mantissa|0x800000,shift=14-halfExponent,base=fullMantissa>>>shift,remainder=fullMantissa&((1<<shift)-1),halfway=1<<(shift-1);half=sign|base;if(remainder>halfway||(remainder===halfway&&(base&1)))half++;}
    } else {const base=sign|(halfExponent<<10)|(mantissa>>>13),remainder=mantissa&0x1fff;half=base;if(remainder>0x1000||(remainder===0x1000&&(base&1)))half++;}
  }
  const halfSign=(half&0x8000)<<16,halfExponent=(half>>>10)&0x1f,halfMantissa=half&0x3ff;
  if(halfExponent===0)bits[0]=halfMantissa===0?halfSign:halfSign|((127-15-9+Math.floor(Math.log2(halfMantissa)))<<23)|((halfMantissa<<(23-Math.floor(Math.log2(halfMantissa))))&0x7fffff);
  else if(halfExponent===0x1f)bits[0]=halfSign|0x7f800000|(halfMantissa?0x400000:0);
  else bits[0]=halfSign|((halfExponent-15+127)<<23)|(halfMantissa<<13);
  return float[0];
}

export function roundToBFloat16(value:number) {
  const float=new Float32Array(1),bits=new Uint32Array(float.buffer);float[0]=value;
  const word=bits[0],roundingBias=0x7fff+((word>>>16)&1);bits[0]=(word+roundingBias)&0xffff0000;
  return float[0];
}

export const roundForPrecision=(value:number,precision:FloatingPrecision)=>precision==='fp32'?Math.fround(value):precision==='fp16'?roundToFloat16(value):roundToBFloat16(value);
export function matrix2x2Trace(left:number[],right:number[],precision:FloatingPrecision) {
  if(left.length!==4||right.length!==4)throw new RangeError('both matrices must be 2 × 2');
  const roundedLeft=left.map(value=>roundForPrecision(value,precision)),roundedRight=right.map(value=>roundForPrecision(value,precision));
  const multiply=(a:number[],b:number[])=>[a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3]];
  const reference=multiply(left,right).map(Math.fround),output=multiply(roundedLeft,roundedRight).map(value=>roundForPrecision(value,precision));
  return {roundedLeft,roundedRight,reference,output,maxAbsoluteError:Math.max(...output.map((value,index)=>Math.abs(value-reference[index]))),bytesPerElement:precision==='fp32'?4:2};
}

export function gradientScalingTrace(gradient:number,scale:number,precision:FloatingPrecision='fp16') {
  const withoutScaling=roundForPrecision(gradient,precision),scaledGradient=gradient*scale,storedScaledGradient=roundForPrecision(scaledGradient,precision);
  const finite=Number.isFinite(storedScaledGradient),restoredGradient=finite?storedScaledGradient/scale:NaN;
  return {gradient,scale,withoutScaling,scaledGradient,storedScaledGradient,restoredGradient,finite};
}

export type ControlFlowMode='eager'|'python-flag'|'tensor-if'|'torch-cond';
export function controlFlowCompilePlan(mode:ControlFlowMode,shape:number,predicate:boolean) {
  const branch=predicate?'positive':'negative';
  if(mode==='eager')return {branch,cacheKey:null,graphBreak:false,guard:null,regions:['runtime forward path'],capturesBothBranches:false};
  if(mode==='python-flag')return {branch,cacheKey:`shape:${shape}|flag:${predicate}`,graphBreak:false,guard:`shape == [${shape}] and flag is ${predicate}`,regions:[`${branch} FX graph`],capturesBothBranches:false};
  if(mode==='tensor-if')return {branch,cacheKey:`shape:${shape}|branch:${branch}`,graphBreak:true,guard:`shape == [${shape}]`,regions:['prefix FX graph','Python branch decision',`${branch} continuation graph`],capturesBothBranches:false};
  return {branch,cacheKey:`shape:${shape}|cond`,graphBreak:false,guard:`shape == [${shape}]`,regions:['FX graph with cond','true branch subgraph','false branch subgraph'],capturesBothBranches:true};
}

export type CompileCornerCase='fixed-loop'|'tensor-loop'|'scalar-item'|'shape-branch'|'cond-contract';
export function compileCornerCasePlan(kind:CompileCornerCase,option:number|boolean) {
  if(kind==='fixed-loop'){
    const iterations=Number(option);
    return {outcome:'captured',guard:`steps == ${iterations}`,graphBreak:false,nodeCount:iterations*2,regions:Array.from({length:iterations},(_,index)=>`iteration ${index+1}`)};
  }
  if(kind==='tensor-loop'){
    const structured=Boolean(option);
    return structured
      ? {outcome:'inference only',guard:'carried tensor metadata stays compatible',graphBreak:false,nodeCount:2,regions:['while_loop condition subgraph','while_loop body subgraph']}
      : {outcome:'Python loop',guard:null,graphBreak:true,nodeCount:2,regions:['compiled prefix','Python tests tensor each iteration']};
  }
  if(kind==='scalar-item'){
    const captureScalar=Boolean(option);
    return captureScalar
      ? {outcome:'scalar captured',guard:'backend must support the scalar expression',graphBreak:false,nodeCount:3,regions:['sum','item as graph scalar','multiply']}
      : {outcome:'scalar escapes to Python',guard:null,graphBreak:true,nodeCount:3,regions:['sum FX region','Python scalar','multiply continuation']};
  }
  if(kind==='shape-branch'){
    const batch=Number(option),small=batch<=16;
    return {outcome:small?'small path':'large path',guard:small?'batch <= 16':'batch > 16',graphBreak:false,nodeCount:2,regions:[small?'small-batch branch':'large-batch branch','specialized FX graph']};
  }
  const compatible=Boolean(option);
  return compatible
    ? {outcome:'captured cond',guard:'branch outputs have compatible metadata',graphBreak:false,nodeCount:3,regions:['cond','true subgraph [N, D]','false subgraph [N, D]']}
    : {outcome:'capture error',guard:null,graphBreak:false,nodeCount:2,regions:['true output [N, D]','false output [N]']};
}

export type NetworkCaptureMode='eager-network'|'python-network'|'cond-network';
export function networkGraphBreakPlan(mode:NetworkCaptureMode,predicate:boolean) {
  const expert=predicate?'expert A':'expert B';
  if(mode==='eager-network')return {expert,graphBreaks:0,capturedBranches:1,regions:['one runtime autograd path'],backward:['loss','head',expert,'stem']};
  if(mode==='python-network')return {expert,graphBreaks:1,capturedBranches:1,regions:['stem + predicate FX','Python decision',`${expert} + head FX`],backward:['loss','compiled head',`compiled ${expert}`,'Python boundary','compiled stem']};
  return {expert,graphBreaks:0,capturedBranches:2,regions:['stem','cond with expert A + expert B','head'],backward:['loss','head',`selected ${expert}`,'stem']};
}
