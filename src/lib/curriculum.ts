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
