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
