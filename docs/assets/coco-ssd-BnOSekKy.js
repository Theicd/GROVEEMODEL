function jp(t,e){for(var n=0;n<e.length;n++){const s=e[n];if(typeof s!="string"&&!Array.isArray(s)){for(const r in s)if(r!=="default"&&!(r in t)){const a=Object.getOwnPropertyDescriptor(s,r);a&&Object.defineProperty(t,r,a.get?a:{enumerable:!0,get:()=>s[r]})}}}return Object.freeze(Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}))}var pt=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function Mp(t){return t&&t.__esModule&&Object.prototype.hasOwnProperty.call(t,"default")?t.default:t}function cr(t){if(t.__esModule)return t;var e=t.default;if(typeof e=="function"){var n=function s(){return this instanceof s?Reflect.construct(e,arguments,this.constructor):e.apply(this,arguments)};n.prototype=e.prototype}else n={};return Object.defineProperty(n,"__esModule",{value:!0}),Object.keys(t).forEach(function(s){var r=Object.getOwnPropertyDescriptor(t,s);Object.defineProperty(n,s,r.get?r:{enumerable:!0,get:function(){return t[s]}})}),n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wp=1e-7,Up=1e-4;class qp{constructor(e,n){this.backend=e,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,n){this.dataIdsCount++,this.data.set(e,n)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class uo{refCount(e){return me("refCount")}incRef(e){return me("incRef")}timerAvailable(){return!0}time(e){return me("time")}read(e){return me("read")}readSync(e){return me("readSync")}readToGPU(e,n){return me("readToGPU")}numDataIds(){return me("numDataIds")}disposeData(e,n){return me("disposeData")}write(e,n,s){return me("write")}move(e,n,s,r,a){return me("move")}createTensorFromGPUData(e,n,s){return me("createTensorFromGPUData")}memory(){return me("memory")}floatPrecision(){return me("floatPrecision")}epsilon(){return this.floatPrecision()===32?Wp:Up}dispose(){return me("dispose")}}function me(t){throw new Error(`'${t}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lo(t){let e=t.length,n=0;for(;e>0;)n=Math.random()*e|0,e--,qn(t,e,n)}function Gp(t,e){if(t.length!==e.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${t.length}Second array length was ${e.length}`);let n=t.length,s=0;for(;n>0;)s=Math.random()*n|0,n--,qn(t,n,s),qn(e,n,s)}function hn(t,e,n){return Math.max(t,Math.min(e,n))}function Hp(t){return t%2===0?t:t+1}function qn(t,e,n){const s=t[e];t[e]=t[n],t[n]=s}function Kp(t){let e=0;for(let n=0;n<t.length;n++)e+=t[n];return e}function Xp(t,e){const n=Math.random();return e*n+(1-n)*t}function Zp(t,e){let n=0;for(let s=0;s<t.length;s++){const r=Number(t[s])-Number(e[s]);n+=r*r}return n}function g(t,e){if(!t)throw new Error(typeof e=="string"?e:e())}function fe(t,e,n=""){g(Re(t,e),()=>n+` Shapes ${t} and ${e} must match`)}function Ft(t){g(t!=null,()=>"The input to the tensor constructor must be a non-null value.")}function W(t){if(t.length===0)return 1;let e=t[0];for(let n=1;n<t.length;n++)e*=t[n];return e}function Jp(t){return t.length===0}function co(t,e){if(t===e)return!0;if(t==null||e==null||t.length!==e.length)return!1;for(let n=0;n<t.length;n++)if(t[n]!==null&&e[n]!==null&&t[n]!==e[n])return!1;return!0}function Re(t,e){if(t===e)return!0;if(t==null||e==null||t.length!==e.length)return!1;for(let n=0;n<t.length;n++)if(t[n]!==e[n])return!1;return!0}function qt(t){return t%1===0}function Yp(t){if(Math.tanh!=null)return Math.tanh(t);if(t===1/0)return 1;if(t===-1/0)return-1;{const e=Math.exp(2*t);return(e-1)/(e+1)}}function Qp(t){const e=Math.ceil(Math.sqrt(t));return[e,Math.ceil(t/e)]}function ef(t){const e=new Uint32Array(t);for(let n=0;n<t;++n)e[n]=n;return lo(e),e}function ln(t,e){return e<=t.length?t:t+" ".repeat(e-t.length)}function tf(t,e=r=>0,n,s){return new Promise((r,a)=>{let o=0;const i=()=>{if(t()){r();return}o++;const u=e(o);if(n!=null&&o>=n){a();return}s!=null?s(i,u):setTimeout(i,u)};i()})}function nf(t,e){let n=1,s=-1;for(let a=0;a<t.length;++a)if(t[a]>=0)n*=t[a];else if(t[a]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${a}`);s=a}else if(t[a]<0)throw Error(`Shapes can not be < 0. Found ${t[a]} at dim ${a}`);if(s===-1){if(e>0&&e!==n)throw Error(`Size(${e}) must match the product of shape ${t}`);return t}if(n===0)throw Error(`Cannot infer the missing size in [${t}] when there are 0 elements`);if(e%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${e} / ${n}`);const r=t.slice();return r[s]=e/n,r}function kn(t,e){const n=e.length;return t=t==null?e.map((s,r)=>r):[].concat(t),g(t.every(s=>s>=-n&&s<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${t}`),g(t.every(s=>qt(s)),()=>`All values in axis param must be integers but got axis ${t}`),t.map(s=>s<0?n+s:s)}function ho(t,e){const n=[],s=[],r=e!=null&&Array.isArray(e)&&e.length===0,a=e==null||r?null:kn(e,t).sort();let o=0;for(let i=0;i<t.length;++i){if(a!=null){if(a[o]===i&&t[i]!==1)throw new Error(`Can't squeeze axis ${i} since its dim '${t[i]}' is not 1`);(a[o]==null||a[o]>i)&&t[i]===1&&(n.push(t[i]),s.push(i)),a[o]<=i&&o++}t[i]!==1&&(n.push(t[i]),s.push(i))}return{newShape:n,keptDims:s}}function po(t,e){return hr(t,e)}function hr(t,e){let n=null;if(t==null||t==="float32")n=new Float32Array(e);else if(t==="int32")n=new Int32Array(e);else if(t==="bool")n=new Uint8Array(e);else if(t==="string")n=new Array(e);else throw new Error(`Unknown data type ${t}`);return n}function fo(t,e){for(let n=0;n<t.length;n++){const s=t[n];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${e} being uploaded contains ${s}.`)}}function mo(t){return t==="bool"||t==="complex64"||t==="float32"||t==="int32"||t==="string"}function sf(t,e){return!(e==="complex64"||e==="float32"&&t!=="complex64"||e==="int32"&&t!=="float32"&&t!=="complex64"||e==="bool"&&t==="bool")}function Gn(t){if(t==="float32"||t==="int32")return 4;if(t==="complex64")return 8;if(t==="bool")return 1;throw new Error(`Unknown dtype ${t}`)}function go(t){if(t==null)return 0;let e=0;return t.forEach(n=>e+=n.length),e}function nt(t){return typeof t=="string"||t instanceof String}function yo(t){return typeof t=="boolean"}function bo(t){return typeof t=="number"}function vn(t){return Array.isArray(t)?vn(t[0]):t instanceof Float32Array?"float32":t instanceof Int32Array||t instanceof Uint8Array||t instanceof Uint8ClampedArray?"int32":bo(t)?"float32":nt(t)?"string":yo(t)?"bool":"float32"}function ot(t){return!!(t&&t.constructor&&t.call&&t.apply)}function Hn(t,e){for(let n=e;n<t;++n)if(t%n===0)return n;return t}function en(t){const e=t.length;if(e<2)return[];const n=new Array(e-1);n[e-2]=t[e-1];for(let s=e-3;s>=0;--s)n[s]=n[s+1]*t[s+1];return n}function wo(t,e,n,s=!1){const r=new Array;if(e.length===1){const a=e[0]*(s?2:1);for(let o=0;o<a;o++)r[o]=n[t+o]}else{const a=e[0],o=e.slice(1),i=o.reduce((u,l)=>u*l)*(s?2:1);for(let u=0;u<a;u++)r[u]=wo(t+u*i,o,n,s)}return r}function $t(t,e,n=!1){if(t.length===0)return e[0];const s=t.reduce((r,a)=>r*a)*(n?2:1);if(s===0)return[];if(s!==e.length)throw new Error(`[${t}] does not match the input size ${e.length}${n?" for a complex tensor":""}.`);return wo(0,t,e,n)}function rf(t,e){if(Array.isArray(t))return t;if(e==="float32")return t instanceof Float32Array?t:new Float32Array(t);if(e==="int32")return t instanceof Int32Array?t:new Int32Array(t);if(e==="bool"||e==="string")return Uint8Array.from(new Int32Array(t));throw new Error(`Unknown dtype ${e}`)}function pr(t,e){const n=os(t,e);for(let s=0;s<n.length;s++)n[s]=1;return n}function os(t,e){if(e==null||e==="float32"||e==="complex64")return new Float32Array(t);if(e==="int32")return new Int32Array(t);if(e==="bool")return new Uint8Array(t);throw new Error(`Unknown data type ${e}`)}function af(t,e){const n=t.reduce((s,r)=>s*r,1);if(e==null||e==="float32")return $t(t,new Float32Array(n));if(e==="int32")return $t(t,new Int32Array(n));if(e==="bool")return $t(t,new Uint8Array(n));throw new Error(`Unknown data type ${e}`)}function Se(t){t.forEach(e=>{g(Number.isInteger(e)&&e>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${t}].`)})}function of(t,e,n){if(e===0)return 0;if(e===1)return t[0];let s=t[t.length-1];for(let r=0;r<t.length-1;++r)s+=n[r]*t[r];return s}function uf(t,e,n){if(e===0)return[];if(e===1)return[t];const s=new Array(e);for(let r=0;r<s.length-1;++r)s[r]=Math.floor(t/n[r]),t-=s[r]*n[r];return s[s.length-1]=t,s}function it(t){return t&&t.then&&typeof t.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Da="tfjsflags";class No{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=lf,this.populateURLFlags()}setPlatform(e,n){this.platform!=null&&(R().getBool("IS_TEST")||R().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=n}registerFlag(e,n,s){if(this.flagRegistry[e]={evaluationFn:n,setHook:s},this.urlFlags[e]!=null){const r=this.urlFlags[e];R().getBool("IS_TEST")||R().getBool("PROD")||console.warn(`Setting feature override from URL ${e}: ${r}.`),this.set(e,r)}}async getAsync(e){return e in this.flags?this.flags[e]:(this.flags[e]=await this.evaluateFlag(e),this.flags[e])}get(e){if(e in this.flags)return this.flags[e];const n=this.evaluateFlag(e);if(it(n))throw new Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=n,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,n){if(this.flagRegistry[e]==null)throw new Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=n,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(n)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw new Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const e=this.getQueryParams(this.global.location.search);Da in e&&e[Da].split(",").forEach(s=>{const[r,a]=s.split(":");this.urlFlags[r]=hf(r,a)})}}function lf(t){const e={};return t.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...s)=>(cf(e,s[0],s[1]),s.join("="))),e}function cf(t,e,n){t[decodeURIComponent(e)]=decodeURIComponent(n||"")}function hf(t,e){const n=e.toLowerCase();return n==="true"||n==="false"?n==="true":`${+n}`===n?+n:e}function R(){return fr}let fr=null;function pf(t){fr=t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Es;function So(){if(Es==null){let t;if(typeof window<"u")t=window;else if(typeof global<"u")t=global;else if(typeof process<"u")t=process;else if(typeof self<"u")t=self;else throw new Error("Could not find a global object");Es=t}return Es}function ff(){const t=So();return t._tfGlobals==null&&(t._tfGlobals=new Map),t._tfGlobals}function dr(t,e){const n=ff();if(n.has(t))return n.get(t);{const s=e();return n.set(t,s),n.get(t)}}const To="Abs",$o="Acos",Eo="Acosh",mr="Add",ko="AddN",vo="All",_o="Any",Io="ArgMax",xo="ArgMin",Ao="Asin",Oo="Asinh",Do="Atan",Fo="Atanh",Co="Atan2",Ro="AvgPool",df="AvgPoolGrad",Po="AvgPool3D",mf="AvgPool3DGrad",Bo="BatchMatMul",Lo="BatchToSpaceND",zo="Bincount",Vo="BitwiseAnd",gf="BroadcastTo",jo="BroadcastArgs",gr="Cast",Mo="Ceil",Wo="ClipByValue",Uo="Complex",qo="ComplexAbs",Go="Concat",Ho="Conv2D",Ko="Conv2DBackpropFilter",Xo="Conv2DBackpropInput",Zo="Conv3D",yf="Conv3DBackpropFilterV2",Jo="Conv3DBackpropInputV2",Yo="Cos",Qo="Cosh",ei="Cumprod",ti="Cumsum",ni="CropAndResize",si="DenseBincount",ri="DepthToSpace",ai="DepthwiseConv2dNative",oi="DepthwiseConv2dNativeBackpropFilter",ii="DepthwiseConv2dNativeBackpropInput",ui="Diag",li="Dilation2D",bf="Dilation2DBackpropInput",wf="Dilation2DBackpropFilter",yr="Draw",ci="RealDiv",hi="Einsum",pi="Elu",Nf="EluGrad",fi="Erf",di="Equal",mi="Exp",gi="ExpandDims",yi="Expm1",bi="FFT",wi="Fill",Ni="FlipLeftRight",Si="Floor",Ti="FloorDiv",$i="FusedBatchNorm",Ei="GatherV2",ki="GatherNd",vi="Greater",_i="GreaterEqual",br="Identity",Ii="IFFT",xi="Imag",Ai="IsFinite",Oi="IsInf",Di="IsNan",Fi="LeakyRelu",Ci="Less",Ri="LessEqual",Pi="LinSpace",Bi="Log",Li="Log1p",zi="LogicalAnd",Vi="LogicalNot",ji="LogicalOr",Sf="LogicalXor",Tf="LogSoftmax",$f="LowerBound",Mi="LRN",Ef="LRNGrad",kf="MatrixBandPart",Wi="Max",Ui="Maximum",qi="MaxPool",vf="MaxPoolGrad",Gi="MaxPool3D",_f="MaxPool3DGrad",Hi="MaxPoolWithArgmax",Ki="Mean",Xi="Min",Zi="Minimum",Ji="MirrorPad",Yi="Mod",Qi="Multinomial",eu="Multiply",tu="Neg",nu="NotEqual",su="NonMaxSuppressionV3",ru="NonMaxSuppressionV4",au="NonMaxSuppressionV5",ou="OnesLike",iu="OneHot",uu="Pack",lu="PadV2",If="Pool",cu="Pow",hu="Prelu",pu="Prod",fu="RaggedGather",du="RaggedRange",mu="RaggedTensorToTensor",gu="Range",yu="Real",bu="Reciprocal",wu="Relu",Nu="Reshape",Su="ResizeNearestNeighbor",xf="ResizeNearestNeighborGrad",Tu="ResizeBilinear",Af="ResizeBilinearGrad",$u="Relu6",Eu="Reverse",ku="Round",vu="Rsqrt",_u="ScatterNd",Iu="TensorScatterUpdate",xu="SearchSorted",Au="Select",Ou="Selu",Du="Slice",Fu="Sin",Cu="Sinh",Ru="Sign",Pu="Sigmoid",Bu="Softplus",Lu="Sqrt",zu="Sum",Vu="SpaceToBatchND",ju="SplitV",Mu="Softmax",Wu="SparseFillEmptyRows",Uu="SparseReshape",qu="SparseSegmentMean",Gu="SparseSegmentSum",Hu="SparseToDense",Ku="SquaredDifference",Of="Square",Xu="StaticRegexReplace",Zu="StridedSlice",Ju="StringNGrams",Yu="StringSplit",Qu="StringToHashBucketFast",el="Sub",tl="Tan",nl="Tanh",wr="Tile",sl="TopK",rl="Transform",jn="Transpose",al="Unique",ol="Unpack",il="UnsortedSegmentSum",Df="UpperBound",ul="ZerosLike",ll="Step",Os="FromPixels",cl="RotateWithOffset",Ds="_FusedMatMul",Fs="FusedConv2D",Cs="FusedDepthwiseConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function et(...t){R().getBool("IS_TEST")||R().getBool("PROD")||console.warn(...t)}function Ff(...t){R().getBool("IS_TEST")||R().getBool("PROD")||console.log(...t)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gt=dr("kernelRegistry",()=>new Map),pn=dr("gradRegistry",()=>new Map);function fn(t,e){const n=Nr(t,e);return Gt.get(n)}function Rs(t){return pn.get(t)}function Kn(t){const e=Gt.entries(),n=[];for(;;){const{done:s,value:r}=e.next();if(s)break;const[a,o]=r,[i]=a.split("_");i===t&&n.push(o)}return n}function hl(t){const{kernelName:e,backendName:n}=t,s=Nr(e,n);Gt.has(s)&&et(`The kernel '${e}' for backend '${n}' is already registered`),Gt.set(s,t)}function Cf(t){const{kernelName:e}=t;pn.has(e)&&R().getBool("DEBUG")&&et(`Overriding the gradient for '${e}'`),pn.set(e,t)}function Rf(t,e){const n=Nr(t,e);if(!Gt.has(n))throw new Error(`The kernel '${t}' for backend '${e}' is not registered`);Gt.delete(n)}function Pf(t){if(!pn.has(t))throw new Error(`The gradient '${t}' for backend is not registered`);pn.delete(t)}function Bf(t,e){Kn(t).forEach(s=>{const r=Object.assign({},s,{backendName:e});hl(r)})}function Nr(t,e){return`${e}_${t}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pl(t){return t instanceof Float32Array||t instanceof Int32Array||t instanceof Uint8Array||t instanceof Uint8ClampedArray}var fl=Z,ve=null;try{ve=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function Z(t,e,n){this.low=t|0,this.high=e|0,this.unsigned=!!n}Z.prototype.__isLong__;Object.defineProperty(Z.prototype,"__isLong__",{value:!0});function Te(t){return(t&&t.__isLong__)===!0}Z.isLong=Te;var Fa={},Ca={};function Ct(t,e){var n,s,r;return e?(t>>>=0,(r=0<=t&&t<256)&&(s=Ca[t],s)?s:(n=J(t,(t|0)<0?-1:0,!0),r&&(Ca[t]=n),n)):(t|=0,(r=-128<=t&&t<128)&&(s=Fa[t],s)?s:(n=J(t,t<0?-1:0,!1),r&&(Fa[t]=n),n))}Z.fromInt=Ct;function _e(t,e){if(isNaN(t))return e?St:Ie;if(e){if(t<0)return St;if(t>=dl)return yl}else{if(t<=-Pa)return we;if(t+1>=Pa)return gl}return t<0?_e(-t,e).neg():J(t%Ht|0,t/Ht|0,e)}Z.fromNumber=_e;function J(t,e,n){return new Z(t,e,n)}Z.fromBits=J;var Xn=Math.pow;function Sr(t,e,n){if(t.length===0)throw Error("empty string");if(t==="NaN"||t==="Infinity"||t==="+Infinity"||t==="-Infinity")return Ie;if(typeof e=="number"?(n=e,e=!1):e=!!e,n=n||10,n<2||36<n)throw RangeError("radix");var s;if((s=t.indexOf("-"))>0)throw Error("interior hyphen");if(s===0)return Sr(t.substring(1),e,n).neg();for(var r=_e(Xn(n,8)),a=Ie,o=0;o<t.length;o+=8){var i=Math.min(8,t.length-o),u=parseInt(t.substring(o,o+i),n);if(i<8){var l=_e(Xn(n,i));a=a.mul(l).add(_e(u))}else a=a.mul(r),a=a.add(_e(u))}return a.unsigned=e,a}Z.fromString=Sr;function Pe(t,e){return typeof t=="number"?_e(t,e):typeof t=="string"?Sr(t,e):J(t.low,t.high,typeof e=="boolean"?e:t.unsigned)}Z.fromValue=Pe;var Ra=65536,Lf=1<<24,Ht=Ra*Ra,dl=Ht*Ht,Pa=dl/2,Ba=Ct(Lf),Ie=Ct(0);Z.ZERO=Ie;var St=Ct(0,!0);Z.UZERO=St;var Vt=Ct(1);Z.ONE=Vt;var ml=Ct(1,!0);Z.UONE=ml;var Ps=Ct(-1);Z.NEG_ONE=Ps;var gl=J(-1,2147483647,!1);Z.MAX_VALUE=gl;var yl=J(-1,-1,!0);Z.MAX_UNSIGNED_VALUE=yl;var we=J(0,-2147483648,!1);Z.MIN_VALUE=we;var I=Z.prototype;I.toInt=function(){return this.unsigned?this.low>>>0:this.low};I.toNumber=function(){return this.unsigned?(this.high>>>0)*Ht+(this.low>>>0):this.high*Ht+(this.low>>>0)};I.toString=function(e){if(e=e||10,e<2||36<e)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(we)){var n=_e(e),s=this.div(n),r=s.mul(n).sub(this);return s.toString(e)+r.toInt().toString(e)}else return"-"+this.neg().toString(e);for(var a=_e(Xn(e,6),this.unsigned),o=this,i="";;){var u=o.div(a),l=o.sub(u.mul(a)).toInt()>>>0,h=l.toString(e);if(o=u,o.isZero())return h+i;for(;h.length<6;)h="0"+h;i=""+h+i}};I.getHighBits=function(){return this.high};I.getHighBitsUnsigned=function(){return this.high>>>0};I.getLowBits=function(){return this.low};I.getLowBitsUnsigned=function(){return this.low>>>0};I.getNumBitsAbs=function(){if(this.isNegative())return this.eq(we)?64:this.neg().getNumBitsAbs();for(var e=this.high!=0?this.high:this.low,n=31;n>0&&!(e&1<<n);n--);return this.high!=0?n+33:n+1};I.isZero=function(){return this.high===0&&this.low===0};I.eqz=I.isZero;I.isNegative=function(){return!this.unsigned&&this.high<0};I.isPositive=function(){return this.unsigned||this.high>=0};I.isOdd=function(){return(this.low&1)===1};I.isEven=function(){return(this.low&1)===0};I.equals=function(e){return Te(e)||(e=Pe(e)),this.unsigned!==e.unsigned&&this.high>>>31===1&&e.high>>>31===1?!1:this.high===e.high&&this.low===e.low};I.eq=I.equals;I.notEquals=function(e){return!this.eq(e)};I.neq=I.notEquals;I.ne=I.notEquals;I.lessThan=function(e){return this.comp(e)<0};I.lt=I.lessThan;I.lessThanOrEqual=function(e){return this.comp(e)<=0};I.lte=I.lessThanOrEqual;I.le=I.lessThanOrEqual;I.greaterThan=function(e){return this.comp(e)>0};I.gt=I.greaterThan;I.greaterThanOrEqual=function(e){return this.comp(e)>=0};I.gte=I.greaterThanOrEqual;I.ge=I.greaterThanOrEqual;I.compare=function(e){if(Te(e)||(e=Pe(e)),this.eq(e))return 0;var n=this.isNegative(),s=e.isNegative();return n&&!s?-1:!n&&s?1:this.unsigned?e.high>>>0>this.high>>>0||e.high===this.high&&e.low>>>0>this.low>>>0?-1:1:this.sub(e).isNegative()?-1:1};I.comp=I.compare;I.negate=function(){return!this.unsigned&&this.eq(we)?we:this.not().add(Vt)};I.neg=I.negate;I.add=function(e){Te(e)||(e=Pe(e));var n=this.high>>>16,s=this.high&65535,r=this.low>>>16,a=this.low&65535,o=e.high>>>16,i=e.high&65535,u=e.low>>>16,l=e.low&65535,h=0,c=0,f=0,d=0;return d+=a+l,f+=d>>>16,d&=65535,f+=r+u,c+=f>>>16,f&=65535,c+=s+i,h+=c>>>16,c&=65535,h+=n+o,h&=65535,J(f<<16|d,h<<16|c,this.unsigned)};I.subtract=function(e){return Te(e)||(e=Pe(e)),this.add(e.neg())};I.sub=I.subtract;I.multiply=function(e){if(this.isZero())return Ie;if(Te(e)||(e=Pe(e)),ve){var n=ve.mul(this.low,this.high,e.low,e.high);return J(n,ve.get_high(),this.unsigned)}if(e.isZero())return Ie;if(this.eq(we))return e.isOdd()?we:Ie;if(e.eq(we))return this.isOdd()?we:Ie;if(this.isNegative())return e.isNegative()?this.neg().mul(e.neg()):this.neg().mul(e).neg();if(e.isNegative())return this.mul(e.neg()).neg();if(this.lt(Ba)&&e.lt(Ba))return _e(this.toNumber()*e.toNumber(),this.unsigned);var s=this.high>>>16,r=this.high&65535,a=this.low>>>16,o=this.low&65535,i=e.high>>>16,u=e.high&65535,l=e.low>>>16,h=e.low&65535,c=0,f=0,d=0,y=0;return y+=o*h,d+=y>>>16,y&=65535,d+=a*h,f+=d>>>16,d&=65535,d+=o*l,f+=d>>>16,d&=65535,f+=r*h,c+=f>>>16,f&=65535,f+=a*l,c+=f>>>16,f&=65535,f+=o*u,c+=f>>>16,f&=65535,c+=s*h+r*l+a*u+o*i,c&=65535,J(d<<16|y,c<<16|f,this.unsigned)};I.mul=I.multiply;I.divide=function(e){if(Te(e)||(e=Pe(e)),e.isZero())throw Error("division by zero");if(ve){if(!this.unsigned&&this.high===-2147483648&&e.low===-1&&e.high===-1)return this;var n=(this.unsigned?ve.div_u:ve.div_s)(this.low,this.high,e.low,e.high);return J(n,ve.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?St:Ie;var s,r,a;if(this.unsigned){if(e.unsigned||(e=e.toUnsigned()),e.gt(this))return St;if(e.gt(this.shru(1)))return ml;a=St}else{if(this.eq(we)){if(e.eq(Vt)||e.eq(Ps))return we;if(e.eq(we))return Vt;var o=this.shr(1);return s=o.div(e).shl(1),s.eq(Ie)?e.isNegative()?Vt:Ps:(r=this.sub(e.mul(s)),a=s.add(r.div(e)),a)}else if(e.eq(we))return this.unsigned?St:Ie;if(this.isNegative())return e.isNegative()?this.neg().div(e.neg()):this.neg().div(e).neg();if(e.isNegative())return this.div(e.neg()).neg();a=Ie}for(r=this;r.gte(e);){s=Math.max(1,Math.floor(r.toNumber()/e.toNumber()));for(var i=Math.ceil(Math.log(s)/Math.LN2),u=i<=48?1:Xn(2,i-48),l=_e(s),h=l.mul(e);h.isNegative()||h.gt(r);)s-=u,l=_e(s,this.unsigned),h=l.mul(e);l.isZero()&&(l=Vt),a=a.add(l),r=r.sub(h)}return a};I.div=I.divide;I.modulo=function(e){if(Te(e)||(e=Pe(e)),ve){var n=(this.unsigned?ve.rem_u:ve.rem_s)(this.low,this.high,e.low,e.high);return J(n,ve.get_high(),this.unsigned)}return this.sub(this.div(e).mul(e))};I.mod=I.modulo;I.rem=I.modulo;I.not=function(){return J(~this.low,~this.high,this.unsigned)};I.and=function(e){return Te(e)||(e=Pe(e)),J(this.low&e.low,this.high&e.high,this.unsigned)};I.or=function(e){return Te(e)||(e=Pe(e)),J(this.low|e.low,this.high|e.high,this.unsigned)};I.xor=function(e){return Te(e)||(e=Pe(e)),J(this.low^e.low,this.high^e.high,this.unsigned)};I.shiftLeft=function(e){return Te(e)&&(e=e.toInt()),(e&=63)===0?this:e<32?J(this.low<<e,this.high<<e|this.low>>>32-e,this.unsigned):J(0,this.low<<e-32,this.unsigned)};I.shl=I.shiftLeft;I.shiftRight=function(e){return Te(e)&&(e=e.toInt()),(e&=63)===0?this:e<32?J(this.low>>>e|this.high<<32-e,this.high>>e,this.unsigned):J(this.high>>e-32,this.high>=0?0:-1,this.unsigned)};I.shr=I.shiftRight;I.shiftRightUnsigned=function(e){if(Te(e)&&(e=e.toInt()),e&=63,e===0)return this;var n=this.high;if(e<32){var s=this.low;return J(s>>>e|n<<32-e,n>>>e,this.unsigned)}else return e===32?J(n,0,this.unsigned):J(n>>>e-32,0,this.unsigned)};I.shru=I.shiftRightUnsigned;I.shr_u=I.shiftRightUnsigned;I.toSigned=function(){return this.unsigned?J(this.low,this.high,!1):this};I.toUnsigned=function(){return this.unsigned?this:J(this.low,this.high,!0)};I.toBytes=function(e){return e?this.toBytesLE():this.toBytesBE()};I.toBytesLE=function(){var e=this.high,n=this.low;return[n&255,n>>>8&255,n>>>16&255,n>>>24,e&255,e>>>8&255,e>>>16&255,e>>>24]};I.toBytesBE=function(){var e=this.high,n=this.low;return[e>>>24,e>>>16&255,e>>>8&255,e&255,n>>>24,n>>>16&255,n>>>8&255,n&255]};Z.fromBytes=function(e,n,s){return s?Z.fromBytesLE(e,n):Z.fromBytesBE(e,n)};Z.fromBytesLE=function(e,n){return new Z(e[0]|e[1]<<8|e[2]<<16|e[3]<<24,e[4]|e[5]<<8|e[6]<<16|e[7]<<24,n)};Z.fromBytesBE=function(e,n){return new Z(e[4]<<24|e[5]<<16|e[6]<<8|e[7],e[0]<<24|e[1]<<16|e[2]<<8|e[3],n)};const bl=Mp(fl),zf=jp({__proto__:null,default:bl},[fl]);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bt=bl||zf;function _n(t){return bt.fromString(t,!0,16)}const wl=_n("c3a5c85c97cb3127"),yt=_n("b492b66fbe98f273"),ce=_n("9ae16a3b2f90404f");function Bs(t){return t.xor(t.shru(47))}function Nl(t,e,n){const s=t.slice(e,e+n);return bt.fromBytes(Array.from(s),!0,!0)}function G(t,e){return Nl(t,e,8)}function La(t,e){return Nl(t,e,4)}function re(t,e){return e===0?t:t.shru(e).or(t.shl(64-e))}function at(t,e,n=_n("9ddfea08eb382d69")){let s=t.xor(e).mul(n);s=s.xor(s.shru(47));let r=e.xor(s).mul(n);return r=r.xor(r.shru(47)),r=r.mul(n),r}function Vf(t,e,n,s,r,a){r=r.add(t),a=re(a.add(r).add(s),21);const o=r;return r=r.add(e),r=r.add(n),a=a.add(re(r,44)),[r.add(s),a.add(o)]}function Ln(t,e,n,s){return Vf(G(t,e),G(t,e+8),G(t,e+16),G(t,e+24),n,s)}function jf(t,e=t.length){if(e>=8){const n=ce.add(e*2),s=G(t,0).add(ce),r=G(t,e-8),a=re(r,37).mul(n).add(s),o=re(s,25).add(r).mul(n);return at(a,o,n)}if(e>=4){const n=ce.add(e*2),s=La(t,0);return at(s.shl(3).add(e),La(t,e-4),n)}if(e>0){const n=t[0],s=t[e>>1],r=t[e-1],a=n+(s<<8),o=e+(r<<2);return Bs(ce.mul(a).xor(wl.mul(o))).mul(ce)}return ce}function Mf(t,e=t.length){const n=ce.add(e*2),s=G(t,0).mul(yt),r=G(t,8),a=G(t,e-8).mul(n),o=G(t,e-16).mul(ce);return at(re(s.add(r),43).add(re(a,30)).add(o),s.add(re(r.add(ce),18)).add(a),n)}function Wf(t,e=t.length){const n=ce.add(e*2),s=G(t,0).mul(ce),r=G(t,8),a=G(t,e-8).mul(n),o=G(t,e-16).mul(ce),i=re(s.add(r),43).add(re(a,30)).add(o),u=at(i,s.add(re(r.add(ce),18)).add(a),n),l=G(t,16).mul(n),h=G(t,24),c=i.add(G(t,e-32)).mul(n),f=u.add(G(t,e-24)).mul(n);return at(re(l.add(h),43).add(re(c,30)).add(f),l.add(re(h.add(s),18)).add(c),n)}function Uf(t,e=t.length){const n=bt.fromNumber(81,!0);if(e<=32)return e<=16?jf(t,e):Mf(t,e);if(e<=64)return Wf(t,e);let s=n,r=n.mul(yt).add(113),a=Bs(r.mul(ce).add(113)).mul(ce),o=[bt.UZERO,bt.UZERO],i=[bt.UZERO,bt.UZERO];s=s.mul(ce).add(G(t,0));let u=0;const l=(e-1>>6)*64,h=l+(e-1&63)-63;do s=re(s.add(r).add(o[0]).add(G(t,u+8)),37).mul(yt),r=re(r.add(o[1]).add(G(t,u+48)),42).mul(yt),s=s.xor(i[1]),r=r.add(o[0]).add(G(t,u+40)),a=re(a.add(i[0]),33).mul(yt),o=Ln(t,u,o[1].mul(yt),s.add(i[0])),i=Ln(t,u+32,a.add(i[1]),r.add(G(t,u+16))),[a,s]=[s,a],u+=64;while(u!==l);const c=yt.add(a.and(255).shl(1));return u=h,i[0]=i[0].add(e-1&63),o[0]=o[0].add(i[0]),i[0]=i[0].add(o[0]),s=re(s.add(r).add(o[0]).add(G(t,u+8)),37).mul(c),r=re(r.add(o[1]).add(G(t,u+48)),42).mul(c),s=s.xor(i[1].mul(9)),r=r.add(o[0].mul(9).add(G(t,u+40))),a=re(a.add(i[0]),33).mul(c),o=Ln(t,u,o[1].mul(c),s.add(i[0])),i=Ln(t,u+32,a.add(i[1]),r.add(G(t,u+16))),[a,s]=[s,a],at(at(o[0],i[0],c).add(Bs(r).mul(wl)).add(a),at(o[1],i[1],c).add(s),c)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qf(t,e){return e==="string"?In(t):is([t],e)}function Gf(t,e){return t instanceof Float32Array&&e==="float32"||t instanceof Int32Array&&e==="int32"||t instanceof Uint8Array&&e==="bool"}function is(t,e){if(e==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(t)&&(t=ut(t)),R().getBool("DEBUG")&&fo(t,e),Gf(t,e))return t;if(e==null||e==="float32"||e==="complex64")return new Float32Array(t);if(e==="int32")return new Int32Array(t);if(e==="bool"){const n=new Uint8Array(t.length);for(let s=0;s<n.length;++s)Math.round(t[s])!==0&&(n[s]=1);return n}else throw new Error(`Unknown data type ${e}`)}function dn(){return R().platform.now()}function Hf(t,e){return R().platform.fetch(t,e)}function In(t,e="utf-8"){return e=e||"utf-8",R().platform.encode(t,e)}function Zn(t,e="utf-8"){return e=e||"utf-8",R().platform.decode(t,e)}function ae(t){return R().platform.isTypedArray!=null?R().platform.isTypedArray(t):pl(t)}function ut(t,e=[],n=!1){if(e==null&&(e=[]),typeof t=="boolean"||typeof t=="number"||typeof t=="string"||it(t)||t==null||ae(t)&&n)e.push(t);else if(Array.isArray(t)||ae(t))for(let s=0;s<t.length;++s)ut(t[s],e,n);else{let s=-1;for(const r of Object.keys(t))/^([1-9]+[0-9]*|0)$/.test(r)&&(s=Math.max(s,Number(r)));for(let r=0;r<=s;r++)ut(t[r],e,n)}return e}const Kf=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:Re,arraysEqualWithNull:co,assert:g,assertNonNegativeIntegerDimensions:Se,assertNonNull:Ft,assertShapesMatch:fe,bytesFromStringArray:go,bytesPerElement:Gn,checkConversionForErrors:fo,clamp:hn,computeStrides:en,convertBackendValuesAndArrayBuffer:rf,createScalarValue:qf,createShuffledIndices:ef,decodeString:Zn,distSquared:Zp,encodeString:In,fetch:Hf,fingerPrint64:Uf,flatten:ut,getArrayFromDType:hr,getTypedArrayFromDType:po,hasEncodingLoss:sf,hexToLong:_n,indexToLoc:uf,inferDtype:vn,inferFromImplicitShape:nf,isBoolean:yo,isFunction:ot,isInt:qt,isNumber:bo,isPromise:it,isScalarShape:Jp,isString:nt,isTypedArray:ae,isValidDtype:mo,locToIndex:of,makeOnesTypedArray:pr,makeZerosNestedTypedArray:af,makeZerosTypedArray:os,nearestDivisor:Hn,nearestLargerEven:Hp,now:dn,parseAxisParam:kn,randUniform:Xp,repeatedTry:tf,rightPad:ln,shuffle:lo,shuffleCombo:Gp,sizeFromShape:W,sizeToSquarishShape:Qp,squeezeShape:ho,sum:Kp,swap:qn,tanh:Yp,toNestedArray:$t,toTypedArray:is},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Xf{constructor(e,n){this.backendTimer=e,this.logger=n,n==null&&(this.logger=new Jf)}profileKernel(e,n,s){let r;const a=()=>{r=s()};let o;const i=dn();if(this.backendTimer.timerAvailable())o=this.backendTimer.time(a);else{a();for(const l of r)l.dataSync();o=Promise.resolve({kernelMs:dn()-i})}if(R().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let l=0;l<r.length;l++){const h=r[l];h.data().then(c=>{Zf(c,h.dtype,e)})}return{kernelName:e,outputs:r,inputs:n,timeMs:o.then(l=>l.kernelMs),extraInfo:o.then(l=>l.getExtraProfileInfo!=null?l.getExtraProfileInfo():"")}}logKernelProfile(e){const{kernelName:n,outputs:s,timeMs:r,inputs:a,extraInfo:o}=e;s.forEach(i=>{Promise.all([i.data(),r,o]).then(u=>{this.logger.logKernelProfile(n,i,u[0],u[1],a,u[2])})})}}function Zf(t,e,n){if(e!=="float32")return!1;for(let s=0;s<t.length;s++){const r=t[s];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}class Jf{logKernelProfile(e,n,s,r,a,o){const i=typeof r=="number"?ln(`${r}ms`,9):r.error,u=ln(e,25),l=n.rank,h=n.size,c=ln(n.shape.toString(),14);let f="";for(const d in a){const y=a[d];if(y!=null){const S=y.shape||n.shape,b=S.length;f+=`${d}: ${b}D ${b>0?S:""} `}}console.log(`%c${u}	%c${i}	%c${l}D ${c}	%c${h}	%c${f}	%c${o}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yf(t,e,n){const s={},r={};for(let u=0;u<e.length;u++)s[e[u].id]=!0;for(let u=0;u<t.length;u++){const l=t[u],h=l.inputs;for(const c in h){const f=h[c];let d=!1;for(let y=0;y<e.length;y++)if(s[f.id]){l.outputs.forEach(S=>s[S.id]=!0),d=!0,r[l.id]=!0;break}if(d)break}}const a={};a[n.id]=!0;const o={};for(let u=t.length-1;u>=0;u--){const l=t[u],h=l.inputs;for(let c=0;c<l.outputs.length;c++)if(a[l.outputs[c].id]){for(const f in h)a[h[f].id]=!0,o[l.id]=!0;break}}const i=[];for(let u=0;u<t.length;u++){const l=t[u];if(r[l.id]&&o[l.id]){const h={};for(const f in l.inputs){const d=l.inputs[f];s[d.id]&&(h[f]=d)}const c=Object.assign({},l);c.inputs=h,c.outputs=l.outputs,i.push(c)}}return i}function Qf(t,e,n,s){for(let r=e.length-1;r>=0;r--){const a=e[r],o=[];if(a.outputs.forEach(u=>{const l=t[u.id];l!=null?o.push(l):o.push(null)}),a.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${a.kernelName}.`);const i=a.gradient(o);for(const u in a.inputs){if(!(u in i))throw new Error(`Cannot backprop through input ${u}. Available gradients found: ${Object.keys(i)}.`);const l=n(()=>i[u]());if(l.dtype!=="float32")throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input ${u} must have 'float32' dtype, but has '${l.dtype}'`);const h=a.inputs[u];if(!Re(l.shape,h.shape))throw new Error(`Error in gradient for op ${a.kernelName}. The gradient of input '${u}' has shape '${l.shape}', which does not match the shape of the input '${h.shape}'`);if(t[h.id]==null)t[h.id]=l;else{const c=t[h.id];t[h.id]=s(c,l),c.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const za=20,rn=3,ks=7;function ed(t,e,n,s){const r=en(e),a=td(t,e,n,r),o=e.length,i=Mn(t,e,n,r,a),u=["Tensor"];return s&&(u.push(`  dtype: ${n}`),u.push(`  rank: ${o}`),u.push(`  shape: [${e}]`),u.push("  values:")),u.push(i.map(l=>"    "+l).join(`
`)),u.join(`
`)}function td(t,e,n,s){const r=W(e),a=s[s.length-1],o=new Array(a).fill(0),i=e.length,u=n==="complex64"?un(t):t;if(i>1)for(let l=0;l<r/a;l++){const h=l*a;for(let c=0;c<a;c++)o[c]=Math.max(o[c],on(u[h+c],0,n).length)}return o}function on(t,e,n){let s;return Array.isArray(t)?s=`${parseFloat(t[0].toFixed(ks))} + ${parseFloat(t[1].toFixed(ks))}j`:nt(t)?s=`'${t}'`:n==="bool"?s=Sl(t):s=parseFloat(t.toFixed(ks)).toString(),ln(s,e)}function Sl(t){return t===0?"false":"true"}function Mn(t,e,n,s,r,a=!0){const o=n==="complex64"?2:1,i=e[0],u=e.length;if(u===0){if(n==="complex64"){const S=un(t);return[on(S[0],0,n)]}return n==="bool"?[Sl(t[0])]:[t[0].toString()]}if(u===1){if(i>za){const b=rn*o;let T=Array.from(t.slice(0,b)),A=Array.from(t.slice((i-rn)*o,i*o));return n==="complex64"&&(T=un(T),A=un(A)),["["+T.map(($,E)=>on($,r[E],n)).join(", ")+", ..., "+A.map(($,E)=>on($,r[i-rn+E],n)).join(", ")+"]"]}return["["+(n==="complex64"?un(t):Array.from(t)).map((b,T)=>on(b,r[T],n)).join(", ")+"]"]}const l=e.slice(1),h=s.slice(1),c=s[0]*o,f=[];if(i>za){for(let S=0;S<rn;S++){const b=S*c,T=b+c;f.push(...Mn(t.slice(b,T),l,n,h,r,!1))}f.push("...");for(let S=i-rn;S<i;S++){const b=S*c,T=b+c;f.push(...Mn(t.slice(b,T),l,n,h,r,S===i-1))}}else for(let S=0;S<i;S++){const b=S*c,T=b+c;f.push(...Mn(t.slice(b,T),l,n,h,r,S===i-1))}const d=u===2?",":"";f[0]="["+(i>0?f[0]+d:"");for(let S=1;S<f.length-1;S++)f[S]=" "+f[S]+d;let y=`,
`;for(let S=2;S<u;S++)y+=`
`;return f[f.length-1]=" "+f[f.length-1]+"]"+(a?"":y),f}function un(t){const e=[];for(let n=0;n<t.length;n+=2)e.push([t[n],t[n+1]]);return e}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Jn{constructor(e,n,s){if(this.dtype=n,this.shape=e.slice(),this.size=W(e),s!=null){const r=s.length;g(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||hr(n,this.size),this.strides=en(e)}set(e,...n){n.length===0&&(n=[0]),g(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const s=this.locToIndex(n);this.values[s]=e}get(...e){e.length===0&&(e=[0]);let n=0;for(const r of e){if(r<0||r>=this.shape[n]){const a=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw new Error(a)}n++}let s=e[e.length-1];for(let r=0;r<e.length-1;++r)s+=this.strides[r]*e[r];return this.values[s]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let n=e[e.length-1];for(let s=0;s<e.length-1;++s)n+=this.strides[s]*e[s];return n}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];const n=new Array(this.shape.length);for(let s=0;s<n.length-1;++s)n[s]=Math.floor(e/this.strides[s]),e-=n[s]*this.strides[s];return n[n.length-1]=e,n}get rank(){return this.shape.length}toTensor(){return Oe().makeTensor(this.values,this.shape,this.dtype)}}let Oe=null,Lt=null;function nd(t){Oe=t}function sd(t){Lt=t}class te{constructor(e,n,s,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=n||"float32",this.size=W(e),this.strides=en(e),this.dataId=s,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const e=await this.data();return Lt.buffer(this.shape,this.dtype,e)}bufferSync(){return Lt.buffer(this.shape,this.dtype,this.dataSync())}async array(){const e=await this.data();return $t(this.shape,e,this.dtype==="complex64")}arraySync(){return $t(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const e=Oe().read(this.dataId);if(this.dtype==="string"){const n=await e;try{return n.map(s=>Zn(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),Oe().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();const e=Oe().readSync(this.dataId);if(this.dtype==="string")try{return e.map(n=>Zn(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();const e=await Oe().read(this.dataId);return this.dtype==="string"?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),Oe().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(e=!1){return Lt.print(this,e)}clone(){return this.throwIfDisposed(),Lt.clone(this)}toString(e=!1){const n=this.dataSync();return ed(n,this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Lt.cast(this,e)}variable(e=!0,n,s){return this.throwIfDisposed(),Oe().makeVariable(this,e,n,s)}}Object.defineProperty(te,Symbol.hasInstance,{value:t=>!!t&&t.data!=null&&t.dataSync!=null&&t.throwIfDisposed!=null});function Tl(){return dr("Tensor",()=>te)}Tl();class mn extends te{constructor(e,n,s,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=n,this.name=s}assign(e){if(e.dtype!==this.dtype)throw new Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!Re(e.shape,this.shape))throw new Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);Oe().disposeTensor(this),this.dataId=e.dataId,Oe().incRef(this,null)}dispose(){Oe().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(mn,Symbol.hasInstance,{value:t=>t instanceof te&&t.assign!=null&&t.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Ls;(function(t){t.R0="R0",t.R1="R1",t.R2="R2",t.R3="R3",t.R4="R4",t.R5="R5",t.R6="R6"})(Ls||(Ls={}));var zs;(function(t){t.float32="float32",t.int32="int32",t.bool="int32",t.complex64="complex64"})(zs||(zs={}));var Vs;(function(t){t.float32="float32",t.int32="int32",t.bool="bool",t.complex64="complex64"})(Vs||(Vs={}));var js;(function(t){t.float32="float32",t.int32="float32",t.bool="float32",t.complex64="complex64"})(js||(js={}));var Ms;(function(t){t.float32="complex64",t.int32="complex64",t.bool="complex64",t.complex64="complex64"})(Ms||(Ms={}));const rd={float32:js,int32:zs,bool:Vs,complex64:Ms};function us(t,e){if(t==="string"||e==="string"){if(t==="string"&&e==="string")return"string";throw new Error(`Can not upcast ${t} with ${e}`)}return rd[t][e]}function ad(t){return us(t,"int32")}function $l(t){return t!=null&&typeof t=="object"&&"texture"in t&&t.texture instanceof WebGLTexture}function El(t){return typeof GPUBuffer<"u"&&t!=null&&typeof t=="object"&&"buffer"in t&&t.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ee(t,e){if(t.dtype===e.dtype)return[t,e];const n=us(t.dtype,e.dtype);return[t.cast(n),e.cast(n)]}function kl(t,e){g(t.dtype===e.dtype,()=>`The dtypes of the first(${t.dtype}) and second(${e.dtype}) input must match`)}function od(t,e){return e.some(n=>n.id===t.id)}function Tr(t){const e=[];return vl(t,e,new Set),e}function vl(t,e,n){if(t==null)return;if(t instanceof te){e.push(t);return}if(!id(t))return;const s=t;for(const r in s){const a=s[r];n.has(a)||(n.add(a),vl(a,e,n))}}function id(t){return Array.isArray(t)||typeof t=="object"}const ud=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:kl,getTensorsInContainer:Tr,isTensorInList:od,makeTypesMatch:ee},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vs(t){return t.kernelName!=null}class Va{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(const e in this.registeredVariables)this.registeredVariables[e].dispose()}}class Kt{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Va}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const e=this.getSortedBackends();for(let n=0;n<e.length;n++){const s=e[n];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:e,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry))if(e in this.registryFactory){const{asyncInit:n}=this.initializeBackend(e);if(n)return null}else return null;return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,n,s=1){return e in this.registryFactory?(et(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:n,priority:s},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw new Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;const{success:n,asyncInit:s}=this.initializeBackend(e);if(!(s?await n:n))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new Xf(this.backendInstance),!0}setupRegisteredKernels(){Kn(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){Kn(e).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[e])})}initializeBackend(e){const n=this.registryFactory[e];if(n==null)throw new Error(`Cannot initialize backend ${e}, no registration found.`);try{const s=n.factory();if(s&&!(s instanceof uo)&&typeof s.then=="function"){const r=++this.pendingBackendInitId,a=s.then(o=>r<this.pendingBackendInitId?!1:(this.registry[e]=o,this.pendingBackendInit=null,!0)).catch(o=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,et(`Initialization of backend ${e} failed`),et(o.stack||o.message)),!1));return this.pendingBackendInit=a,{success:a,asyncInit:!0}}else return this.registry[e]=s,{success:!0,asyncInit:!1}}catch(s){return et(`Initialization of backend ${e} failed`),et(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw new Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,n)=>this.registryFactory[n].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){const e=this.getSortedBackends();for(let n=0;n<e.length;n++){const s=e[n],{success:r,asyncInit:a}=this.initializeBackend(s);if(a||r)return{name:s,asyncInit:a}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,n){const s=this.state.tensorInfo.get(n),r=s.backend,a=this.readSync(n),o=r.refCount(n);r.disposeData(n,!0),s.backend=e,e.move(n,a,s.shape,s.dtype,o),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,n){let s=null;if(n==null){if(typeof e!="function")throw new Error("Please provide a function to tidy()");n=e}else{if(typeof e!="string"&&!(e instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=e}let r;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(r),()=>(r=n(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(e,n,s){e();try{const r=s();return n(),r}catch(r){throw n(),r}}nextTensorId(){return Kt.nextTensorId++}nextVariableId(){return Kt.nextVariableId++}clone(e){const n=N.runKernel(br,{x:e}),s={x:e},r=o=>({x:()=>{const i="float32",u={x:o},l={dtype:i};return N.runKernel(gr,u,l)}}),a=[];return this.addTapeNode(this.state.activeScope.name,s,[n],r,a,{}),n}runKernel(e,n,s){if(this.backendName==null&&this.backend,!(fn(e,this.backendName)!=null))throw new Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:n,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,n,s){const r=this.backend.numDataIds();let a=0;s.forEach(u=>{a+=u.dtype==="complex64"?3:1});const o=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],i=r-n-a-o;if(i>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${i} data ids) after running '${e}'`)}runKernelFunc(e){let n,s=[];const r=this.isTapeOn(),a=this.state.numBytes,o=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let i;this.backendName==null&&this.backend;let u;const l=vs(e)?e.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(vs(e)){const{kernelName:y,inputs:S,attrs:b}=e;this.backendName==null&&this.backend;const T=fn(y,this.backendName);g(T!=null,()=>`Cannot find registered kernel '${y}' for backend '${this.backendName}'`),i=()=>{const A=this.backend.numDataIds();u=T.kernelFunc({inputs:S,attrs:b,backend:this.backend});const $=Array.isArray(u)?u:[u];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(y,A,$);const E=$.map(v=>v.rank!=null?v:this.makeTensorFromTensorInfo(v));if(r){const v=this.getTensorsForGradient(y,S,E);s=this.saveTensorsForBackwardMode(v)}return E}}else{const{forwardFunc:y}=e,S=b=>{r&&(s=b.map(T=>this.keep(this.clone(T))))};i=()=>{const b=this.backend.numDataIds();u=this.tidy(()=>y(this.backend,S));const T=Array.isArray(u)?u:[u];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(l,b,T),T}}const{inputs:h,attrs:c}=e,f=vs(e)?null:e.backwardsFunc;let d;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=i():(d=this.profiler.profileKernel(l,h,()=>i()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(d),n=d.outputs)}),r&&this.addTapeNode(l,h,n,f,s,c),this.state.profiling&&this.state.activeProfile.kernels.push({name:l,bytesAdded:this.state.numBytes-a,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-o,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(y=>h[y]!=null?h[y].shape:null),outputShapes:n.map(y=>y.shape),kernelTimeMs:d.timeMs,extraInfo:d.extraInfo}),Array.isArray(u)?n:n[0]}saveTensorsForBackwardMode(e){return e.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(e,n,s){const r=Rs(e);if(r!=null){const a=r.inputsToSave||[],o=r.outputsToSave||[];let i;r.saveAllInputs?(g(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),i=Object.keys(n).map(l=>n[l])):i=a.map(l=>n[l]);const u=s.filter((l,h)=>o[h]);return i.concat(u)}return[]}makeTensor(e,n,s,r){if(e==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",r=r||this.backend;let a=e;s==="string"&&nt(e[0])&&(a=e.map(u=>In(u)));const o=r.write(a,n,s),i=new te(n,s,o,this.nextTensorId());if(this.trackTensor(i,r),s==="string"){const u=this.state.tensorInfo.get(o),l=go(a);this.state.numBytes+=l-u.bytes,u.bytes=l}return i}makeTensorFromDataId(e,n,s,r){s=s||"float32";const a={dataId:e,shape:n,dtype:s};return this.makeTensorFromTensorInfo(a,r)}makeTensorFromTensorInfo(e,n){const{dataId:s,shape:r,dtype:a}=e,o=new te(r,a,s,this.nextTensorId());return this.trackTensor(o,n),o}makeVariable(e,n=!0,s,r){s=s||this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));const a=new mn(e,n,s,this.nextTensorId());if(this.state.registeredVariables[a.name]!=null)throw new Error(`Variable with name ${a.name} was already registered`);return this.state.registeredVariables[a.name]=a,this.incRef(a,this.backend),a}trackTensor(e,n){this.state.numTensors++,e.dtype==="string"&&this.state.numStringTensors++;let s=0;e.dtype!=="complex64"&&e.dtype!=="string"&&(s=e.size*Gn(e.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:n||this.backend,dtype:e.dtype,shape:e.shape,bytes:s})),e instanceof mn||this.track(e)}incRef(e,n){this.trackTensor(e,n),this.backend.incRef(e.dataId)}removeDataId(e,n){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===n&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;const n=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),e.dtype!=="complex64"&&e.dtype!=="string"){const s=e.size*Gn(e.dtype);this.state.numBytes-=s}n.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,n.backend)}disposeVariables(){for(const e in this.state.registeredVariables){const n=this.state.registeredVariables[e];this.disposeVariable(n)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){const e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons==null&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;const n=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,n,s,r,a,o){const i={id:this.state.nextTapeNodeId++,kernelName:e,inputs:n,outputs:s,saved:a},u=Rs(e);u!=null&&(r=u.gradFunc),r!=null&&(i.gradient=l=>(l=l.map((h,c)=>{if(h==null){const f=s[c],d=os(f.size,f.dtype);return this.makeTensor(d,f.shape,f.dtype)}return h}),r(l.length>1?l:l[0],a,o))),this.state.activeTape.push(i)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(n.name=e),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(e){const n=Tr(e),s=new Set(n.map(a=>a.id));for(let a=0;a<this.state.activeScope.track.length;a++){const o=this.state.activeScope.track[a];!o.kept&&!s.has(o.id)&&o.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(a=>{!a.kept&&a.scopeId===r.id&&this.track(a)})}gradients(e,n,s,r=!1){if(g(n.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const a=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));g(a instanceof te,()=>"The result y returned by f() must be a tensor.");const o=Yf(this.state.activeTape,n,a);if(!r&&o.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const i={};i[a.id]=s??ld(a.shape),Qf(i,o,l=>this.tidy(l),cd);const u=n.map(l=>i[l.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(l=>{for(const h of l.saved)h.dispose()}),this.state.activeTape=null),{value:a,grads:u}})}customGrad(e){return g(ot(e),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{g(n.every(i=>i instanceof te),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const r={};n.forEach((i,u)=>{r[u]=i});const a=(i,u)=>(s=e(...n,u),g(s.value instanceof te,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),g(ot(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),o=(i,u)=>{const l=s.gradFunc(i,u),h=Array.isArray(l)?l:[l];g(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),g(h.every(f=>f instanceof te),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const c={};return h.forEach((f,d)=>{c[d]=()=>f}),c};return this.runKernelFunc({forwardFunc:a,backwardsFunc:o,inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,n){return this.state.tensorInfo.get(e).backend.readToGPU(e,n)}async time(e){const n=dn(),s=await this.backend.time(e);return s.wallMs=dn()-n,s}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Va;for(const e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}Kt.nextTensorId=0;Kt.nextVariableId=0;function ld(t){const e=pr(W(t),"float32");return N.makeTensor(e,t,"float32")}function _l(){const t=So();if(t._tfengine==null){const e=new No(t);t._tfengine=new Kt(e)}return pf(t._tfengine.ENV),nd(()=>t._tfengine),t._tfengine}const N=_l();function cd(t,e){const n={a:t,b:e};return N.runKernel(mr,n)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hd(){return typeof navigator<"u"&&navigator!=null}let Ws;function pd(t){Ws=t}function fd(t){if(Ws!==void 0)return Ws;if(t||hd()){if(t||(t=navigator),t.product==="ReactNative")return!0;const e=t.userAgent||t.vendor||(typeof window<"u"?window.opera:"");if(!e){const n=t;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(e)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(e.substr(0,4))}return!1}function Il(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const dd=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:Il,isMobile:fd,mockIsMobile:pd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const de=R();de.registerFlag("DEBUG",()=>!1,t=>{t&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});de.registerFlag("IS_BROWSER",()=>Il());de.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");de.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));de.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));de.registerFlag("PROD",()=>!1);de.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>de.getBool("DEBUG"));de.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);de.registerFlag("IS_TEST",()=>!1);de.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>de.getBool("DEBUG"));de.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);de.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);de.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ze(t,e){let n=t;if(ae(t))return e==="string"?[]:[t.length];if($l(t)){const r=t.channels||"RGBA";return[t.height,t.width*r.length]}else if(El(t))return[t.buffer.size/(e==null?4:Gn(e))];if(!Array.isArray(t))return[];const s=[];for(;Array.isArray(n)||ae(n)&&e!=="string";)s.push(n.length),n=n[0];return Array.isArray(t)&&R().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&xl(t,s,[]),s}function xl(t,e,n){if(n=n||[],!Array.isArray(t)&&!ae(t)){g(e.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${e[0]} elements`);return}g(e.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${t.length} elements`),g(t.length===e[0],()=>`Element arr[${n.join("][")}] should have ${e[0]} elements, but has ${t.length} elements`);const s=e.slice(1);for(let r=0;r<t.length;++r)xl(t[r],s,n.concat(r))}function ja(t,e,n,s){if(t!=="string_or_numeric"){if(t==null)throw new Error("Expected dtype cannot be null.");if(t!=="numeric"&&t!==e||t==="numeric"&&e==="string")throw new Error(`Argument '${n}' passed to '${s}' must be ${t} tensor, but got ${e} tensor`)}}function m(t,e,n,s="numeric"){if(t instanceof Tl())return ja(s,t.dtype,e,n),t;let r=vn(t);if(r!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(r=s),ja(s,r,e,n),t==null||!ae(t)&&!Array.isArray(t)&&typeof t!="number"&&typeof t!="boolean"&&typeof t!="string"){const u=t==null?"null":t.constructor.name;throw new Error(`Argument '${e}' passed to '${n}' must be a Tensor or TensorLike, but got '${u}'`)}const a=ze(t,r);!ae(t)&&!Array.isArray(t)&&(t=[t]);const i=r!=="string"?is(t,r):ut(t,[],!0);return N.makeTensor(i,a,r)}function gn(t,e,n,s="numeric"){if(!Array.isArray(t))throw new Error(`Argument ${e} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return t.map((a,o)=>m(a,`${e}[${o}]`,n,s))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $r="__op";function w(t){const e=Object.keys(t);if(e.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${e.length} keys.`);let n=e[0];const s=t[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+$r;const r=(...a)=>{N.startScope(n);try{const o=s(...a);return it(o)&&console.error("Cannot return a Promise inside of tidy."),N.endScope(o),o}catch(o){throw N.endScope(null),o}};return Object.defineProperty(r,"name",{value:n,configurable:!0}),r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function md(t,e){const n=m(t,"real","complex"),s=m(e,"imag","complex");fe(n.shape,s.shape,`real and imag shapes, ${n.shape} and ${s.shape}, must match in call to tf.complex().`);const r={real:n,imag:s};return N.runKernel(Uo,r)}const Je=w({complex_:md});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ft(t,e,n,s){if(s==null)s=vn(t);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(El(t)||$l(t)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return N.backend.createTensorFromGPUData(t,e||n,s)}if(!ae(t)&&!Array.isArray(t)&&typeof t!="number"&&typeof t!="boolean"&&typeof t!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(e!=null){Se(e);const r=W(e),a=W(n);g(r===a,()=>`Based on the provided shape, [${e}], the tensor should have ${r} values but has ${a}`);for(let o=0;o<n.length;++o){const i=n[o],u=o===n.length-1?i!==W(e.slice(o)):!0;g(n[o]===e[o]||!u,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${e}). `)}}return!ae(t)&&!Array.isArray(t)&&(t=[t]),e=e||n,t=s!=="string"?is(t,s):ut(t,[],!0),N.makeTensor(t,e,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fe(t,e,n){const s=ze(t,n);return ft(t,e,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vt={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class Be{static join(e){return new Be(e).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(s=>ae(s)?s.buffer:s),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let n=0;for(let s=0;s<e.length;s++){const r=e[s];s!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const a=n+r.byteLength;this.shards.push({buffer:r,start:n,end:a}),n=a}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,n=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(e=isNaN(Number(e))?0:e,n=isNaN(Number(n))?0:n,e=Math.max(0,e),n=Math.min(this.byteLength,n),n<=e)return new ArrayBuffer(0);const s=this.findShardForByte(e);if(s===-1)throw new Error(`Could not find start shard for byte ${e}`);const r=n-e,a=new ArrayBuffer(r),o=new Uint8Array(a);let i=0;for(let u=s;u<this.shards.length;u++){const l=this.shards[u],c=e+i-l.start,f=i,y=Math.min(n,l.end)-l.start,S=new Uint8Array(l.buffer,c,y-c);if(o.set(S,f),i+=S.length,n<l.end)break}return a}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function n(r){return e<r.start?-1:e>=r.end?1:0}if(n(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=gd(this.shards,n);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function gd(t,e){let n=0,s=t.length;for(;n<=s;){const r=Math.floor((s-n)/2)+n,a=e(t[r]);if(a===0)return r;a<0?s=r:n=r+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yd(){R().set("PROD",!0)}function bd(){R().set("DEBUG",!0)}function wd(){R().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function Nd(t){R().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(t+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function Sd(){N.disposeVariables()}function Td(){return N}function $d(){return N.memory()}function Ed(t){return N.profile(t)}function j(t,e){return N.tidy(t,e)}function pe(t){Tr(t).forEach(n=>n.dispose())}function De(t){return N.keep(t)}function kd(t){return N.time(t)}function vd(t){return N.setBackend(t)}function _d(){return N.ready()}function Al(){return N.backendName}function Id(t){N.removeBackend(t)}function xd(t){return N.findBackend(t)}function Ad(t){return N.findBackendFactory(t)}function Od(t,e,n=1){return N.registerBackend(t,e,n)}function Ol(){return N.backend}function Dd(t,e){R().setPlatform(t,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lt=4;async function Fd(t,e){const n=[],s=[],r=Array.isArray(t)?t.map(o=>o.name):Object.keys(t);for(let o=0;o<r.length;++o){const i=r[o],u=Array.isArray(t)?t[o].tensor:t[i];if(u.dtype!=="float32"&&u.dtype!=="int32"&&u.dtype!=="bool"&&u.dtype!=="string"&&u.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${i}': ${u.dtype}`);const l={name:i,shape:u.shape,dtype:u.dtype};if(u.dtype==="string"){const h=new Promise(async c=>{const f=await u.bytes(),d=f.reduce((b,T)=>b+T.length,0)+lt*f.length,y=new Uint8Array(d);let S=0;for(let b=0;b<f.length;b++){const T=f[b],A=new Uint8Array(new Uint32Array([T.length]).buffer);y.set(A,S),S+=lt,y.set(T,S),S+=T.length}c(y)});s.push(h)}else s.push(u.data());e!=null&&(l.group=e),n.push(l)}const a=await Promise.all(s);return{data:Pd(a),specs:n}}function Dl(t,e){const n=new Be(t),s={};let r=0;for(const a of e){const o=Cd(a,(i,u)=>n.slice(r+i,r+u));s[a.name]=Fl(a,n.slice(r,r+o)),r+=o}return s}function Cd(t,e){const n=W(t.shape);let s;if("quantization"in t){const r=t.quantization;s=vt[r.dtype]}else if(t.dtype==="string"){let r=0;for(let a=0;a<n;a++)r+=lt+new Uint32Array(e(r,r+lt))[0];return r}else s=vt[t.dtype];return n*s}async function Rd(t,e){const n=W(t.shape);let s;if("quantization"in t){const r=t.quantization;s=vt[r.dtype]}else if(t.dtype==="string"){let r=0;for(let a=0;a<n;a++)r+=lt+new Uint32Array(await e(r,r+lt))[0];return r}else s=vt[t.dtype];return n*s}function Fl(t,e){const n=t.name,s=t.dtype,r=t.shape,a=W(r);let o,i=0;if("quantization"in t){const u=t.quantization;if(u.dtype==="uint8"||u.dtype==="uint16"){if(!("min"in u&&"scale"in u))throw new Error(`Weight ${t.name} with quantization ${u.dtype} doesn't have corresponding metadata min and scale.`)}else if(u.dtype==="float16"){if(s!=="float32")throw new Error(`Weight ${t.name} is quantized with ${u.dtype} which only supports weights of type float32 not ${s}.`)}else throw new Error(`Weight ${t.name} has unknown quantization dtype ${u.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const l=vt[u.dtype],h=u.dtype==="uint8"?new Uint8Array(e):new Uint16Array(e);if(s==="float32")if(u.dtype==="uint8"||u.dtype==="uint16"){o=new Float32Array(h.length);for(let c=0;c<h.length;c++){const f=h[c];o[c]=f*u.scale+u.min}}else if(u.dtype==="float16")o=Wd()(h);else throw new Error(`Unsupported quantization type ${u.dtype} for weight type float32.`);else if(s==="int32"){if(u.dtype!=="uint8"&&u.dtype!=="uint16")throw new Error(`Unsupported quantization type ${u.dtype} for weight type int32.`);o=new Int32Array(h.length);for(let c=0;c<h.length;c++){const f=h[c];o[c]=Math.round(f*u.scale+u.min)}}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);i+=a*l}else if(s==="string"){const u=W(t.shape);o=[];for(let l=0;l<u;l++){const h=new Uint32Array(e.slice(i,i+lt))[0];i+=lt;const c=new Uint8Array(e.slice(i,i+h));o.push(c),i+=h}}else{const u=vt[s];if(s==="float32")o=new Float32Array(e);else if(s==="int32")o=new Int32Array(e);else if(s==="bool")o=new Uint8Array(e);else if(s==="complex64"){o=new Float32Array(e);const l=new Float32Array(o.length/2),h=new Float32Array(o.length/2);for(let y=0;y<l.length;y++)l[y]=o[y*2],h[y]=o[y*2+1];const c=Fe(l,r,"float32"),f=Fe(h,r,"float32"),d=Je(c,f);return c.dispose(),f.dispose(),d}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);i+=a*u}return Fe(o,r,s)}async function Ma(t,e,n){let s=new Uint8Array(e);for(;s.byteLength<n;){const{done:r,value:a}=await t.read();if(r&&a==null){const i=n-s.byteLength;throw new Error(`Reader is done but ${i} bytes are still expected`)}const o=new Uint8Array(s.length+a.byteLength);o.set(s,0),o.set(new Uint8Array(a),s.length),s=o}return s.buffer}async function Cl(t,e){const n={},s=t.getReader();let r=new ArrayBuffer(0);for(const a of e){const o=await Rd(a,async(l,h)=>(r=await Ma(s,r,h),r.slice(l,h)));r=await Ma(s,r,o);const i=r.slice(0,o);r=r.slice(o);const u=Fl(a,i);if(n[a.name]=u,Al()==="webgpu"){const l=Ol();"uploadToGPU"in l&&W(u.shape)>=R().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&l.uploadToGPU(u.dataId)}}return n}function Pd(t){if(t===null)throw new Error(`Invalid input value: ${JSON.stringify(t)}`);let e=0;const n=[];t.forEach(a=>{if(e+=a.byteLength,n.push(a.byteLength===a.buffer.byteLength?a:new a.constructor(a)),!(a instanceof Float32Array||a instanceof Int32Array||a instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${a.constructor.name}`)});const s=new Uint8Array(e);let r=0;return n.forEach(a=>{s.set(new Uint8Array(a.buffer),r),r+=a.byteLength}),s.buffer}const Er=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function Wa(t){return Er?Buffer.byteLength(t,"utf8"):new Blob([t]).size}function Bd(t){if(Er)return Buffer.from(t).toString("base64");const e=new Uint8Array(t);let n="";for(let s=0,r=e.length;s<r;s++)n+=String.fromCharCode(e[s]);return btoa(n)}function Ld(t){if(Er){const s=Buffer.from(t,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const e=atob(t),n=new Uint8Array(e.length);for(let s=0;s<e.length;++s)n.set([e.charCodeAt(s)],s);return n.buffer}function zd(t){return Be.join(t)}function Ua(t){const e="/";for(t=t.trim();t.endsWith(e);)t=t.slice(0,t.length-1);const n=t.split(e);return n[n.length-1]}function Rl(t,e){const n={modelTopology:t.modelTopology,format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,weightsManifest:e};return t.signature!=null&&(n.signature=t.signature),t.userDefinedMetadata!=null&&(n.userDefinedMetadata=t.userDefinedMetadata),t.modelInitializer!=null&&(n.modelInitializer=t.modelInitializer),t.initializerSignature!=null&&(n.initializerSignature=t.initializerSignature),t.trainingConfig!=null&&(n.trainingConfig=t.trainingConfig),n}function kr(t,e,n){const s={modelTopology:t.modelTopology,format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy};if(t.trainingConfig!=null&&(s.trainingConfig=t.trainingConfig),t.weightsManifest!=null){if(!e)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=e,s.weightData=n}return t.signature!=null&&(s.signature=t.signature),t.userDefinedMetadata!=null&&(s.userDefinedMetadata=t.userDefinedMetadata),t.modelInitializer!=null&&(s.modelInitializer=t.modelInitializer),t.initializerSignature!=null&&(s.initializerSignature=t.initializerSignature),s}async function vr(t,e){let n,s;return t.weightsManifest!=null&&([n,s]=await e(t.weightsManifest)),kr(t,n,s)}function xn(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:t.modelTopology==null?0:Wa(JSON.stringify(t.modelTopology)),weightSpecsBytes:t.weightSpecs==null?0:Wa(JSON.stringify(t.weightSpecs)),weightDataBytes:t.weightData==null?0:new Be(t.weightData).byteLength}}function Yn(t){const e=[];for(const n of t)e.push(...n.weights);return e}function Vd(){const t=n=>{let s=n<<13,r=0;for(;!(s&8388608);)r-=8388608,s<<=1;return s&=-8388609,r+=947912704,s|r},e=new Uint32Array(2048);e[0]=0;for(let n=1;n<1024;n++)e[n]=t(n);for(let n=1024;n<2048;n++)e[n]=939524096+(n-1024<<13);return e}function jd(){const t=new Uint32Array(64);t[0]=0,t[31]=1199570944,t[32]=2147483648,t[63]=3347054592;for(let e=1;e<31;e++)t[e]=e<<23;for(let e=33;e<63;e++)t[e]=2147483648+(e-32<<23);return t}function Md(){const t=new Uint32Array(64);for(let e=0;e<64;e++)t[e]=1024;return t[0]=t[32]=0,t}function Wd(){const t=Vd(),e=jd(),n=Md();return s=>{const r=new ArrayBuffer(4*s.length),a=new Uint32Array(r);for(let o=0;o<s.length;o++){const i=s[o],u=t[n[i>>10]+(i&1023)]+e[i>>10];a[o]=u}return new Float32Array(r)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Q{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return Q.instance==null&&(Q.instance=new Q),Q.instance}static registerSaveRouter(e){Q.getInstance().saveRouters.push(e)}static registerLoadRouter(e){Q.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return Q.getHandlers(e,"save")}static getLoadHandlers(e,n){return Q.getHandlers(e,"load",n)}static getHandlers(e,n,s){const r=[];return(n==="load"?Q.getInstance().loadRouters:Q.getInstance().saveRouters).forEach(o=>{const i=o(e,s);i!==null&&r.push(i)}),r}}const Ud=t=>Q.registerSaveRouter(t),qd=t=>Q.registerLoadRouter(t),Gd=t=>Q.getSaveHandlers(t),Hd=(t,e)=>Q.getLoadHandlers(t,e);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Us="tensorflowjs",qs=1,Tt="models_store",st="model_info_store";function Pl(){if(!R().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const t=typeof window>"u"?self:window,e=t.indexedDB||t.mozIndexedDB||t.webkitIndexedDB||t.msIndexedDB||t.shimIndexedDB;if(e==null)throw new Error("The current browser does not appear to support IndexedDB.");return e}function Gs(t){const e=t.result;e.createObjectStore(Tt,{keyPath:"modelPath"}),e.createObjectStore(st,{keyPath:"modelPath"})}class _t{constructor(e){if(this.indexedDB=Pl(),e==null||!e)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,n){return new Promise((s,r)=>{const a=this.indexedDB.open(Us,qs);a.onupgradeneeded=()=>Gs(a),a.onsuccess=()=>{const o=a.result;if(n==null){const i=o.transaction(Tt,"readonly"),l=i.objectStore(Tt).get(this.modelPath);l.onsuccess=()=>{if(l.result==null)return o.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(l.result.modelArtifacts)},l.onerror=h=>(o.close(),r(l.error)),i.oncomplete=()=>o.close()}else{n.weightData=Be.join(n.weightData);const i=xn(n),u=o.transaction(st,"readwrite");let l=u.objectStore(st),h;try{h=l.put({modelPath:this.modelPath,modelArtifactsInfo:i})}catch(f){return r(f)}let c;h.onsuccess=()=>{c=o.transaction(Tt,"readwrite");const f=c.objectStore(Tt);let d;try{d=f.put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:i})}catch(y){return r(y)}d.onsuccess=()=>s({modelArtifactsInfo:i}),d.onerror=y=>{l=u.objectStore(st);const S=l.delete(this.modelPath);S.onsuccess=()=>(o.close(),r(d.error)),S.onerror=b=>(o.close(),r(d.error))}},h.onerror=f=>(o.close(),r(h.error)),u.oncomplete=()=>{c==null?o.close():c.oncomplete=()=>o.close()}}},a.onerror=o=>r(a.error)})}}_t.URL_SCHEME="indexeddb://";const Bl=t=>R().getBool("IS_BROWSER")&&!Array.isArray(t)&&t.startsWith(_t.URL_SCHEME)?Kd(t.slice(_t.URL_SCHEME.length)):null;Q.registerSaveRouter(Bl);Q.registerLoadRouter(Bl);function Kd(t){return new _t(t)}function Xd(t){return t.startsWith(_t.URL_SCHEME)?t.slice(_t.URL_SCHEME.length):t}class Zd{constructor(){this.indexedDB=Pl()}async listModels(){return new Promise((e,n)=>{const s=this.indexedDB.open(Us,qs);s.onupgradeneeded=()=>Gs(s),s.onsuccess=()=>{const r=s.result,a=r.transaction(st,"readonly"),i=a.objectStore(st).getAll();i.onsuccess=()=>{const u={};for(const l of i.result)u[l.modelPath]=l.modelArtifactsInfo;e(u)},i.onerror=u=>(r.close(),n(i.error)),a.oncomplete=()=>r.close()},s.onerror=r=>n(s.error)})}async removeModel(e){return e=Xd(e),new Promise((n,s)=>{const r=this.indexedDB.open(Us,qs);r.onupgradeneeded=()=>Gs(r),r.onsuccess=()=>{const a=r.result,o=a.transaction(st,"readwrite"),i=o.objectStore(st),u=i.get(e);let l;u.onsuccess=()=>{if(u.result==null)return a.close(),s(new Error(`Cannot find model with path '${e}' in IndexedDB.`));{const h=i.delete(e),c=()=>{l=a.transaction(Tt,"readwrite");const d=l.objectStore(Tt).delete(e);d.onsuccess=()=>n(u.result.modelArtifactsInfo),d.onerror=y=>s(u.error)};h.onsuccess=c,h.onerror=f=>(c(),a.close(),s(u.error))}},u.onerror=h=>(a.close(),s(u.error)),o.oncomplete=()=>{l==null?a.close():l.oncomplete=()=>a.close()}},r.onerror=a=>s(r.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ke="/",zt="tensorflowjs_models",Ll="info",Jd="model_topology",Yd="weight_specs",Qd="weight_data",em="model_metadata";function zl(t){return{info:[zt,t,Ll].join(Ke),topology:[zt,t,Jd].join(Ke),weightSpecs:[zt,t,Yd].join(Ke),weightData:[zt,t,Qd].join(Ke),modelMetadata:[zt,t,em].join(Ke)}}function Vl(t){for(const e of Object.values(t))window.localStorage.removeItem(e)}function tm(t){const e=t.split(Ke);if(e.length<3)throw new Error(`Invalid key format: ${t}`);return e.slice(1,e.length-1).join(Ke)}function nm(t){return t.startsWith(It.URL_SCHEME)?t.slice(It.URL_SCHEME.length):t}class It{constructor(e){if(!R().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,e==null||!e)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=zl(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(e.modelTopology),s=JSON.stringify(e.weightSpecs),r=xn(e),a=Be.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,Bd(a));const o={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature!=null?e.signature:void 0,userDefinedMetadata:e.userDefinedMetadata!=null?e.userDefinedMetadata:void 0,modelInitializer:e.modelInitializer!=null?e.modelInitializer:void 0,initializerSignature:e.initializerSignature!=null?e.initializerSignature:void 0,trainingConfig:e.trainingConfig!=null?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(o)),{modelArtifactsInfo:r}}catch{throw Vl(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=s;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=r;const a=this.LS.getItem(this.keys.modelMetadata);if(a!=null){const i=JSON.parse(a);n.format=i.format,n.generatedBy=i.generatedBy,n.convertedBy=i.convertedBy,i.signature!=null&&(n.signature=i.signature),i.userDefinedMetadata!=null&&(n.userDefinedMetadata=i.userDefinedMetadata),i.modelInitializer!=null&&(n.modelInitializer=i.modelInitializer),i.initializerSignature!=null&&(n.initializerSignature=i.initializerSignature),i.trainingConfig!=null&&(n.trainingConfig=i.trainingConfig)}const o=this.LS.getItem(this.keys.weightData);if(o==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=Ld(o),n}}It.URL_SCHEME="localstorage://";const jl=t=>R().getBool("IS_BROWSER")&&!Array.isArray(t)&&t.startsWith(It.URL_SCHEME)?sm(t.slice(It.URL_SCHEME.length)):null;Q.registerSaveRouter(jl);Q.registerLoadRouter(jl);function sm(t){return new It(t)}class rm{constructor(){g(R().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),g(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const e={},n=zt+Ke,s=Ke+Ll;for(let r=0;r<this.LS.length;++r){const a=this.LS.key(r);if(a.startsWith(n)&&a.endsWith(s)){const o=tm(a);e[o]=JSON.parse(this.LS.getItem(a))}}return e}async removeModel(e){e=nm(e);const n=zl(e);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${e}'`);const s=JSON.parse(this.LS.getItem(n.info));return Vl(n),s}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jt="://";class le{constructor(){this.managers={}}static getInstance(){return le.instance==null&&(le.instance=new le),le.instance}static registerManager(e,n){g(e!=null,()=>"scheme must not be undefined or null."),e.endsWith(jt)&&(e=e.slice(0,e.indexOf(jt))),g(e.length>0,()=>"scheme must not be an empty string.");const s=le.getInstance();g(s.managers[e]==null,()=>`A model store manager is already registered for scheme '${e}'.`),s.managers[e]=n}static getManager(e){const n=le.getInstance().managers[e];if(n==null)throw new Error(`Cannot find model manager for scheme '${e}'`);return n}static getSchemes(){return Object.keys(le.getInstance().managers)}}function Wn(t){if(t.indexOf(jt)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${le.getSchemes().join(",")}`);return{scheme:t.split(jt)[0],path:t.split(jt)[1]}}async function Ml(t,e,n=!1){g(t!==e,()=>`Old path and new path are the same: '${t}'`);const s=Q.getLoadHandlers(t);g(s.length>0,()=>`Copying failed because no load handler is found for source URL ${t}.`),g(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${t}.`);const r=s[0],a=Q.getSaveHandlers(e);g(a.length>0,()=>`Copying failed because no save handler is found for destination URL ${e}.`),g(a.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${e}.`);const o=a[0],i=Wn(t).scheme,u=Wn(t).path,l=i===Wn(t).scheme,h=await r.load();n&&l&&await le.getManager(i).removeModel(u);const c=await o.save(h);return n&&!l&&await le.getManager(i).removeModel(u),c.modelArtifactsInfo}async function am(){const t=le.getSchemes(),e={};for(const n of t){const s=await le.getManager(n).listModels();for(const r in s){const a=n+jt+r;e[a]=s[r]}}return e}async function om(t){const e=Wn(t);return le.getManager(e.scheme).removeModel(e.path)}async function im(t,e){return Ml(t,e,!1)}async function um(t,e){return Ml(t,e,!0)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class lm{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,n){return fetch(e,n)}now(){return performance.now()}encode(e,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,n){return new TextDecoder(n).decode(e)}setTimeoutCustom(e,n){if(typeof window>"u"||!R().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(e,n);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const r=this.functionRefs[s.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return pl(e)}}if(R().get("IS_BROWSER")){R().setPlatform("browser",new lm);try{le.registerManager(It.URL_SCHEME,new rm)}catch{}try{le.registerManager(_t.URL_SCHEME,new Zd)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cm={importFetch:()=>require("node-fetch")};let _s;class hm{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(e,n){return R().global.fetch!=null?R().global.fetch(e,n):(_s==null&&(_s=cm.importFetch()),_s(e,n))}now(){const e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(e)}decode(e,n){return e.length===0?"":new this.util.TextDecoder(n).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}}R().get("IS_NODE")&&!R().get("IS_BROWSER")&&R().setPlatform("node",new hm);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ve(t,e="float32",n){return e=e||"float32",Se(t),new Jn(t,e,n)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pm(t,e){const n=m(t,"x","cast");if(!mo(e))throw new Error(`Failed to cast to unknown dtype ${e}`);if(e==="string"&&n.dtype!=="string"||e!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:n},r={dtype:e};return N.runKernel(gr,s,r)}const X=w({cast_:pm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fm(t){const n={x:m(t,"x","clone","string_or_numeric")};return N.runKernel(br,n)}const Xe=w({clone_:fm});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _r(t,e=!1){console.log(t.toString(e))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */_l();const dm={buffer:Ve,cast:X,clone:Xe,print:_r};sd(dm);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mm(t,e){let n=m(t,"a","add"),s=m(e,"b","add");[n,s]=ee(n,s);const r={a:n,b:s};return N.runKernel(mr,r)}const C=w({add_:mm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gm(t,e){let n=m(t,"a","floorDiv"),s=m(e,"b","floorDiv");[n,s]=ee(n,s);const r={a:n,b:s};return N.runKernel(Ti,r)}const Ir=w({floorDiv_:gm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ym(t,e){let n=m(t,"a","div"),s=m(e,"b","div");if([n,s]=ee(n,s),n.dtype==="int32"&&s.dtype==="int32")return Ir(n,s);const r={a:n,b:s},a={};return N.runKernel(ci,r,a)}const K=w({div_:ym});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bm(t,e){let n=m(t,"a","mul"),s=m(e,"b","mul");[n,s]=ee(n,s);const r={a:n,b:s};return N.runKernel(eu,r)}const x=w({mul_:bm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wm(t){const e=m(t,"x","abs");if(e.dtype==="complex64"){const n={x:e};return N.runKernel(qo,n)}else{const n={x:e};return N.runKernel(To,n)}}const be=w({abs_:wm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nm(t){const n={x:m(t,"x","acos")};return N.runKernel($o,n)}const Wl=w({acos_:Nm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sm(t){const n={x:m(t,"x","acosh")};return N.runKernel(Eo,n)}const Ul=w({acosh_:Sm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tm(t){g(Array.isArray(t),()=>"The argument passed to tf.addN() must be a list of tensors"),g(t.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${t.length}`);const e=t.map((r,a)=>m(r,`tensors${a}`,"addN")),n=e[0];e.forEach(r=>{if(r.dtype!==n.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),e.forEach(r=>{if(!Re(r.shape,n.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const s=e;return N.runKernel(ko,s)}const ql=w({addN_:Tm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $m(t,e=null,n=!1){const r={x:m(t,"x","all","bool")},a={axis:e,keepDims:n};return N.runKernel(vo,r,a)}const Gl=w({all_:$m});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Em(t,e=null,n=!1){const r={x:m(t,"x","any","bool")},a={axis:e,keepDims:n};return N.runKernel(_o,r,a)}const Hl=w({any_:Em});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function km(t,e=0){const s={x:m(t,"x","argMax")},r={axis:e};return N.runKernel(Io,s,r)}const Kl=w({argMax_:km});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vm(t,e=0){const s={x:m(t,"x","argMin")},r={axis:e};return N.runKernel(xo,s,r)}const Xl=w({argMin_:vm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _m(t){const n={x:m(t,"x","asin")};return N.runKernel(Ao,n)}const Zl=w({asin_:_m});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Im(t){const n={x:m(t,"x","asinh")};return N.runKernel(Oo,n)}const Jl=w({asinh_:Im});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xm(t){const n={x:m(t,"x","atan")};return N.runKernel(Do,n)}const Yl=w({atan_:xm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Am(t,e){let n=m(t,"a","atan2"),s=m(e,"b","atan2");[n,s]=ee(n,s);const r={a:n,b:s};return N.runKernel(Co,r)}const Ql=w({atan2_:Am});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Om(t){const n={x:m(t,"x","atanh")};return N.runKernel(Fo,n)}const ec=w({atanh_:Om});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dm(t,e,n,s,r="NHWC",a){const o=t[3],i=[...e,o],u=sc(r);return An(t,i,n,a,s,null,null,u)}function tc(t,e,n,s,r,a,o="channelsLast"){const[i,u]=yn(e);let l;if(o==="channelsLast")l=[i,u,t[3],t[3]];else if(o==="channelsFirst")l=[i,u,t[1],t[1]];else throw new Error(`Unknown dataFormat ${o}`);return An(t,l,n,s,r,a,!1,o)}function Fm(t,e,n,s,r,a,o="NDHWC"){const[i,u,l]=Hs(e);let h,c;if(o==="NDHWC")c="channelsLast",h=[i,u,l,t[4],t[4]];else if(o==="NCDHW")c="channelsFirst",h=[i,u,l,t[1],t[1]];else throw new Error(`Unknown dataFormat ${o}`);return nc(t,h,n,s,r,!1,c,a)}function An(t,e,n,s,r,a,o=!1,i="channelsLast"){let[u,l,h,c]=[-1,-1,-1,-1];if(i==="channelsLast")[u,l,h,c]=t;else if(i==="channelsFirst")[u,c,l,h]=t;else throw new Error(`Unknown dataFormat ${i}`);const[f,d,,y]=e,[S,b]=yn(n),[T,A]=yn(s),$=Mt(f,T),E=Mt(d,A),{padInfo:v,outHeight:_,outWidth:O}=Pm(r,l,h,S,b,$,E,a,i),D=o?y*c:y;let F;return i==="channelsFirst"?F=[u,D,_,O]:i==="channelsLast"&&(F=[u,_,O,D]),{batchSize:u,dataFormat:i,inHeight:l,inWidth:h,inChannels:c,outHeight:_,outWidth:O,outChannels:D,padInfo:v,strideHeight:S,strideWidth:b,filterHeight:f,filterWidth:d,effectiveFilterHeight:$,effectiveFilterWidth:E,dilationHeight:T,dilationWidth:A,inShape:t,outShape:F,filterShape:e}}function nc(t,e,n,s,r,a=!1,o="channelsLast",i){let[u,l,h,c,f]=[-1,-1,-1,-1,-1];if(o==="channelsLast")[u,l,h,c,f]=t;else if(o==="channelsFirst")[u,f,l,h,c]=t;else throw new Error(`Unknown dataFormat ${o}`);const[d,y,S,,b]=e,[T,A,$]=Hs(n),[E,v,_]=Hs(s),O=Mt(d,E),D=Mt(y,v),F=Mt(S,_),{padInfo:B,outDepth:P,outHeight:M,outWidth:U}=Bm(r,l,h,c,T,A,$,O,D,F,i),Y=a?b*f:b;let se;return o==="channelsFirst"?se=[u,Y,P,M,U]:o==="channelsLast"&&(se=[u,P,M,U,Y]),{batchSize:u,dataFormat:o,inDepth:l,inHeight:h,inWidth:c,inChannels:f,outDepth:P,outHeight:M,outWidth:U,outChannels:Y,padInfo:B,strideDepth:T,strideHeight:A,strideWidth:$,filterDepth:d,filterHeight:y,filterWidth:S,effectiveFilterDepth:O,effectiveFilterHeight:D,effectiveFilterWidth:F,dilationDepth:E,dilationHeight:v,dilationWidth:_,inShape:t,outShape:se,filterShape:e}}function Cm(t,e,n,s,r){s==null&&(s=xr(t,e,n));const a=t[0],o=t[1],i=bn((a-e+2*s)/n+1,r),u=bn((o-e+2*s)/n+1,r);return[i,u]}function Rm(t,e,n,s,r,a){r==null&&(r=xr(t,e[0],s[0]));const o=[0,0,0,n];for(let i=0;i<3;i++)t[i]+2*r>=e[i]&&(o[i]=bn((t[i]-e[i]+2*r)/s[i]+1,a));return o}function xr(t,e,n,s=1){const r=Mt(e,s);return Math.floor((t[0]*(n-1)-n+r)/2)}function yn(t){return typeof t=="number"?[t,t,t]:t.length===2?[t[0],t[1],1]:t}function Hs(t){return typeof t=="number"?[t,t,t]:t}function Mt(t,e){return e<=1?t:t+(t-1)*(e-1)}function Pm(t,e,n,s,r,a,o,i,u){let l,h,c;if(typeof t=="number"){l={top:t,bottom:t,left:t,right:t,type:t===0?"VALID":"NUMBER"};const d=Cm([e,n],a,s,t,i);h=d[0],c=d[1]}else if(t==="same"){h=Math.ceil(e/s),c=Math.ceil(n/r);const f=Math.max(0,(h-1)*s+a-e),d=Math.max(0,(c-1)*r+o-n),y=Math.floor(f/2),S=f-y,b=Math.floor(d/2),T=d-b;l={top:y,bottom:S,left:b,right:T,type:"SAME"}}else if(t==="valid")l={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((e-a+1)/s),c=Math.ceil((n-o+1)/r);else if(typeof t=="object"){const f=u==="channelsLast"?t[1][0]:t[2][0],d=u==="channelsLast"?t[1][1]:t[2][1],y=u==="channelsLast"?t[2][0]:t[3][0],S=u==="channelsLast"?t[2][1]:t[3][1];l={top:f,bottom:d,left:y,right:S,type:f===0&&d===0&&y===0&&S===0?"VALID":"EXPLICIT"},h=bn((e-a+f+d)/s+1,i),c=bn((n-o+y+S)/r+1,i)}else throw Error(`Unknown padding parameter: ${t}`);return{padInfo:l,outHeight:h,outWidth:c}}function Bm(t,e,n,s,r,a,o,i,u,l,h){let c,f,d,y;if(t==="valid"&&(t=0),typeof t=="number"){c={top:t,bottom:t,left:t,right:t,front:t,back:t,type:t===0?"VALID":"NUMBER"};const b=Rm([e,n,s,1],[i,u,l],1,[r,a,o],t,h);f=b[0],d=b[1],y=b[2]}else if(t==="same"){f=Math.ceil(e/r),d=Math.ceil(n/a),y=Math.ceil(s/o);const S=(f-1)*r+i-e,b=(d-1)*a+u-n,T=(y-1)*o+l-s,A=Math.floor(S/2),$=S-A,E=Math.floor(b/2),v=b-E,_=Math.floor(T/2),O=T-_;c={top:E,bottom:v,left:_,right:O,front:A,back:$,type:"SAME"}}else throw Error(`Unknown padding parameter: ${t}`);return{padInfo:c,outDepth:f,outHeight:d,outWidth:y}}function bn(t,e){if(!e)return Math.trunc(t);switch(e){case"round":return Math.round(t);case"ceil":return Math.ceil(t);case"floor":return Math.floor(t);default:throw new Error(`Unknown roundingMode ${e}`)}}function wn(t){const[e,n,s]=yn(t);return e===1&&n===1&&s===1}function Ye(t,e){return wn(t)||wn(e)}function xt(t){return yn(t).every(e=>e>0)}function sc(t){if(t==="NHWC")return"channelsLast";if(t==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${t}`)}function Ae(t,e,n){if(n!=null){if(typeof e=="string")throw Error(`Error in ${t}: pad must be an integer when using dimRoundingMode ${n} but got pad ${e}.`);if(typeof e=="number")g(qt(e),()=>`Error in ${t}: pad must be an integer when using dimRoundingMode ${n} but got pad ${e}.`);else if(typeof e=="object")e.forEach(s=>{s.forEach(r=>{g(qt(r),()=>`Error in ${t}: pad must be an integer when using dimRoundingMode ${n} but got pad ${r}.`)})});else throw Error(`Error in ${t}: Unknown padding parameter: ${e}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lm(t,e){const s={x:m(t,"x","reshape","string_or_numeric")},r={shape:e};return N.runKernel(Nu,s,r)}const k=w({reshape_:Lm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zm(t,e,n,s,r){const a=m(t,"x","avgPool","float32"),o=1;g(Ye(n,o),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`);let i=a,u=!1;a.rank===3&&(u=!0,i=k(a,[1,a.shape[0],a.shape[1],a.shape[2]])),g(i.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${i.rank}.`),Ae("avgPool",s,r);const l={x:i},h={filterSize:e,strides:n,pad:s,dimRoundingMode:r};let c=N.runKernel(Ro,l,h);return c=X(c,a.dtype),u?k(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Ar=w({avgPool_:zm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vm(t,e,n,s,r,a="NDHWC"){const o=m(t,"x","avgPool3d","float32");let i=o,u=!1;o.rank===4&&(u=!0,i=k(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),g(i.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${i.rank}.`),g(a==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),g(typeof n=="number"&&n>0||Array.isArray(n)&&n[0]>0&&n[1]>0&&n[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${n}'`),Ae("avgPool3d",s,r);const l={x:i},h={filterSize:e,strides:n,pad:s,dimRoundingMode:r,dataFormat:a};let c=N.runKernel(Po,l,h);return c=X(c,i.dtype),u?k(c,[c.shape[1],c.shape[2],c.shape[3],c.shape[4]]):c}const rc=w({avgPool3d_:Vm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jm(t,e=0){g(t.length>=1,()=>"Pass at least one tensor to concat");const n=gn(t,"tensors","concat","string_or_numeric");if(n[0].dtype==="complex64"&&n.forEach(a=>{if(a.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${a.dtype}. `)}),n.length===1)return Xe(n[0]);const s=n,r={axis:e};return N.runKernel(Go,s,r)}const ue=w({concat_:jm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mm(t,e,n=!1,s=!1){let r=m(t,"a","matMul"),a=m(e,"b","matMul");[r,a]=ee(r,a);const o={a:r,b:a},i={transposeA:n,transposeB:s};return N.runKernel(Bo,o,i)}const V=w({matMul_:Mm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wm(t){const n={x:m(t,"x","sigmoid","float32")};return N.runKernel(Pu,n)}const Et=w({sigmoid_:Wm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Um(t,e,n){const s=m(t,"x","slice","string_or_numeric");if(s.rank===0)throw new Error("Slicing scalar is not possible");const r={x:s},a={begin:e,size:n};return N.runKernel(Du,r,a)}const q=w({slice_:Um});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qm(t){const n={x:m(t,"x","tanh","float32")};return N.runKernel(nl,n)}const Qn=w({tanh_:qm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gm(t,e,n,s,r,a){const o=m(t,"forgetBias","basicLSTMCell"),i=m(e,"lstmKernel","basicLSTMCell"),u=m(n,"lstmBias","basicLSTMCell"),l=m(s,"data","basicLSTMCell"),h=m(r,"c","basicLSTMCell"),c=m(a,"h","basicLSTMCell"),f=ue([l,c],1),d=V(f,i),y=C(d,u),S=y.shape[0],b=y.shape[1]/4,T=[S,b],A=q(y,[0,0],T),$=q(y,[0,b],T),E=q(y,[0,b*2],T),v=q(y,[0,b*3],T),_=C(x(Et(A),Qn($)),x(h,Et(C(o,E)))),O=x(Qn(_),Et(v));return[_,O]}const ac=w({basicLSTMCell_:Gm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hm(t,e,n){const s=m(t,"x","batchToSpaceND"),r=e.reduce((i,u)=>i*u);g(s.rank>=1+e.length,()=>`input rank is ${s.rank} but should be > than blockShape.length ${e.length}`),g(n.length===e.length,()=>`crops.length is ${n.length} but should be equal to blockShape.length  ${e.length}`),g(s.shape[0]%r===0,()=>`input tensor batch is ${s.shape[0]} but is not divisible by the product of the elements of blockShape ${e.join(" * ")} === ${r}`);const a={x:s},o={blockShape:e,crops:n};return N.runKernel(Lo,a,o)}const Or=w({batchToSpaceND_:Hm});function Km(t){let e;return t.rank===0||t.rank===1?e=k(t,[1,1,1,t.size]):t.rank===2?e=k(t,[1,1,t.shape[0],t.shape[1]]):t.rank===3?e=k(t,[1,t.shape[0],t.shape[1],t.shape[2]]):e=t,e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xm(t,e,n,s,r,a){a==null&&(a=.001);const o=m(t,"x","batchNorm"),i=m(e,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;r!=null&&(l=m(r,"scale","batchNorm"));let h;s!=null&&(h=m(s,"offset","batchNorm")),g(i.rank===u.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),g(h==null||i.rank===h.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),g(l==null||i.rank===l.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const f={x:Km(o),scale:l,offset:h,mean:i,variance:u},d={varianceEpsilon:a},y=N.runKernel($i,f,d);return k(y,o.shape)}const On=w({batchNorm_:Xm});function Zm(t,e,n,s,r,a){const o=m(t,"x","batchNorm"),i=m(e,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;r!=null&&(l=m(r,"scale","batchNorm"));let h;return s!=null&&(h=m(s,"offset","batchNorm")),g(o.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${o.rank}.`),g(i.rank===2||i.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${i.rank}.`),g(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${u.rank}.`),l!=null&&g(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${l.rank}.`),h!=null&&g(h.rank===2||h.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${h.rank}.`),On(o,i,u,h,l,a)}const oc=w({batchNorm2d_:Zm});function Jm(t,e,n,s,r,a){const o=m(t,"x","batchNorm"),i=m(e,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;r!=null&&(l=m(r,"scale","batchNorm"));let h;return s!=null&&(h=m(s,"offset","batchNorm")),g(o.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${o.rank}.`),g(i.rank===3||i.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${i.rank}.`),g(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${u.rank}.`),l!=null&&g(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${l.rank}.`),h!=null&&g(h.rank===3||h.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${h.rank}.`),On(o,i,u,h,l,a)}const ic=w({batchNorm3d_:Jm});function Ym(t,e,n,s,r,a){const o=m(t,"x","batchNorm"),i=m(e,"mean","batchNorm"),u=m(n,"variance","batchNorm");let l;r!=null&&(l=m(r,"scale","batchNorm"));let h;return s!=null&&(h=m(s,"offset","batchNorm")),g(o.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${o.rank}.`),g(i.rank===4||i.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${i.rank}.`),g(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${u.rank}.`),l!=null&&g(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${l.rank}.`),h!=null&&g(h.rank===4||h.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${h.rank}.`),On(o,i,u,h,l,a)}const uc=w({batchNorm4d_:Ym});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qm(t,e,n){const s=m(t,"x","bincount"),r=m(e,"weights","bincount");g(s.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${s.dtype}`),g(n>=0,()=>`size must be non-negative, but got ${n}.`),g(r.size===s.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${s.shape}, weights shape: ${r.shape}.`);const a={x:s,weights:r},o={size:n};return N.runKernel(zo,a,o)}const Dr=w({bincount_:Qm});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eg(t,e){const n=m(t,"x","bitwiseAnd"),s=m(e,"y","bitwiseAnd");if(!Re(n.shape,s.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${n.shape}, y: ${s.shape}`);if(n.dtype!=="int32"||s.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${n.dtype} and type of y: ${s.dtype}`);const r={a:n,b:s};return N.runKernel(Vo,r)}const lc=w({bitwiseAnd_:eg});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tg(t,e){const n=m(t,"s0","broadcastArgs","int32"),s=m(e,"s1","broadcastArgs","int32");if(n.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${n.rank}`);if(s.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${s.rank}`);const r={s0:n,s1:s};return N.runKernel(jo,r)}const cc=w({broadcastArgs_:tg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ng(t,e){let n=m(t,"broadcastTo","x");const s=n.shape;if(Se(e),e.length<n.rank)throw new Error(`broadcastTo(): shape.length=${e.length} < input.rank=${n.rank}.`);if(e.length>n.rank){const l=n.shape.slice();for(;l.length<e.length;)l.unshift(1);n=k(n,l)}const r=n.shape,a=Array.from(e);for(let l=e.length-1;l>=0;l--)if(r[l]===e[l])a[l]=1;else if(n.shape[l]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${e}].`);if(a.map((l,h)=>l>1?h:-1).filter(l=>l>=0).length===0)return Xe(n);const i={x:n},u={reps:a};return N.runKernel(wr,i,u)}const cn=w({broadcastTo_:ng});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sg(t){const n={x:m(t,"x","ceil","float32")};return N.runKernel(Mo,n)}const hc=w({ceil_:sg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tn(t,e,n){Se(t),n=n||vn(e);const s={shape:t,value:e,dtype:n};return N.runKernel(wi,{},s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rg(t,e,n){const s=m(t,"x","clipByValue");if(g(e<=n,()=>`Error in clip: min (${e}) must be less than or equal to max (${n}).`),e===n)return tn(s.shape,e,s.dtype);const r={x:s},a={clipValueMin:e,clipValueMax:n};return N.runKernel(Wo,r,a)}const pc=w({clipByValue_:rg});function ag(t){return ue(t,0)}const fc=w({concat1d_:ag});function og(t,e){return ue(t,e)}const dc=w({concat2d_:og});function ig(t,e){return ue(t,e)}const mc=w({concat3d_:ig});function ug(t,e){return ue(t,e)}const gc=w({concat4d_:ug});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lg(t,e,n,s,r="NHWC",a=[1,1],o){const i=m(t,"x","conv2d","float32"),u=m(e,"filter","conv2d","float32");let l=i,h=!1;i.rank===3&&(h=!0,l=k(i,[1,i.shape[0],i.shape[1],i.shape[2]])),g(l.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${l.rank}.`),g(u.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${u.rank}.`),Ae("conv2d",s,o);const c=r==="NHWC"?l.shape[3]:l.shape[1];g(c===u.shape[2],()=>`Error in conv2d: depth of input (${c}) must match input depth for filter ${u.shape[2]}.`),g(Ye(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),g(xt(a),()=>"Error in conv2D: Dilated rates should be larger than 0."),g(xt(n),()=>"Error in conv2D: Strides should be larger than 0.");const f={x:l,filter:u},d={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o},y=N.runKernel(Ho,f,d);return h?k(y,[y.shape[1],y.shape[2],y.shape[3]]):y}const Dn=w({conv2d_:lg});function cg(t,e,n,s,r="NWC",a=1,o){const i=m(t,"x","conv1d"),u=m(e,"filter","conv1d");let l=i,h=!1;i.rank===2&&(h=!0,l=k(i,[1,i.shape[0],i.shape[1]])),g(l.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${l.rank}.`),g(u.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${u.rank}.`),Ae("conv1d",s,o),g(l.shape[2]===u.shape[1],()=>`Error in conv1d: depth of input (${l.shape[2]}) must match input depth for filter ${u.shape[1]}.`),g(Ye(n,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${n} and dilation '${a}'`),g(xt(a),()=>"Error in conv1D: Dilated rates should be larger than 0."),g(xt(n),()=>"Error in conv1D: Stride should be larger than 0."),g(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const c=k(u,[1,u.shape[0],u.shape[1],u.shape[2]]),f=k(l,[l.shape[0],1,l.shape[1],l.shape[2]]),b=Dn(f,c,[1,n],s,"NHWC",[1,a],o);return h?k(b,[b.shape[2],b.shape[3]]):k(b,[b.shape[0],b.shape[2],b.shape[3]])}const yc=w({conv1d_:cg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hg(t,e,n,s,r,a="NHWC",o){g(t.length===e.rank,()=>`Length of inShape (${t.length}) and rank of dy (${e.rank}) must match`);let i=t,u=e,l=!1;e.rank===3&&(l=!0,u=k(e,[1,e.shape[0],e.shape[1],e.shape[2]]),i=[1,t[0],t[1],t[2]]),g(i.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${i.length}.`),g(u.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${u.rank}`),g(n.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${n.rank}`);const h=a==="NHWC"?i[3]:i[1],c=a==="NHWC"?u.shape[3]:u.shape[1];g(h===n.shape[2],()=>`Error in conv2dDerInput: depth of input (${h}) must match input depth for filter ${n.shape[2]}.`),g(c===n.shape[3],()=>`Error in conv2dDerInput: depth of output (${c}) must match output depth for filter ${n.shape[3]}.`),Ae("conv2dDerInput",r,o);const f={dy:u,filter:n},d={strides:s,pad:r,dataFormat:a,dimRoundingMode:o,inputShape:i},y=N.runKernel(Xo,f,d);return l?k(y,[y.shape[1],y.shape[2],y.shape[3]]):y}const bc=w({conv2DBackpropInput_:hg});function pg(t,e,n,s,r,a){const o=m(t,"x","conv2dTranspose"),i=m(e,"filter","conv2dTranspose");return bc(n,o,i,s,r,"NHWC",a)}const wc=w({conv2dTranspose_:pg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fg(t,e,n,s,r="NDHWC",a=[1,1,1]){const o=m(t,"x","conv3d"),i=m(e,"filter","conv3d");let u=o,l=!1;o.rank===4&&(l=!0,u=k(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),g(u.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${u.rank}.`),g(i.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${i.rank}.`),g(u.shape[4]===i.shape[3],()=>`Error in conv3d: depth of input (${u.shape[4]}) must match input depth for filter ${i.shape[3]}.`),g(Ye(n,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),g(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`),g(xt(a),()=>"Error in conv3D: Dilated rates should be larger than 0."),g(xt(n),()=>"Error in conv3D: Strides should be larger than 0.");const h={x:u,filter:i},c={strides:n,pad:s,dataFormat:r,dilations:a},f=N.runKernel(Zo,h,c);return l?k(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const Nc=w({conv3d_:fg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dg(t,e,n,s,r){g(t.length===e.rank,()=>`Length of inShape (${t.length}) and rank of dy (${e.rank}) must match`);let a=t,o=e,i=!1;e.rank===4&&(i=!0,o=k(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]),a=[1,t[0],t[1],t[2],t[3]]);const u=a[4],l=o.shape[4];g(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),g(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),g(n.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${n.rank}`),g(u===n.shape[3],()=>`Error in conv3dDerInput: depth of input (${u}) must match input depth for filter ${n.shape[3]}.`),g(l===n.shape[4],()=>`Error in conv3dDerInput: depth of output (${l}) must match output depth for filter ${n.shape[4]}.`);const h={dy:o,filter:n},c={pad:r,strides:s,inputShape:a},f=N.runKernel(Jo,h,c);return i?k(f,[f.shape[1],f.shape[2],f.shape[3],f.shape[4]]):f}const mg=w({conv3DBackpropInput_:dg});function gg(t,e,n,s,r){const a=m(t,"x","conv3dTranspose"),o=m(e,"filter","conv3dTranspose");return mg(n,a,o,s,r)}const Sc=w({conv3dTranspose_:gg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yg(t){const n={x:m(t,"x","cos","float32")};return N.runKernel(Yo,n)}const Tc=w({cos_:yg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bg(t){const n={x:m(t,"x","cosh","float32")};return N.runKernel(Qo,n)}const $c=w({cosh_:bg});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wg(t,e=0,n=!1,s=!1){const a={x:m(t,"x","cumprod")},o={axis:e,exclusive:n,reverse:s};return N.runKernel(ei,a,o)}const Ec=w({cumprod_:wg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ng(t,e=0,n=!1,s=!1){const a={x:m(t,"x","cumsum")},o={axis:e,exclusive:n,reverse:s};return N.runKernel(ti,a,o)}const kc=w({cumsum_:Ng});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sg(t,e,n,s=!1){const r=m(t,"x","denseBincount"),a=m(e,"weights","denseBincount");g(r.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${r.dtype}`),g(r.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${r.rank}.`),g(n>=0,()=>`size must be non-negative, but got ${n}.`),g(a.size===r.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${r.shape}, weights shape: ${a.shape}.`);const o={x:r,weights:a},i={size:n,binaryOutput:s};return N.runKernel(si,o,i)}const vc=w({denseBincount_:Sg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tg(t,e,n="NHWC"){const s=m(t,"x","depthToSpace","float32"),r=n==="NHWC"?s.shape[1]:s.shape[2],a=n==="NHWC"?s.shape[2]:s.shape[3],o=n==="NHWC"?s.shape[3]:s.shape[1];g(e>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${e}`),g(r*e>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${e}  for depthToSpace with input shape
    ${s.shape}`),g(a*e>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${e} for depthToSpace with input shape
        ${s.shape}`),g(o%(e*e)===0,()=>`Dimension size must be evenly divisible by ${e*e} but is ${o} for depthToSpace with input shape ${s.shape}`);const i={x:s},u={blockSize:e,dataFormat:n};return N.runKernel(ri,i,u)}const _c=w({depthToSpace_:Tg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $g(t,e,n,s,r="NHWC",a=[1,1],o){const i=m(t,"x","depthwiseConv2d","float32"),u=m(e,"filter","depthwiseConv2d","float32");let l=i,h=!1;i.rank===3&&(h=!0,l=k(i,[1,i.shape[0],i.shape[1],i.shape[2]])),g(l.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${l.rank}.`),g(u.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${u.rank}.`);const c=r==="NHWC"?l.shape[3]:l.shape[1];g(c===u.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${c}) must match the inChannels dimension in filter ${u.shape[2]}.`),Ae("depthwiseConv2d",s,o);const f={x:l,filter:u},d={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o},y=N.runKernel(ai,f,d);return h?k(y,[y.shape[1],y.shape[2],y.shape[3]]):y}const ls=w({depthwiseConv2d_:$g});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Eg(t){const n={x:m(t,"x","diag")};return N.runKernel(ui,n)}const Ic=w({diag_:Eg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kg(t,e,n,s,r=[1,1],a="NHWC"){const o=m(t,"x","dilation2d"),i=m(e,"filter","dilation2d");g(o.rank===3||o.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${o.rank}.`),g(i.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${i.rank}.`),g(a==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let u=o,l=!1;o.rank===3&&(u=k(o,[1,o.shape[0],o.shape[1],o.shape[2]]),l=!0),g(u.shape[3]===i.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${u.shape[3]} vs ${i.shape[2]}`);const h={x:u,filter:i},c={strides:n,pad:s,dilations:r},f=N.runKernel(li,h,c);return l?k(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const xc=w({dilation2d_:kg});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ac(t,e){const n=t.length,s=[];for(let r=0;r<n;r++){const a=n-1-r,o=t[a]||1;(e[e.length-1-r]||1)>1&&o===1&&s.unshift(a)}return s}function Fr(t,e){const n=[];for(let s=0;s<e.length;s++){const r=t[t.length-s-1],a=e.length-s-1,o=e[a];(r==null||r===1&&o>1)&&n.unshift(a)}return n}function ne(t,e){const n=Math.max(t.length,e.length),s=new Array(n);for(let r=0;r<n;r++){let a=t[t.length-r-1];a==null&&(a=1);let o=e[e.length-r-1];if(o==null&&(o=1),a===1)s[n-r-1]=o;else if(o===1)s[n-r-1]=a;else if(a!==o){const i=`Operands could not be broadcast together with shapes ${t} and ${e}.`;throw Error(i)}else s[n-r-1]=a}return s}const vg=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:ne,getBroadcastDims:Ac,getReductionAxes:Fr},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _g(t,e){let n=m(t,"a","equal","string_or_numeric"),s=m(e,"b","equal","string_or_numeric");[n,s]=ee(n,s),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(di,r)}const Cr=w({equal_:_g});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ig(t,e,n){const s=m(e,"a","where"),r=m(n,"b","where"),a=m(t,"condition","where","bool"),o=ne(ne(a.shape,s.shape),r.shape),i=cn(a,o),u=cn(s,o),l=cn(r,o),h={condition:i,t:u,e:l};return N.runKernel(Au,h)}const Ze=w({where_:Ig});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xg(t){const n={x:m(t,"x","zerosLike")};return N.runKernel(ul,n)}const Ne=w({zerosLike_:xg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ag(t,e){let n=m(t,"a","div"),s=m(e,"b","div");[n,s]=ee(n,s);const r=K(n,s),a=Ne(r),o=Cr(s,a);return Ze(o,a,r)}const Oc=w({divNoNan_:Ag});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Og(t,e){const n=m(t,"t1","dot"),s=m(e,"t2","dot");g((n.rank===1||n.rank===2)&&(s.rank===1||s.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${n.rank} and ${s.rank}.`);const r=n.rank===1?n.size:n.shape[1],a=s.rank===1?s.size:s.shape[0];if(g(r===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${r} and ${a}.`),n.rank===1&&s.rank===1){const o=k(n,[1,-1]),i=k(s,[-1,1]),u=V(o,i);return k(u,[])}else if(n.rank===1&&s.rank===2){const o=k(n,[1,-1]),i=k(s,[s.shape[0],s.shape[1]]),u=V(o,i);return k(u,[u.size])}else if(n.rank===2&&s.rank===1){const o=k(s,[-1,1]),i=V(n,o);return k(i,[i.size])}else{const o=k(s,[s.shape[0],s.shape[1]]);return V(n,o)}}const Dc=w({dot_:Og});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dg(t,...e){const n=e.map((r,a)=>m(r,`tensors${a}`,"einsum")),s={equation:t};return N.runKernel(hi,n,s)}const wt=w({einsum_:Dg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fg(t){const n={x:m(t,"x","elu","float32")};return N.runKernel(pi,n)}const Rr=w({elu_:Fg});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cg(t,e){const n=m(t,"x","ensureShape","string_or_numeric");if(!co(n.shape,e))throw new Error(`EnsureShape: Shape of tensor ${n.shape} is not compatible with expected shape ${e}`);return t}const Fc=w({ensureShape_:Cg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rg(t){let e=m(t,"x","erf");g(e.dtype==="int32"||e.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),e.dtype==="int32"&&(e=X(e,"float32"));const n={x:e};return N.runKernel(fi,n)}const Cc=w({erf_:Rg});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pr(t,e){for(let n=0;n<t.length;++n)if(t[t.length-n-1]!==e-1-n)return!1;return!0}function Rc(t,e,n){const s=t.length+e.length,r=[];let a=0,o=0;for(let i=0;i<s;i++)n.indexOf(i)===-1?r.push(t[a++]):r.push(e[o++]);return r}function Pg(t,e){const n=[],s=t.length;for(let a=0;a<s;a++)e.indexOf(a)===-1&&n.push(t[a]);const r=e.map(a=>t[a]);return[n,r]}function Fn(t,e){const n=e.map(s=>1);return Rc(t,n,e)}function Bg(t,e,n){g(Pr(e,n),()=>`${t} supports only inner-most axes for now. Got axes ${e} and rank-${n} input.`)}function Lg(t,e){if(Pr(t,e))return null;const n=[];for(let s=0;s<e;++s)t.indexOf(s)===-1&&n.push(s);return t.forEach(s=>n.push(s)),n}function zg(t){return t.map((e,n)=>[n,e]).sort((e,n)=>e[1]-n[1]).map(e=>e[0])}function Vg(t,e){const n=[];for(let s=e-t;s<e;++s)n.push(s);return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jg(t,e=null,n=!1){const r={x:m(t,"x","max")},a={reductionIndices:e,keepDims:n};return N.runKernel(Wi,r,a)}const kt=w({max_:jg});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mg(t,e=null,n=!1){const r={x:m(t,"x","min")},a={axis:e,keepDims:n};return N.runKernel(Xi,r,a)}const es=w({min_:Mg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wg(t,e){let n=m(t,"base","pow"),s=m(e,"exp","pow");[n,s]=ee(n,s);const r={a:n,b:s};return N.runKernel(cu,r)}const Xt=w({pow_:Wg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z(t,e){if((ae(t)&&e!=="string"||Array.isArray(t))&&e!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(e==="string"&&ae(t)&&!(t instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return ft(t,[],[],e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ug(t){const n={x:m(t,"x","sqrt","float32")};return N.runKernel(Lu,n)}const je=w({sqrt_:Ug});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qg(t){const e=m(t,"x","square"),n={};return N.runKernel("Square",{x:e},n)}const xe=w({square_:qg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gg(t,e=null,n=!1){let s=m(t,"x","sum");s.dtype==="bool"&&(s=X(s,"int32"));const r={x:s},a={axis:e,keepDims:n};return N.runKernel(zu,r,a)}const H=w({sum_:Gg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hg(t,e="euclidean",n=null,s=!1){t=m(t,"x","norm");const r=Pc(t,e,n);let a=r.shape;if(s){const o=kn(n,t.shape);a=Fn(r.shape,o)}return k(r,a)}function Pc(t,e,n=null){if(t.rank===0)return be(t);if(t.rank!==1&&n===null)return Pc(k(t,[-1]),e,n);if(t.rank===1||typeof n=="number"||Array.isArray(n)&&n.length===1){if(e===1)return H(be(t),n);if(e===1/0)return kt(be(t),n);if(e===-1/0)return es(be(t),n);if(e==="euclidean"||e===2)return je(H(Xt(be(t),z(2,"int32")),n));throw new Error(`Error in norm: invalid ord value: ${e}`)}if(Array.isArray(n)&&n.length===2){if(e===1)return kt(H(be(t),n[0]),n[1]-1);if(e===1/0)return kt(H(be(t),n[1]),n[0]);if(e===-1/0)return es(H(be(t),n[1]),n[0]);if(e==="fro"||e==="euclidean")return je(H(xe(t),n));throw new Error(`Error in norm: invalid ord value: ${e}`)}throw new Error(`Error in norm: invalid axis: ${n}`)}const Cn=w({norm_:Hg});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kg(t,e=null,n=!1){return Cn(t,"euclidean",e,n)}const Bc=w({euclideanNorm_:Kg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xg(t){const n={x:m(t,"x","exp")};return N.runKernel(mi,n)}const ct=w({exp_:Xg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zg(t,e=0){const n=m(t,"x","expandDims","string_or_numeric");g(e<=n.rank,()=>"Axis must be <= rank of the tensor");const s={input:n},r={dim:e};return N.runKernel(gi,s,r)}const qe=w({expandDims_:Zg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jg(t){const n={x:m(t,"x","expm1")};return N.runKernel(yi,n)}const Lc=w({expm1_:Jg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yg(t,e){const n=m(t,"x","tile","string_or_numeric");g(n.rank===e.length,()=>`Error in transpose: rank of input ${n.rank} must match length of reps ${e}.`);const s={x:n},r={reps:e};return N.runKernel(wr,s,r)}const Wt=w({tile_:Yg});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qg(t,e,n,s="float32"){e==null&&(e=t);const r=Ve([t,e],s),a=t<=e?t:e;for(let i=0;i<a;++i)r.set(1,i,i);const o=k(r.toTensor(),[t,e]);if(n==null)return o;if(n.length===1)return Wt(qe(o,0),[n[0],1,1]);if(n.length===2)return Wt(qe(qe(o,0),0),[n[0],n[1],1,1]);if(n.length===3)return Wt(qe(qe(qe(o,0),0),0),[n[0],n[1],n[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${n.length}D.`)}const Br=w({eye_:Qg});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ey(t){const n={x:m(t,"x","floor","float32")};return N.runKernel(Si,n)}const Lr=w({floor_:ey});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ty(t,e,n=0,s=0){const r=m(t,"x","gather"),a=m(e,"indices","gather","int32"),o={x:r,indices:a},i={axis:n,batchDims:s};return N.runKernel(Ei,o,i)}const zr=w({gather_:ty});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ny(t,e){let n=m(t,"a","greater","string_or_numeric"),s=m(e,"b","greater","string_or_numeric");[n,s]=ee(n,s),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(vi,r)}const Rn=w({greater_:ny});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sy(t,e){let n=m(t,"a","greaterEqual","string_or_numeric"),s=m(e,"b","greaterEqual","string_or_numeric");[n,s]=ee(n,s),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(_i,r)}const Vr=w({greaterEqual_:sy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ry(t){const n={input:m(t,"input","imag")};return N.runKernel(xi,n)}const Pn=w({imag_:ry});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ay(t){const n={x:m(t,"x","isFinite")};return N.runKernel(Ai,n)}const zc=w({isFinite_:ay});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oy(t){const n={x:m(t,"x","isInf")};return N.runKernel(Oi,n)}const Vc=w({isInf_:oy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iy(t){const n={x:m(t,"x","isNaN")};return N.runKernel(Di,n)}const jc=w({isNaN_:iy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uy(t,e=.2){const s={x:m(t,"x","leakyRelu")},r={alpha:e};return N.runKernel(Fi,s,r)}const jr=w({leakyRelu_:uy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ly(t,e){let n=m(t,"a","less","string_or_numeric"),s=m(e,"b","less","string_or_numeric");[n,s]=ee(n,s),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(Ci,r)}const ts=w({less_:ly});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cy(t,e){let n=m(t,"a","lessEqual","string_or_numeric"),s=m(e,"b","lessEqual","string_or_numeric");[n,s]=ee(n,s),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(Ri,r)}const cs=w({lessEqual_:cy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mc(t,e,n){if(n<=0)throw new Error("The number of values should be positive.");const s={start:t,stop:e,num:n};return N.runKernel(Pi,{},s)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hy(t,e=5,n=1,s=1,r=.5){const a=m(t,"x","localResponseNormalization");g(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),g(qt(e),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${e}.`);let o=a,i=!1;a.rank===3&&(i=!0,o=k(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const u={x:o},l={depthRadius:e,bias:n,alpha:s,beta:r},h=N.runKernel(Mi,u,l);return i?k(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const Wc=w({localResponseNormalization_:hy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function py(t){const n={x:m(t,"x","log","float32")};return N.runKernel(Bi,n)}const Zt=w({log_:py});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fy(t){const n={x:m(t,"x","log1p")};return N.runKernel(Li,n)}const Mr=w({log1p_:fy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dy(t){return g(ot(t),()=>"The f passed in grad(f) must be a function"),(e,n)=>{const s=m(e,"x","tf.grad","string_or_numeric"),r=n!=null?m(n,"dy","tf.grad"):null;return N.tidy(()=>{const{value:a,grads:o}=N.gradients(()=>t(s),[s],r);return r!=null&&fe(a.shape,r.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),hs(o),o[0]})}}function my(t){return g(ot(t),()=>"The f passed in grads(f) must be a function"),(e,n)=>{g(Array.isArray(e),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const s=gn(e,"args","tf.grads","string_or_numeric"),r=n!=null?m(n,"dy","tf.grads"):null;return N.tidy(()=>{const{value:a,grads:o}=N.gradients(()=>t(...s),s,r);return r!=null&&fe(a.shape,r.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),hs(o),o})}}function gy(t){return g(ot(t),()=>"The f passed in valueAndGrad(f) must be a function"),(e,n)=>{g(e instanceof te,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),g(n==null||n instanceof te,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:s,value:r}=N.gradients(()=>t(e),[e],n);return hs(s),{grad:s[0],value:r}}}function yy(t){return g(ot(t),()=>"The f passed in valueAndGrads(f) must be a function"),(e,n)=>{g(Array.isArray(e)&&e.every(r=>r instanceof te),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),g(n==null||n instanceof te,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const s=N.gradients(()=>t(...e),e,n);return n!=null&&fe(s.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),hs(s.grads),s}}function Uc(t,e){g(ot(t),()=>"The f passed in variableGrads(f) must be a function"),g(e==null||Array.isArray(e)&&e.every(l=>l instanceof mn),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=e!=null;if(!n){e=[];for(const l in N.registeredVariables)e.push(N.registeredVariables[l])}const s=n?e.filter(l=>!l.trainable):null,r=e.length;e=e.filter(l=>l.trainable),g(e.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const a=!0,{value:o,grads:i}=N.gradients(t,e,null,a);g(i.some(l=>l!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),g(o.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${o.rank} tensor`);const u={};return e.forEach((l,h)=>{i[h]!=null&&(u[l.name]=i[h])}),s!=null&&s.forEach(l=>u[l.name]=null),{value:o,grads:u}}function Me(t){return N.customGrad(t)}function hs(t){if(t.filter(n=>n==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function by(t){const n={x:m(t,"x","neg")};return N.runKernel(tu,n)}const Ce=w({neg_:by});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wy(t){const n={x:m(t,"x","softplus")};return N.runKernel(Bu,n)}const Wr=w({softplus_:wy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ny(t){const e=m(t,"x","logSigmoid");return Me(s=>({value:Ce(Wr(Ce(s))),gradFunc:o=>x(o,Et(Ce(s)))}))(e)}const qc=w({logSigmoid_:Ny});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sy(t,e){let n=m(t,"a","sub"),s=m(e,"b","sub");[n,s]=ee(n,s);const r={a:n,b:s};return N.runKernel(el,r)}const L=w({sub_:Sy});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ty(t,e=-1){const n=m(t,"logits","logSoftmax");if(e===-1&&(e=n.rank-1),e!==n.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and axis was ${e}`);return Me((r,a)=>{const i=kt(r,e,!0),u=L(r,i),l=L(X(u,"float32"),Zt(H(ct(u),e,!0)));return a([l]),{value:l,gradFunc:(c,f)=>{const[d]=f,y=!0,S=ct(d);return L(c,x(H(c,e,y),S))}}})(n)}const Gc=w({logSoftmax_:Ty});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $y(t,e=null,n=!1){const s=m(t,"x","logSumExp"),r=kn(e,s.shape),a=kt(s,r,!0),o=L(s,a),i=ct(o),u=H(i,r),l=Zt(u),h=C(k(a,l.shape),l);if(n){const c=Fn(h.shape,r);return k(h,c)}return h}const Ur=w({logSumExp_:$y});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ey(t,e){const n=m(t,"a","logicalAnd","bool"),s=m(e,"b","logicalAnd","bool");ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(zi,r)}const Nn=w({logicalAnd_:Ey});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ky(t){const n={x:m(t,"x","logicalNot","bool")};return N.runKernel(Vi,n)}const qr=w({logicalNot_:ky});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vy(t,e){const n=m(t,"a","logicalOr","bool"),s=m(e,"b","logicalOr","bool");ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(ji,r)}const Gr=w({logicalOr_:vy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _y(t,e){const n=m(t,"a","logicalXor","bool"),s=m(e,"b","logicalXor","bool");return ne(n.shape,s.shape),Nn(Gr(t,e),qr(Nn(t,e)))}const Hc=w({logicalXor_:_y});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zn=2147483648;function Iy(t,e,n="left"){const s=m(t,"sortedSequence","searchSorted"),r=m(e,"values","searchSorted"),a=s.shape[s.shape.length-1],o=r.shape[r.shape.length-1],i=k(s,[-1,a]),u=k(r,[-1,o]);if(i.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(i.shape[0]!==u.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(W(u.shape)>=zn)throw new Error(`values tensor size must less than ${zn}`);if(i.shape[1]>=zn)throw new Error(`trailing dim_size must less than ${zn} for int32 output type, was ${i.shape[1]}`);const l={sortedSequence:i,values:u},h={side:n};return N.runKernel(xu,l,h)}const ps=w({searchSorted_:Iy});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kc(t,e){return ps(t,e,"left")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xy(t,e,n,s,r){const a=m(t,"x","maxPool"),o=1;let i=a,u=!1;a.rank===3&&(u=!0,i=k(a,[1,a.shape[0],a.shape[1],a.shape[2]])),g(i.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${i.rank}.`),g(Ye(n,o),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${n} and dilations '${o}'`),Ae("maxPool",s,r);const l={x:i},h={filterSize:e,strides:n,pad:s,dimRoundingMode:r},c=N.runKernel(qi,l,h);return u?k(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const Hr=w({maxPool_:xy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ay(t,e=[1,1,1],n,s,r,a="NDHWC"){const o=m(t,"x","maxPool3d");let i=o,u=!1;o.rank===4&&(u=!0,i=k(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),g(i.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${i.rank}.`),g(a==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),Ae("maxPool3d",s,r);const l={x:i},h={filterSize:e,strides:n,pad:s,dimRoundingMode:r,dataFormat:a},c=N.runKernel(Gi,l,h);return u?k(c,[c.shape[1],c.shape[2],c.shape[3],c.shape[4]]):c}const Xc=w({maxPool3d_:Ay});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oy(t,e,n,s,r=!1){const o={x:m(t,"x","maxPoolWithArgmax")},i={filterSize:e,strides:n,pad:s,includeBatchInIndex:r},u=N.runKernel(Hi,o,i);return{result:u[0],indexes:u[1]}}const Zc=w({maxPoolWithArgmax_:Oy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dy(t,e){let n=m(t,"a","maximum"),s=m(e,"b","maximum");[n,s]=ee(n,s),n.dtype==="bool"&&(n=X(n,"int32"),s=X(s,"int32")),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(Ui,r)}const Kr=w({maximum_:Dy});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fy(t,e=null,n=!1){const r={x:m(t,"x","mean")},a={axis:e,keepDims:n};return N.runKernel(Ki,r,a)}const Sn=w({mean_:Fy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function At(t,e="float32"){if(Se(t),e==="complex64"){const s=At(t,"float32"),r=At(t,"float32");return Je(s,r)}const n=os(W(t),e);return N.makeTensor(n,t,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rt(t,e="float32"){if(Se(t),e==="complex64"){const s=rt(t,"float32"),r=At(t,"float32");return Je(s,r)}const n=pr(W(t),e);return N.makeTensor(n,t,e)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jc(t,e,{indexing:n="xy"}={}){if(n!=="xy"&&n!=="ij")throw new TypeError(`${n} is not a valid third argument to meshgrid`);if(t===void 0)return[];let s=m(t,"x","meshgrid",t instanceof te?t.dtype:"float32");if(e===void 0)return[s];let r=m(e,"y","meshgrid",e instanceof te?e.dtype:"float32");const a=W(s.shape),o=W(r.shape);return n==="xy"?(s=k(s,[1,-1]),r=k(r,[-1,1]),[V(rt([o,1],s.dtype),s),V(r,rt([1,a],r.dtype))]):(s=k(s,[-1,1]),r=k(r,[1,-1]),[V(s,rt([1,o],s.dtype)),V(rt([a,1],r.dtype),r)])}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cy(t,e){let n=m(t,"a","minimum"),s=m(e,"b","minimum");[n,s]=ee(n,s),n.dtype==="bool"&&(n=X(n,"int32"),s=X(s,"int32")),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(Zi,r)}const Tn=w({minimum_:Cy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ry(t,e,n){g(n==="reflect"||n==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${n}.`);const s=m(t,"x","mirrorPad");if(s.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");g(e.length===s.rank,()=>`Padding doesn't match input. Must be ${s.rank}. Got ${e.length}.`);const r=n==="reflect"?1:0;for(let i=0;i<s.rank;i++)g(e[i].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),g(e[i][0]>=0&&e[i][0]<=s.shape[i]-r&&e[i][1]>=0&&e[i][1]<=s.shape[i]-r,()=>`Padding in dimension ${i} cannot be greater than or equal to ${s.shape[i]-r} or less than 0 for input of shape ${s.shape}`);const a={paddings:e,mode:n},o={x:s};return N.runKernel(Ji,o,a)}const Yc=w({mirrorPad_:Ry});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Py(t,e){let n=m(t,"a","mod"),s=m(e,"b","mod");[n,s]=ee(n,s);const r={a:n,b:s};return N.runKernel(Yi,r)}const Qc=w({mod_:Py});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function By(t,e=null,n=!1){t=m(t,"x","moments");const s=kn(e,t.shape),r=Sn(t,s,n);let a=r.shape;n||(a=Fn(r.shape,s));const o=xe(L(X(t,"float32"),k(r,a))),i=Sn(o,s,n);return{mean:r,variance:i}}const eh=w({moments_:By});function Ly(t,e,n,s){const r=m(e,"data","multiRNNCell"),a=gn(n,"c","multiRNNCell"),o=gn(s,"h","multiRNNCell");let i=r;const u=[];for(let c=0;c<t.length;c++){const f=t[c](i,a[c],o[c]);u.push(f[0]),u.push(f[1]),i=f[1]}const l=[],h=[];for(let c=0;c<u.length;c+=2)l.push(u[c]),h.push(u[c+1]);return[l,h]}const th=w({multiRNNCell_:Ly});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zy(t,e,n,s=!1){const r=m(t,"logits","multinomial"),a=r.size,o=r.rank;if(a<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(o>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${o}`);n=n||Math.random();const u={logits:o===1?k(r,[1,-1]):r},l={numSamples:e,seed:n,normalized:s},h=N.runKernel(Qi,u,l);return o===1?k(h,[h.size]):h}const nh=w({multinomial_:zy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vy(t,e){let n=m(t,"a","notEqual","string_or_numeric"),s=m(e,"b","notEqual","string_or_numeric");[n,s]=ee(n,s),ne(n.shape,s.shape);const r={a:n,b:s};return N.runKernel(nu,r)}const Xr=w({notEqual_:Vy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jy(t,e,n=1,s=0,r="int32"){if(e<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${e}`);const o={indices:m(t,"indices","oneHot","int32")},i={dtype:r,depth:e,onValue:n,offValue:s};return N.runKernel(iu,o,i)}const ns=w({oneHot_:jy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function My(t){const n={x:m(t,"x","onesLike")};return N.runKernel(ou,n)}const sh=w({onesLike_:My});function Wy(t,e){const n=m(t,"v1","outerProduct"),s=m(e,"v2","outerProduct");g(n.rank===1&&s.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${n.rank} and ${s.rank}.`);const r=k(n,[-1,1]),a=k(s,[1,-1]);return V(r,a)}const rh=w({outerProduct_:Wy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uy(t,e,n=0){const s=m(t,"x","pad");if(s.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:e,constantValue:n},a={x:s};return N.runKernel(lu,a,r)}const nn=w({pad_:Uy});function qy(t,e,n=0){return g(e.length===2,()=>"Invalid number of paddings. Must be length of 2."),nn(t,[e],n)}const ah=w({pad1d_:qy});function Gy(t,e,n=0){return g(e.length===2&&e[0].length===2&&e[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),nn(t,e,n)}const oh=w({pad2d_:Gy});function Hy(t,e,n=0){return g(e.length===3&&e[0].length===2&&e[1].length===2&&e[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),nn(t,e,n)}const ih=w({pad3d_:Hy});function Ky(t,e,n=0){return g(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),nn(t,e,n)}const uh=w({pad4d_:Ky});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xy(t,e,n){const s=m(t,"x","spaceToBatchND");g(s.rank>=1+e.length,()=>`input rank ${s.rank} should be > than [blockShape] ${e.length}`),g(n.length===e.length,()=>`paddings.shape[0] ${n.length} must be equal to [blockShape] ${e.length}`),g(s.shape.reduce((o,i,u)=>u>0&&u<=e.length?o&&(i+n[u-1][0]+n[u-1][1])%e[u-1]===0:o,!0),()=>`input spatial dimensions ${s.shape.slice(1)} with paddings ${n.toString()} must be divisible by blockShapes ${e.toString()}`);const r={x:s},a={blockShape:e,paddings:n};return N.runKernel(Vu,r,a)}const Zr=w({spaceToBatchND_:Xy});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zy(t,e,n,s,r,a,o){r==null&&(r=[1,1]),a==null&&(a=1),s===0&&(s="valid");const i=m(t,"x","maxPool");let u=i,l=!1;i.rank===3&&(l=!0,u=k(i,[1,i.shape[0],i.shape[1],i.shape[2]])),g(Ye(a,r),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${r}'`);const h=tc(u.shape,e,a,r,s),c=[h.dilationHeight,h.dilationWidth];let f;s==="same"?f=Yy([h.filterHeight,h.filterWidth],c):f=[[0,0],[0,0]];const d=c[0]===1&&c[1]===1,[y,S]=Jy([h.inHeight,h.inWidth],c,f),b=d?s:"valid",T=d?u:Zr(u,c,y),$=(n==="avg"?()=>Ar(T,e,a,b,o):()=>Hr(T,e,a,b,o))(),E=d?$:Or($,c,S);return l?k(E,[E.shape[1],E.shape[2],E.shape[3]]):E}function Jy(t,e,n){const s=n.map(h=>h[0]),r=n.map(h=>h[1]),a=t.concat(s,r),o=e.map((h,c)=>(h-a[c]%h)%h),i=r.map((h,c)=>h+o[c]),u=e.map((h,c)=>[s[c],i[c]]),l=e.map((h,c)=>[0,o[c]]);return[u,l]}function Yy(t,e){const s=t.map((o,i)=>o+(o-1)*(e[i]-1)).map(o=>o-1),r=s.map(o=>Math.floor(o/2)),a=s.map((o,i)=>o-r[i]);return s.map((o,i)=>[r[i],a[i]])}const lh=w({pool_:Zy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qy(t,e){const n=m(t,"x","prelu"),s=m(e,"alpha","prelu"),r={x:n,alpha:s};return N.runKernel(hu,r)}const Jr=w({prelu_:Qy});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eb(t,e=null,n=!1){let s=m(t,"x","prod");s.dtype==="bool"&&(s=X(s,"int32"));const r={x:s},a={axis:e,keepDims:n};return N.runKernel(pu,r,a)}const ch=w({prod_:eb});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tb(t,e,n,s){const r=t.map((h,c)=>m(h,`tensors${c}`,"raggedGather","int32")),a=m(e,"paramsDenseValues","raggedGather"),o=m(n,"indices","raggedGather","int32"),i={paramsNestedSplits:r,paramsDenseValues:a,indices:o},u={outputRaggedRank:s},l=N.runKernel(fu,i,u);return{outputNestedSplits:l.slice(0,l.length-1),outputDenseValues:l[l.length-1]}}const hh=w({raggedGather_:tb});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nb(t,e,n){const s=m(t,"starts","raggedRange"),r=m(e,"limits","raggedRange",s.dtype),a=m(n,"deltas","raggedRange",s.dtype),o={starts:s,limits:r,deltas:a},i=N.runKernel(du,o);return{rtNestedSplits:i[0],rtDenseValues:i[1]}}const ph=w({raggedRange_:nb});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sb(t,e,n,s,r){const a=m(t,"shape","raggedTensorToTensor","int32"),o=m(e,"values","raggedTensorToTensor"),i=m(n,"defaultValue","raggedTensorToTensor",o.dtype),u=s.map((c,f)=>m(c,`tensors${f}`,"raggedTensorToTensor","int32")),l={shape:a,values:o,defaultValue:i,rowPartitionTensors:u},h={rowPartitionTypes:r};return N.runKernel(mu,l,h)}const fh=w({raggedTensorToTensor_:sb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rb(t,e,n){Se(t);const s=W(t);let r=null;if(n==null||n==="float32")r=new Float32Array(s);else if(n==="int32")r=new Int32Array(s);else if(n==="bool")r=new Uint8Array(s);else throw new Error(`Unknown data type ${n}`);for(let a=0;a<s;a++)r[a]=e();return N.makeTensor(r,t,n)}const dh=w({rand_:rb});var Yr={exports:{}};Yr.exports;(function(t){(function(e,n,s){function r(u){var l=this,h=i();l.next=function(){var c=2091639*l.s0+l.c*23283064365386963e-26;return l.s0=l.s1,l.s1=l.s2,l.s2=c-(l.c=c|0)},l.c=1,l.s0=h(" "),l.s1=h(" "),l.s2=h(" "),l.s0-=h(u),l.s0<0&&(l.s0+=1),l.s1-=h(u),l.s1<0&&(l.s1+=1),l.s2-=h(u),l.s2<0&&(l.s2+=1),h=null}function a(u,l){return l.c=u.c,l.s0=u.s0,l.s1=u.s1,l.s2=u.s2,l}function o(u,l){var h=new r(u),c=l&&l.state,f=h.next;return f.int32=function(){return h.next()*4294967296|0},f.double=function(){return f()+(f()*2097152|0)*11102230246251565e-32},f.quick=f,c&&(typeof c=="object"&&a(c,h),f.state=function(){return a(h,{})}),f}function i(){var u=4022871197,l=function(h){h=String(h);for(var c=0;c<h.length;c++){u+=h.charCodeAt(c);var f=.02519603282416938*u;u=f>>>0,f-=u,f*=u,u=f>>>0,f-=u,u+=f*4294967296}return(u>>>0)*23283064365386963e-26};return l}n&&n.exports?n.exports=o:this.alea=o})(pt,t)})(Yr);var ab=Yr.exports,Qr={exports:{}};Qr.exports;(function(t){(function(e,n,s){function r(i){var u=this,l="";u.x=0,u.y=0,u.z=0,u.w=0,u.next=function(){var c=u.x^u.x<<11;return u.x=u.y,u.y=u.z,u.z=u.w,u.w^=u.w>>>19^c^c>>>8},i===(i|0)?u.x=i:l+=i;for(var h=0;h<l.length+64;h++)u.x^=l.charCodeAt(h)|0,u.next()}function a(i,u){return u.x=i.x,u.y=i.y,u.z=i.z,u.w=i.w,u}function o(i,u){var l=new r(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},c.int32=l.next,c.quick=c,h&&(typeof h=="object"&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xor128=o})(pt,t)})(Qr);var ob=Qr.exports,ea={exports:{}};ea.exports;(function(t){(function(e,n,s){function r(i){var u=this,l="";u.next=function(){var c=u.x^u.x>>>2;return u.x=u.y,u.y=u.z,u.z=u.w,u.w=u.v,(u.d=u.d+362437|0)+(u.v=u.v^u.v<<4^(c^c<<1))|0},u.x=0,u.y=0,u.z=0,u.w=0,u.v=0,i===(i|0)?u.x=i:l+=i;for(var h=0;h<l.length+64;h++)u.x^=l.charCodeAt(h)|0,h==l.length&&(u.d=u.x<<10^u.x>>>4),u.next()}function a(i,u){return u.x=i.x,u.y=i.y,u.z=i.z,u.w=i.w,u.v=i.v,u.d=i.d,u}function o(i,u){var l=new r(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},c.int32=l.next,c.quick=c,h&&(typeof h=="object"&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xorwow=o})(pt,t)})(ea);var ib=ea.exports,ta={exports:{}};ta.exports;(function(t){(function(e,n,s){function r(i){var u=this;u.next=function(){var h=u.x,c=u.i,f,d;return f=h[c],f^=f>>>7,d=f^f<<24,f=h[c+1&7],d^=f^f>>>10,f=h[c+3&7],d^=f^f>>>3,f=h[c+4&7],d^=f^f<<7,f=h[c+7&7],f=f^f<<13,d^=f^f<<9,h[c]=d,u.i=c+1&7,d};function l(h,c){var f,d=[];if(c===(c|0))d[0]=c;else for(c=""+c,f=0;f<c.length;++f)d[f&7]=d[f&7]<<15^c.charCodeAt(f)+d[f+1&7]<<13;for(;d.length<8;)d.push(0);for(f=0;f<8&&d[f]===0;++f);for(f==8?d[7]=-1:d[f],h.x=d,h.i=0,f=256;f>0;--f)h.next()}l(u,i)}function a(i,u){return u.x=i.x.slice(),u.i=i.i,u}function o(i,u){i==null&&(i=+new Date);var l=new r(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},c.int32=l.next,c.quick=c,h&&(h.x&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xorshift7=o})(pt,t)})(ta);var ub=ta.exports,na={exports:{}};na.exports;(function(t){(function(e,n,s){function r(i){var u=this;u.next=function(){var h=u.w,c=u.X,f=u.i,d,y;return u.w=h=h+1640531527|0,y=c[f+34&127],d=c[f=f+1&127],y^=y<<13,d^=d<<17,y^=y>>>15,d^=d>>>12,y=c[f]=y^d,u.i=f,y+(h^h>>>16)|0};function l(h,c){var f,d,y,S,b,T=[],A=128;for(c===(c|0)?(d=c,c=null):(c=c+"\0",d=0,A=Math.max(A,c.length)),y=0,S=-32;S<A;++S)c&&(d^=c.charCodeAt((S+32)%c.length)),S===0&&(b=d),d^=d<<10,d^=d>>>15,d^=d<<4,d^=d>>>13,S>=0&&(b=b+1640531527|0,f=T[S&127]^=d+b,y=f==0?y+1:0);for(y>=128&&(T[(c&&c.length||0)&127]=-1),y=127,S=4*128;S>0;--S)d=T[y+34&127],f=T[y=y+1&127],d^=d<<13,f^=f<<17,d^=d>>>15,f^=f>>>12,T[y]=d^f;h.w=b,h.X=T,h.i=y}l(u,i)}function a(i,u){return u.i=i.i,u.w=i.w,u.X=i.X.slice(),u}function o(i,u){i==null&&(i=+new Date);var l=new r(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},c.int32=l.next,c.quick=c,h&&(h.X&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.xor4096=o})(pt,t)})(na);var lb=na.exports,sa={exports:{}};sa.exports;(function(t){(function(e,n,s){function r(i){var u=this,l="";u.next=function(){var c=u.b,f=u.c,d=u.d,y=u.a;return c=c<<25^c>>>7^f,f=f-d|0,d=d<<24^d>>>8^y,y=y-c|0,u.b=c=c<<20^c>>>12^f,u.c=f=f-d|0,u.d=d<<16^f>>>16^y,u.a=y-c|0},u.a=0,u.b=0,u.c=-1640531527,u.d=1367130551,i===Math.floor(i)?(u.a=i/4294967296|0,u.b=i|0):l+=i;for(var h=0;h<l.length+20;h++)u.b^=l.charCodeAt(h)|0,u.next()}function a(i,u){return u.a=i.a,u.b=i.b,u.c=i.c,u.d=i.d,u}function o(i,u){var l=new r(i),h=u&&u.state,c=function(){return(l.next()>>>0)/4294967296};return c.double=function(){do var f=l.next()>>>11,d=(l.next()>>>0)/4294967296,y=(f+d)/(1<<21);while(y===0);return y},c.int32=l.next,c.quick=c,h&&(typeof h=="object"&&a(h,l),c.state=function(){return a(l,{})}),c}n&&n.exports?n.exports=o:this.tychei=o})(pt,t)})(sa);var cb=sa.exports,mh={exports:{}};const hb={},pb=Object.freeze(Object.defineProperty({__proto__:null,default:hb},Symbol.toStringTag,{value:"Module"})),fb=cr(pb);(function(t){(function(e,n,s){var r=256,a=6,o=52,i="random",u=s.pow(r,a),l=s.pow(2,o),h=l*2,c=r-1,f;function d(E,v,_){var O=[];v=v==!0?{entropy:!0}:v||{};var D=T(b(v.entropy?[E,$(n)]:E??A(),3),O),F=new y(O),B=function(){for(var P=F.g(a),M=u,U=0;P<l;)P=(P+U)*r,M*=r,U=F.g(1);for(;P>=h;)P/=2,M/=2,U>>>=1;return(P+U)/M};return B.int32=function(){return F.g(4)|0},B.quick=function(){return F.g(4)/4294967296},B.double=B,T($(F.S),n),(v.pass||_||function(P,M,U,Y){return Y&&(Y.S&&S(Y,F),P.state=function(){return S(F,{})}),U?(s[i]=P,M):P})(B,D,"global"in v?v.global:this==s,v.state)}function y(E){var v,_=E.length,O=this,D=0,F=O.i=O.j=0,B=O.S=[];for(_||(E=[_++]);D<r;)B[D]=D++;for(D=0;D<r;D++)B[D]=B[F=c&F+E[D%_]+(v=B[D])],B[F]=v;(O.g=function(P){for(var M,U=0,Y=O.i,se=O.j,Ee=O.S;P--;)M=Ee[Y=c&Y+1],U=U*r+Ee[c&(Ee[Y]=Ee[se=c&se+M])+(Ee[se]=M)];return O.i=Y,O.j=se,U})(r)}function S(E,v){return v.i=E.i,v.j=E.j,v.S=E.S.slice(),v}function b(E,v){var _=[],O=typeof E,D;if(v&&O=="object")for(D in E)try{_.push(b(E[D],v-1))}catch{}return _.length?_:O=="string"?E:E+"\0"}function T(E,v){for(var _=E+"",O,D=0;D<_.length;)v[c&D]=c&(O^=v[c&D]*19)+_.charCodeAt(D++);return $(v)}function A(){try{var E;return f&&(E=f.randomBytes)?E=E(r):(E=new Uint8Array(r),(e.crypto||e.msCrypto).getRandomValues(E)),$(E)}catch{var v=e.navigator,_=v&&v.plugins;return[+new Date,e,_,e.screen,$(n)]}}function $(E){return String.fromCharCode.apply(0,E)}if(T(s.random(),n),t.exports){t.exports=d;try{f=fb}catch{}}else s["seed"+i]=d})(typeof self<"u"?self:pt,[],Math)})(mh);var db=mh.exports,mb=ab,gb=ob,yb=ib,bb=ub,wb=lb,Nb=cb,Rt=db;Rt.alea=mb;Rt.xor128=gb;Rt.xorwow=yb;Rt.xorshift7=bb;Rt.xor4096=wb;Rt.tychei=Nb;var ra=Rt;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sb=.001,gh=.1;function Tb(t,e,n){return n==null&&(n=aa()),Ks(t,e,(s,r)=>oa(s,r,n))}function aa(){return N.backend.floatPrecision()===32?Sb:gh}function Ks(t,e,n){let s=!0;if((ae(t)||ae(e))&&(s=!1),ae(t)&&ae(e)&&(s=!0),s){const o=t.constructor.name,i=e.constructor.name;if(o!==i)throw new Error(`Arrays are of different type. Actual: ${o}. Expected: ${i}`)}if(Array.isArray(t)&&Array.isArray(e)){const o=ze(t),i=ze(e);if(!Re(o,i))throw new Error(`Arrays have different shapes. Actual: [${o}]. Expected: [${i}]`)}const r=ae(t)?t:ut(t),a=ae(e)?e:ut(e);if(r.length!==a.length)throw new Error(`Arrays have different lengths actual: ${r.length} vs expected: ${a.length}.
Actual:   ${r}.
Expected: ${a}.`);for(let o=0;o<a.length;++o){const i=r[o],u=a[o];if(!n(i,u))throw new Error(`Arrays differ: actual[${o}] = ${i}, expected[${o}] = ${u}.
Actual:   ${r}.
Expected: ${a}.`)}typeof expect<"u"&&expect().nothing()}function $b(t,e){t().then(()=>e.fail(),()=>e()),typeof expect<"u"&&expect().nothing()}function Eb(t,e){const n=typeof e=="string"||typeof e=="number"||typeof e=="boolean"?[e]:e;return nt(t)||nt(t[0])||nt(e)||nt(e[0])?Ks(t,n,(s,r)=>s==r):Ks(t,e,(s,r)=>oa(s,r,0))}function kb(t,e,n){if(n==null&&(n=aa()),!oa(t,e,n))throw new Error(`Numbers differ: actual === ${t}, expected === ${e}`);typeof expect<"u"&&expect().nothing()}function oa(t,e,n){return!isFinite(t)&&!isFinite(e)?!0:!(isNaN(t)||isNaN(e)||Math.abs(t-e)>n)}function vb(t,e,n){for(let s=0;s<t.length;s++)if(t[s]<e||t[s]>n)throw new Error(`Value out of range:${t[s]} low: ${e}, high: ${n}`)}function _b(t,e){const n=new Float32Array(t),s=new Float32Array(e);if(n.length!==s.length)throw new Error(`Expected ArrayBuffer to be of length ${s.length}, but it was ${n.length}`);for(let r=0;r<s.length;r++)if(n[r]!==s[r])throw new Error(`Expected ArrayBuffer value at ${r} to be ${s[r]} but got ${n[r]} instead`)}function yh(t){for(let e=0;e<t.length;e++){const n=t[e];Array.isArray(n)?yh(n):t[e]=In(n)}return t}function Ib(t){const e=document.createElement("video");return"playsInline"in e&&(e.playsInline=!0),e.muted=!0,e.loop=!0,e.style.position="fixed",e.style.left="0px",e.style.top="0px",e.preload="auto",e.appendChild(t),new Promise(n=>{e.addEventListener("loadeddata",s=>n(e)),e.load()})}async function xb(t){await t.play(),"requestVideoFrameCallback"in t&&await new Promise(e=>{t.requestVideoFrameCallback(e)})}const Ab=Object.freeze(Object.defineProperty({__proto__:null,TEST_EPSILON_FLOAT16:gh,createVideoElement:Ib,encodeStrings:yh,expectArrayBuffersEqual:_b,expectArraysClose:Tb,expectArraysEqual:Eb,expectNumbersClose:kb,expectPromiseToFail:$b,expectValuesInRange:vb,play:xb,testEpsilon:aa},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ia{constructor(e,n,s,r,a){this.mean=e,this.stdDev=n,this.dtype=s,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const o=a||Math.random();this.random=ra.alea(o.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let e,n,s=!1;for(;!s;){let r,a,o;do r=2*this.random()-1,a=2*this.random()-1,o=r*r+a*a;while(o>=1||o===0);const i=Math.sqrt(-2*Math.log(o)/o);e=this.mean+this.stdDev*r*i,n=this.mean+this.stdDev*a*i,(!this.truncated||this.isValidTruncated(e))&&(s=!0)}return(!this.truncated||this.isValidTruncated(n))&&(this.nextVal=this.convertValue(n)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype==="float32"?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}}class Ob{constructor(e,n,s,r){this.alpha=e,this.beta=1/n,this.dtype=s;const a=r||Math.random();this.randu=ra.alea(a.toString()),this.randn=new ia(0,1,s,!1,this.randu()),e<1?this.d=e+2/3:this.d=e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,n,s,r,a,o;for(;;){do r=this.randn.nextValue(),o=1+this.c*r;while(o<=0);if(o*=o*o,e=r*r,n=1-.331*e*e,s=.5*e+this.d*(1-o+Math.log(o)),a=this.randu(),a<n||Math.log(a)<s)break}return o=1/this.beta*this.d*o,this.alpha<1&&(o*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(o)}convertValue(e){return this.dtype==="float32"?e:Math.round(e)}}class Db{constructor(e=0,n=1,s,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=e,this.range=n-e,this.dtype=s,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${e} - ${n} <= 1 and dtype is not float`);this.random=ra.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fb(t,e,n=1,s="float32",r){if(Se(t),n==null&&(n=1),s==null&&(s="float32"),s!=="float32"&&s!=="int32")throw new Error(`Unsupported data type ${s}`);const a=new Ob(e,n,s,r),o=Ve(t,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const bh=w({randomGamma_:Fb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cb(t,e=0,n=1,s,r){if(Se(t),s!=null&&s==="bool")throw new Error(`Unsupported data type ${s}`);const a=new ia(e,n,s,!1,r),o=Ve(t,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const ua=w({randomNormal_:Cb});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rb(t,e,n){if(e!=null&&e==="bool")throw new Error(`Unsupported data type ${e}`);return ua(t,0,1,e,n)}const wh=w({randomStandardNormal_:Rb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pb(t,e=0,n=1,s="float32",r){Se(t);const a=Ve(t,s),o=new Db(e,n,null,r);for(let i=0;i<a.values.length;i++)a.values[i]=o.nextValue();return a.toTensor()}const fs=w({randomUniform_:Pb});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bb(t,e,n,s){return fs(t,e,n,"int32",s)}const Nh=w({randomUniformInt_:Bb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jt(t,e,n=1,s="float32"){if(n===0)throw new Error("Cannot have a step of zero");const r={start:t,stop:e,step:n,dtype:s};return N.runKernel(gu,{},r)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lb(t){const n={input:m(t,"input","real")};return N.runKernel(yu,n)}const Yt=w({real_:Lb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zb(t){const n={x:m(t,"x","reciprocal")};return N.runKernel(bu,n)}const Sh=w({reciprocal_:zb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vb(t){const n={x:m(t,"x","relu")};return N.runKernel(wu,n)}const Bn=w({relu_:Vb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jb(t){const n={x:m(t,"x","relu6")};return N.runKernel($u,n)}const la=w({relu6_:jb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mb(t,e){const s={x:m(t,"x","reverse")},r={dims:e};return N.runKernel(Eu,s,r)}const ht=w({reverse_:Mb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wb(t){const e=m(t,"x","reverse");return g(e.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${e.rank}.`),ht(e,0)}const Th=w({reverse1d_:Wb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ub(t,e){const n=m(t,"x","reverse");return g(n.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${n.rank}.`),ht(n,e)}const $h=w({reverse2d_:Ub});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qb(t,e){const n=m(t,"x","reverse");return g(n.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${n.rank}.`),ht(n,e)}const Eh=w({reverse3d_:qb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gb(t,e){const n=m(t,"x","reverse");return g(n.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${n.rank}.`),ht(n,e)}const kh=w({reverse4d_:Gb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hb(t){const n={x:m(t,"x","round")};return N.runKernel(ku,n)}const ca=w({round_:Hb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kb(t){const n={x:m(t,"x","rsqrt","float32")};return N.runKernel(vu,n)}const vh=w({rsqrt_:Kb});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xb(t){const n={x:m(t,"x","selu")};return N.runKernel(Ou,n)}const _h=w({selu_:Xb});function Zb(t,e,n,s,r,a=[1,1],o="NHWC"){const i=m(t,"x","separableConv2d"),u=m(e,"depthwiseFilter","separableConv2d"),l=m(n,"pointwiseFilter","separableConv2d");let h=i,c=!1;if(i.rank===3&&(c=!0,h=k(i,[1,i.shape[0],i.shape[1],i.shape[2]])),o==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");g(h.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${h.rank}.`),g(u.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${u.rank}.`),g(l.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${u.rank}.`),g(l.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${l.shape[0]}.`),g(l.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${l.shape[1]}.`);const f=u.shape[2],d=u.shape[3];g(l.shape[2]===f*d,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${f*d}, but got ${l.shape[2]}.`);const y=ls(h,u,s,r,o,a),b=Dn(y,l,1,"valid",o);return c?k(b,[b.shape[1],b.shape[2],b.shape[3]]):b}const Ih=w({separableConv2d_:Zb});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Jb(t,e){const n=m(t,"x","setdiff1d"),s=m(e,"y","setdiff1d");g(n.dtype===s.dtype,()=>`x and y should have the same dtype, but got x (${n.dtype}) and y (${s.dtype}).`),g(n.rank===1,()=>`x should be 1D tensor, but got x (${n.shape}).`),g(s.rank===1,()=>`y should be 1D tensor, but got y (${s.shape}).`);const r=await n.data(),a=await s.data(),o=new Set(a);let i=0;for(let h=0;h<r.length;h++)o.has(r[h])||i++;const u=new Jn([i],n.dtype),l=new Jn([i],"int32");for(let h=0,c=0;h<r.length;h++)o.has(r[h])||(u.values[c]=r[h],l.values[c]=h,c++);return[u.toTensor(),l.toTensor()]}const xh=Jb;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yb(t){const n={x:m(t,"x","sign")};return N.runKernel(Ru,n)}const Ah=w({sign_:Yb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qb(t){const n={x:m(t,"x","sin","float32")};return N.runKernel(Fu,n)}const Oh=w({sin_:Qb});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function e0(t){const n={x:m(t,"x","sinh")};return N.runKernel(Cu,n)}const Dh=w({sinh_:e0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function t0(t,e,n){const s=m(t,"x","slice1d");return g(s.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${s.rank} tensor`),q(s,[e],[n])}const Fh=w({slice1d_:t0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function n0(t,e,n){const s=m(t,"x","slice2d");return g(s.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${s.rank} tensor`),q(s,e,n)}const Ch=w({slice2d_:n0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function s0(t,e,n){const s=m(t,"x","slice3d");return g(s.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${s.rank} tensor`),q(s,e,n)}const Rh=w({slice3d_:s0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function r0(t,e,n){const s=m(t,"x","slice4d");return g(s.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${s.rank} tensor`),q(s,e,n)}const Ph=w({slice4d_:r0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function a0(t,e=-1){const n=m(t,"logits","softmax","float32");if(e===-1&&(e=n.rank-1),e!==n.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${n.rank} and dim was ${e}`);const s={logits:n},r={dim:e};return N.runKernel(Mu,s,r)}const Bh=w({softmax_:a0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function o0(t){g(t.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${t.dtype}.`);const e={input:t};return N.runKernel(bi,e)}const ds=w({fft_:o0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function i0(t){g(t.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${t.dtype}.`);const e={input:t};return N.runKernel(Ii,e)}const $n=w({ifft_:i0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function u0(t){const e=t.shape[t.shape.length-1],n=t.size/e;let s;if(e<=2){const r=k(t,[n,e]);s=$n(r)}else{const r=[n,2*(e-1)],a=k(Yt(t),[n,e]),o=k(Pn(t),[n,e]),i=ht(q(a,[0,1],[n,e-2]),1),u=x(ht(q(o,[0,1],[n,e-2]),1),z(-1)),l=ue([a,i],1),h=ue([o,u],1),c=k(Je(l,h),[r[0],r[1]]);s=$n(c)}if(s=Yt(s),t.rank===3&&t.shape[0]!==0){const r=s,a=t.shape[0];s=k(s,[a,s.shape[0]/a,s.shape[1]]),r.dispose()}return s}const ha=w({irfft_:u0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function l0(t,e,n=0){const r={x:m(t,"x","split")},a={numOrSizeSplits:e,axis:n};return N.runKernel(ju,r,a)}const Qt=w({split_:l0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function c0(t,e){g(t.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${t.dtype}`);let n=t.shape[t.shape.length-1];const s=t.size/n;let r;if(e!=null&&e<n){const y=t.shape.map(b=>0),S=t.shape.map(b=>b);S[t.shape.length-1]=e,r=q(t,y,S),n=e}else if(e!=null&&e>n){const y=t.shape.map(S=>S);y[t.shape.length-1]=e-n,r=ue([t,At(y)],t.shape.length-1),n=e}else r=t;const a=Ne(r),o=k(Je(r,a),[s,n]),i=ds(o),u=Math.floor(n/2)+1,l=Yt(i),h=Pn(i),c=Qt(l,[u,n-u],l.shape.length-1),f=Qt(h,[u,n-u],h.shape.length-1),d=r.shape.slice();return d[r.shape.length-1]=u,k(Je(c[0],f[0]),d)}const ms=w({rfft_:c0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function h0(t,e){let n=m(t,"a","squaredDifference"),s=m(e,"b","squaredDifference");[n,s]=ee(n,s),ne(n.shape,s.shape);const r={a:n,b:s},a={};return N.runKernel(Ku,r,a)}const pa=w({squaredDifference_:h0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function p0(t,e){const n=m(t,"x","squeeze","string_or_numeric");return k(n,ho(n.shape,e).newShape)}const gs=w({squeeze_:p0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function f0(t,e=0){const n=gn(t,"tensors","stack","string_or_numeric");g(n.length>=1,()=>"Pass at least one tensor to tf.stack"),n.length>0&&g(e<=n[0].rank,()=>"Axis must be <= rank of the tensor");const s=n,r={axis:e};return N.runKernel(uu,s,r)}const We=w({stack_:f0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function d0(t,e=0){const s={x:m(t,"x","step")},r={alpha:e};return N.runKernel(ll,s,r)}const fa=w({step_:d0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function m0(t,e,n,s,r=0,a=0,o=0,i=0,u=0){const h={x:m(t,"x","stridedSlice","string_or_numeric")},c={begin:e,end:n,strides:s,beginMask:r,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:u};return N.runKernel(Zu,h,c)}const Lh=w({stridedSlice_:m0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function g0(t){const n={x:m(t,"x","tan","float32")};return N.runKernel(tl,n)}const zh=w({tan_:g0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $e(t,e){Ft(t);const n=ze(t,e);if(n.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return ft(t,null,n,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ut(t,e,n){if(Ft(t),e!=null&&e.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const s=ze(t,n);if(s.length!==2&&s.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return ft(t,e,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function da(t,e,n){if(Ft(t),e!=null&&e.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const s=ze(t,n);if(s.length!==3&&s.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return ft(t,e,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vh(t,e,n){if(Ft(t),e!=null&&e.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const s=ze(t,n);if(s.length!==4&&s.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return ft(t,e,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jh(t,e,n){if(Ft(t),e!=null&&e.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const s=ze(t,n);if(s.length!==5&&s.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return ft(t,e,s,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mh(t,e,n){if(Ft(t),e!=null&&e.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const s=ze(t,n);if(s.length!==6&&s.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(s.length===1&&e==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return e=e||s,ft(t,e,s,n)}function ma(t,e,n){const s=e.rank>1?e.shape[e.rank-1]:1,r=e.rank>1?e.rank-1:1,a=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${e.shape}, shape: ${t}, sliceDim: ${s}, and batchDim: ${r}.`;if(n.rank<r)throw new Error(a+` update.rank < ${r}. `);if(t.length<s+(n.rank-r))throw new Error(a+` Output shape length < ${s+(n.rank-r)}`);if(n.rank!==r+t.length-s)throw new Error(a+` update.rank != ${r+t.length-s}`);for(let o=0;o<r;++o)if(n.shape[o]!==e.shape[o])throw new Error(a+` updates.shape[${o}] (${n.shape[o]}) != indices.shape[${o}] (${e.shape[o]}).`);for(let o=0;o<n.rank-r;++o)if(n.shape[o+r]!==t[o+s])throw new Error(a+` updates.shape[${o+r}] (${n.shape[o+r]}) != shape[${o+r}] (${t[o+r]})`)}function ys(t,e,n){if(e.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${e.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(e.size===0)throw new Error(`Indices specified for empty output. indices shape: ${e.shape}`);if(t.size===0)throw new Error(`Updates specified for empty output. updates shape: ${t.shape}`)}ma(n,e,t)}function Wh(t,e,n){const s=e.shape.length,r=s>1?e.shape[s-1]:1,a=n.length;let o=1;for(let c=r;c<a;++c)o*=n[c];const i=r<1?1:r,u=W(e.shape)/i,l=[...en(n.slice(0,r)),1],h=W(n);return{sliceRank:r,numUpdates:u,sliceSize:o,strides:l,outputSize:h}}const y0=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:Wh,validateInput:ys,validateUpdateShape:ma},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function b0(t,e,n){const s=m(t,"tensor","tensorScatterupdate"),r=m(e,"indices","tensorScatterupdate","int32"),a=m(n,"updates","tensorScatterupdate");if(ys(a,r,s.shape),s.dtype!==a.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${s.dtype} and ${a.dtype}.`);const o={tensor:s,indices:r,updates:a},i={};return N.runKernel(Iu,o,i)}const Uh=w({tensorScatterUpdate_:b0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function w0(t,e=1,n=!0){const s=m(t,"x","topk");if(s.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const r=s.shape[s.shape.length-1];if(e<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${e}`);if(e>r)throw new Error(`'k' passed to topk() must be <= the last dimension (${r}) but got ${e}`);const a={x:s},o={k:e,sorted:n},[i,u]=N.runKernel(sl,a,o);return{values:i,indices:u}}const qh=w({topk_:w0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function N0(t,e=0,n=1,s,r){if(Se(t),s!=null&&s==="bool")throw new Error("Unsupported data type $ { dtype }");const a=new ia(e,n,s,!0,r),o=Ve(t,s);for(let i=0;i<o.values.length;i++)o.values[i]=a.nextValue();return o.toTensor()}const Gh=w({truncatedNormal_:N0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function S0(t,e=0){const n=m(t,"x","unique","string_or_numeric");g(n.rank>0,()=>"The input tensor must be at least 1D");const s={x:n},r={axis:e},[a,o]=N.runKernel(al,s,r);return{values:a,indices:o}}const Hh=w({unique_:S0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function T0(t,e,n){const s=m(t,"x","unsortedSegmentSum"),r=m(e,"segmentIds","unsortedSegmentSum","int32");g(qt(n),()=>"numSegments must be of dtype int");const a={x:s,segmentIds:r},o={numSegments:n};return N.runKernel(il,a,o)}const Kh=w({unsortedSegmentSum_:T0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $0(t,e=0){const n=m(t,"x","unstack","string_or_numeric");g(e>=-n.shape.length&&e<n.shape.length,()=>`Axis = ${e} is not in [-${n.shape.length}, ${n.shape.length})`);const s={value:n},r={axis:e};return N.runKernel(ol,s,r)}const dt=w({unstack_:$0});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xh(t,e){return ps(t,e,"right")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zh(t,e=!0,n,s){return N.makeVariable(t,e,n,s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jh(t,e){const n=[];for(let a=0;a<e.length;a++)e[a]&&n.push(a);const s=Ve(t,"int32"),r=Ve([n.length,t.length],"int32");for(let a=0;a<n.length;a++){const o=s.indexToLoc(n[a]),i=a*t.length;r.values.set(o,i)}return r.toTensor()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function E0(t){const e=m(t,"condition","whereAsync","bool"),n=await e.data(),s=Jh(e.shape,n);return t!==e&&e.dispose(),s}const ga=E0;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function k0(t,e,n){const s=m(t,"tensor","boolMask"),r=m(e,"mask","boolMask","bool"),a=n??0,o=r.rank,i=s.shape;g(o>0,()=>"mask cannot be scalar"),fe(i.slice(a,a+o),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let u=1;for(let S=a;S<a+o;S++)u*=i[S];const l=i.slice(0,a).concat([u],i.slice(a+o)),h=k(s,l),c=k(r,[-1]),f=await ga(c),d=gs(f,[1]),y=zr(h,d,a);return t!==s&&s.dispose(),e!==r&&r.dispose(),d.dispose(),h.dispose(),c.dispose(),f.dispose(),y}const Yh=k0;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function v0(t,e,n){const s=m(t,"x","transpose");if(e==null&&(e=s.shape.map((o,i)=>i).reverse()),g(s.rank===e.length,()=>`Error in transpose: rank of input ${s.rank} must match length of perm ${e}.`),e.forEach(o=>{g(o>=0&&o<s.rank,()=>`All entries in 'perm' must be between 0 and ${s.rank-1} but got ${e}`)}),s.rank<=1)return s.clone();const r={x:s},a={perm:e};return s.dtype==="complex64"?j(()=>{let o=Yt(s),i=Pn(s);return o=N.runKernel(jn,{x:o},a),i=N.runKernel(jn,{x:i},a),n&&(i=Ce(i)),Je(o,i)}):N.runKernel(jn,r,a)}const En=w({transpose_:v0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _0(t,e,n,s,r=!0){const a=m(t,"v","movingAverage"),o=m(e,"x","movingAverage"),i=m(n,"decay","movingAverage");kl(a,o),g(Re(a.shape,o.shape),()=>"Shape mismatch in v and x");const u=z(1),l=L(u,i);let h=x(L(o,a),l);if(r){g(s!=null,()=>"When using zeroDebias: true, step is required.");const c=m(s,"step","movingAverage");h=K(h,L(u,Xt(i,c)))}return C(a,h)}const Qh=w({movingAverage_:_0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function I0(t,e,n){Se(n);const s=m(t,"indices","scatterND","int32"),r=m(e,"updates","scatterND");ys(r,s,n);const a={indices:s,updates:r},o={shape:n};return N.runKernel(_u,a,o)}const ep=w({scatterND_:I0});function x0(t,e,n,s){if(t.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${t.shape}.`);const r=t.rank>0?t.shape[0]:1,a=t.rank>1?t.shape[1]:1;if(n.length!==a)throw new Error(`outputShape has incorrect number of elements:, ${n.length}, should be: ${a}.`);const o=e.size;if(!(e.rank===0||e.rank===1&&o===r))throw new Error(`sparseValues has incorrect shape ${e.shape}, should be [] or [${r}]`);if(e.dtype!==s.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function A0(t,e,n,s=0){Se(n);const r=m(t,"sparseIndices","sparseToDense","int32"),a=m(e,"sparseValues","sparseToDense","string_or_numeric"),o=m(s,"defaultValue","sparseToDense",a.dtype);x0(r,a,n,o);const i={sparseIndices:r,sparseValues:a,defaultValue:o},u={outputShape:n};return N.runKernel(Hu,i,u)}const tp=w({sparseToDense_:A0});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function O0(t,e){const n=m(e,"indices","gatherND","int32"),r={params:m(t,"x","gatherND","string_or_numeric"),indices:n};return N.runKernel(ki,r)}const np=w({gatherND_:O0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D0(t,e){if(e==null)return t.shape.slice();if(Re(t.shape,e))return e;if(t.shape.length===e.length){const n=[];for(let s=0;s<t.shape.length;s++)e[s]==null&&t.shape[s]!=null?n.push(t.shape[s]):n.push(e[s]);return n}return e}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function F0(t,e,n,s){const r=m(t,"x","dropout");if(g(r.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${r.dtype} tensor instead.`),g(e>=0&&e<1,()=>`rate must be a float in the range [0, 1), but got ${e}.`),e===0)return t instanceof te?r.clone():r;const a=D0(r,n),o=1-e,i=K(Lr(C(fs(a,0,1,"float32",s),o)),o);return x(r,i)}const sp=w({dropout_:F0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ya(t){return Math.floor(Math.pow(2,Math.ceil(Math.log(t)/Math.log(2))))}function bs(t,e,n){const s=1-t%2,r=new Float32Array(t);for(let a=0;a<t;++a){const o=2*Math.PI*a/(t+s-1);r[a]=e-n*Math.cos(o)}return $e(r,"float32")}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function C0(t,e,n=1){const s=m(t,"predictions","inTopK"),r=m(e,"targets","inTopK");g(s.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${s.rank}`),g(s.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${s.rank} and targets rank ${r.rank}`),fe(s.shape.slice(0,s.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const a=s.shape[s.shape.length-1];g(n>0&&n<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${n}`);const o=await s.data(),i=await r.data(),[u,l]=[o.length/a,a],h=po("bool",u);for(let c=0;c<u;c++){const f=c*l,d=o.subarray(f,f+l),y=[];for(let S=0;S<d.length;S++)y.push({value:d[S],index:S});y.sort((S,b)=>b.value-S.value),h[c]=0;for(let S=0;S<n;S++)if(y[S].index===i[c]){h[c]=1;break}}return t!==s&&s.dispose(),e!==r&&r.dispose(),Fe(h,r.shape,"bool")}const rp=C0;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function R0(t,e,n,s,r,a="NHWC",o){let i=t;t.rank===3&&(i=k(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let u=e;u.rank===3&&(u=k(e,[1,e.shape[0],e.shape[1],e.shape[2]])),g(i.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${i.shape}.`),g(u.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${u.shape}.`),g(n.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${n}.`);const l=a==="NHWC"?i.shape[3]:i.shape[1],h=a==="NHWC"?u.shape[3]:u.shape[1];g(l===n[2],()=>`Error in conv2dDerFilter: depth of input ${l}) must match input depth in filter (${n[2]}.`),g(h===n[3],()=>`Error in conv2dDerFilter: depth of dy (${h}) must match output depth for filter (${n[3]}).`),Ae("conv2dDerFilter",r,o);const c={x:i,dy:u},f={strides:s,pad:r,dataFormat:a,dimRoundingMode:o,filterShape:n};return N.runKernel(Ko,c,f)}const P0=w({conv2DBackpropFilter_:R0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ws(t,e,n){if(n==null||n==="linear")return t;if(n==="relu")return x(t,fa(e));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function Ns(t,e){let n=e;const s=Fr(t.shape,e.shape);return s.length>0&&(n=H(n,s)),k(n,t.shape)}function Ss(t,e,n,s){if(e==="linear")return t;if(e==="relu")return Bn(t);if(e==="elu")return Rr(t);if(e==="relu6")return la(t);if(e==="prelu")return Jr(t,n);if(e==="leakyrelu")return jr(t,s);if(e==="sigmoid")return Et(t);throw new Error(`Unknown fused activation ${e}.`)}const Ts=(t,e)=>!(t>0)||e==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function B0({x:t,filter:e,strides:n,pad:s,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:l,leakyreluAlpha:h}){if(u=u||"linear",Ts(N.state.gradientDepth,u)===!1){g(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let _=Dn(t,e,n,s,r,a,o);return i!=null&&(_=C(_,i)),Ss(_,u,l,h)}const c=m(t,"x","conv2d","float32"),f=m(e,"filter","conv2d","float32");let d=c,y=!1;c.rank===3&&(y=!0,d=k(c,[1,c.shape[0],c.shape[1],c.shape[2]])),g(d.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${d.rank}.`),g(f.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${f.rank}.`),Ae("fused conv2d",s,o);const S=r==="NHWC"?d.shape[3]:d.shape[1];g(f.shape[2]===S,()=>`Error in conv2d: depth of input (${S}) must match input depth for filter ${f.shape[2]}.`),g(Ye(n,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`);const b=An(d.shape,f.shape,n,a,s,o);let T;i!=null&&(T=m(i,"bias","fused conv2d"),[T]=ee(T,c),r==="NHWC"?ne(b.outShape,T.shape):(g(T.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${T.shape.length}.`),g(T.shape.length===0||T.shape[0]===b.outChannels||T.shape[0]===1,()=>`Error in fused conv2d: bias shape (${T.shape}) is not compatible with the number of output channels (${b.outChannels})`)));let A;if(l!=null){const _=l.shape;if(g(_.length<=1||_.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${_.length}.`),_.length===1)g(_[0]===1||_[0]===b.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${_}) is not compatible with the number of output channels (${b.outChannels}).`);else if(_.length===3)try{ne(_,b.outShape)}catch{const D=`Error in fused conv2d: PReLU activation weights (${_}) is not compatible with the output shape of the conv2d (${b.outShape}).`;throw Error(D)}A=m(l,"prelu weights","fused conv2d")}const $=(_,O)=>{g(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[D,F,B,P]=O,M=ws(_,B,u);g(wn(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);const U=bc(F.shape,M,D,n,s),Y=P0(F,M,D.shape,n,s),se=[U,Y];if(P!=null){const Ee=Ns(P,M);se.push(Ee)}return se},E={x:d,filter:f,bias:T,preluActivationWeights:A},v={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:h};return i==null?Me((O,D,F)=>{let B=N.runKernel(Fs,E,v);return F([D,O,B]),y&&(B=k(B,[B.shape[1],B.shape[2],B.shape[3]])),{value:B,gradFunc:$}})(d,f):Me((O,D,F,B)=>{let P=N.runKernel(Fs,E,v);return B([D,O,P,F]),y&&(P=k(P,[P.shape[1],P.shape[2],P.shape[3]])),{value:P,gradFunc:$}})(d,f,T)}const L0=w({fusedConv2d_:B0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z0(t,e,n,s,r,a=[1,1],o){let i=t;t.rank===3&&(i=k(t,[1,t.shape[0],t.shape[1],t.shape[2]]));let u=e;u.rank===3&&(u=k(e,[1,e.shape[0],e.shape[1],e.shape[2]]));const l={x:i,dy:u},h={strides:s,pad:r,dimRoundingMode:o,dilations:a,filterShape:n};return N.runKernel(oi,l,h)}const V0=w({depthwiseConv2dNativeBackpropFilter_:z0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function j0(t,e,n,s,r,a=[1,1],o){let i=e,u=!1;e.rank===3&&(u=!0,i=k(e,[1,e.shape[0],e.shape[1],e.shape[2]]));const l={dy:i,filter:n},h={strides:s,pad:r,dimRoundingMode:o,dilations:a,inputShape:t},c=N.runKernel(ii,l,h);return u?k(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const M0=w({depthwiseConv2dNativeBackpropInput_:j0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function W0({x:t,filter:e,strides:n,pad:s,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:o,bias:i,activation:u="linear",preluActivationWeights:l,leakyreluAlpha:h}){if(Ts(N.state.gradientDepth,u)===!1){let v=ls(t,e,n,s,r,a,o);return i!=null&&(v=C(v,i)),Ss(v,u,l,h)}const c=m(t,"x","depthwiseConv2d","float32"),f=m(e,"filter","depthwiseConv2d","float32");let d=c,y=!1;c.rank===3&&(y=!0,d=k(c,[1,c.shape[0],c.shape[1],c.shape[2]])),g(d.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${d.rank}.`),g(f.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${f.rank}.`),g(d.shape[3]===f.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${d.shape[3]}) must match the inChannels dimension in filter ${f.shape[2]}.`),a==null&&(a=[1,1]),g(Ye(n,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${n} and dilations '${a}'`),Ae("fused depthwiseConv2d",s,o);const S=An(d.shape,f.shape,n,a,s,o,!0);let b;i!=null&&(b=m(i,"bias","fused conv2d"),[b]=ee(b,c),ne(S.outShape,b.shape));let T;l!=null&&(T=m(l,"prelu weights","fused depthwiseConv2d"));const A=(v,_)=>{g(wn(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[O,D,F,B]=_,P=ws(v,F,u),M=M0(D.shape,P,O,n,s,a,o),U=V0(D,P,O.shape,n,s,a,o);if(B!=null){const Y=Ns(b,P);return[M,U,Y]}return[M,U]},$={x:d,filter:f,bias:b,preluActivationWeights:T},E={strides:n,pad:s,dataFormat:r,dilations:a,dimRoundingMode:o,activation:u,leakyreluAlpha:h};return i==null?Me((_,O,D)=>{let F=N.runKernel(Cs,$,E);return D([O,_,F]),y&&(F=k(F,[F.shape[1],F.shape[2],F.shape[3]])),{value:F,gradFunc:A}})(d,f):Me((_,O,D,F)=>{let B=N.runKernel(Cs,$,E);return F([O,_,B,D]),y&&(B=k(B,[B.shape[1],B.shape[2],B.shape[3]])),{value:B,gradFunc:A}})(d,f,b)}const U0=w({fusedDepthwiseConv2d_:W0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function q0({a:t,b:e,transposeA:n=!1,transposeB:s=!1,bias:r,activation:a="linear",preluActivationWeights:o,leakyreluAlpha:i=.2}){if(Ts(N.state.gradientDepth,a)===!1){let P=V(t,e,n,s);return r!=null&&(P=C(P,r)),Ss(P,a,o,i)}let u=m(t,"a","fused matMul"),l=m(e,"b","fused matMul");[u,l]=ee(u,l);const h=n?u.shape[u.rank-2]:u.shape[u.rank-1],c=s?l.shape[l.rank-1]:l.shape[l.rank-2],f=n?u.shape[u.rank-1]:u.shape[u.rank-2],d=s?l.shape[l.rank-2]:l.shape[l.rank-1],y=u.shape.slice(0,-2),S=l.shape.slice(0,-2),b=W(y),T=W(S);g(h===c,()=>`Error in fused matMul: inner shapes (${h}) and (${c}) of Tensors with shapes ${u.shape} and ${l.shape} and transposeA=${n} and transposeB=${s} must match.`);const $=ne(u.shape.slice(0,-2),l.shape.slice(0,-2)).concat([f,d]),E=n?k(u,[b,h,f]):k(u,[b,f,h]),v=s?k(l,[T,d,c]):k(l,[T,c,d]);let _;r!=null&&(_=m(r,"bias","fused matMul"),[_]=ee(_,u),ne($,_.shape));let O;o!=null&&(O=m(o,"prelu weights","fused matMul"));const D=(P,M)=>{const[U,Y,se,Ee]=M,Ue=ws(k(P,se.shape),se,a);let Pt,Bt;if(!n&&!s?(Pt=V(Ue,Y,!1,!0),Bt=V(U,Ue,!0,!1)):!n&&s?(Pt=V(Ue,Y,!1,!1),Bt=V(Ue,U,!0,!1)):n&&!s?(Pt=V(Y,Ue,!1,!0),Bt=V(U,Ue,!1,!1)):(Pt=V(Y,Ue,!0,!0),Bt=V(Ue,U,!0,!0)),r!=null){const Vp=Ns(Ee,Ue);return[Pt,Bt,Vp]}else return[Pt,Bt]},F={a:E,b:v,bias:_,preluActivationWeights:O},B={transposeA:n,transposeB:s,activation:a,leakyreluAlpha:i};return r==null?Me((M,U,Y)=>{const se=N.runKernel(Ds,F,B);return Y([M,U,se]),{value:k(se,$),gradFunc:D}})(E,v):Me((M,U,Y,se)=>{const Ee=N.runKernel(Ds,F,B);return se([M,U,Ee,Y]),{value:k(Ee,$),gradFunc:D}})(E,v,_)}const G0=w({fusedMatMul_:q0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ap=Object.freeze(Object.defineProperty({__proto__:null,conv2d:L0,depthwiseConv2d:U0,matMul:G0},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function H0(t){return bs(t,.54,.46)}const K0=w({hammingWindow_:H0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function X0(t){return bs(t,.5,.5)}const op=w({hannWindow_:X0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Z0(t,e,n,s=!1,r=0){let a=0;const o=[];for(;a+e<=t.size;)o.push(q(t,a,e)),a+=n;if(s)for(;a<t.size;){const i=a+e-t.size,u=ue([q(t,a,e-i),tn([i],r)]);o.push(u),a+=n}return o.length===0?Ut([],[0,e]):k(ue(o),[o.length,e])}const ip=w({frame_:Z0});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function J0(t,e,n,s,r=op){s==null&&(s=ya(e));const a=ip(t,e,n),o=x(a,r(e));return ms(o,s)}const Y0=w({stft_:J0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Q0(t,e,n,s,r="bilinear",a=0){const o=m(t,"image","cropAndResize"),i=m(e,"boxes","cropAndResize","float32"),u=m(n,"boxInd","cropAndResize","int32"),l=i.shape[0];g(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),g(i.rank===2&&i.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${l},4] but had shape ${i.shape}.`),g(u.rank===1&&u.shape[0]===l,()=>`Error in cropAndResize: boxInd must be have size [${l}] but had shape ${i.shape}.`),g(s.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${s.length}.`),g(s[0]>=1&&s[1]>=1,()=>`cropSize must be atleast [1,1], but was ${s}`),g(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const h={image:o,boxes:i,boxInd:u},c={method:r,extrapolationValue:a,cropSize:s};return N.runKernel(ni,h,c)}const ew=w({cropAndResize_:Q0});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tw(t){const e=m(t,"image","flipLeftRight","float32");g(e.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${e.rank}.`);const n={image:e};return N.runKernel(Ni,n,{})}const nw=w({flipLeftRight_:tw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sw(t){const e=m(t,"image","grayscaleToRGB"),n=e.rank-1,s=e.shape[n];g(e.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${e.rank}.`),g(s===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${s}.`);const r=new Array(e.rank);return r.fill(1,0,n),r[n]=3,Wt(e,r)}const rw=w({grayscaleToRGB_:sw});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aw(t){const e=m(t,"image","RGBToGrayscale"),n=e.rank-1,s=e.shape[n];g(e.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${e.rank}.`),g(s===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${s}.`);const r=e.dtype,a=X(e,"float32"),o=$e([.2989,.587,.114]);let i;switch(e.rank){case 2:i=wt("ij,j->i",a,o);break;case 3:i=wt("ijk,k->ij",a,o);break;case 4:i=wt("ijkl,l->ijk",a,o);break;case 5:i=wt("ijklm,m->ijkl",a,o);break;case 6:i=wt("ijklmn,n->ijklm",a,o);break;default:throw new Error("Not a valid tensor rank.")}return i=qe(i,-1),X(i,r)}const ow=w({rgbToGrayscale_:aw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iw(t,e,n=0,s=.5){const r=m(t,"image","rotateWithOffset","float32");g(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const a={image:r},o={radians:e,fillValue:n,center:s};return N.runKernel(cl,a,o)}const uw=w({rotateWithOffset_:iw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sn(t,e,n,s,r,a){s==null&&(s=.5),r==null&&(r=Number.NEGATIVE_INFINITY),a==null&&(a=0);const o=t.shape[0];return n=Math.min(n,o),g(0<=s&&s<=1,()=>`iouThreshold must be in [0, 1], but was '${s}'`),g(t.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${t.rank}'`),g(t.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${t.shape[1]}`),g(e.rank===1,()=>"scores must be a 1D tensor"),g(e.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${e.shape[0]}`),g(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:a}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lw(t,e,n,s=.5,r=Number.NEGATIVE_INFINITY){const a=m(t,"boxes","nonMaxSuppression","float32"),o=m(e,"scores","nonMaxSuppression","float32"),i=sn(a,o,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const u={maxOutputSize:n,iouThreshold:s,scoreThreshold:r};return N.runKernel(su,{boxes:a,scores:o},u)}const cw=w({nonMaxSuppression_:lw});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hw(t,e,n){const s=pw(t,e,n),r=s<0?-(s+1):s;t.splice(r,0,e)}function pw(t,e,n){return dw(t,e,n||fw)}function fw(t,e){return t>e?1:t<e?-1:0}function dw(t,e,n){let s=0,r=t.length,a=0,o=!1;for(;s<r;){a=s+(r-s>>>1);const i=n(e,t[a]);i>0?s=a+1:(r=a,o=!i)}return o?s:-s-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function up(t,e,n,s,r){return ba(t,e,n,s,r,0)}function lp(t,e,n,s,r,a){return ba(t,e,n,s,r,0,!1,a,!0)}function cp(t,e,n,s,r,a){return ba(t,e,n,s,r,a,!0)}function ba(t,e,n,s,r,a,o=!1,i=!1,u=!1){const l=[];for(let b=0;b<e.length;b++)e[b]>r&&l.push({score:e[b],boxIndex:b,suppressBeginIndex:0});l.sort(qa);const h=a>0?-.5/a:0,c=[],f=[];for(;c.length<n&&l.length>0;){const b=l.pop(),{score:T,boxIndex:A,suppressBeginIndex:$}=b;if(T<r)break;let E=!1;for(let v=c.length-1;v>=$;--v){const _=mw(t,A,c[v]);if(_>=s){E=!0;break}if(b.score=b.score*gw(s,h,_),b.score<=r)break}b.suppressBeginIndex=c.length,E||(b.score===T?(c.push(A),f.push(b.score)):b.score>r&&hw(l,b,qa))}const d=c.length,y=n-d;i&&y>0&&(c.push(...new Array(y).fill(0)),f.push(...new Array(y).fill(0)));const S={selectedIndices:c};return o&&(S.selectedScores=f),u&&(S.validOutputs=d),S}function mw(t,e,n){const s=t.subarray(e*4,e*4+4),r=t.subarray(n*4,n*4+4),a=Math.min(s[0],s[2]),o=Math.min(s[1],s[3]),i=Math.max(s[0],s[2]),u=Math.max(s[1],s[3]),l=Math.min(r[0],r[2]),h=Math.min(r[1],r[3]),c=Math.max(r[0],r[2]),f=Math.max(r[1],r[3]),d=(i-a)*(u-o),y=(c-l)*(f-h);if(d<=0||y<=0)return 0;const S=Math.max(a,l),b=Math.max(o,h),T=Math.min(i,c),A=Math.min(u,f),$=Math.max(T-S,0)*Math.max(A-b,0);return $/(d+y-$)}function gw(t,e,n){const s=Math.exp(e*n*n);return n<=t?s:0}function qa(t,e){return t.score-e.score||t.score===e.score&&e.boxIndex-t.boxIndex}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function yw(t,e,n,s=.5,r=Number.NEGATIVE_INFINITY){const a=m(t,"boxes","nonMaxSuppressionAsync"),o=m(e,"scores","nonMaxSuppressionAsync"),i=sn(a,o,n,s,r);n=i.maxOutputSize,s=i.iouThreshold,r=i.scoreThreshold;const u=await Promise.all([a.data(),o.data()]),l=u[0],h=u[1],{selectedIndices:c}=up(l,h,n,s,r);return a!==t&&a.dispose(),o!==e&&o.dispose(),$e(c,"int32")}const bw=yw;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ww(t,e,n,s=.5,r=Number.NEGATIVE_INFINITY,a=0){const o=m(t,"boxes","nonMaxSuppression"),i=m(e,"scores","nonMaxSuppression"),u=sn(o,i,n,s,r,a);n=u.maxOutputSize,s=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const l={boxes:o,scores:i},h={maxOutputSize:n,iouThreshold:s,scoreThreshold:r,softNmsSigma:a},c=N.runKernel(au,l,h);return{selectedIndices:c[0],selectedScores:c[1]}}const Nw=w({nonMaxSuppressionWithScore_:ww});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Sw(t,e,n,s=.5,r=Number.NEGATIVE_INFINITY,a=0){const o=m(t,"boxes","nonMaxSuppressionAsync"),i=m(e,"scores","nonMaxSuppressionAsync"),u=sn(o,i,n,s,r,a);n=u.maxOutputSize,s=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const l=await Promise.all([o.data(),i.data()]),h=l[0],c=l[1],{selectedIndices:f,selectedScores:d}=cp(h,c,n,s,r,a);return o!==t&&o.dispose(),i!==e&&i.dispose(),{selectedIndices:$e(f,"int32"),selectedScores:$e(d)}}const Tw=Sw;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $w(t,e,n,s=.5,r=Number.NEGATIVE_INFINITY,a=!1){const o=m(t,"boxes","nonMaxSuppression"),i=m(e,"scores","nonMaxSuppression"),u=sn(o,i,n,s,r,null),l=u.maxOutputSize,h=u.iouThreshold,c=u.scoreThreshold,f={boxes:o,scores:i},d={maxOutputSize:l,iouThreshold:h,scoreThreshold:c,padToMaxOutputSize:a},y=N.runKernel(ru,f,d);return{selectedIndices:y[0],validOutputs:y[1]}}const Ew=w({nonMaxSuppressionPadded_:$w});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function kw(t,e,n,s=.5,r=Number.NEGATIVE_INFINITY,a=!1){const o=m(t,"boxes","nonMaxSuppressionAsync"),i=m(e,"scores","nonMaxSuppressionAsync"),u=sn(o,i,n,s,r,null),l=u.maxOutputSize,h=u.iouThreshold,c=u.scoreThreshold,[f,d]=await Promise.all([o.data(),i.data()]),{selectedIndices:y,validOutputs:S}=lp(f,d,l,h,c,a);return o!==t&&o.dispose(),i!==e&&i.dispose(),{selectedIndices:$e(y,"int32"),validOutputs:z(S,"int32")}}const vw=kw;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _w(t,e,n=!1,s=!1){const r=m(t,"images","resizeBilinear");g(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),g(e.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${e}.`),g(s===!1||n===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let a=r,o=!1;r.rank===3&&(o=!0,a=k(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:s,size:e},l=N.runKernel(Tu,i,u);return o?k(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const Iw=w({resizeBilinear_:_w});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xw(t,e,n=!1,s=!1){const r=m(t,"images","resizeNearestNeighbor");g(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),g(e.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${e}.`),g(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),g(s===!1||n===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let a=r,o=!1;r.rank===3&&(o=!0,a=k(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const i={images:a},u={alignCorners:n,halfPixelCenters:s,size:e},l=N.runKernel(Su,i,u);return o?k(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const Aw=w({resizeNearestNeighbor_:xw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ow(t,e="binary",n=!1,s=.5){const r=m(t,"image","threshold"),a=.2989,o=.587,i=.114,u=r.shape[0]*r.shape[1];let l=x($e([s]),255),h,c,f,d;if(g(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),g(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),g(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),g(e==="otsu"||e==="binary",()=>`Method must be binary or otsu, but was ${e}`),r.shape[2]===3){[h,c,f]=Qt(r,[1,1,1],-1);const b=x(h,a),T=x(c,o),A=x(f,i);d=C(C(b,T),A)}else d=t;if(e==="otsu"){const b=Dr(X(ca(d),"int32"),Fe([]),256);l=Dw(b,u)}const y=n?cs(d,l):Rn(d,l);return X(x(y,255),"int32")}function Dw(t,e){let n=$e([-1]),s=$e([0]),r=$e([0]),a,o,i,u,l,h;for(let c=0;c<t.size-1;c++){a=q(t,0,c+1),o=q(t,c+1),l=K(H(a),e),h=K(H(o),e);const f=H(x(a,Jt(0,a.size)));i=K(f,H(a));const d=tn(o.shape,a.size),y=C(Jt(0,o.size),d),S=x(o,y);u=K(H(S),H(o));const b=L(i,u),T=L(i,u),A=x(l,h);r=x(x(A,b),T);const $=Rn(r,s);s=Ze($,r,s),n=Ze($,$e([c]),n)}return n}const Fw=w({threshold_:Ow});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cw(t,e,n="nearest",s="constant",r=0,a){const o=m(t,"image","transform","float32"),i=m(e,"transforms","transform","float32");g(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),g(i.rank===2&&(i.shape[0]===o.shape[0]||i.shape[0]===1)&&i.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),g(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);const u={image:o,transforms:i},l={interpolation:n,fillMode:s,fillValue:r,outputShape:a};return N.runKernel(rl,u,l)}const Rw=w({transform_:Cw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pw(t,e,n){const s=m(t,"a","bandPart");g(s.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${s.rank}.`);const r=s.shape,[a,o]=s.shape.slice(-2);let i,u;typeof e=="number"?(g(e%1===0,()=>`bandPart(): numLower must be an integer, got ${e}.`),g(e<=a,()=>`bandPart(): numLower (${e}) must not be greater than the number of rows (${a}).`),i=m(e<0?a:e,"numLower","bandPart")):(g(e.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),i=Ze(ts(e,0),a,Tn(e,a))),typeof n=="number"?(g(n%1===0,()=>`bandPart(): numUpper must be an integer, got ${n}.`),g(n<=o,()=>`bandPart(): numUpper (${n}) must not be greater than the number of columns (${o}).`),u=m(n<0?o:n,"numUpper","bandPart")):(g(n.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),u=Ze(ts(n,0),o,Tn(n,o)));const l=k(Jt(0,a,1,"int32"),[-1,1]),h=Jt(0,o,1,"int32"),c=L(l,h),f=Nn(cs(c,i),Vr(c,Ce(u))),d=At([a,o],s.dtype);return k(We(dt(k(s,[-1,a,o])).map(y=>Ze(f,y,d))),r)}const Bw=w({bandPart_:Pw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lw(t){let e;if(Array.isArray(t)){e=!1,g(t!=null&&t.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=t[0].shape[0];for(let a=1;a<t.length;++a)g(t[a].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${t[a].shape[0]} vs. ${r})`)}else e=!0,t=Qt(t,t.shape[0],0).map(r=>gs(r,[0]));g(t.length<=t[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${t.length}) exceeds number of dimensions (${t[0].shape[0]}).`);const n=[],s=t;for(let r=0;r<t.length;++r)n.push(N.tidy(()=>{let a=s[r];if(r>0)for(let o=0;o<r;++o){const i=x(H(x(n[o],a)),n[o]);a=L(a,i)}return K(a,Cn(a,"euclidean"))}));return e?We(n,0):n}const zw=w({gramSchmidt_:Lw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vw(t,e=!1){if(g(t.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${t.rank}`),t.rank===2)return Ga(t,e);{const n=t.shape.slice(0,t.shape.length-2).reduce((u,l)=>u*l),s=dt(k(t,[n,t.shape[t.shape.length-2],t.shape[t.shape.length-1]]),0),r=[],a=[];s.forEach(u=>{const[l,h]=Ga(u,e);r.push(l),a.push(h)});const o=k(We(r,0),t.shape),i=k(We(a,0),t.shape);return[o,i]}}function Ga(t,e=!1){return N.tidy(()=>{g(t.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${t.shape.length}D Tensor.`);const n=t.shape[0],s=t.shape[1];let r=Br(n),a=Xe(t);const o=Ut([[1]],[1,1]);let i=Xe(o);const u=n>=s?s:n;for(let l=0;l<u;++l){const h=a,c=i,f=r;[i,a,r]=N.tidy(()=>{const d=q(a,[l,l],[n-l,1]),y=Cn(d),S=q(a,[l,l],[1,1]),b=Ze(Rn(S,0),Ut([[-1]]),Ut([[1]])),T=L(S,x(b,y)),A=K(d,T);A.shape[0]===1?i=Xe(o):i=ue([o,q(A,[1,0],[A.shape[0]-1,A.shape[1]])],0);const $=Ce(K(V(b,T),y)),E=q(a,[l,0],[n-l,s]),v=x($,i),_=En(i);if(l===0)a=L(E,V(v,V(_,E)));else{const F=L(E,V(v,V(_,E)));a=ue([q(a,[0,0],[l,s]),F],0)}const O=En(v),D=q(r,[0,l],[n,r.shape[1]-l]);if(l===0)r=L(D,V(V(D,i),O));else{const F=L(D,V(V(D,i),O));r=ue([q(r,[0,0],[n,l]),F],1)}return[i,a,r]}),pe([h,c,f])}return!e&&n>s&&(r=q(r,[0,0],[n,s]),a=q(a,[0,0],[s,s])),[r,a]})}const jw=w({qr_:Vw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var he;(function(t){t[t.NONE=0]="NONE",t[t.MEAN=1]="MEAN",t[t.SUM=2]="SUM",t[t.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(he||(he={}));function Mw(t,e,n=he.SUM_BY_NONZERO_WEIGHTS){const s=m(t,"losses","computeWeightedLoss");let r=null;e!=null&&(r=m(e,"weights","computeWeightedLoss"));const a=r==null?s:x(s,r);if(n===he.NONE)return a;if(n===he.SUM)return H(a);if(n===he.MEAN){if(r==null)return Sn(a);{const o=s.size/r.size,i=K(H(a),H(r));return o>1?K(i,z(o)):i}}if(n===he.SUM_BY_NONZERO_WEIGHTS){if(r==null)return K(H(a),z(s.size));{const o=x(r,rt(s.shape)),i=X(H(Xr(o,z(0))),"float32");return K(H(a),i)}}throw Error(`Unknown reduction: ${n}`)}const Qe=w({computeWeightedLoss_:Mw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ww(t,e,n,s=he.SUM_BY_NONZERO_WEIGHTS){const r=m(t,"labels","absoluteDifference"),a=m(e,"predictions","absoluteDifference");let o=null;n!=null&&(o=m(n,"weights","absoluteDifference")),fe(r.shape,a.shape,"Error in absoluteDifference: ");const i=be(L(r,a));return Qe(i,o,s)}const Uw=w({absoluteDifference_:Ww});function qw(t,e,n,s,r=he.SUM_BY_NONZERO_WEIGHTS){const a=m(t,"labels","cosineDistance"),o=m(e,"predictions","cosineDistance");let i=null;s!=null&&(i=m(s,"weights","cosineDistance")),fe(a.shape,o.shape,"Error in cosineDistance: ");const u=z(1),l=L(u,H(x(a,o),n,!0));return Qe(l,i,r)}const Gw=w({cosineDistance_:qw});function Hw(t,e,n,s=he.SUM_BY_NONZERO_WEIGHTS){let r=m(t,"labels","hingeLoss");const a=m(e,"predictions","hingeLoss");let o=null;n!=null&&(o=m(n,"weights","hingeLoss")),fe(r.shape,a.shape,"Error in hingeLoss: ");const i=z(1);r=L(x(z(2),r),i);const u=Bn(L(i,x(r,a)));return Qe(u,o,s)}const Kw=w({hingeLoss_:Hw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xw(t,e,n,s=1,r=he.SUM_BY_NONZERO_WEIGHTS){const a=m(t,"labels","huberLoss"),o=m(e,"predictions","huberLoss");let i=null;n!=null&&(i=m(n,"weights","huberLoss")),fe(a.shape,o.shape,"Error in huberLoss: ");const u=z(s),l=be(L(o,a)),h=Tn(l,u),c=L(l,h),f=C(x(z(.5),xe(h)),x(u,c));return Qe(f,i,r)}const Zw=w({huberLoss_:Xw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jw(t,e,n,s=1e-7,r=he.SUM_BY_NONZERO_WEIGHTS){const a=m(t,"labels","logLoss"),o=m(e,"predictions","logLoss");let i=null;n!=null&&(i=m(n,"weights","logLoss")),fe(a.shape,o.shape,"Error in logLoss: ");const u=z(1),l=z(s),h=Ce(x(a,Zt(C(o,l)))),c=x(L(u,a),Zt(C(L(u,o),l))),f=L(h,c);return Qe(f,i,r)}const Yw=w({logLoss_:Jw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qw(t,e,n,s=he.SUM_BY_NONZERO_WEIGHTS){const r=m(t,"labels","meanSquaredError"),a=m(e,"predictions","meanSquaredError");let o=null;n!=null&&(o=m(n,"weights","meanSquaredError")),fe(r.shape,a.shape,"Error in meanSquaredError: ");const i=pa(r,a);return Qe(i,o,s)}const e1=w({meanSquaredError_:Qw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function t1(t,e){const n=m(t,"labels","sigmoidCrossEntropyWithLogits"),s=m(e,"logits","sigmoidCrossEntropyWithLogits");fe(n.shape,s.shape,"Error in sigmoidCrossEntropyWithLogits: ");const r=Bn(s),a=x(s,n),o=Mr(ct(Ce(be(s))));return C(L(r,a),o)}function n1(t,e,n,s=0,r=he.SUM_BY_NONZERO_WEIGHTS){let a=m(t,"multiClassLabels","sigmoidCrossEntropy");const o=m(e,"logits","sigmoidCrossEntropy");let i=null;if(n!=null&&(i=m(n,"weights","sigmoidCrossEntropy")),fe(a.shape,o.shape,"Error in sigmoidCrossEntropy: "),s>0){const l=z(s),h=z(1),c=z(.5);a=C(x(a,L(h,l)),x(c,l))}const u=t1(a,o);return Qe(u,i,r)}const s1=w({sigmoidCrossEntropy_:n1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function r1(t,e,n=-1){if(n===-1&&(n=e.rank-1),n!==e.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${e.rank} and dim was ${n}`);return Me((r,a,o)=>{const u=Ur(a,[n],!0),l=L(X(a,"float32"),u);o([r,l]);const h=Ce(x(l,r));return{value:H(h,[n]),gradFunc:(d,y)=>{const[S,b]=y,T=Fn(d.shape,[n]);return[x(k(d,T),L(X(S,"float32"),ct(b))),x(k(d,T),L(ct(b),X(S,"float32")))]}}})(t,e)}function a1(t,e,n,s=0,r=he.SUM_BY_NONZERO_WEIGHTS){let a=m(t,"onehotLabels","softmaxCrossEntropy");const o=m(e,"logits","softmaxCrossEntropy");let i=null;if(n!=null&&(i=m(n,"weights","softmaxCrossEntropy")),fe(a.shape,o.shape,"Error in softmaxCrossEntropy: "),s>0){const l=z(s),h=z(1),c=z(a.shape[1]);a=C(x(a,L(h,l)),K(l,c))}const u=r1(a,o);return Qe(u,i,r)}const o1=w({softmaxCrossEntropy_:a1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function i1(t,e,n,s){const r=m(t,"indices","sparseFillEmptyRows","int32"),a=m(e,"values","sparseFillEmptyRows"),o=m(n,"denseShape","sparseFillEmptyRows","int32"),i=m(s,"defaultValue","sparseFillEmptyRows",a.dtype);if(r.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${r.shape}`);if(a.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${a.shape}`);if(o.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${o.shape}`);if(i.rank!==0)throw new Error(`Default value should be a scalar but received shape ${i.shape}`);const u={indices:r,values:a,denseShape:o,defaultValue:i},l=N.runKernel(Wu,u);return{outputIndices:l[0],outputValues:l[1],emptyRowIndicator:l[2],reverseIndexMap:l[3]}}const u1=w({sparseFillEmptyRows_:i1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function l1(t,e,n){const s=m(t,"inputIndices","sparseReshape","int32"),r=m(e,"inputShape","sparseReshape","int32"),a=m(n,"newShape","sparseReshape","int32");if(s.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${s.shape}`);if(r.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${r.shape}`);if(a.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${a.shape}`);const o={inputIndices:s,inputShape:r,newShape:a},i=N.runKernel(Uu,o);return{outputIndices:i[0],outputShape:i[1]}}const c1=w({sparseReshape_:l1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function h1(t,e,n){const s=m(t,"data","sparseSegmentMean"),r=m(e,"indices","sparseSegmentMean","int32"),a=m(n,"segmentIds","sparseSegmentMean","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);const o={data:s,indices:r,segmentIds:a};return N.runKernel(qu,o)}const p1=w({sparseSegmentMean_:h1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function f1(t,e,n){const s=m(t,"data","sparseSegmentSum"),r=m(e,"indices","sparseSegmentSum","int32"),a=m(n,"segmentIds","sparseSegmentSum","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);const o={data:s,indices:r,segmentIds:a};return N.runKernel(Gu,o)}const d1=w({sparseSegmentSum_:f1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function m1(t,e,n,s,r,a,o,i){const u=m(t,"data","stringNGrams","string");if(u.dtype!=="string")throw new Error("Data must be of datatype string");if(u.shape.length!==1)throw new Error(`Data must be a vector, saw: ${u.shape}`);const l=m(e,"dataSplits","stringNGrams");if(l.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const h={separator:n,nGramWidths:s,leftPad:r,rightPad:a,padWidth:o,preserveShortSequences:i},c={data:u,dataSplits:l},f=N.runKernel(Ju,c,h);return{nGrams:f[0],nGramsSplits:f[1]}}const g1=w({stringNGrams_:m1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function y1(t,e,n=!0){const s=m(t,"input","stringSplit","string"),r=m(e,"delimiter","stringSplit","string");if(s.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${s.shape}`);if(r.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${r.shape}`);const a={skipEmpty:n},o={input:s,delimiter:r},i=N.runKernel(Yu,o,a);return{indices:i[0],values:i[1],shape:i[2]}}const b1=w({stringSplit_:y1});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function w1(t,e){const n=m(t,"input","stringToHashBucketFast","string"),s={numBuckets:e};if(e<=0)throw new Error("Number of buckets must be at least 1");const r={input:n};return N.runKernel(Qu,r,s)}const N1=w({stringToHashBucketFast_:w1});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function S1(t,e,n,s=!0){const r=m(t,"input","staticRegexReplace","string"),a={pattern:e,rewrite:n,replaceGlobal:s};return N.runKernel(Xu,{x:r},a)}const T1=w({staticRegexReplace_:S1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const hp={fft:ds,ifft:$n,rfft:ms,irfft:ha},pp={hammingWindow:K0,hannWindow:op,frame:ip,stft:Y0},fp={flipLeftRight:nw,grayscaleToRGB:rw,resizeNearestNeighbor:Aw,resizeBilinear:Iw,rgbToGrayscale:ow,rotateWithOffset:uw,cropAndResize:ew,nonMaxSuppression:cw,nonMaxSuppressionAsync:bw,nonMaxSuppressionWithScore:Nw,nonMaxSuppressionWithScoreAsync:Tw,nonMaxSuppressionPadded:Ew,nonMaxSuppressionPaddedAsync:vw,threshold:Fw,transform:Rw},dp={bandPart:Bw,gramSchmidt:zw,qr:jw},mp={absoluteDifference:Uw,computeWeightedLoss:Qe,cosineDistance:Gw,hingeLoss:Kw,huberLoss:Zw,logLoss:Yw,meanSquaredError:e1,sigmoidCrossEntropy:s1,softmaxCrossEntropy:o1},gp={sparseFillEmptyRows:u1,sparseReshape:c1,sparseSegmentMean:p1,sparseSegmentSum:d1},yp={stringNGrams:g1,stringSplit:b1,stringToHashBucketFast:N1,staticRegexReplace:T1};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $1=new Map,Xs=new Map;class bp{getClassName(){return this.constructor.className}static fromConfig(e,n){return new e(n)}}class tt{constructor(){this.classNameMap={}}static getMap(){return tt.instance==null&&(tt.instance=new tt),tt.instance}static register(e){tt.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function wp(t,e,n){g(t.className!=null,()=>"Class being registered does not have the static className property defined."),g(typeof t.className=="string",()=>"className is required to be a string, but got type "+typeof t.className),g(t.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof e>"u"&&(e="Custom"),typeof n>"u"&&(n=t.className);const s=n,r=e+">"+s;return tt.register(t),$1.set(r,t),Xs.set(t,r),t}function E1(t){return Xs.has(t)?Xs.get(t):t.className}const k1=Object.freeze(Object.defineProperty({__proto__:null,Serializable:bp,SerializationMap:tt,getRegisteredName:E1,registerClass:wp},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class mt extends bp{minimize(e,n=!1,s){const{value:r,grads:a}=this.computeGradients(e,s);if(s!=null){const o=s.map(i=>({name:i.name,tensor:a[i.name]}));this.applyGradients(o)}else this.applyGradients(a);return pe(a),n?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,n){return Uc(e,n)}dispose(){this.iterations_!=null&&pe(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:z(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(mt,Symbol.hasInstance,{value:t=>t.minimize!=null&&t.computeGradients!=null&&t.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class wa extends mt{static get className(){return"Adadelta"}constructor(e,n,s=null){super(),this.learningRate=e,this.rho=n,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=N.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const a=N.registeredVariables[s],o=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accum_grad`,variable:j(()=>Ne(a).variable(o))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${s}/accum_var`,variable:j(()=>Ne(a).variable(o))});const i=Array.isArray(e)?e[r].tensor:e[s];if(i==null)return;const u=this.accumulatedGrads[r].variable,l=this.accumulatedUpdates[r].variable;j(()=>{const h=C(x(u,this.rho),x(xe(i),1-this.rho)),c=x(K(je(C(l,this.epsilon)),je(C(u,this.epsilon))),i),f=C(x(l,this.rho),x(xe(c),1-this.rho));u.assign(h),l.assign(f);const d=C(x(c,-this.learningRate),a);a.assign(d)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(pe(this.accumulatedGrads.map(e=>e.variable)),pe(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){const e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(e){e=await this.extractIterations(e);const n=e.length/2,s=!1;this.accumulatedGrads=e.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedUpdates=e.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,n){return new e(n.learningRate,n.rho,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Na extends mt{static get className(){return"Adagrad"}constructor(e,n=.1){super(),this.learningRate=e,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const a=N.registeredVariables[s];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accumulator`,variable:j(()=>tn(a.shape,this.initialAccumulatorValue).variable(!1))});const o=Array.isArray(e)?e[r].tensor:e[s];if(o==null)return;const i=this.accumulatedGrads[r].variable;j(()=>{const u=C(i,xe(o));i.assign(u);const l=C(x(K(o,je(C(u,N.backend.epsilon()))),-this.learningRate),a);a.assign(l)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&pe(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const n=!1;this.accumulatedGrads=e.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,n){return new e(n.learningRate,n.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Sa extends mt{static get className(){return"Adam"}constructor(e,n,s,r=null){super(),this.learningRate=e,this.beta1=n,this.beta2=s,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],j(()=>{this.accBeta1=z(n).variable(),this.accBeta2=z(s).variable()}),r==null&&(this.epsilon=N.backend.epsilon())}applyGradients(e){const n=Array.isArray(e)?e.map(s=>s.name):Object.keys(e);j(()=>{const s=L(1,this.accBeta1),r=L(1,this.accBeta2);n.forEach((a,o)=>{const i=N.registeredVariables[a],u=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${a}/m`,variable:j(()=>Ne(i).variable(u))}),this.accumulatedSecondMoment[o]==null&&(this.accumulatedSecondMoment[o]={originalName:`${a}/v`,variable:j(()=>Ne(i).variable(u))});const l=Array.isArray(e)?e[o].tensor:e[a];if(l==null)return;const h=this.accumulatedFirstMoment[o].variable,c=this.accumulatedSecondMoment[o].variable,f=C(x(h,this.beta1),x(l,1-this.beta1)),d=C(x(c,this.beta2),x(xe(l),1-this.beta2)),y=K(f,s),S=K(d,r);h.assign(f),c.assign(d);const b=C(x(K(y,C(je(S),this.epsilon)),-this.learningRate),i);i.assign(b)}),this.accBeta1.assign(x(this.accBeta1,this.beta1)),this.accBeta2.assign(x(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&pe(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&pe(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(e){e=await this.extractIterations(e),j(()=>{this.accBeta1.assign(Xt(this.beta1,this.iterations_+1)),this.accBeta2.assign(Xt(this.beta2,this.iterations_+1))});const n=e.length/2,s=!1;this.accumulatedFirstMoment=e.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedSecondMoment=e.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,n){return new e(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ta extends mt{static get className(){return"Adamax"}constructor(e,n,s,r=null,a=0){super(),this.learningRate=e,this.beta1=n,this.beta2=s,this.epsilon=r,this.decay=a,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],j(()=>{this.iteration=z(0).variable(),this.accBeta1=z(n).variable()}),r==null&&(this.epsilon=N.backend.epsilon())}applyGradients(e){const n=Array.isArray(e)?e.map(s=>s.name):Object.keys(e);j(()=>{const s=L(1,this.accBeta1),r=K(-this.learningRate,C(x(this.iteration,this.decay),1));n.forEach((a,o)=>{const i=N.registeredVariables[a],u=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${a}/m`,variable:Ne(i).variable(u)}),this.accumulatedWeightedInfNorm[o]==null&&(this.accumulatedWeightedInfNorm[o]={originalName:`${a}/v`,variable:Ne(i).variable(u)});const l=Array.isArray(e)?e[o].tensor:e[a];if(l==null)return;const h=this.accumulatedFirstMoment[o].variable,c=this.accumulatedWeightedInfNorm[o].variable,f=C(x(h,this.beta1),x(l,1-this.beta1)),d=x(c,this.beta2),y=be(l),S=Kr(d,y);h.assign(f),c.assign(S);const b=C(x(K(r,s),K(f,C(S,this.epsilon))),i);i.assign(b)}),this.iteration.assign(C(this.iteration,1)),this.accBeta1.assign(x(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&pe(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&pe(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,n){return new e(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $s extends mt{static get className(){return"SGD"}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const a=Array.isArray(e)?e[r].tensor:e[s];if(a==null)return;const o=N.registeredVariables[s];j(()=>{const i=C(x(this.c,a),o);o.assign(i)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=De(z(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,n){return new e(n.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $a extends $s{static get className(){return"Momentum"}constructor(e,n,s=!1){super(e),this.learningRate=e,this.momentum=n,this.useNesterov=s,this.accumulations=[],this.m=z(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const a=N.registeredVariables[s];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${s}/momentum`,variable:j(()=>Ne(a).variable(!1))});const o=this.accumulations[r].variable,i=Array.isArray(e)?e[r].tensor:e[s];i!=null&&j(()=>{let u;const l=C(x(this.m,o),i);this.useNesterov?u=C(x(this.c,C(i,x(l,this.m))),a):u=C(x(this.c,l),a),o.assign(l),a.assign(u)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&pe(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const n=!1;this.accumulations=e.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,n){return new e(n.learningRate,n.momentum,n.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ea extends mt{static get className(){return"RMSProp"}constructor(e,n=.9,s=0,r=null,a=!1){if(super(),this.learningRate=e,this.decay=n,this.momentum=s,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=a,r==null&&(this.epsilon=N.backend.epsilon()),e==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){(Array.isArray(e)?e.map(s=>s.name):Object.keys(e)).forEach((s,r)=>{const a=N.registeredVariables[s],o=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${s}/rms`,variable:j(()=>Ne(a).variable(o))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${s}/momentum`,variable:j(()=>Ne(a).variable(o))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${s}/mg`,variable:j(()=>Ne(a).variable(o))});const i=Array.isArray(e)?e[r].tensor:e[s];if(i==null)return;const u=this.accumulatedMeanSquares[r].variable,l=this.accumulatedMoments[r].variable;j(()=>{const h=C(x(u,this.decay),x(xe(i),1-this.decay));if(this.centered){const c=this.accumulatedMeanGrads[r].variable,f=C(x(c,this.decay),x(i,1-this.decay)),d=K(x(i,this.learningRate),je(L(h,C(xe(f),this.epsilon)))),y=C(x(l,this.momentum),d);u.assign(h),c.assign(f),l.assign(y);const S=L(a,y);a.assign(S)}else{const c=C(x(u,this.decay),x(xe(i),1-this.decay)),f=C(x(l,this.momentum),K(x(i,this.learningRate),je(C(c,this.epsilon))));u.assign(c),l.assign(f);const d=L(a,f);a.assign(d)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&pe(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&pe(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&pe(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(e){e=await this.extractIterations(e);const n=this.centered?e.length/3:e.length/2,s=!1;this.accumulatedMeanSquares=e.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedMoments=e.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=e.slice(n*2,n*3).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,n){return new e(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const v1=[wa,Na,Sa,Ta,$a,Ea,$s];function _1(){for(const t of v1)wp(t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const I1="model",x1=".json",A1=".weights.bin";function Ha(t){return new Promise(e=>setTimeout(e)).then(t)}class Ot{constructor(e){if(!R().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");e.startsWith(Ot.URL_SCHEME)&&(e=e.slice(Ot.URL_SCHEME.length)),(e==null||e.length===0)&&(e=I1),this.modelJsonFileName=e+x1,this.weightDataFileName=e+A1}async save(e){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const n=Be.join(e.weightData),s=window.URL.createObjectURL(new Blob([n],{type:"application/octet-stream"}));if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:e.weightSpecs}],a=Rl(e,r),o=window.URL.createObjectURL(new Blob([JSON.stringify(a)],{type:"application/json"})),i=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(i.download=this.modelJsonFileName,i.href=o,await Ha(()=>i.dispatchEvent(new MouseEvent("click"))),e.weightData!=null){const u=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;u.download=this.weightDataFileName,u.href=s,await Ha(()=>u.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:xn(e)}}}}Ot.URL_SCHEME="downloads://";class O1{constructor(e){if(e==null||e.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${e}`);this.jsonFile=e[0],this.weightsFiles=e.slice(1)}async load(){return new Promise((e,n)=>{const s=new FileReader;s.onload=r=>{const a=JSON.parse(r.target.result),o=a.modelTopology;if(o==null){n(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(a.weightsManifest==null){n(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){e({modelTopology:o});return}const u=vr(a,l=>this.loadWeights(l));e(u)},s.onerror=r=>n(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(e){const n=[],s=[];for(const o of e)n.push(...o.weights),s.push(...o.paths);const r=this.checkManifestAndWeightFiles(e),a=s.map(o=>this.loadWeightsFile(o,r[o]));return Promise.all(a).then(o=>[n,o])}loadWeightsFile(e,n){return new Promise((s,r)=>{const a=new FileReader;a.onload=o=>{const i=o.target.result;s(i)},a.onerror=o=>r(`Failed to weights data from file of path '${e}'.`),a.readAsArrayBuffer(n)})}checkManifestAndWeightFiles(e){const n=[],s=this.weightsFiles.map(a=>Ua(a.name)),r={};for(const a of e)a.paths.forEach(o=>{const i=Ua(o);if(n.indexOf(i)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${i}'`);if(n.push(i),s.indexOf(i)===-1)throw new Error(`Weight file with basename '${i}' is not provided.`);r[o]=this.weightsFiles[s.indexOf(i)]});if(n.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${n.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const D1=t=>R().getBool("IS_BROWSER")&&!Array.isArray(t)&&t.startsWith(Ot.URL_SCHEME)?F1(t.slice(Ot.URL_SCHEME.length)):null;Q.registerSaveRouter(D1);function F1(t="model"){return new Ot(t)}function C1(t){return new O1(t)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ka(t,e,n,s){o(t),n=n??0,s=s??1,i(n,s);let r=0;const a=u=>(u.then(l=>{const h=n+ ++r/t.length*(s-n);return e(h),l}),u);function o(u){g(u!=null&&Array.isArray(u)&&u.length>0,()=>"promises must be a none empty array")}function i(u,l){g(u>=0&&u<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${u}`),g(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${l}`),g(l>=u,()=>`startFraction must be no more than endFraction, but got startFraction ${u} and endFraction ${l}`)}return Promise.all(t.map(a))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Np(t,e){e==null&&(e={});const n=e.fetchFunc==null?R().platform.fetch:e.fetchFunc,s=t.map(c=>n(c,e.requestInit,{isBinary:!0})),i=(e.onProgress==null?await Promise.all(s):await Ka(s,e.onProgress,0,.5)).map(c=>c.arrayBuffer());return e.onProgress==null?await Promise.all(i):await Ka(i,e.onProgress,.5,1)}function R1(t,e){var n;const s=e.fetchFunc==null?R().platform.fetch:e.fetchFunc;let r=0,a;return(n=e.onProgress)===null||n===void 0||n.call(e,0),new ReadableStream({pull:async o=>{for(var i;r<t.length;){a||(a=(await s(t[r],e.requestInit,{isBinary:!0})).body.getReader());const{done:u,value:l}=await a.read();if(u){r++,a=void 0,(i=e.onProgress)===null||i===void 0||i.call(e,r/t.length);continue}o.enqueue(l);return}o.close()}})}async function P1(t,e="",n,s){return Sp(o=>Np(o,{requestInit:s}))(t,e,n)}function Sp(t){return async(e,n="",s)=>{const r=e.map(()=>!1),a={},o=s!=null?s.map(()=>!1):[],i=[];if(e.forEach((d,y)=>{let S=0;d.weights.forEach(b=>{const T="quantization"in b?b.quantization.dtype:b.dtype,A=vt[T]*W(b.shape),$=()=>{r[y]=!0,a[y]==null&&(a[y]=[]),a[y].push({manifestEntry:b,groupOffset:S,sizeBytes:A})};s!=null?s.forEach((E,v)=>{E===b.name&&($(),o[v]=!0)}):$(),i.push(b.name),S+=A})}),!o.every(d=>d)){const d=s.filter((y,S)=>!o[S]);throw new Error(`Could not find weights in manifest with names: ${d.join(", ")}. 
Manifest JSON has weights with names: ${i.join(", ")}.`)}const u=r.reduce((d,y,S)=>(y&&d.push(S),d),[]),l=[];u.forEach(d=>{e[d].paths.forEach(y=>{const S=n+(n.endsWith("/")?"":"/")+y;l.push(S)})});const h=await t(l),c={};let f=0;return u.forEach(d=>{const y=e[d].paths.length,S=new Be(h.slice(f,f+y));a[d].forEach(T=>{const A=S.slice(T.groupOffset,T.groupOffset+T.sizeBytes),$=Dl(A,[T.manifestEntry]);for(const E in $)c[E]=$[E]}),f+=y}),c}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const B1="application/octet-stream",L1="application/json";class ka{constructor(e,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(g(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=R().platform.fetch,g(e!=null&&e.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(e)&&g(e.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${e.length}).`),this.path=e,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{},this.loadOptions=n}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:e.weightSpecs}],r=Rl(e,s);if(n.body.append("model.json",new Blob([JSON.stringify(r)],{type:L1}),"model.json"),e.weightData!=null){const o=Be.join(e.weightData);n.body.append("model.weights.bin",new Blob([o],{type:B1}),"model.weights.bin")}const a=await this.fetch(this.path,n);if(a.ok)return{modelArtifactsInfo:xn(e),responses:[a]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${a.status}.`)}async loadModelJSON(){const e=await this.fetch(this.path,this.requestInit);if(!e.ok)throw new Error(`Request to ${this.path} failed with status code ${e.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await e.json()}catch{let o=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?o+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":o+=" Please make sure the server is serving valid JSON for this request.",new Error(o)}const s=n.modelTopology,r=n.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return n}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const e=await this.loadModelJSON();return vr(e,n=>this.loadWeights(n))}async loadStream(){const e=await this.loadModelJSON(),n=await this.getWeightUrls(e.weightsManifest),s=Yn(e.weightsManifest),r=()=>R1(n,this.loadOptions);return Object.assign(Object.assign({},e),{weightSpecs:s,getWeightStream:r})}async getWeightUrls(e){const n=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=z1(n),a=this.weightPathPrefix||s,o=[],i=[];for(const u of e)for(const l of u.paths)this.weightUrlConverter!=null?i.push(this.weightUrlConverter(l)):o.push(a+l+r);return this.weightUrlConverter&&o.push(...await Promise.all(i)),o}async loadWeights(e){const n=await this.getWeightUrls(e),s=Yn(e),r=await Np(n,this.loadOptions);return[s,r]}}ka.URL_SCHEME_REGEX=/^https?:\/\//;function z1(t){const e=t.lastIndexOf("/"),n=t.lastIndexOf("?"),s=t.substring(0,e),r=n>e?t.substring(n):"";return[s+"/",r]}function Zs(t){return t.match(ka.URL_SCHEME_REGEX)!=null}const Tp=(t,e)=>{if(typeof fetch>"u"&&(e==null||e.fetchFunc==null))return null;{let n=!0;if(Array.isArray(t)?n=t.every(s=>Zs(s)):n=Zs(t),n)return va(t,e)}return null};Q.registerSaveRouter(Tp);Q.registerLoadRouter(Tp);function va(t,e){return new ka(t,e)}function V1(t,e){return va(t,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Is{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class $p{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class j1{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=n=>Promise.resolve(e.save(n)))}}function M1(t,e,n,s){const r=arguments;return new j1(ss(...r))}function ss(t,e,n,s){return arguments.length===1?t.modelTopology!=null||t.weightSpecs!=null?new Is(t):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Is({modelTopology:t})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new Is({modelTopology:t,weightSpecs:e,weightData:n,trainingConfig:s}))}function W1(t){return new $p(t)}function U1(t){return new $p(t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _a=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:Be,browserFiles:C1,browserHTTPRequest:V1,concatenateArrayBuffers:zd,copyModel:im,decodeWeights:Dl,decodeWeightsStream:Cl,encodeWeights:Fd,fromMemory:M1,fromMemorySync:ss,getLoadHandlers:Hd,getModelArtifactsForJSON:vr,getModelArtifactsForJSONSync:kr,getModelArtifactsInfoForJSON:xn,getSaveHandlers:Gd,getWeightSpecs:Yn,http:va,isHTTPScheme:Zs,listModels:am,loadWeights:P1,moveModel:um,registerLoadRouter:qd,registerSaveRouter:Ud,removeModel:om,weightsLoaderFactory:Sp,withSaveHandler:W1,withSaveHandlerSync:U1},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function q1(t,e,n){const s=m(t,"labels","confusionMatrix"),r=m(e,"predictions","confusionMatrix");g(n==null||n>0&&Number.isInteger(n),()=>`If provided, numClasses must be a positive integer, but got ${n}`),g(s.rank===1,()=>`Expected the rank of labels to be 1, but got ${s.rank}`),g(r.rank===1,()=>`Expected the rank of predictions to be 1, but got ${r.rank}`),g(s.shape[0]===r.shape[0],()=>`Mismatch in the number of examples: ${s.shape[0]} vs. ${r.shape[0]}. Labels and predictions should have the same number of elements.`),g(n>0&&Number.isInteger(n),()=>`numClasses is required to be a positive integer, but got ${n}`);const a=ns(X(s,"int32"),n),o=ns(X(r,"int32"),n),i=En(a),u=V(i,o);return X(u,"int32")}const G1=w({confusionMatrix_:q1});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const H1=Object.freeze(Object.defineProperty({__proto__:null,confusionMatrix:G1},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let gt,Xa=!1;function Ep(t,e=3){if(e>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(t==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let n=!1,s=!1,r=!1,a=!1,o=!1,i=!1;if(t.data instanceof Uint8Array)n=!0;else if(typeof ImageData<"u"&&t instanceof ImageData)s=!0;else if(typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement)r=!0;else if(typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement)a=!0;else if(t.getContext!=null)o=!0;else if(typeof ImageBitmap<"u"&&t instanceof ImageBitmap)i=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${t.constructor.name}`);if(fn(Os,N.backendName)!=null){const y={pixels:t},S={numChannels:e};return N.runKernel(Os,y,S)}const[l,h]=r?[t.videoWidth,t.videoHeight]:[t.width,t.height];let c;if(o)c=t.getContext("2d").getImageData(0,0,l,h).data;else if(s||n)c=t.data;else if(a||r||i){if(gt==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")gt=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else gt=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});gt.canvas.width=l,gt.canvas.height=h,gt.drawImage(t,0,0,l,h),c=gt.getImageData(0,0,l,h).data}let f;if(e===4)f=new Int32Array(c);else{const y=l*h;f=new Int32Array(y*e);for(let S=0;S<y;S++)for(let b=0;b<e;++b)f[S*e+b]=c[S*4+b]}return da(f,[h,l,e],"int32")}function K1(t){return t!=null&&t.data instanceof Uint8Array}function X1(){return typeof window<"u"&&typeof ImageBitmap<"u"&&window.hasOwnProperty("createImageBitmap")}function Z1(t){return t!=null&&t.width!==0&&t.height!==0}function J1(t){return X1()&&!(t instanceof ImageBitmap)&&Z1(t)&&!K1(t)}async function Y1(t,e=3){let n=null;if(R().getBool("WRAP_TO_IMAGEBITMAP")&&J1(t)){let s;try{s=await createImageBitmap(t,{premultiplyAlpha:"none"})}catch{s=null}s!=null&&s.width===t.width&&s.height===t.height?n=s:n=t}else n=t;return Ep(n,e)}function kp(t){if(t.rank!==2&&t.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${t.rank}.`);const e=t.rank===2?1:t.shape[2];if(e>4||e===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${e}`);if(t.dtype!=="float32"&&t.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${t.dtype}. Please use float32 or int32 tensors.`)}function Q1(t){const e=(t==null?void 0:t.alpha)||1;if(e>1||e<0)throw new Error(`Alpha value ${e} is suppoed to be in range [0 - 1].`)}async function eN(t,e){let n=m(t,"img","toPixels");if(!(t instanceof te)){const l=n;n=X(l,"int32"),l.dispose()}kp(n);const[s,r]=n.shape.slice(0,2),a=n.rank===2?1:n.shape[2],o=await n.data(),i=n.dtype==="float32"?255:1,u=new Uint8ClampedArray(r*s*4);for(let l=0;l<s*r;++l){const h=[0,0,0,255];for(let f=0;f<a;f++){const d=o[l*a+f];if(n.dtype==="float32"){if(d<0||d>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${d}.`)}else if(n.dtype==="int32"&&(d<0||d>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${d}.`);a===1?(h[0]=d*i,h[1]=d*i,h[2]=d*i):h[f]=d*i}const c=l*4;u[c+0]=Math.round(h[0]),u[c+1]=Math.round(h[1]),u[c+2]=Math.round(h[2]),u[c+3]=Math.round(h[3])}if(e!=null){Xa||fn(yr,N.backendName)!=null&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),Xa=!0),e.width=r,e.height=s;const l=e.getContext("2d"),h=new ImageData(u,r,s);l.putImageData(h,0,0)}return n!==t&&n.dispose(),u}function tN(t,e,n){let s=m(t,"img","draw");if(!(t instanceof te)){const o=s;s=X(o,"int32"),o.dispose()}kp(s),Q1(n==null?void 0:n.imageOptions);const r={image:s},a={canvas:e,options:n};N.runKernel(yr,r,a)}const nN=w({fromPixels_:Ep}),sN=Object.freeze(Object.defineProperty({__proto__:null,draw:tN,fromPixels:nN,fromPixelsAsync:Y1,toPixels:eN},Symbol.toStringTag,{value:"Module"}));function vp(t,e){const n=t.shape.length,s=e.shape.length;if(n<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(e.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${e.dtype}.`);if(e.shape[s-1]>n)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${e.shape[s-1]} vs. ${n}`);if(W(t.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${t.shape}.`);const r=e.shape,a=r[r.length-1];let o=1;for(let c=0;c<r.length-1;++c)o*=r[c];const i=t.shape,u=r.slice();u.pop();let l=1;for(let c=a;c<n;++c)l*=i[c],u.push(i[c]);const h=[...en(t.shape).map(c=>c/l),1].slice(0,a);return[u,o,l,h]}const rN=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:vp},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Js=-2,aN=-1;function oN(t,e,n){const s=t.shape.length;g(s===e.length,()=>`Error in slice${s}D: Length of begin ${e} must match the rank of the array (${s}).`),g(s===n.length,()=>`Error in slice${s}D: Length of size ${n} must match the rank of the array (${s}).`);for(let r=0;r<s;++r)g(e[r]+n[r]<=t.shape[r],()=>`Error in slice${s}D: begin[${r}] + size[${r}] (${e[r]+n[r]}) would overflow input.shape[${r}] (${t.shape[r]})`)}function iN(t){const e=[];let n=0;for(;t>0;)t&1&&e.push(n),t/=2,n++;return e}function uN(t,e,n){const s=[];for(let r=0;r<t.length;r++)s[r]=Math.ceil((e[r]-t[r])/n[r]);return s}function _p(t,e,n,s){const r=[...t];for(let a=r.length;a<s.length;a++)r.push(1);for(let a=0;a<n;a++)a===0?r[e]=1:(r.splice(e,0,1),r.pop());return r}function Ip(t,e,n){return n<=t?n:n-(e-1)}function xp(t,e){const n=[];for(let s=0;s<t;s++)n.push(e+s);return n}function lN(t,e,n,s,r,a,o,i,u){const l=t.length;let h=new Array(l),c=new Array(l),f=new Array(l);if(e.length&&n>0){const d=e[0],y=n+1;h=Ap(o,d,y,s,t),c=Op(i,d,y,r,t),f=_p(a,d,y,t)}else for(let d=0;d<l;d++)h[d]=Fp(o,s,a,t,d,u),c[d]=Cp(i,r,a,t,d,u),f[d]=Dp(a,d,u);return{begin:h,end:c,strides:f}}function Ap(t,e,n,s,r){const a=[...r],o=xp(n,e);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=0;else{const u=Ip(e,n,i);let l=s[u];t&1<<u&&(l=0),a[i]=l}return a}function Op(t,e,n,s,r){const a=[...r],o=xp(n,e);for(let i=0;i<a.length;i++)if(o.indexOf(i)>-1)a[i]=Number.MAX_SAFE_INTEGER;else{const u=Ip(e,n,i);let l=s[u];t&1<<u&&(l=Number.MAX_SAFE_INTEGER),a[i]=l}for(let i=0;i<a.length;i++){const u=r[i];a[i]<0&&(a[i]+=u),a[i]=hn(0,a[i],r[i])}return a}function Dp(t,e,n){let s=t[e];return(n&1<<e||s==null)&&(s=1),s}function Fp(t,e,n,s,r,a){let o=e[r];const i=n[r]||1;(t&1<<r||a&1<<r||o==null)&&(i>0?o=Number.MIN_SAFE_INTEGER:o=Number.MAX_SAFE_INTEGER);const u=s[r];return o<0&&(o+=u),o=hn(0,o,u-1),o}function Cp(t,e,n,s,r,a){let o=e[r];const i=n[r]||1;(t&1<<r||a&1<<r||o==null)&&(i>0?o=Number.MAX_SAFE_INTEGER:o=Number.MIN_SAFE_INTEGER);const u=s[r];return o<0&&(o+=u),i>0?o=hn(0,o,u):o=hn(-1,o,u-1),o}function cN(t,e,n){let s=n.length;for(let r=0;r<n.length;r++)if(n[r]>1){s=r;break}for(let r=s+1;r<n.length;r++)if(e[r]>0||n[r]!==t[r])return!1;return!0}function hN(t,e){let n=t.length>0?t[t.length-1]:1;for(let s=0;s<t.length-1;s++)n+=t[s]*e[s];return n}function pN(t,e,n){let s;const r=t.shape.length;typeof e=="number"?s=[e,...new Array(r-1).fill(0)]:e.length<r?s=e.concat(new Array(r-e.length).fill(0)):s=e.slice(),s.forEach(o=>{g(o!==-1,()=>"slice() does not support negative begin indexing.")});let a;return n==null?a=new Array(r).fill(-1):typeof n=="number"?a=[n,...new Array(r-1).fill(-1)]:n.length<r?a=n.concat(new Array(r-n.length).fill(-1)):a=n,a=a.map((o,i)=>o>=0?o:(g(o===-1,()=>`Negative size values should be exactly -1 but got ${o} for the slice() size at index ${i}.`),t.shape[i]-s[i])),[s,a]}function fN(t,e,n,s,r,a,o,i,u){let l;if(s==null?(l=new Array(e.length),l.fill(1)):l=s,o!=null&&o&o-1)throw new Error("Multiple ellipses in slice is not allowed.");let h=!1;const c={dims:l.length,numAddAxisAfterEllipsis:0,begin:e.slice(),end:n.slice(),strides:l.slice(),beginMask:r,endMask:a,ellipsisMask:o,newAxisMask:i,shrinkAxisMask:u};for(let $=0;$<c.dims;$++)h&&1<<$&i&&c.numAddAxisAfterEllipsis++,1<<$&o&&(h=!0);h||(c.ellipsisMask|=1<<c.dims,c.dims++);const f={dims:t.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};dN(c,f);let d=!0,y=!0,S=!0;const b=[],T=[];for(let $=0;$<t.length;++$){if(f.strides[$]===0)throw Error(`strides[${$}] must be non-zero`);const E=!!(f.shrinkAxisMask&1<<$),v=t[$];if(v===-1){b.push(E?1:-1);continue}const _=[f.beginMask&1<<$,f.endMask&1<<$],O=[f.strides[$]>0?0:-1,f.strides[$]>0?v:v-1];if(E&&f.strides[$]<=0)throw Error("only stride 1 allowed on non-range indexing.");S=S&&f.strides[$]===1;const D=!!(f.beginMask&1<<$&&f.endMask&1<<$);if(f.beginValid&&f.endValid){if(E){const M=f.begin[$]<0?v+f.begin[$]:f.begin[$];if(f.begin[$]=M,f.end[$]=f.begin[$]+1,M<0||M>=v)throw Error(`slice index ${f.begin[$]} of dimension ${$} out of bounds.`)}else f.begin[$]=Za(f.begin[$],0,f.strides[$],v,_,O),f.end[$]=Za(f.end[$],1,f.strides[$],v,_,O);const P=f.strides[$]===1&&f.begin[$]===0&&f.end[$]===v;d=d&&P,y=y&&($===0&&f.strides[$]===1||P)}else d=d&&f.strides[$]===1&&D,y=y&&($===0&&f.strides[$]===1||D);let F,B=!1;if(f.beginValid&&f.endValid?(F=f.end[$]-f.begin[$],B=!0):E?(F=1,B=!0):D&&v>=0&&(f.strides[$]<0?F=-v:F=v,B=!0),B){let P;F===0||F<0!=f.strides[$]<0?P=0:P=Math.trunc(F/f.strides[$])+(F%f.strides[$]!==0?1:0),b.push(P)}else b.push(-1)}for(let $=0;$<f.finalShapeGatherIndices.length;++$){const E=f.finalShapeGatherIndices[$];E>=0?T.push(b[E]):E===Js&&T.push(1)}return{finalShapeSparse:T.filter(($,E)=>f.finalShapeGatherIndices[E]!==Js),finalShape:T,isIdentity:d,sliceDim0:y,isSimpleSlice:S,begin:f.begin,end:f.end,strides:f.strides}}function dN(t,e){e.beginMask=0,e.endMask=0,e.shrinkAxisMask=0;let n=0;e.beginValid=t.begin!=null,e.endValid=t.end!=null,e.begin=new Array(e.dims),e.end=new Array(e.dims),e.strides=new Array(e.dims),e.finalShapeGatherIndices=[],e.finalShapeGatherIndicesSparse=[],e.inputShapeGatherIndicesSparse=new Array(e.dims);for(let s=0;s<t.dims;s++)if(1<<s&t.ellipsisMask){const r=Math.min(e.dims-(t.dims-s)+1+t.numAddAxisAfterEllipsis,e.dims);for(;n<r;n++)e.begin[n]=0,e.end[n]=0,e.strides[n]=1,e.beginMask|=1<<n,e.endMask|=1<<n,e.finalShapeGatherIndices.push(n),e.finalShapeGatherIndicesSparse.push(-1),e.inputShapeGatherIndicesSparse[n]=s}else if(1<<s&t.newAxisMask)e.finalShapeGatherIndices.push(Js),e.finalShapeGatherIndicesSparse.push(-1);else{if(n===e.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${e.dims} dims, ${e.begin.length}.`);t.begin!=null&&(e.begin[n]=t.begin[s]),t.end!=null&&(e.end[n]=t.end[s]),e.strides[n]=t.strides[s],t.beginMask&1<<s&&(e.beginMask|=1<<n),t.endMask&1<<s&&(e.endMask|=1<<n),t.shrinkAxisMask&1<<s?(e.finalShapeGatherIndices.push(aN),e.finalShapeGatherIndicesSparse.push(-1),e.shrinkAxisMask|=1<<n):(e.finalShapeGatherIndices.push(n),e.finalShapeGatherIndicesSparse.push(s)),e.inputShapeGatherIndicesSparse[n]=s,n++}}function Za(t,e,n,s,r,a){if(r[e])return n>0?a[e]:a[e+1&1];{const o=t<0?s+t:t;return o<a[0]?a[0]:o>a[1]?a[1]:o}}const Rp=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:oN,computeFlatOffset:hN,computeOutShape:uN,getNormalizedAxes:lN,isSliceContinous:cN,maskToAxes:iN,parseSliceParams:pN,sliceInfo:fN,startForAxis:Fp,startIndicesWithElidedDims:Ap,stopForAxis:Cp,stopIndicesWithElidedDims:Op,stridesForAxis:Dp,stridesWithElidedDims:_p},Symbol.toStringTag,{value:"Module"}));/** @license See the LICENSE file. */const mN="4.22.0";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Pp{static sgd(e){return new $s(e)}static momentum(e,n,s=!1){return new $a(e,n,s)}static rmsprop(e,n=.9,s=0,r=null,a=!1){return new Ea(e,n,s,r,a)}static adam(e=.001,n=.9,s=.999,r=null){return new Sa(e,n,s,r)}static adadelta(e=.001,n=.95,s=null){return new wa(e,n,s)}static adamax(e=.002,n=.9,s=.999,r=null,a=0){return new Ta(e,n,s,r,a)}static adagrad(e,n=.1){return new Na(e,n)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const gN=Pp;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const yN=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:t=>t();function bN(){return new Promise(t=>yN(()=>t()))}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wN(t,e){const n=t[0].length;t.forEach((r,a)=>{g(r.length===n,()=>`Error in concat${n}D: rank of tensors[${a}] must be the same as the rank of the rest (${n})`)}),g(e>=0&&e<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);const s=t[0];t.forEach((r,a)=>{for(let o=0;o<n;o++)g(o===e||r[o]===s[o],()=>`Error in concat${n}D: Shape of tensors[${a}] (${r}) does not match the shape of the rest (${s}) along the non-concatenated axis ${a}.`)})}function NN(t,e){const n=t[0].slice();for(let s=1;s<t.length;s++)n[e]+=t[s][e];return n}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Le;(function(t){t[t.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",t[t.VALUE_ROWIDS=1]="VALUE_ROWIDS",t[t.ROW_LENGTHS=2]="ROW_LENGTHS",t[t.ROW_SPLITS=3]="ROW_SPLITS",t[t.ROW_LIMITS=4]="ROW_LIMITS",t[t.ROW_STARTS=5]="ROW_STARTS"})(Le||(Le={}));function SN(t,e,n){let s=new Array;if(n==null&&e==null)return s;if(e==null)for(;s.length<t+n.length;)s.push(-1);else s=e.slice();if(n==null)return s;if(t+n.length!==s.length)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.rank = ${t+n.length}, but shape.rank = ${s.length}`);for(let r=1;r<n.length;++r){const a=n[r],o=s[s.length-n.length+r],i=s[o];if(a>=0)if(i>=0){if(i!==a)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.shape[${r+t}] = ${a} but shape[${r+t}] = ${i}`)}else s[o]=a}return s}function TN(t){const e={FIRST_DIM_SIZE:Le.FIRST_DIM_SIZE,VALUE_ROWIDS:Le.VALUE_ROWIDS,ROW_LENGTHS:Le.ROW_LENGTHS,ROW_SPLITS:Le.ROW_SPLITS,ROW_LIMITS:Le.ROW_LIMITS,ROW_STARTS:Le.ROW_STARTS},n=[];for(const s of t)if(s in e)n.push(e[s]);else break;return n}function $N(t){return t.length===0?0:t[0]===Le.FIRST_DIM_SIZE?t.length-1:t.length}function EN(t,e){if(t==null||e==null)return;const n=t.length,s=e.length;if(n>=s)throw new Error(`defaultValue.shape=${t} and ragged tensor flatValues.shape=${e}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${s})`);for(let r=0;r<Math.min(n,s-1);++r){const a=t[r],o=e[r+1];if(a>=0&&o>=0&&a!==1&&a!==o)throw new Error(`defaultValue.shape=${t}, and ragged tensor input flatValues.shape=${e} are incompatible: defaultValue.shape[${r-t.length}] = ${a} but ragged tensor input.flatValues.shape[${r-t.length}] = ${o}`)}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ia=30;function kN(t){return t<=Ia?t:Hn(t,Math.floor(Math.sqrt(t)))}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vN(t,e,n){const s=n*(typeof t=="number"?t:t[0]),r=e*(typeof t=="number"?t:t[1]);return[s,r]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _N(t,e,n,s=!0){let r=[];if(s)r=r.concat(e.slice(0)),r.push(t[0]/n),r=r.concat(t.slice(1));else{r=r.concat(t[0]);const a=e.length;for(let o=0;o<a;++o)r=r.concat([t[o+1]/e[o],e[o]]);r=r.concat(t.slice(a+1))}return r}function IN(t,e,n=!0){const s=[];if(n){s.push(e);for(let r=e+1;r<t;++r)r<=2*e?(s.push(r),s.push(r-(e+1))):s.push(r)}else{const r=[],a=[];for(let o=1;o<t;++o)o>=e*2+1||o%2===1?a.push(o):r.push(o);s.push(...r),s.push(0),s.push(...a)}return s}function xN(t,e,n,s=!0){const r=[];s?r.push(t[0]/n):r.push(t[0]*n);for(let a=1;a<t.length;++a)a<=e.length?s?r.push(e[a-1]*t[a]):r.push(t[a]/e[a-1]):r.push(t[a]);return r}function AN(t,e){const n=[0];for(let s=0;s<e;++s)n.push(t[s][0]);return n}function ON(t,e,n){const s=t.slice(0,1);for(let r=0;r<n;++r)s.push(t[r+1]-e[r][0]-e[r][1]);return s}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const DN=1.7580993408473768,FN=1.0507009873554805;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const CN=.3275911,RN=.254829592,PN=-.284496736,BN=1.421413741,LN=-1.453152027,zN=1.061405429;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function VN(t,e){if(t.length!==e.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${t.length}, imag: ${e.length}.`);const n=new Float32Array(t.length*2);for(let s=0;s<n.length;s+=2)n[s]=t[s/2],n[s+1]=e[s/2];return n}function jN(t){const e=new Float32Array(t.length/2),n=new Float32Array(t.length/2);for(let s=0;s<t.length;s+=2)e[s/2]=t[s],n[s/2]=t[s+1];return{real:e,imag:n}}function MN(t){const e=Math.ceil(t.length/4),n=new Float32Array(e),s=new Float32Array(e);for(let r=0;r<t.length;r+=4)n[Math.floor(r/4)]=t[r],s[Math.floor(r/4)]=t[r+1];return{real:n,imag:s}}function WN(t){const e=Math.floor(t.length/4),n=new Float32Array(e),s=new Float32Array(e);for(let r=2;r<t.length;r+=4)n[Math.floor(r/4)]=t[r],s[Math.floor(r/4)]=t[r+1];return{real:n,imag:s}}function UN(t,e){const n=t[e*2],s=t[e*2+1];return{real:n,imag:s}}function qN(t,e,n,s){t[s*2]=e,t[s*2+1]=n}function GN(t,e){const n=new Float32Array(t/2),s=new Float32Array(t/2);for(let r=0;r<Math.ceil(t/2);r++){const a=(e?2:-2)*Math.PI*(r/t);n[r]=Math.cos(a),s[r]=Math.sin(a)}return{real:n,imag:s}}function HN(t,e,n){const s=(n?2:-2)*Math.PI*(t/e),r=Math.cos(s),a=Math.sin(s);return{real:r,imag:a}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xs="->",KN=/->/g,Ja=",",Ya="...";function XN(t,e){t=t.replace(/\s/g,"");const n=(t.length-t.replace(KN,"").length)/xs.length;if(n<1)throw new Error("Equations without an arrow are not supported.");if(n>1)throw new Error(`Equation must contain exactly one arrow ("${xs}").`);const[s,r]=t.split(xs);g(s.indexOf(Ya)===-1,()=>`The ellipsis notation ("${Ya}") is not supported yet.`);const a=s.split(Ja),o=a.length;if(e!==o)throw new Error(`Expected ${o} input tensors, received ${e}`);if(o>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const i=[];for(let f=0;f<r.length;++f){const d=r[f];if(!a.some(y=>y.indexOf(d)!==-1))throw new Error(`Output subscripts contain the label ${d} not present in the input subscripts.`);i.indexOf(d)===-1&&i.push(d)}for(let f=0;f<s.length;++f){const d=s[f];i.indexOf(d)===-1&&d!==Ja&&i.push(d)}const u=new Array(a.length);for(let f=0;f<o;++f){if(new Set(a[f].split("")).size!==a[f].length)throw new Error(`Found duplicate axes in input component ${a[f]}. Support for duplicate axes in input is not implemented yet.`);u[f]=[];for(let d=0;d<a[f].length;++d)u[f].push(i.indexOf(a[f][d]))}const l=i.length,h=r.length,c=[];for(let f=h;f<l;++f)c.push(f);return{allDims:i,summedDims:c,idDims:u}}function ZN(t,e){let n=new Array(t);n.fill(-1);for(let r=0;r<e.length;++r)n[e[r]]=r;const s=[];for(let r=0;r<t;++r)n[r]===-1&&s.push(r);return n=n.filter(r=>r!==-1),{permutationIndices:n,expandDims:s}}function JN(t,e,n){const s=new Array(t);for(let r=0;r<n.length;++r){const a=n[r].shape;for(let o=0;o<e[r].length;++o)s[e[r][o]]===void 0?s[e[r][o]]=a[o]:g(s[e[r][o]]===a[o],()=>`Expected dimension ${s[e[r][o]]} at axis ${o} of input shaped ${JSON.stringify(a)}, but got dimension ${a[o]}`)}}function YN(t,e){const n=t,s=[];let r=0;t.length===0&&n.push(-1),r=t.length+1;for(let o=0;o<r;++o)s.push([]);const a=[];for(let o=0;o<n.length;++o){const i=n[o],u=eS(e,i);for(const l of u)a.indexOf(l)===-1&&(s[o].push(l),a.push(l))}return{path:n,steps:s}}function QN(t){return t.every((e,n)=>e===n)}function eS(t,e){const n=[];for(let s=0;s<t.length;++s)(t[s].length===0||t[s].indexOf(e)!==-1||e===-1)&&n.push(s);return n}function tS(t,e,n=0){let s=[];if(typeof e=="number")g(t.shape[n]%e===0,()=>"Number of splits must evenly divide the axis."),s=new Array(e).fill(t.shape[n]/e);else{const r=e.reduce((o,i)=>(i===-1&&(o+=1),o),0);g(r<=1,()=>"There should be only one negative value in split array.");const a=e.indexOf(-1);if(a!==-1){const o=e.reduce((i,u)=>u>0?i+u:i);e[a]=t.shape[n]-o}g(t.shape[n]===e.reduce((o,i)=>o+i),()=>"The sum of sizes must match the size of the axis dimension."),s=e}return s}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nS(t){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${t}`}function sS(t,e){return`indices(${t}, 0) is invalid: ${e} < 0`}function rS(t,e,n){return`indices(${t}, 0) is invalid: ${e} >= ${n}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aS(t,e){return`only one output dimension may be -1, not both ${t} and ${e}`}function oS(t,e){return`size ${t} must be non-negative, not ${e}`}function iS(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function uS(t,e){const n=W(t),s=W(e);return`Input to reshape is a SparseTensor with ${n}
  dense values, but the requested shape requires a multiple of ${s}. inputShape=${t} outputShape= ${e}`}function lS(t,e){const n=W(t),s=W(e);return`Input to reshape is a tensor with ${n} dense values, but the requested shape has ${s}. inputShape=${t} outputShape=${e}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cS(){return"segment ids must be >= 0"}function hS(){return"segment ids are not increasing"}function pS(t,e){return`Segment id ${t} out of range [0, ${e}), possibly because segmentIds input is not sorted.`}function fS(t,e,n){return`Bad: indices[${t}] == ${e} out of range [0, ${n})`}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dS(t,e){let n=!1,s;for(t<=Ia?(s=t,n=!0):s=Hn(t,Math.floor(Math.sqrt(t)));!n;)s>e||s===t?n=!0:s=Hn(t,s+1);return s}function mS(t,e,n){const s=[],r=t.length;for(let a=0;a<r;a++)a!==e?s.push(t[a]):s.push(n);return s}function gS(t,e,n,s){const r=e.shape.length,a=t.shape.length;if(s!==0&&(s<-r||s>r))throw new Error(`Expect batchDims in the range of [-${r}, ${r}], but got ${s}`);if(s<0&&(s+=r),s>a)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${a}).`);if(n<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${n}).`);for(let c=0;c<s;++c)if(t.shape[c]!==e.shape[c])throw new Error(`x.shape[${c}]: ${t.shape[c]} should be equal to indices.shape[${c}]: ${e.shape[c]}.`);const o=t.shape[n],i=[];let u=1,l=1,h=1;for(let c=0;c<s;++c)i.push(t.shape[c]),u*=t.shape[c];for(let c=s;c<n;c++)i.push(t.shape[c]),l*=t.shape[c];for(let c=s;c<r;c++)i.push(e.shape[c]);for(let c=n+1;c<a;c++)i.push(t.shape[c]),h*=t.shape[c];return{batchSize:u,sliceSize:h,outerSize:l,dimSize:o,outputShape:i}}const yS=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:gS,computeOutShape:mS,segOpComputeOptimalWindowSize:dS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bS(t){try{return t.map(e=>Zn(e))}catch(e){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function wS(t){return t.map(e=>In(e))}const NS=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:RN,ERF_A2:PN,ERF_A3:BN,ERF_A4:LN,ERF_A5:zN,ERF_P:CN,PARALLELIZE_THRESHOLD:Ia,get RowPartitionType(){return Le},SELU_SCALE:FN,SELU_SCALEALPHA:DN,applyActivation:Ss,assertAndGetBroadcastShape:ne,assertAxesAreInnerMostDims:Bg,assertParamsConsistent:wN,assignToTypedArray:qN,axesAreInnerMostDims:Pr,calculateShapes:Wh,checkEinsumDimSizes:JN,checkPadOnDimRoundingMode:Ae,combineLocations:Rc,combineRaggedTensorToTensorShapes:SN,complexWithEvenIndex:MN,complexWithOddIndex:WN,computeConv2DInfo:An,computeConv3DInfo:nc,computeDefaultPad:xr,computeDilation2DInfo:Dm,computeOptimalWindowSize:kN,computeOutAndReduceShapes:Pg,computeOutShape:NN,computePool2DInfo:tc,computePool3DInfo:Fm,convertConv2DDataFormat:sc,decodeEinsumEquation:XN,eitherStridesOrDilationsAreOne:Ye,expandShapeToKeepDim:Fn,exponent:HN,exponents:GN,fromStringArrayToUint8:wS,fromUint8ToStringArray:bS,getAxesPermutation:Lg,getBroadcastDims:Ac,getComplexWithIndex:UN,getEinsumComputePath:YN,getEinsumPermutation:ZN,getFusedBiasGradient:Ns,getFusedDyActivation:ws,getImageCenter:vN,getInnerMostAxes:Vg,getPermuted:IN,getRaggedRank:$N,getReductionAxes:Fr,getReshaped:_N,getReshapedPermuted:xN,getRowPartitionTypesHelper:TN,getSliceBeginCoords:AN,getSliceSize:ON,getSparseFillEmptyRowsIndicesDenseShapeMismatch:nS,getSparseFillEmptyRowsNegativeIndexErrorMessage:sS,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:rS,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:iS,getSparseReshapeInputOutputMismatchErrorMessage:lS,getSparseReshapeInputOutputMultipleErrorMessage:uS,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:aS,getSparseReshapeNegativeOutputDimErrorMessage:oS,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:fS,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:cS,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:hS,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:pS,getUndoAxesPermutation:zg,isIdentityPermutation:QN,log:Ff,mergeRealAndImagArrays:VN,prepareAndValidate:vp,prepareSplitSize:tS,segment_util:yS,shouldFuse:Ts,slice_util:Rp,splitRealAndImagArrays:jN,stridesOrDilationsArePositive:xt,tupleValuesAreOne:wn,upcastType:us,validateDefaultValueShape:EN,validateInput:ys,validateUpdateShape:ma,warn:et},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const SS=Object.freeze(Object.defineProperty({__proto__:null,nonMaxSuppressionV3Impl:up,nonMaxSuppressionV4Impl:lp,nonMaxSuppressionV5Impl:cp,whereImpl:Jh},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */_1();const TS=Object.freeze(Object.defineProperty({__proto__:null,Abs:To,Acos:$o,Acosh:Eo,AdadeltaOptimizer:wa,AdagradOptimizer:Na,AdamOptimizer:Sa,AdamaxOptimizer:Ta,Add:mr,AddN:ko,All:vo,Any:_o,ArgMax:Io,ArgMin:xo,Asin:Ao,Asinh:Oo,Atan:Do,Atan2:Co,Atanh:Fo,AvgPool:Ro,AvgPool3D:Po,AvgPool3DGrad:mf,AvgPoolGrad:df,BatchMatMul:Bo,BatchToSpaceND:Lo,Bincount:zo,BitwiseAnd:Vo,BroadcastArgs:jo,BroadcastTo:gf,Cast:gr,Ceil:Mo,ClipByValue:Wo,Complex:Uo,ComplexAbs:qo,Concat:Go,Conv2D:Ho,Conv2DBackpropFilter:Ko,Conv2DBackpropInput:Xo,Conv3D:Zo,Conv3DBackpropFilterV2:yf,Conv3DBackpropInputV2:Jo,Cos:Yo,Cosh:Qo,CropAndResize:ni,Cumprod:ei,Cumsum:ti,DataStorage:qp,DenseBincount:si,DepthToSpace:ri,DepthwiseConv2dNative:ai,DepthwiseConv2dNativeBackpropFilter:oi,DepthwiseConv2dNativeBackpropInput:ii,Diag:ui,Dilation2D:li,Dilation2DBackpropFilter:wf,Dilation2DBackpropInput:bf,Draw:yr,get ENV(){return fr},Einsum:hi,Elu:pi,EluGrad:Nf,Environment:No,Equal:di,Erf:fi,Exp:mi,ExpandDims:gi,Expm1:yi,FFT:bi,Fill:wi,FlipLeftRight:Ni,Floor:Si,FloorDiv:Ti,FromPixels:Os,FusedBatchNorm:$i,FusedConv2D:Fs,FusedDepthwiseConv2D:Cs,GatherNd:ki,GatherV2:Ei,Greater:vi,GreaterEqual:_i,IFFT:Ii,Identity:br,Imag:xi,IsFinite:Ai,IsInf:Oi,IsNan:Di,KernelBackend:uo,LRN:Mi,LRNGrad:Ef,LeakyRelu:Fi,Less:Ci,LessEqual:Ri,LinSpace:Pi,Log:Bi,Log1p:Li,LogSoftmax:Tf,LogicalAnd:zi,LogicalNot:Vi,LogicalOr:ji,LogicalXor:Sf,LowerBound:$f,MatrixBandPart:kf,Max:Wi,MaxPool:qi,MaxPool3D:Gi,MaxPool3DGrad:_f,MaxPoolGrad:vf,MaxPoolWithArgmax:Hi,Maximum:Ui,Mean:Ki,Min:Xi,Minimum:Zi,MirrorPad:Ji,Mod:Yi,MomentumOptimizer:$a,Multinomial:Qi,Multiply:eu,Neg:tu,NonMaxSuppressionV3:su,NonMaxSuppressionV4:ru,NonMaxSuppressionV5:au,NotEqual:nu,OP_SCOPE_SUFFIX:$r,OneHot:iu,OnesLike:ou,Optimizer:mt,OptimizerConstructors:Pp,Pack:uu,PadV2:lu,Pool:If,Pow:cu,Prelu:hu,Prod:pu,RMSPropOptimizer:Ea,RaggedGather:fu,RaggedRange:du,RaggedTensorToTensor:mu,Range:gu,get Rank(){return Ls},Real:yu,RealDiv:ci,Reciprocal:bu,get Reduction(){return he},Relu:wu,Relu6:$u,Reshape:Nu,ResizeBilinear:Tu,ResizeBilinearGrad:Af,ResizeNearestNeighbor:Su,ResizeNearestNeighborGrad:xf,Reverse:Eu,RotateWithOffset:cl,Round:ku,Rsqrt:vu,SGDOptimizer:$s,ScatterNd:_u,SearchSorted:xu,Select:Au,Selu:Ou,Sigmoid:Pu,Sign:Ru,Sin:Fu,Sinh:Cu,Slice:Du,Softmax:Mu,Softplus:Bu,SpaceToBatchND:Vu,SparseFillEmptyRows:Wu,SparseReshape:Uu,SparseSegmentMean:qu,SparseSegmentSum:Gu,SparseToDense:Hu,SplitV:ju,Sqrt:Lu,Square:Of,SquaredDifference:Ku,StaticRegexReplace:Xu,Step:ll,StridedSlice:Zu,StringNGrams:Ju,StringSplit:Yu,StringToHashBucketFast:Qu,Sub:el,Sum:zu,Tan:tl,Tanh:nl,Tensor:te,TensorBuffer:Jn,TensorScatterUpdate:Iu,Tile:wr,TopK:sl,Transform:rl,Transpose:jn,Unique:al,Unpack:ol,UnsortedSegmentSum:il,UpperBound:Df,Variable:mn,ZerosLike:ul,_FusedMatMul:Ds,abs:be,acos:Wl,acosh:Ul,add:C,addN:ql,all:Gl,any:Hl,argMax:Kl,argMin:Xl,asin:Zl,asinh:Jl,atan:Yl,atan2:Ql,atanh:ec,avgPool:Ar,avgPool3d:rc,backend:Ol,backend_util:NS,basicLSTMCell:ac,batchNorm:On,batchNorm2d:oc,batchNorm3d:ic,batchNorm4d:uc,batchToSpaceND:Or,bincount:Dr,bitwiseAnd:lc,booleanMaskAsync:Yh,broadcastArgs:cc,broadcastTo:cn,broadcast_util:vg,browser:sN,buffer:Ve,cast:X,ceil:hc,clipByValue:pc,clone:Xe,complex:Je,concat:ue,concat1d:fc,concat2d:dc,concat3d:mc,concat4d:gc,conv1d:yc,conv2d:Dn,conv2dTranspose:wc,conv3d:Nc,conv3dTranspose:Sc,copyRegisteredKernels:Bf,cos:Tc,cosh:$c,cosineWindow:bs,cumprod:Ec,cumsum:kc,customGrad:Me,denseBincount:vc,deprecationWarn:Nd,depthToSpace:_c,depthwiseConv2d:ls,device_util:dd,diag:Ic,dilation2d:xc,disableDeprecationWarnings:wd,dispose:pe,disposeVariables:Sd,div:K,divNoNan:Oc,dot:Dc,dropout:sp,einsum:wt,elu:Rr,enableDebugMode:bd,enableProdMode:yd,enclosingPowerOfTwo:ya,engine:Td,ensureShape:Fc,env:R,equal:Cr,erf:Cc,euclideanNorm:Bc,exp:ct,expandDims:qe,expm1:Lc,eye:Br,fft:ds,fill:tn,findBackend:xd,findBackendFactory:Ad,floor:Lr,floorDiv:Ir,fused:ap,gather:zr,gatherND:np,gather_util:rN,getBackend:Al,getGradient:Rs,getKernel:fn,getKernelsForBackend:Kn,grad:dy,grads:my,greater:Rn,greaterEqual:Vr,ifft:$n,imag:Pn,image:fp,inTopKAsync:rp,io:_a,irfft:ha,isFinite:zc,isInf:Vc,isNaN:jc,keep:De,kernel_impls:SS,leakyRelu:jr,less:ts,lessEqual:cs,linalg:dp,linspace:Mc,localResponseNormalization:Wc,log:Zt,log1p:Mr,logSigmoid:qc,logSoftmax:Gc,logSumExp:Ur,logicalAnd:Nn,logicalNot:qr,logicalOr:Gr,logicalXor:Hc,losses:mp,lowerBound:Kc,matMul:V,math:H1,max:kt,maxPool:Hr,maxPool3d:Xc,maxPoolWithArgmax:Zc,maximum:Kr,mean:Sn,memory:$d,meshgrid:Jc,min:es,minimum:Tn,mirrorPad:Yc,mod:Qc,moments:eh,movingAverage:Qh,mul:x,multiRNNCell:th,multinomial:nh,neg:Ce,nextFrame:bN,norm:Cn,notEqual:Xr,oneHot:ns,ones:rt,onesLike:sh,op:w,outerProduct:rh,pad:nn,pad1d:ah,pad2d:oh,pad3d:ih,pad4d:uh,pool:lh,pow:Xt,prelu:Jr,print:_r,prod:ch,profile:Ed,raggedGather:hh,raggedRange:ph,raggedTensorToTensor:fh,rand:dh,randomGamma:bh,randomNormal:ua,randomStandardNormal:wh,randomUniform:fs,randomUniformInt:Nh,range:Jt,ready:_d,real:Yt,reciprocal:Sh,registerBackend:Od,registerGradient:Cf,registerKernel:hl,relu:Bn,relu6:la,removeBackend:Id,reshape:k,reverse:ht,reverse1d:Th,reverse2d:$h,reverse3d:Eh,reverse4d:kh,rfft:ms,round:ca,rsqrt:vh,scalar:z,scatterND:ep,scatter_util:y0,searchSorted:ps,selu:_h,separableConv2d:Ih,serialization:k1,setBackend:vd,setPlatform:Dd,setdiff1dAsync:xh,sigmoid:Et,sign:Ah,signal:pp,sin:Oh,sinh:Dh,slice:q,slice1d:Fh,slice2d:Ch,slice3d:Rh,slice4d:Ph,slice_util:Rp,softmax:Bh,softplus:Wr,spaceToBatchND:Zr,sparse:gp,sparseToDense:tp,spectral:hp,split:Qt,sqrt:je,square:xe,squaredDifference:pa,squeeze:gs,stack:We,step:fa,stridedSlice:Lh,string:yp,sub:L,sum:H,sumOutType:ad,tan:zh,tanh:Qn,tensor:Fe,tensor1d:$e,tensor2d:Ut,tensor3d:da,tensor4d:Vh,tensor5d:jh,tensor6d:Mh,tensorScatterUpdate:Uh,tensor_util:ud,test_util:Ab,tidy:j,tile:Wt,time:kd,topk:qh,train:gN,transpose:En,truncatedNormal:Gh,unique:Hh,unregisterGradient:Pf,unregisterKernel:Rf,unsortedSegmentSum:Kh,unstack:dt,upcastType:us,upperBound:Xh,util:Kf,valueAndGrad:gy,valueAndGrads:yy,variable:Zh,variableGrads:Uc,version_core:mN,where:Ze,whereAsync:ga,zeros:At,zerosLike:Ne},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $S=R();$S.registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,t=>{t&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */var ge;(function(t){t[t.DT_INVALID=0]="DT_INVALID",t[t.DT_FLOAT=1]="DT_FLOAT",t[t.DT_DOUBLE=2]="DT_DOUBLE",t[t.DT_INT32=3]="DT_INT32",t[t.DT_UINT8=4]="DT_UINT8",t[t.DT_INT16=5]="DT_INT16",t[t.DT_INT8=6]="DT_INT8",t[t.DT_STRING=7]="DT_STRING",t[t.DT_COMPLEX64=8]="DT_COMPLEX64",t[t.DT_INT64=9]="DT_INT64",t[t.DT_BOOL=10]="DT_BOOL",t[t.DT_QINT8=11]="DT_QINT8",t[t.DT_QUINT8=12]="DT_QUINT8",t[t.DT_QINT32=13]="DT_QINT32",t[t.DT_BFLOAT16=14]="DT_BFLOAT16",t[t.DT_QINT16=15]="DT_QINT16",t[t.DT_QUINT16=16]="DT_QUINT16",t[t.DT_UINT16=17]="DT_UINT16",t[t.DT_COMPLEX128=18]="DT_COMPLEX128",t[t.DT_HALF=19]="DT_HALF",t[t.DT_RESOURCE=20]="DT_RESOURCE",t[t.DT_VARIANT=21]="DT_VARIANT",t[t.DT_UINT32=22]="DT_UINT32",t[t.DT_UINT64=23]="DT_UINT64",t[t.DT_FLOAT_REF=101]="DT_FLOAT_REF",t[t.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",t[t.DT_INT32_REF=103]="DT_INT32_REF",t[t.DT_UINT8_REF=104]="DT_UINT8_REF",t[t.DT_INT16_REF=105]="DT_INT16_REF",t[t.DT_INT8_REF=106]="DT_INT8_REF",t[t.DT_STRING_REF=107]="DT_STRING_REF",t[t.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",t[t.DT_INT64_REF=109]="DT_INT64_REF",t[t.DT_BOOL_REF=110]="DT_BOOL_REF",t[t.DT_QINT8_REF=111]="DT_QINT8_REF",t[t.DT_QUINT8_REF=112]="DT_QUINT8_REF",t[t.DT_QINT32_REF=113]="DT_QINT32_REF",t[t.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",t[t.DT_QINT16_REF=115]="DT_QINT16_REF",t[t.DT_QUINT16_REF=116]="DT_QUINT16_REF",t[t.DT_UINT16_REF=117]="DT_UINT16_REF",t[t.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",t[t.DT_HALF_REF=119]="DT_HALF_REF",t[t.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",t[t.DT_VARIANT_REF=121]="DT_VARIANT_REF",t[t.DT_UINT32_REF=122]="DT_UINT32_REF",t[t.DT_UINT64_REF=123]="DT_UINT64_REF"})(ge||(ge={}));var Qa;(function(t){(function(e){e[e.LEGACY=0]="LEGACY",e[e.V1=1]="V1",e[e.V2=2]="V2"})(t.CheckpointFormatVersion||(t.CheckpointFormatVersion={}))})(Qa||(Qa={}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xa={};function ES(t,e){const n={tfOpName:t,category:"custom",inputs:[],attrs:[],customExecutor:e};xa[t]=n}function Bp(t){return xa[t]}function kS(t){delete xa[t]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function p(t,e,n,s,r){const a=e.inputParams[t];if(a&&a.inputIndexStart!==void 0){const i=a.inputIndexStart,u=a.inputIndexEnd===0?void 0:a.inputIndexEnd===void 0?i+1:a.inputIndexEnd,l=i<0?e.inputNames.length+i:i;if(a.type==="tensor")return oe(e.inputNames[l],n,s,r);if(a.type==="tensors"){const f=e.inputs.slice(i,u);return e.inputNames.slice(i,u).filter((y,S)=>{var b;return((b=f[S])===null||b===void 0?void 0:b.op)!=="NoOp"}).map(y=>oe(y,n,s,r))}const h=oe(e.inputNames[l],n,s,r),c=h.dataSync();return a.type==="number"?c[0]:$t(h.shape,c)}const o=e.attrParams[t];return o&&o.value}function oe(t,e,n,s){const[r,a]=ye(t,n);if(s!=null){const i=s.getHashTableHandleByName(r);if(i!=null)return i}const o=n.currentContextIds.find(i=>!!e[rs(r,i)]);return o!==void 0?e[rs(r,o)][a]:void 0}function eo(t,e,n){return e[rs(t,n.currentContextId)]}function Ge(t,e){const[n,s,r]=ye(t,e);return[rs(n,e&&e.currentContextId),s,r]}function rs(t,e){return e?`${t}-${e}`:t}function ye(t,e){if(t==="")return["",0,void 0];const n=e!=null&&e.parseNodeNameCache!=null;if(n){const a=e.parseNodeNameCache.get(t);if(a!=null)return a}const s=t.split(":");let r;if(s.length===1)r=[t,0,void 0];else{const a=s[0],o=s.length===3?s[1]:void 0,i=Number(s[s.length-1]);r=[a,i,o]}return n&&e.parseNodeNameCache.set(t,r),r}function Un(t,e,n){let s=p("pad",t,e,n);if(s==="explicit"){s=p("explicitPaddings",t,e,n);const r=[[0,0],[0,0],[0,0],[0,0]];for(let a=0;a<4;a++)r[a][0]=s[a*2],r[a][1]=s[a*2+1];return r}return s}function He(t){return t.kept?t:Xe(t)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vS=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],_S=Object.freeze(Object.defineProperty({__proto__:null,json:vS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const IS=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],xS=Object.freeze(Object.defineProperty({__proto__:null,json:IS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const AS=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],OS=Object.freeze(Object.defineProperty({__proto__:null,json:AS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const DS=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],FS=Object.freeze(Object.defineProperty({__proto__:null,json:DS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const CS=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],RS=Object.freeze(Object.defineProperty({__proto__:null,json:CS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const PS=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],BS=Object.freeze(Object.defineProperty({__proto__:null,json:PS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const LS=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],zS=Object.freeze(Object.defineProperty({__proto__:null,json:LS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const VS=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],jS=Object.freeze(Object.defineProperty({__proto__:null,json:VS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const MS=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],WS=Object.freeze(Object.defineProperty({__proto__:null,json:MS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const US=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],qS=Object.freeze(Object.defineProperty({__proto__:null,json:US},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const GS=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],HS=Object.freeze(Object.defineProperty({__proto__:null,json:GS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const KS=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],XS=Object.freeze(Object.defineProperty({__proto__:null,json:KS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ZS=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],JS=Object.freeze(Object.defineProperty({__proto__:null,json:ZS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const YS=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],QS=Object.freeze(Object.defineProperty({__proto__:null,json:YS},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const eT=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],tT=Object.freeze(Object.defineProperty({__proto__:null,json:eT},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nT=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],sT=Object.freeze(Object.defineProperty({__proto__:null,json:nT},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rT=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],aT=Object.freeze(Object.defineProperty({__proto__:null,json:rT},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const oT=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],iT=Object.freeze(Object.defineProperty({__proto__:null,json:oT},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uT=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],lT=Object.freeze(Object.defineProperty({__proto__:null,json:uT},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class to{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[_S,xS,OS,FS,RS,BS,zS,jS,WS,qS,HS,XS,JS,QS,tT,sT,aT,iT,lT],n=[].concat(...e.map(s=>s.json));this.opMappers=n.reduce((s,r)=>(s[r.tfOpName]=r,s),{})}transformGraph(e,n={}){const s=e.node,r=[],a=[],o=[],i=s.reduce((S,b)=>(S[b.name]=this.mapNode(b),b.op.startsWith("Placeholder")?r.push(S[b.name]):b.op==="Const"?a.push(S[b.name]):(b.input==null||b.input.length===0)&&o.push(S[b.name]),S),{});let u=[];const l=[];let h={},c={};n!=null&&(h=this.mapSignatureEntries(n.inputs),c=this.mapSignatureEntries(n.outputs));const f=Object.keys(i);f.forEach(S=>{const b=i[S];b.inputNames.forEach((T,A)=>{const[$,,E]=Ge(T),v=i[$];if(v.outputs!=null){const _=v.outputs.indexOf(E);if(_!==-1){const O=`${$}:${_}`;b.inputNames[A]=O}}b.inputs.push(v),v.children.push(b)})}),Object.keys(c).length===0?f.forEach(S=>{const b=i[S];b.children.length===0&&l.push(b)}):Object.keys(c).forEach(S=>{const[b]=Ge(S),T=i[b];T!=null&&(T.signatureKey=c[S],l.push(T))}),Object.keys(h).length>0?Object.keys(h).forEach(S=>{const[b]=Ge(S),T=i[b];T&&(T.signatureKey=h[S],u.push(T))}):u=r;let d={};e.library!=null&&e.library.function!=null&&(d=e.library.function.reduce((S,b)=>(S[b.signature.name]=this.mapFunction(b),S),{}));const y={nodes:i,inputs:u,outputs:l,weights:a,placeholders:r,signature:n,functions:d};return o.length>0&&(y.initNodes=o),y}mapSignatureEntries(e){return Object.keys(e||{}).reduce((n,s)=>(n[e[s].name]=s,n),{})}mapNode(e){const n=Bp(e.op)||this.opMappers[e.op]||{};e.attr==null&&(e.attr={});const s={name:e.name,op:e.op,category:n.category,inputNames:(e.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:n.outputs};return n.inputs!=null&&(s.inputParams=n.inputs.reduce((r,a)=>(r[a.name]={type:a.type,inputIndexStart:a.start,inputIndexEnd:a.end},r),{})),n.attrs!=null&&(s.attrParams=n.attrs.reduce((r,a)=>{const o=a.type;let i;switch(a.type){case"string":i=Ys(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Ys(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"string[]":i=ar(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=ar(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"number":i=er(e.attr,a.tfName,a.defaultValue||0),i===void 0&&a.tfDeprecatedName&&(i=er(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"number[]":i=rr(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=rr(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool":i=Qs(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=Qs(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool[]":i=ir(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=ir(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape":i=sr(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=sr(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape[]":i=or(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=or(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype":i=tr(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=tr(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype[]":i=nr(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=nr(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"func":i=no(e.attr,a.tfName,a.defaultValue),i===void 0&&a.tfDeprecatedName&&(i=no(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${a.type} for op: ${e.op}`)}return r[a.name]={value:i,type:o},r},{})),s}mapFunction(e){const n=e.nodeDef,s=[],r=[];let a={};n!=null&&(a=n.reduce((c,f)=>(c[f.name]=this.mapNode(f),f.op==="Const"&&r.push(c[f.name]),c),{}));const o=[],i=[];e.signature.inputArg.forEach(c=>{const[f]=Ge(c.name),d={name:f,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:Aa(c.type),type:"dtype"}},children:[]};d.signatureKey=c.name,o.push(d),a[f]=d}),Object.keys(a).forEach(c=>{const f=a[c];f.inputNames.forEach((d,y)=>{const[S,,b]=Ge(d),T=a[S];if(T.outputs!=null){const A=T.outputs.indexOf(b);if(A!==-1){const $=`${S}:${A}`;f.inputNames[y]=$}}f.inputs.push(T),T.children.push(f)})});const l=e.ret;e.signature.outputArg.forEach(c=>{const[f,d]=Ge(l[c.name]),y=a[f];y!=null&&(y.defaultOutput=d,i.push(y))});const h=this.mapArgsToSignature(e);return{nodes:a,inputs:o,outputs:i,weights:r,placeholders:s,signature:h}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((n,s)=>(n[s.name]=this.mapArgToTensorInfo(s),n),{}),outputs:e.signature.outputArg.reduce((n,s)=>(n[s.name]=this.mapArgToTensorInfo(s,e.ret),n),{})}}mapArgToTensorInfo(e,n){let s=e.name;return n!=null&&(s=n[s]),{name:s,dtype:e.type}}}function cT(t){const e=R().global;if(typeof e.atob<"u")return e.atob(t);if(typeof Buffer<"u")return new Buffer(t,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function Lp(t,e){const n=Array.isArray(t)?String.fromCharCode.apply(null,t):cT(t);return e?n:n.toLowerCase()}function Ys(t,e,n,s=!1){const r=t[e];return r!=null?Lp(r.s,s):n}function Qs(t,e,n){const s=t[e];return s?s.b:n}function er(t,e,n){const s=t[e]||{},r=s.i!=null?s.i:s.f!=null?s.f:n;return typeof r=="number"?r:parseInt(r,10)}function Aa(t){switch(typeof t=="string"&&(t=ge[t]),t){case ge.DT_FLOAT:case ge.DT_HALF:return"float32";case ge.DT_INT32:case ge.DT_INT64:case ge.DT_INT8:case ge.DT_UINT8:return"int32";case ge.DT_BOOL:return"bool";case ge.DT_DOUBLE:return"float32";case ge.DT_STRING:return"string";case ge.DT_COMPLEX64:case ge.DT_COMPLEX128:return"complex64";default:return null}}function no(t,e,n){const s=t[e];return s&&s.func?s.func.name:n}function tr(t,e,n){const s=t[e];return s&&s.type?Aa(s.type):n}function nr(t,e,n){const s=t[e];return s&&s.list&&s.list.type?s.list.type.map(r=>Aa(r)):n}function zp(t){if(!t.unknownRank)return t.dim!=null?t.dim.map(e=>typeof e.size=="number"?e.size:parseInt(e.size,10)):[]}function sr(t,e,n){const s=t[e];return s&&s.shape?zp(s.shape):n}function rr(t,e,n){const s=t[e];return s?((s.list.f&&s.list.f.length?s.list.f:s.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):n}function ar(t,e,n,s=!1){const r=t[e];return r&&r.list&&r.list.s?r.list.s.map(a=>Lp(a,s)):n}function or(t,e,n){const s=t[e];return s&&s.list&&s.list.shape?s.list.shape.map(r=>zp(r)):n}function ir(t,e,n){const s=t[e];return s&&s.list&&s.list.b?s.list.b:n}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hT{constructor(e,n,s){this.node=e,this.tensorMap=n,this.context=s,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(r=>this.getInput(r)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((r,a)=>(r[a]=this.getAttr(a),r),{}))}getInput(e){return oe(e,this.tensorMap,this.context)}getAttr(e,n){const s=this.node.rawAttrs[e];if(s.tensor!=null)return oe(e,this.tensorMap,this.context);if(s.i!=null||s.f!=null)return er(this.node.rawAttrs,e,n);if(s.s!=null)return Ys(this.node.rawAttrs,e,n);if(s.b!=null)return Qs(this.node.rawAttrs,e,n);if(s.shape!=null)return sr(this.node.rawAttrs,e,n);if(s.type!=null)return tr(this.node.rawAttrs,e,n);if(s.list!=null){if(s.list.i!=null||s.list.f!=null)return rr(this.node.rawAttrs,e,n);if(s.list.s!=null)return ar(this.node.rawAttrs,e,n);if(s.list.shape!=null)return or(this.node.rawAttrs,e,n);if(s.list.b!=null)return ir(this.node.rawAttrs,e,n);if(s.list.type!=null)return nr(this.node.rawAttrs,e,n)}return n}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ie=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:$r,abs:be,acos:Wl,acosh:Ul,add:C,addN:ql,all:Gl,any:Hl,argMax:Kl,argMin:Xl,asin:Zl,asinh:Jl,atan:Yl,atan2:Ql,atanh:ec,avgPool:Ar,avgPool3d:rc,basicLSTMCell:ac,batchNorm:On,batchNorm2d:oc,batchNorm3d:ic,batchNorm4d:uc,batchToSpaceND:Or,bincount:Dr,bitwiseAnd:lc,booleanMaskAsync:Yh,broadcastArgs:cc,broadcastTo:cn,buffer:Ve,cast:X,ceil:hc,clipByValue:pc,clone:Xe,complex:Je,concat:ue,concat1d:fc,concat2d:dc,concat3d:mc,concat4d:gc,conv1d:yc,conv2d:Dn,conv2dTranspose:wc,conv3d:Nc,conv3dTranspose:Sc,cos:Tc,cosh:$c,cosineWindow:bs,cumprod:Ec,cumsum:kc,denseBincount:vc,depthToSpace:_c,depthwiseConv2d:ls,diag:Ic,dilation2d:xc,div:K,divNoNan:Oc,dot:Dc,dropout:sp,einsum:wt,elu:Rr,enclosingPowerOfTwo:ya,ensureShape:Fc,equal:Cr,erf:Cc,euclideanNorm:Bc,exp:ct,expandDims:qe,expm1:Lc,eye:Br,fft:ds,fill:tn,floor:Lr,floorDiv:Ir,fused:ap,gather:zr,gatherND:np,greater:Rn,greaterEqual:Vr,ifft:$n,imag:Pn,image:fp,inTopKAsync:rp,irfft:ha,isFinite:zc,isInf:Vc,isNaN:jc,leakyRelu:jr,less:ts,lessEqual:cs,linalg:dp,linspace:Mc,localResponseNormalization:Wc,log:Zt,log1p:Mr,logSigmoid:qc,logSoftmax:Gc,logSumExp:Ur,logicalAnd:Nn,logicalNot:qr,logicalOr:Gr,logicalXor:Hc,losses:mp,lowerBound:Kc,matMul:V,max:kt,maxPool:Hr,maxPool3d:Xc,maxPoolWithArgmax:Zc,maximum:Kr,mean:Sn,meshgrid:Jc,min:es,minimum:Tn,mirrorPad:Yc,mod:Qc,moments:eh,movingAverage:Qh,mul:x,multiRNNCell:th,multinomial:nh,neg:Ce,norm:Cn,notEqual:Xr,oneHot:ns,ones:rt,onesLike:sh,op:w,outerProduct:rh,pad:nn,pad1d:ah,pad2d:oh,pad3d:ih,pad4d:uh,pool:lh,pow:Xt,prelu:Jr,print:_r,prod:ch,raggedGather:hh,raggedRange:ph,raggedTensorToTensor:fh,rand:dh,randomGamma:bh,randomNormal:ua,randomStandardNormal:wh,randomUniform:fs,randomUniformInt:Nh,range:Jt,real:Yt,reciprocal:Sh,relu:Bn,relu6:la,reshape:k,reverse:ht,reverse1d:Th,reverse2d:$h,reverse3d:Eh,reverse4d:kh,rfft:ms,round:ca,rsqrt:vh,scalar:z,scatterND:ep,searchSorted:ps,selu:_h,separableConv2d:Ih,setdiff1dAsync:xh,sigmoid:Et,sign:Ah,signal:pp,sin:Oh,sinh:Dh,slice:q,slice1d:Fh,slice2d:Ch,slice3d:Rh,slice4d:Ph,softmax:Bh,softplus:Wr,spaceToBatchND:Zr,sparse:gp,sparseToDense:tp,spectral:hp,split:Qt,sqrt:je,square:xe,squaredDifference:pa,squeeze:gs,stack:We,step:fa,stridedSlice:Lh,string:yp,sub:L,sum:H,tan:zh,tanh:Qn,tensor:Fe,tensor1d:$e,tensor2d:Ut,tensor3d:da,tensor4d:Vh,tensor5d:jh,tensor6d:Mh,tensorScatterUpdate:Uh,tile:Wt,topk:qh,transpose:En,truncatedNormal:Gh,unique:Hh,unsortedSegmentSum:Kh,unstack:dt,upperBound:Xh,variable:Zh,where:Ze,whereAsync:ga,zeros:At,zerosLike:Ne},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const pT=(t,e,n,s=ie)=>{switch(t.op){case"BiasAdd":case"AddV2":case"Add":return[s.add(p("a",t,e,n),p("b",t,e,n))];case"AddN":return[s.addN(p("tensors",t,e,n))];case"FloorMod":case"Mod":return[s.mod(p("a",t,e,n),p("b",t,e,n))];case"Mul":return[s.mul(p("a",t,e,n),p("b",t,e,n))];case"RealDiv":case"Div":return[s.div(p("a",t,e,n),p("b",t,e,n))];case"DivNoNan":return[s.divNoNan(p("a",t,e,n),p("b",t,e,n))];case"FloorDiv":return[s.floorDiv(p("a",t,e,n),p("b",t,e,n))];case"Sub":return[s.sub(p("a",t,e,n),p("b",t,e,n))];case"Minimum":return[s.minimum(p("a",t,e,n),p("b",t,e,n))];case"Maximum":return[s.maximum(p("a",t,e,n),p("b",t,e,n))];case"Pow":return[s.pow(p("a",t,e,n),p("b",t,e,n))];case"SquaredDifference":return[s.squaredDifference(p("a",t,e,n),p("b",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fT=(t,e,n,s=ie)=>{switch(t.op){case"Abs":case"ComplexAbs":return[s.abs(p("x",t,e,n))];case"Acos":return[s.acos(p("x",t,e,n))];case"Acosh":return[s.acosh(p("x",t,e,n))];case"Asin":return[s.asin(p("x",t,e,n))];case"Asinh":return[s.asinh(p("x",t,e,n))];case"Atan":return[s.atan(p("x",t,e,n))];case"Atan2":return[s.atan2(p("x",t,e,n),p("y",t,e,n))];case"Atanh":return[s.atanh(p("x",t,e,n))];case"Ceil":return[s.ceil(p("x",t,e,n))];case"Complex":return[s.complex(p("real",t,e,n),p("imag",t,e,n))];case"Cos":return[s.cos(p("x",t,e,n))];case"Cosh":return[s.cosh(p("x",t,e,n))];case"Elu":return[s.elu(p("x",t,e,n))];case"Erf":return[s.erf(p("x",t,e,n))];case"Exp":return[s.exp(p("x",t,e,n))];case"Expm1":return[s.expm1(p("x",t,e,n))];case"Floor":return[s.floor(p("x",t,e,n))];case"Log":return[s.log(p("x",t,e,n))];case"Log1p":return[s.log1p(p("x",t,e,n))];case"Imag":return[s.imag(p("x",t,e,n))];case"Neg":return[s.neg(p("x",t,e,n))];case"Reciprocal":return[s.reciprocal(p("x",t,e,n))];case"Real":return[s.real(p("x",t,e,n))];case"Relu":return[s.relu(p("x",t,e,n))];case"Round":return[s.round(p("x",t,e,n))];case"Selu":return[s.selu(p("x",t,e,n))];case"Sigmoid":return[s.sigmoid(p("x",t,e,n))];case"Sin":return[s.sin(p("x",t,e,n))];case"Sign":return[s.sign(p("x",t,e,n))];case"Sinh":return[s.sinh(p("x",t,e,n))];case"Softplus":return[s.softplus(p("x",t,e,n))];case"Sqrt":return[s.sqrt(p("x",t,e,n))];case"Square":return[s.square(p("x",t,e,n))];case"Tanh":return[s.tanh(p("x",t,e,n))];case"Tan":return[s.tan(p("x",t,e,n))];case"ClipByValue":return[s.clipByValue(p("x",t,e,n),p("clipValueMin",t,e,n),p("clipValueMax",t,e,n))];case"Relu6":return[s.relu6(p("x",t,e,n))];case"Rsqrt":return[s.rsqrt(oe(t.inputNames[0],e,n))];case"LeakyRelu":return[s.leakyRelu(p("x",t,e,n),p("alpha",t,e,n))];case"Prelu":return[s.prelu(p("x",t,e,n),p("alpha",t,e,n))];case"IsNan":return[s.isNaN(oe(t.inputNames[0],e,n))];case"IsInf":return[s.isInf(oe(t.inputNames[0],e,n))];case"IsFinite":return[s.isFinite(oe(t.inputNames[0],e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ke(t,e,n=""){if(!(typeof t=="number"||typeof e=="number")){g(t.length===e.length,()=>n+` Shapes ${t} and ${e} must match`);for(let s=0;s<t.length;s++){const r=t[s],a=e[s];g(r<0||a<0||r===a,()=>n+` Shapes ${t} and ${e} must match`)}}}function so(t){return!(typeof t=="number"||t.some(e=>e<0))}function an(t,e,n){let s=ur(t,n);const r=!so(s);if(r&&e.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${s}`);if(r&&e.forEach(a=>{s=ur(a.shape,s)}),!so(s))throw new Error(`Non-fully-defined elementShape: ${s}`);return s}function ur(t,e){if(typeof t=="number")return e;if(typeof e=="number")return t;if(t.length!==e.length)throw new Error(`Incompatible ranks during merge: ${t} vs. ${e}`);const n=[];for(let s=0;s<t.length;++s){const r=t[s],a=e[s];if(r>=0&&a>=0&&r!==a)throw new Error(`Incompatible shape during merge: ${t} vs. ${e}`);n[s]=r>=0?r:a}return n}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class dT{constructor(e,n,s,r,a,o,i){this.name=e,this.dtype=n,this.maxSize=s,this.elementShape=r,this.identicalElementShapes=a,this.dynamicSize=o,this.clearAfterRead=i,this.tensors=[],this.closed_=!1,this.idTensor=z(0),De(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(n=>{(e==null||!e.has(n.tensor.id))&&n.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const n=this.tensors[e];if(n.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(n.cleared=!0),n.read=!0,n.tensor}readMany(e){return e.map(n=>this.read(n))}write(e,n){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const s=this.tensors[e]||{};if(n.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${n.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=n.shape),ke(this.elementShape,n.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),s.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(s.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);s.tensor=n,De(n),s.written=!0,this.tensors[e]=s}writeMany(e,n){if(e.length!==n.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${n.length}.`);e.forEach((s,r)=>this.write(s,n[r]))}gather(e,n){if(n&&n!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${n}`);if(e)e=e.slice(0,this.size());else{e=[];for(let r=0;r<this.size();r++)e.push(r)}if(e.length===0)return Fe([],[0].concat(this.elementShape));const s=this.readMany(e);return ke(this.elementShape,s[0].shape,"TensorArray shape mismatch: "),We(s,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return Fe([],[0].concat(this.elementShape));const n=[];for(let r=0;r<this.size();r++)n.push(r);const s=this.readMany(n);return ke(this.elementShape,s[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${s[0].shape})`),ue(s,0)}scatter(e,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);if(e.length!==n.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${n.shape[0]}`);const s=Math.max(...e);if(!this.dynamicSize&&s>=this.maxSize)throw new Error(`Max index must be < array size (${s}  vs. ${this.maxSize})`);this.writeMany(e,dt(n,0))}split(e,n){if(n.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${n.dtype}`);let s=0;const r=e.map(u=>(s+=u,s));if(s!==n.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${n.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const a=s===0?0:n.size/s,o=[];j(()=>{n=k(n,[1,s,a]);for(let u=0;u<e.length;++u){const h=[0,u===0?0:r[u-1],0],c=[1,e[u],a];o[u]=k(q(n,h,c),this.elementShape)}return o});const i=[];for(let u=0;u<e.length;u++)i[u]=u;this.writeMany(i,o)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Dt{get id(){return this.idTensor.id}constructor(e,n,s,r=-1){this.tensors=e,this.elementShape=n,this.elementDtype=s,e!=null&&e.forEach(a=>{if(s!==a.dtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${a.dtype}`);ke(n,a.shape,"TensorList shape mismatch: "),De(a)}),this.idTensor=z(0),this.maxNumElements=r,De(this.idTensor)}copy(){return new Dt([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(n=>{(e==null||!e.has(n.id))&&n.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,n,s=-1){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(s!==-1&&this.tensors.length!==s)throw new Error(`Operation expected a list with ${s} elements but got a list with ${this.tensors.length} elements.`);ke(e,this.elementShape,"TensorList shape mismatch: ");const r=an(this.elementShape,this.tensors,e);return j(()=>{const a=this.tensors.map(o=>k(o,r));return We(a,0)})}popBack(e,n){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const s=an(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,ke(r.shape,e,"TensorList shape mismatch: "),k(r,s)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(ke(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");De(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(this.maxNumElements!==-1&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const n=new Dt([],this.elementShape,this.elementDtype,this.maxNumElements);n.tensors.length=e;for(let s=0;s<Math.min(this.tensors.length,e);++s)n.tensors[s]=this.tensors[s];return n}getItem(e,n,s){if(s!==this.elementDtype)throw new Error(`Invalid data types; op elements ${s}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw new Error(`element at index ${e} is null.`);ke(this.tensors[e].shape,n,"TensorList shape mismatch: ");const r=an(this.elementShape,this.tensors,n);return k(this.tensors[e],r)}setItem(e,n){if(n.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);ke(this.elementShape,n.shape,"TensorList shape mismatch: "),De(n),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=n}gather(e,n,s){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);ke(this.elementShape,s,"TensorList shape mismatch: "),e=e.slice(0,this.size());const r=an(this.elementShape,this.tensors,s);return e.length===0?Fe([],[0].concat(r)):j(()=>{const a=e.map(o=>k(this.tensors[o],r));return We(a,0)})}concat(e,n){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);ke(this.elementShape,n,"TensorList shape mismatch: ");const s=an(this.elementShape,this.tensors,n);return this.size()===0?Fe([],[0].concat(s)):j(()=>{const r=this.tensors.map(a=>k(a,s));return ue(r,0)})}}function mT(t,e,n){const s=t.dtype;if(t.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${t.shape}`);if(t.dtype!==n)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${n}`);const r=t.shape.slice(1);ke(r,e,"TensorList shape mismatch: ");const a=dt(t);return new Dt(a,e,s)}function gT(t,e,n,s){return new Dt([],t,e,s)}function yT(t,e,n,s){if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const r=Math.max(...e);if(s!=null&&s!==-1&&r>=s)throw new Error(`Max index must be < array size (${r}  vs. ${s})`);const a=new Dt([],n,t.dtype,s),o=dt(t,0);return e.forEach((i,u)=>{a.setItem(i,o[u])}),a}function bT(t,e,n){let s=0;const r=e.map(h=>(s+=h,s));if(s!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${s}, and tensor's shape is: ${t.shape}`);const a=t.shape.slice(1),o=ur(a,n),i=s===0?0:t.size/s,u=j(()=>{const h=[];t=k(t,[1,s,i]);for(let c=0;c<e.length;++c){const d=[0,c===0?0:r[c-1],0],y=[1,e[c],i];h[c]=k(q(t,d,y),o)}return t.dispose(),h}),l=new Dt([],n,t.dtype,e.length);for(let h=0;h<u.length;h++)l.setItem(h,u[h]);return l}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wT=async(t,e,n)=>{switch(t.op){case"If":case"StatelessIf":{const s=p("thenBranch",t,e,n),r=p("elseBranch",t,e,n),a=p("cond",t,e,n),o=p("args",t,e,n);return(await a.data())[0]?n.functionMap[s].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap):n.functionMap[r].executeFunctionAsync(o,n.tensorArrayMap,n.tensorListMap)}case"While":case"StatelessWhile":{const s=p("body",t,e,n),r=p("cond",t,e,n),a=p("args",t,e,n),o=await n.functionMap[r].executeFunctionAsync(a,n.tensorArrayMap,n.tensorListMap),i=a.map(h=>h.id);let u=await o[0].data();o.forEach(h=>{!h.kept&&i.indexOf(h.id)===-1&&h.dispose()});let l=a;for(;u[0];){const h=l;l=await n.functionMap[s].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);const c=l.map(d=>d.id);h.forEach(d=>{!d.kept&&i.indexOf(d.id)===-1&&c.indexOf(d.id)===-1&&d.dispose()});const f=await n.functionMap[r].executeFunctionAsync(l,n.tensorArrayMap,n.tensorListMap);u=await f[0].data(),f.forEach(d=>{!d.kept&&i.indexOf(d.id)===-1&&c.indexOf(d.id)===-1&&d.dispose()})}return l}case"LoopCond":{const s=p("pred",t,e,n);return[He(s)]}case"Switch":{const s=p("pred",t,e,n);let r=p("data",t,e,n);return r.kept||(r=He(r)),(await s.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const s=t.inputNames.find(r=>oe(r,e,n)!==void 0);if(s){const r=oe(s,e,n);return[He(r)]}return}case"Enter":{const s=p("frameName",t,e,n),r=p("tensor",t,e,n);return n.enterFrame(s),[He(r)]}case"Exit":{const s=p("tensor",t,e,n);return n.exitFrame(),[He(s)]}case"NextIteration":{const s=p("tensor",t,e,n);return n.nextIteration(),[He(s)]}case"TensorArrayV3":{const s=p("size",t,e,n),r=p("dtype",t,e,n),a=p("elementShape",t,e,n),o=p("dynamicSize",t,e,n),i=p("clearAfterRead",t,e,n),u=p("identicalElementShapes",t,e,n),l=p("name",t,e,n),h=new dT(l,r,s,a,u,o,i);return n.addTensorArray(h),[h.idTensor,z(1)]}case"TensorArrayWriteV3":{const s=p("tensorArrayId",t,e,n),r=p("index",t,e,n),a=p("tensor",t,e,n),o=n.getTensorArray(s.id);return o.write(r,a),[o.idTensor]}case"TensorArrayReadV3":{const s=p("tensorArrayId",t,e,n),r=p("index",t,e,n);return[n.getTensorArray(s.id).read(r)]}case"TensorArrayGatherV3":{const s=p("tensorArrayId",t,e,n),r=p("indices",t,e,n),a=p("dtype",t,e,n);return[n.getTensorArray(s.id).gather(r,a)]}case"TensorArrayScatterV3":{const s=p("tensorArrayId",t,e,n),r=p("indices",t,e,n),a=p("tensor",t,e,n),o=n.getTensorArray(s.id);return o.scatter(r,a),[o.idTensor]}case"TensorArrayConcatV3":{const s=p("tensorArrayId",t,e,n),r=n.getTensorArray(s.id),a=p("dtype",t,e,n);return[r.concat(a)]}case"TensorArraySplitV3":{const s=p("tensorArrayId",t,e,n),r=p("tensor",t,e,n),a=p("lengths",t,e,n),o=n.getTensorArray(s.id);return o.split(a,r),[o.idTensor]}case"TensorArraySizeV3":{const s=p("tensorArrayId",t,e,n),r=n.getTensorArray(s.id);return[z(r.size(),"int32")]}case"TensorArrayCloseV3":{const s=p("tensorArrayId",t,e,n),r=n.getTensorArray(s.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const s=p("tensorListId",t,e,n),r=p("index",t,e,n),a=p("tensor",t,e,n),o=n.getTensorList(s.id);return o.setItem(r,a),[o.idTensor]}case"TensorListGetItem":{const s=p("tensorListId",t,e,n),r=p("index",t,e,n),a=p("elementShape",t,e,n),o=p("elementDType",t,e,n);return[n.getTensorList(s.id).getItem(r,a,o)]}case"TensorListScatterV2":case"TensorListScatter":{const s=p("indices",t,e,n),r=p("tensor",t,e,n),a=p("elementShape",t,e,n),o=p("numElements",t,e,n),i=yT(r,s,a,o);return n.addTensorList(i),[i.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const s=p("elementShape",t,e,n),r=p("elementDType",t,e,n);let a;t.op==="TensorListReserve"?a="numElements":a="maxNumElements";const o=p(a,t,e,n),i=t.op==="TensorListReserve"?-1:o,u=gT(s,r,o,i);return n.addTensorList(u),[u.idTensor]}case"TensorListGather":{const s=p("tensorListId",t,e,n),r=p("indices",t,e,n),a=p("elementShape",t,e,n),o=p("elementDType",t,e,n);return[n.getTensorList(s.id).gather(r,o,a)]}case"TensorListStack":{const s=p("tensorListId",t,e,n),r=p("elementShape",t,e,n),a=p("elementDType",t,e,n),o=p("numElements",t,e,n);return[n.getTensorList(s.id).stack(r,a,o)]}case"TensorListFromTensor":{const s=p("tensor",t,e,n),r=p("elementShape",t,e,n),a=p("elementDType",t,e,n),o=mT(s,r,a);return n.addTensorList(o),[o.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const s=p("tensorListId",t,e,n),r=n.getTensorList(s.id),a=p("dtype",t,e,n),o=p("elementShape",t,e,n);return[r.concat(a,o)]}case"TensorListPushBack":{const s=p("tensorListId",t,e,n),r=p("tensor",t,e,n),a=n.getTensorList(s.id);return a.pushBack(r),[a.idTensor]}case"TensorListPopBack":{const s=p("tensorListId",t,e,n),r=p("elementShape",t,e,n),a=p("elementDType",t,e,n);return[n.getTensorList(s.id).popBack(r,a)]}case"TensorListSplit":{const s=p("tensor",t,e,n),r=p("elementShape",t,e,n),a=p("lengths",t,e,n),o=bT(s,a,r);return n.addTensorList(o),[o.idTensor]}case"TensorListLength":{const s=p("tensorListId",t,e,n),r=n.getTensorList(s.id);return[z(r.size(),"int32")]}case"TensorListResize":{const s=p("tensorListId",t,e,n),r=p("size",t,e,n),o=n.getTensorList(s.id).resize(r);return n.addTensorList(o),[o.idTensor]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ro(t,e,n){const[s,r]=p("fusedOps",t,e,n),a=s==="biasadd",o=!a,i=r==="prelu",u=s==="fusedbatchnorm",l=p("numArgs",t,e,n);if(a){if(i&&l!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&a&&l!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(u)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const h=p("strides",t,e,n),c=Un(t,e,n),f=p("dataFormat",t,e,n).toUpperCase(),d=p("dilations",t,e,n);let[y,S]=p("args",t,e,n);o&&(S=y,y=void 0);const b=p("leakyreluAlpha",t,e,n);return{stride:h,pad:c,dataFormat:f,dilations:d,biasArg:y,preluArg:S,activationFunc:r,leakyreluAlpha:b}}const NT=(t,e,n,s=ie)=>{switch(t.op){case"Conv1D":{const r=p("stride",t,e,n),a=p("pad",t,e,n),o=p("dataFormat",t,e,n).toUpperCase(),i=p("dilation",t,e,n);return[s.conv1d(p("x",t,e,n),p("filter",t,e,n),r,a,o,i)]}case"Conv2D":{const r=p("strides",t,e,n),a=Un(t,e,n),o=p("dataFormat",t,e,n).toUpperCase(),i=p("dilations",t,e,n);return[s.conv2d(p("x",t,e,n),p("filter",t,e,n),[r[1],r[2]],a,o,[i[1],i[2]])]}case"_FusedConv2D":{const{stride:r,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:l,activationFunc:h,leakyreluAlpha:c}=ro(t,e,n);return[s.fused.conv2d({x:p("x",t,e,n),filter:p("filter",t,e,n),strides:[r[1],r[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:h,preluActivationWeights:l,leakyreluAlpha:c})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:a,dataFormat:o,dilations:i,biasArg:u,preluArg:l,activationFunc:h,leakyreluAlpha:c}=ro(t,e,n);return[s.fused.depthwiseConv2d({x:p("x",t,e,n),filter:p("filter",t,e,n),strides:[r[1],r[2]],pad:a,dataFormat:o,dilations:[i[1],i[2]],bias:u,activation:h,preluActivationWeights:l,leakyreluAlpha:c})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=p("outputShape",t,e,n),a=p("strides",t,e,n),o=Un(t,e,n);return[s.conv2dTranspose(p("x",t,e,n),p("filter",t,e,n),r,[a[1],a[2]],o)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=p("strides",t,e,n),a=Un(t,e,n),o=p("dilations",t,e,n),i=p("dataFormat",t,e,n).toUpperCase();return[s.depthwiseConv2d(p("input",t,e,n),p("filter",t,e,n),[r[1],r[2]],a,i,[o[1],o[2]])]}case"Conv3D":{const r=p("strides",t,e,n),a=p("pad",t,e,n),o=p("dataFormat",t,e,n).toUpperCase(),i=p("dilations",t,e,n);return[s.conv3d(p("x",t,e,n),p("filter",t,e,n),[r[1],r[2],r[3]],a,o,[i[1],i[2],i[3]])]}case"AvgPool":{const r=p("strides",t,e,n),a=p("pad",t,e,n),o=p("kernelSize",t,e,n);return[s.avgPool(p("x",t,e,n),[o[1],o[2]],[r[1],r[2]],a)]}case"MaxPool":{const r=p("strides",t,e,n),a=p("pad",t,e,n),o=p("kernelSize",t,e,n);return[s.maxPool(p("x",t,e,n),[o[1],o[2]],[r[1],r[2]],a)]}case"MaxPoolWithArgmax":{const r=p("strides",t,e,n),a=p("pad",t,e,n),o=p("kernelSize",t,e,n),i=p("includeBatchInIndex",t,e,n),{result:u,indexes:l}=s.maxPoolWithArgmax(p("x",t,e,n),[o[1],o[2]],[r[1],r[2]],a,i);return[u,l]}case"AvgPool3D":{const r=p("strides",t,e,n),a=p("pad",t,e,n),o=p("kernelSize",t,e,n);return[s.avgPool3d(p("x",t,e,n),[o[1],o[2],o[3]],[r[1],r[2],r[3]],a)]}case"MaxPool3D":{const r=p("strides",t,e,n),a=p("pad",t,e,n),o=p("kernelSize",t,e,n);return[s.maxPool3d(p("x",t,e,n),[o[1],o[2],o[3]],[r[1],r[2],r[3]],a)]}case"Dilation2D":{const r=p("strides",t,e,n),a=p("pad",t,e,n),o=p("dilations",t,e,n),i=r[1],u=r[2],l=o[1],h=o[2];return[s.dilation2d(p("x",t,e,n),p("filter",t,e,n),[i,u],a,[l,h],"NHWC")]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ST=(t,e,n,s=ie)=>{switch(t.op){case"Fill":{const r=p("shape",t,e,n),a=p("dtype",t,e,n),o=p("value",t,e,n);return[s.fill(r,o,a)]}case"LinSpace":{const r=p("start",t,e,n),a=p("stop",t,e,n),o=p("num",t,e,n);return[s.linspace(r,a,o)]}case"Multinomial":{const r=p("logits",t,e,n),a=p("numSamples",t,e,n),o=p("seed",t,e,n);return[s.multinomial(r,a,o)]}case"OneHot":{const r=p("indices",t,e,n),a=p("depth",t,e,n),o=p("onValue",t,e,n),i=p("offValue",t,e,n),u=p("dtype",t,e,n);return[s.oneHot(r,a,o,i,u)]}case"Ones":return[s.ones(p("shape",t,e,n),p("dtype",t,e,n))];case"OnesLike":return[s.onesLike(p("x",t,e,n))];case"RandomStandardNormal":return[s.randomStandardNormal(p("shape",t,e,n),p("dtype",t,e,n),p("seed",t,e,n))];case"RandomUniform":return[s.randomUniform(p("shape",t,e,n),p("minval",t,e,n),p("maxval",t,e,n),p("dtype",t,e,n))];case"RandomUniformInt":return[s.randomUniformInt(p("shape",t,e,n),p("minval",t,e,n),p("maxval",t,e,n),p("seed",t,e,n))];case"Range":{const r=p("start",t,e,n),a=p("stop",t,e,n),o=p("step",t,e,n);return[s.range(r,a,o,p("dtype",t,e,n))]}case"TruncatedNormal":{const r=p("shape",t,e,n),a=p("mean",t,e,n),o=p("stdDev",t,e,n),i=p("seed",t,e,n);return[s.truncatedNormal(r,a,o,p("dtype",t,e,n),i)]}case"Zeros":return[s.zeros(p("shape",t,e,n),p("dtype",t,e,n))];case"ZerosLike":return[s.zerosLike(p("x",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function As(t,e,n){const s=p("boxes",t,e,n),r=p("scores",t,e,n),a=p("maxOutputSize",t,e,n),o=p("iouThreshold",t,e,n),i=p("scoreThreshold",t,e,n),u=p("softNmsSigma",t,e,n);return{boxes:s,scores:r,maxOutputSize:a,iouThreshold:o,scoreThreshold:i,softNmsSigma:u}}const TT=async(t,e,n,s,r=ie)=>{switch(t.op){case"NonMaxSuppressionV5":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:l,softNmsSigma:h}=As(t,e,n),c=await r.image.nonMaxSuppressionWithScoreAsync(a,o,i,u,l,h);return[c.selectedIndices,c.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:l}=As(t,e,n),h=p("padToMaxOutputSize",t,e,n),c=await r.image.nonMaxSuppressionPaddedAsync(a,o,i,u,l,h);return[c.selectedIndices,c.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:a,scores:o,maxOutputSize:i,iouThreshold:u,scoreThreshold:l}=As(t,e,n);return[await r.image.nonMaxSuppressionAsync(a,o,i,u,l)]}case"Where":{const a=r.cast(p("condition",t,e,n),"bool"),o=[await r.whereAsync(a)];return a.dispose(),o}case"ListDiff":return r.setdiff1dAsync(p("x",t,e,n),p("y",t,e,n));default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $T=(t,e,n,s=ie)=>{switch(t.op){case"LowerBound":{const r=p("sortedSequence",t,e,n),a=p("values",t,e,n);return[s.lowerBound(r,a)]}case"TopKV2":{const r=p("x",t,e,n),a=p("k",t,e,n),o=p("sorted",t,e,n),i=s.topk(r,a,o);return[i.values,i.indices]}case"UpperBound":{const r=p("sortedSequence",t,e,n),a=p("values",t,e,n);return[s.upperBound(r,a)]}case"Unique":{const r=p("x",t,e,n),a=s.unique(r);return[a.values,a.indices]}case"UniqueV2":{const r=p("x",t,e,n),a=p("axis",t,e,n),o=s.unique(r,a);return[o.values,o.indices]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ET=(t,e,n,s=ie)=>{switch(t.op){case"Const":return e[t.name];case"PlaceholderWithDefault":const r=p("default",t,e,n);return[oe(t.name,e,n)||r];case"Placeholder":return[oe(t.name,e,n)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const h=p("x",t,e,n);return[He(h)]}case"IdentityN":return p("x",t,e,n).map(h=>He(h));case"Snapshot":const a=p("x",t,e,n);return[He(a)];case"Shape":return[s.tensor1d(p("x",t,e,n).shape,"int32")];case"ShapeN":return p("x",t,e,n).map(h=>s.tensor1d(h.shape));case"Size":return[s.scalar(p("x",t,e,n).size,"int32")];case"Rank":return[s.scalar(p("x",t,e,n).rank,"int32")];case"NoOp":return[s.scalar(1)];case"Print":const o=p("x",t,e,n),i=p("data",t,e,n),u=p("message",t,e,n),l=p("summarize",t,e,n);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(u);for(let h=0;h<i.length;h++)console.log(Array.prototype.slice.call(i[h].dataSync()).slice(0,l));return[o];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class kT{get id(){return this.handle.id}constructor(e,n){this.keyDType=e,this.valueDType=n,this.handle=z(0),this.tensorMap=new Map,De(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return z(this.size(),"int32")}async import(e,n){this.checkKeyAndValueTensor(e,n);const s=await e.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),j(()=>{const r=dt(n),a=s.length,o=r.length;g(a===o,()=>`The number of elements doesn't match, keys has ${a} elements, the values has ${o} elements.`);for(let i=0;i<a;i++){const u=s[i],l=r[i];De(l),this.tensorMap.set(u,l)}return this.handle})}async find(e,n){this.checkKeyAndValueTensor(e,n);const s=await e.data();return j(()=>{const r=[];for(let a=0;a<s.length;a++){const o=s[a],i=this.findWithDefault(o,n);r.push(i)}return We(r)})}findWithDefault(e,n){const s=this.tensorMap.get(e);return s??n}checkKeyAndValueTensor(e,n){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(n.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${n.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vT=async(t,e,n,s)=>{switch(t.op){case"HashTable":case"HashTableV2":{const r=s.getHashTableHandleByName(t.name);if(r!=null)return[r];{const a=p("keyDType",t,e,n),o=p("valueDType",t,e,n),i=new kT(a,o);return s.addHashTable(t.name,i),[i.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const r=p("tableHandle",t,e,n,s),a=p("keys",t,e,n),o=p("values",t,e,n);return[await s.getHashTableById(r.id).import(a,o)]}case"LookupTableFind":case"LookupTableFindV2":{const r=p("tableHandle",t,e,n,s),a=p("keys",t,e,n),o=p("defaultValue",t,e,n);return[await s.getHashTableById(r.id).find(a,o)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=p("tableHandle",t,e,n,s);return[s.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _T=(t,e,n,s=ie)=>{switch(t.op){case"ResizeBilinear":{const r=p("images",t,e,n),a=p("size",t,e,n),o=p("alignCorners",t,e,n),i=p("halfPixelCenters",t,e,n);return[s.image.resizeBilinear(r,[a[0],a[1]],o,i)]}case"ResizeNearestNeighbor":{const r=p("images",t,e,n),a=p("size",t,e,n),o=p("alignCorners",t,e,n),i=p("halfPixelCenters",t,e,n);return[s.image.resizeNearestNeighbor(r,[a[0],a[1]],o,i)]}case"CropAndResize":{const r=p("image",t,e,n),a=p("boxes",t,e,n),o=p("boxInd",t,e,n),i=p("cropSize",t,e,n),u=p("method",t,e,n),l=p("extrapolationValue",t,e,n);return[s.image.cropAndResize(r,a,o,i,u,l)]}case"ImageProjectiveTransformV3":{const r=p("images",t,e,n),a=p("transforms",t,e,n),o=p("outputShape",t,e,n),i=p("fillValue",t,e,n),u=p("interpolation",t,e,n),l=p("fillMode",t,e,n);return[s.image.transform(r,a,u.toLowerCase(),l.toLowerCase(),i,o)]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const IT=(t,e,n,s=ie)=>{switch(t.op){case"Equal":return[s.equal(p("a",t,e,n),p("b",t,e,n))];case"NotEqual":return[s.notEqual(p("a",t,e,n),p("b",t,e,n))];case"Greater":return[s.greater(p("a",t,e,n),p("b",t,e,n))];case"GreaterEqual":return[s.greaterEqual(p("a",t,e,n),p("b",t,e,n))];case"Less":return[s.less(p("a",t,e,n),p("b",t,e,n))];case"LessEqual":return[s.lessEqual(p("a",t,e,n),p("b",t,e,n))];case"LogicalAnd":return[s.logicalAnd(p("a",t,e,n),p("b",t,e,n))];case"LogicalNot":return[s.logicalNot(p("a",t,e,n))];case"LogicalOr":return[s.logicalOr(p("a",t,e,n),p("b",t,e,n))];case"Select":case"SelectV2":return[s.where(p("condition",t,e,n),p("a",t,e,n),p("b",t,e,n))];case"BitwiseAnd":return[s.bitwiseAnd(p("a",t,e,n),p("b",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xT=(t,e,n,s=ie)=>{switch(t.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[s.matMul(p("a",t,e,n),p("b",t,e,n),p("transposeA",t,e,n),p("transposeB",t,e,n))];case"Einsum":return[s.einsum(p("equation",t,e,n),...p("tensors",t,e,n))];case"Transpose":return[s.transpose(p("x",t,e,n),p("perm",t,e,n))];case"_FusedMatMul":const[r,a]=p("fusedOps",t,e,n),o=r==="biasadd",i=a==="prelu",u=p("numArgs",t,e,n),l=p("leakyreluAlpha",t,e,n);if(o){if(i&&u!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!i&&u!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[h,c]=p("args",t,e,n);return[s.fused.matMul({a:p("a",t,e,n),b:p("b",t,e,n),transposeA:p("transposeA",t,e,n),transposeB:p("transposeB",t,e,n),bias:h,activation:a,preluActivationWeights:c,leakyreluAlpha:l})];case"MatrixBandPart":return[s.linalg.bandPart(p("a",t,e,n),p("numLower",t,e,n),p("numUpper",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const AT=(t,e,n,s=ie)=>{switch(t.op){case"EuclideanNorm":return[s.euclideanNorm(p("x",t,e,n),p("axis",t,e,n),p("keepDims",t,e,n))];case"FusedBatchNorm":case"FusedBatchNormV2":return[s.batchNorm(p("x",t,e,n),p("mean",t,e,n),p("variance",t,e,n),p("offset",t,e,n),p("scale",t,e,n),p("epsilon",t,e,n))];case"FusedBatchNormV3":return[s.batchNorm(p("x",t,e,n),p("mean",t,e,n),p("variance",t,e,n),p("offset",t,e,n),p("scale",t,e,n),p("epsilon",t,e,n))];case"LRN":return[s.localResponseNormalization(p("x",t,e,n),p("radius",t,e,n),p("bias",t,e,n),p("alpha",t,e,n),p("beta",t,e,n))];case"Softmax":return[s.softmax(p("x",t,e,n))];case"LogSoftmax":return[s.logSoftmax(p("x",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const OT=(t,e,n,s=ie)=>{switch(t.op){case"RaggedGather":{const{outputNestedSplits:r,outputDenseValues:a}=s.raggedGather(p("paramsNestedSplits",t,e,n),p("paramsDenseValues",t,e,n),p("indices",t,e,n),p("outputRaggedRank",t,e,n));return r.concat(a)}case"RaggedRange":{const{rtNestedSplits:r,rtDenseValues:a}=s.raggedRange(p("starts",t,e,n),p("limits",t,e,n),p("splits",t,e,n));return[r,a]}case"RaggedTensorToTensor":return[s.raggedTensorToTensor(p("shape",t,e,n),p("values",t,e,n),p("defaultValue",t,e,n),p("rowPartitionTensors",t,e,n),p("rowPartitionTypes",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const DT=(t,e,n,s=ie)=>{switch(t.op){case"Max":{const i=p("axis",t,e,n),u=p("keepDims",t,e,n);return[s.max(p("x",t,e,n),i,u)]}case"Mean":{const i=p("axis",t,e,n),u=p("keepDims",t,e,n);return[s.mean(p("x",t,e,n),i,u)]}case"Min":{const i=p("axis",t,e,n),u=p("keepDims",t,e,n);return[s.min(p("x",t,e,n),i,u)]}case"Sum":{const i=p("axis",t,e,n),u=p("keepDims",t,e,n);return[s.sum(p("x",t,e,n),i,u)]}case"All":{const i=p("axis",t,e,n),u=p("keepDims",t,e,n);return[s.all(p("x",t,e,n),i,u)]}case"Any":{const i=p("axis",t,e,n),u=p("keepDims",t,e,n);return[s.any(p("x",t,e,n),i,u)]}case"ArgMax":{const i=p("axis",t,e,n);return[s.argMax(p("x",t,e,n),i)]}case"ArgMin":{const i=p("axis",t,e,n);return[s.argMin(p("x",t,e,n),i)]}case"Prod":{const i=p("axis",t,e,n),u=p("keepDims",t,e,n);return[s.prod(p("x",t,e,n),i,u)]}case"Cumprod":{const i=p("axis",t,e,n),u=p("exclusive",t,e,n),l=p("reverse",t,e,n);return[s.cumprod(p("x",t,e,n),i,u,l)]}case"Cumsum":{const i=p("axis",t,e,n),u=p("exclusive",t,e,n),l=p("reverse",t,e,n);return[s.cumsum(p("x",t,e,n),i,u,l)]}case"Bincount":const r=p("x",t,e,n),a=p("weights",t,e,n),o=p("size",t,e,n);return[s.bincount(r,a,o)];case"DenseBincount":{const i=p("x",t,e,n),u=p("weights",t,e,n),l=p("size",t,e,n),h=p("binaryOutput",t,e,n);return[s.denseBincount(i,u,l,h)]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const FT=(t,e,n,s=ie)=>{switch(t.op){case"ConcatV2":case"Concat":{const r=p("n",t,e,n),a=p("axis",t,e,n);let o=p("tensors",t,e,n);return o=o.slice(0,r),[s.concat(o,a)]}case"Gather":{const r=p("x",t,e,n),a=p("indices",t,e,n);return[s.gather(r,s.cast(a,"int32"),0)]}case"GatherV2":{const r=p("axis",t,e,n),a=p("batchDims",t,e,n),o=p("x",t,e,n),i=p("indices",t,e,n);return[s.gather(o,s.cast(i,"int32"),r,a)]}case"Reverse":{const r=p("dims",t,e,n),a=[];for(let i=0;i<r.length;i++)r[i]&&a.push(i);const o=p("x",t,e,n);return[s.reverse(o,a)]}case"ReverseV2":{const r=p("axis",t,e,n),a=p("x",t,e,n);return[s.reverse(a,r)]}case"Slice":{const r=p("begin",t,e,n),a=p("size",t,e,n);return[s.slice(p("x",t,e,n),r,a)]}case"StridedSlice":{const r=p("begin",t,e,n),a=p("end",t,e,n),o=p("strides",t,e,n),i=p("beginMask",t,e,n),u=p("endMask",t,e,n),l=p("ellipsisMask",t,e,n),h=p("newAxisMask",t,e,n),c=p("shrinkAxisMask",t,e,n),f=p("x",t,e,n);return[s.stridedSlice(f,r,a,o,i,u,l,h,c)]}case"Pack":return j(()=>{const r=p("axis",t,e,n),a=p("tensors",t,e,n),o=a[0].shape,i=s.squeeze(a[0]).shape,u=a.map(l=>{const h=Re(l.shape,o);if(!h&&!Re(s.squeeze(l).shape,i))throw new Error("the input tensors shape does not match");return h?l:s.reshape(l,o)});return[s.stack(u,r)]});case"Unpack":{const r=p("axis",t,e,n),a=p("tensor",t,e,n);return s.unstack(a,r)}case"Tile":{const r=p("reps",t,e,n);return[s.tile(p("x",t,e,n),r)]}case"Split":case"SplitV":{const r=p("axis",t,e,n),a=p("numOrSizeSplits",t,e,n),o=p("x",t,e,n);return s.split(o,a,r)}case"ScatterNd":{const r=p("indices",t,e,n),a=p("values",t,e,n),o=p("shape",t,e,n);return[s.scatterND(r,a,o)]}case"GatherNd":{const r=p("x",t,e,n),a=p("indices",t,e,n);return[s.gatherND(r,a)]}case"SparseToDense":{const r=p("sparseIndices",t,e,n),a=p("outputShape",t,e,n),o=p("sparseValues",t,e,n),i=p("defaultValue",t,e,n);return[s.sparseToDense(r,o,a,o.dtype===i.dtype?i:s.cast(i,o.dtype))]}case"TensorScatterUpdate":{const r=p("indices",t,e,n),a=p("values",t,e,n),o=p("tensor",t,e,n);return[s.tensorScatterUpdate(o,r,a)]}default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const CT=(t,e,n,s=ie)=>{switch(t.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:a,emptyRowIndicator:o,reverseIndexMap:i}=s.sparse.sparseFillEmptyRows(p("indices",t,e,n),p("values",t,e,n),p("denseShape",t,e,n),p("defaultValue",t,e,n));return[r,a,o,i]}case"SparseReshape":{const{outputIndices:r,outputShape:a}=s.sparse.sparseReshape(p("inputIndices",t,e,n),p("inputShape",t,e,n),p("newShape",t,e,n));return[r,a]}case"SparseSegmentMean":return[s.sparse.sparseSegmentMean(p("data",t,e,n),p("indices",t,e,n),p("segmentIds",t,e,n))];case"SparseSegmentSum":return[s.sparse.sparseSegmentSum(p("data",t,e,n),p("indices",t,e,n),p("segmentIds",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const RT=(t,e,n,s=ie)=>{switch(t.op){case"FFT":return[s.fft(p("x",t,e,n))];case"IFFT":return[s.ifft(p("x",t,e,n))];case"RFFT":return[s.rfft(p("x",t,e,n))];case"IRFFT":return[s.irfft(p("x",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const PT=(t,e,n,s=ie)=>{switch(t.op){case"StaticRegexReplace":return[s.string.staticRegexReplace(p("input",t,e,n),p("pattern",t,e,n),p("rewrite",t,e,n),p("replaceGlobal",t,e,n))];case"StringNGrams":{const{nGrams:r,nGramsSplits:a}=s.string.stringNGrams(p("data",t,e,n),p("dataSplits",t,e,n),p("separator",t,e,n),p("nGramWidths",t,e,n),p("leftPad",t,e,n),p("rightPad",t,e,n),p("padWidth",t,e,n),p("preserveShortSequences",t,e,n));return[r,a]}case"StringSplit":{const{indices:r,values:a,shape:o}=s.string.stringSplit(p("input",t,e,n),p("delimiter",t,e,n),p("skipEmpty",t,e,n));return[r,a,o]}case"StringToHashBucketFast":return[s.string.stringToHashBucketFast(p("input",t,e,n),p("numBuckets",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const BT=(t,e,n,s=ie)=>{switch(t.op){case"Cast":return[s.cast(p("x",t,e,n),p("dtype",t,e,n))];case"ExpandDims":{const r=p("axis",t,e,n);return[s.expandDims(p("x",t,e,n),r)]}case"Squeeze":{const r=p("axis",t,e,n);return[s.squeeze(p("x",t,e,n),r)]}case"Reshape":return[s.reshape(p("x",t,e,n),p("shape",t,e,n))];case"EnsureShape":return[s.ensureShape(p("x",t,e,n),p("shape",t,e,n))];case"MirrorPad":return[s.mirrorPad(p("x",t,e,n),p("padding",t,e,n),p("mode",t,e,n))];case"PadV2":case"Pad":return[s.pad(p("x",t,e,n),p("padding",t,e,n),p("constantValue",t,e,n))];case"SpaceToBatchND":{const r=p("blockShape",t,e,n),a=p("paddings",t,e,n);return[s.spaceToBatchND(p("x",t,e,n),r,a)]}case"BatchToSpaceND":{const r=p("blockShape",t,e,n),a=p("crops",t,e,n);return[s.batchToSpaceND(p("x",t,e,n),r,a)]}case"DepthToSpace":{const r=p("blockSize",t,e,n),a=p("dataFormat",t,e,n).toUpperCase();return[s.depthToSpace(p("x",t,e,n),r,a)]}case"BroadcastTo":return[s.broadcastTo(p("x",t,e,n),p("shape",t,e,n))];case"BroadcastArgs":return[s.broadcastArgs(p("s0",t,e,n),p("s1",t,e,n))];default:throw TypeError(`Node type ${t.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ao(t,e,n,s,r=j){const a=((o,i,u)=>{switch(o.category){case"arithmetic":return r(()=>pT(o,i,u));case"basic_math":return r(()=>fT(o,i,u));case"control":return wT(o,i,u);case"convolution":return r(()=>NT(o,i,u));case"creation":return r(()=>ST(o,i,u));case"dynamic":return TT(o,i,u);case"evaluation":return r(()=>$T(o,i,u));case"image":return r(()=>_T(o,i,u));case"graph":return r(()=>ET(o,i,u));case"logical":return r(()=>IT(o,i,u));case"matrices":return r(()=>xT(o,i,u));case"normalization":return r(()=>AT(o,i,u));case"ragged":return r(()=>OT(o,i,u));case"reduction":return r(()=>DT(o,i,u));case"slice_join":return r(()=>FT(o,i,u));case"sparse":return r(()=>CT(o,i,u));case"spectral":return r(()=>RT(o,i,u));case"string":return r(()=>PT(o,i,u));case"transformation":return r(()=>BT(o,i,u));case"hash_table":return vT(o,i,u,s);case"custom":const l=Bp(o.op);if(l&&l.customExecutor)return l.customExecutor(new hT(o,i,u));throw TypeError(`Custom op ${o.op} is not registered.`);default:throw TypeError(`Unknown op '${o.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(t,e,n);return it(a)?a.then(o=>[].concat(o)):[].concat(a)}class oo{constructor(e={},n={},s={},r={},a){this.weightMap=e,this.tensorArrayMap=n,this.tensorListMap=s,this.functionMap=r,this.parseNodeNameCache=a,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,n){return{id:e,frameName:n,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let n=0;n<this.contexts.length-1;n++){const s=this.contexts.slice(0,this.contexts.length-n);e.push(this.contextIdforContexts(s))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(n=>n.id===0&&n.iterationId===0?"":`${n.frameName}-${n.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const n in this.tensorArrayMap)this.tensorArrayMap[n].clearAndClose(e);for(const n in this.tensorListMap)this.tensorListMap[n].clearAndClose(e)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function io(t,e,n,s){const r=new Set,a=[];let o=null,i=null;const u=new Set,l=new Set(Object.keys(t).map(f=>ye(f)[0]));s=s||[];const h=new Set(s.map(f=>ye(f.name)[0])),c=[...e];for(;c.length>0;){const f=c.pop();if((Nt(f)||qT(f)||GT(f))&&o==null&&(o=f,i=o.children.map(d=>d.name).filter(d=>r.has(d))),r.add(f.name),n[f.name]==null&&!l.has(f.name)&&!h.has(f.name)){if(f.inputs.length===0){a.push(f.name);continue}f.inputs.forEach(d=>{u.has(d.name)||(u.add(d.name),c.push(d))})}}return{inputs:t,outputs:e,usedNodes:r,missingInputs:a,dynamicNode:o,syncInputs:i}}function LT(t,e){const{usedNodes:n,inputs:s}=e,r=Object.keys(s).map(b=>ye(b)[0]).map(b=>t.nodes[b]),a=t.initNodes||[],o=b=>n.has(typeof b=="string"?b:b.name);function i(b){return[...new Map(b.map(T=>[T.name,T])).values()]}const u=i([...r,...t.weights,...a]).filter(o),l=i([...u,...Object.values(t.nodes)]).filter(o),h=new Map(l.map(b=>[b.name,b])),c={};for(const b of l){c[b.name]=c[b.name]||0;for(const T of b.children)o(T)||(c[T.name]=Number.POSITIVE_INFINITY),c[T.name]=(c[T.name]||0)+1}const f=Object.entries(c).filter(([,b])=>b===0).map(([b])=>b),d=[...f];for(;f.length>0;){const b=f.pop(),T=h.get(b);for(const A of T.children.filter(o))--c[A.name]===0&&(d.push(A.name),f.push(A.name))}const y=d.map(b=>h.get(b)),S=zT(y,u);return VT(S,u),S}function zT(t,e){const n=new Map(t.map(o=>[o.name,o])),s=e.map(o=>o.name),r=new Set(s);for(;s.length>0;){const o=s.pop(),i=n.get(o);for(const u of i.children)!n.has(u.name)||r.has(u.name)||(r.add(u.name),s.push(u.name))}return t.filter(o=>r.has(o.name))}class Vn extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}function VT(t,e){const n=new Map(t.map((i,u)=>[i.name,u])),s=new Set(e.map(i=>i.name)),r=i=>s.has(typeof i=="string"?i:i.name),a=new Set(t.map(i=>i.name)),o=i=>a.has(typeof i=="string"?i:i.name);for(const i of t){for(const u of i.children.filter(o)){if(!n.has(u.name))throw new Vn(`Child ${u.name} of node ${i.name} is unreachable.`);if(n.get(i.name)>n.get(u.name))throw new Vn(`Node ${i.name} is scheduled to run after its child ${u.name}.`)}if(!r(i))for(const u of i.inputs){if(!n.has(u.name))throw new Vn(`Input ${u.name} of node ${i.name} is unreachable.`);if(n.get(u.name)>n.get(i.name))throw new Vn(`Node ${i.name} is scheduled to run before its input ${u.name}.`)}}}function jT(t){const e=new Map(t.map((i,u)=>[i.name,u])),n=Number.MAX_SAFE_INTEGER,s=t.map((i,u)=>Nt(i)?n:u),r=i=>{const u=s[e.get(i.name)];return u??-1},a=t.map((i,u)=>i.children.map(r).reduce((l,h)=>Math.max(l,h),s[u])),o=new Map;for(let i=0;i<t.length;++i){const u=a[i];if(u===n)continue;const l=t[i],h=t[u];o.has(h.name)||o.set(h.name,[]),o.get(h.name).push(l)}return o}const MT=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),WT=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),UT=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function Nt(t){return MT.has(t.op)}function qT(t){return WT.has(t.op)}function GT(t){return UT.has(t.op)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class as{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const n=Object.keys(e).map(s=>e[s].map(r=>r.id));this._weightIds=[].concat(...n),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const n=e.signatureKey||e.name;return e.defaultOutput?`${n}:${e.defaultOutput}`:n})}get functions(){return Object.keys(this._functions).reduce((e,n)=>(e[n]=this._functions[n].signature,e),{})}constructor(e,n){this.graph=e,this.parent=n,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,e.functions!=null&&Object.keys(e.functions).forEach(s=>{this._functionExecutorMap[s]=new as(e.functions[s],this)})}getCompilationKey(e,n){const s=e.map(a=>a.name).sort(),r=n.map(a=>a.name).sort();return s.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(e,n){const s=io(e,n,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:a,syncInputs:o}=s;if(a!=null)throw new Error(`This execution contains the node '${a.name}', which has the dynamic op '${a.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${o}]`);if(r.length>0){const l=n.map(c=>c.name),h=Object.keys(e);throw new Error(`Cannot compute the outputs [${l}] from the provided inputs [${h}]. Missing the following inputs: [${r}]`)}const i=LT(this.graph,s),u=jT(i);return{orderedNodes:i,nodeLiveUntilMap:u}}cloneAndKeepTensor(e){if(e==null)return null;const n=e.clone();return De(n),n}cloneTensorList(e){return e?e.map(s=>this.cloneAndKeepTensor(s)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([n,s])=>[n,this.cloneTensorList(s)]))}execute(e,n){this.disposeIntermediateTensors(),e=this.mapInputs(e);const s=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),n=this.mapOutputs(n),this.checkOutputs(n);const r=s.map(f=>this.graph.nodes[ye(f)[0]]),a=n.map(f=>ye(f)[0]),o=new Set(a);let i=a.map(f=>this.graph.nodes[f]);i.length===0&&(i=this._outputs);const u=this.getCompilationKey(r,i);let l=this.compiledMap.get(u);l==null&&(l=this.compile(e,i),this.compiledMap.set(u,l));try{this.keepIntermediateTensors=R().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(f){this.keepIntermediateTensors=!1,console.warn(f.message)}const h={},c={};return j(()=>{const f=new oo(this.weightMap,h,c,this.functionExecutorMap,this.parseNodeNameCache),d=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(T=>{const[A,$]=ye(T,f),E=[];E[$]=e[T],d[A]=E,this.keepIntermediateTensors&&(this.clonedTensorsMap[A]=this.cloneTensorList(E))});const y=this.getFrozenTensorIds(d),{orderedNodes:S,nodeLiveUntilMap:b}=l;for(const T of S){if(d[T.name])continue;const A=ao(T,d,f,this._resourceManager);if(it(A))throw new Error(`The execution of the op '${T.op}' returned a promise. Please use model.executeAsync() instead.`);d[T.name]=A,this.keepIntermediateTensors&&(this.clonedTensorsMap[T.name]=this.cloneTensorList(A)),this.checkTensorForDisposalWithNodeLiveUntilInfo(T,d,f,y,o,b.get(T.name))}return this.parent==null&&f.dispose(y),n.map(T=>oe(T,d,f))})}getFrozenTensorIds(e){const n=[].concat.apply([],Object.keys(e).map(s=>e[s]).map(s=>s.map(r=>r.id)));return new Set(n)}checkTensorForDisposal(e,n,s,r,a,o,i){if(!(Nt(n)||o.has(e))){for(const u of s[e])u!=null&&(i[u.id]=(i[u.id]||0)+n.children.length);for(const u of n.inputs){if(Nt(u))continue;const l=eo(u.name,s,r);if(l!=null)for(const h of l){if(!h||h.kept||a.has(h.id))continue;const c=i[h.id];c===1?(h.dispose(),delete i[h.id]):c!=null&&i[h.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,n,s,r,a,o){function i(u){return Nt(u)||a.has(u.name)}if(!(Nt(e)||o==null))for(const u of o){if(i(u))continue;const l=eo(u.name,n,s);for(const h of l)!h||h.kept||r.has(h.id)||h.dispose()}}async executeAsync(e,n){return this._executeAsync(e,n)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const n of e)n&&!n.isDisposed&&n.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,n,s=!1,r={},a={}){this.disposeIntermediateTensors(),s||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),n=this.mapOutputs(n),this.checkOutputs(n));try{this.keepIntermediateTensors=R().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(f){this.keepIntermediateTensors=!1,console.warn(f.message)}const o=new oo(this.weightMap,r,a,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const i=await this.executeWithControlFlow(e,o,n,s),u=n.map(f=>oe(f,i,o)),l=u.map(f=>f.id),h=Object.keys(e).map(f=>e[f].id),c=new Set([...l,...h,...this.weightIds]);return Object.values(i).forEach(f=>{f.forEach(d=>{d&&!d.isDisposed&&!c.has(d.id)&&d.dispose()})}),this.parent==null&&o.dispose(c),u}async executeFunctionAsync(e,n,s){const r=e.reduce((a,o,i)=>(a[this.inputs[i].name]=o,a),{});return this._executeAsync(r,this.outputNodes,!0,n,s)}async executeWithControlFlow(e,n,s,r){const a=Object.keys(e),o=a.map(E=>this.graph.nodes[ye(E)[0]]),i=s.map(E=>ye(E)[0]),u=new Set(i);let l=i.map(E=>this.graph.nodes[E]);l.length===0&&(l=this._outputs);const{usedNodes:h,missingInputs:c,dynamicNode:f,syncInputs:d}=io(e,l,this.weightMap,this._initNodes),y=[...o,...this.graph.weights,...this._initNodes||[]].map(E=>({node:E,contexts:n.currentContext})),S=Object.assign({},this.weightMap);Object.keys(e).forEach(E=>{const[v,_]=ye(E),O=[];O[_]=e[E],S[v]=O});const b={},T=this.getFrozenTensorIds(S),A={};for(;y.length>0;){const E=this.processStack(o,y,n,S,A,T,u,b,h);await Promise.all(E)}f==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const $=l.filter(E=>!Nt(E)&&!oe(E.name,S,n)).map(E=>E.name);if($.length>0){let E="";throw f!=null&&(E=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${d}]`),new Error(`Cannot compute the outputs [${$}] from the provided inputs [${a}]. Consider providing the following inputs: [${c}]. ${E}`)}return S}processStack(e,n,s,r,a,o,i,u,l){const h=[];for(;n.length>0;){const c=n.pop();s.currentContext=c.contexts;let f="";if(c.node.op==="Enter"&&p("isConstant",c.node,r,s)&&([f]=Ge(c.node.name,s)),r[c.node.name]==null){const d=ao(c.node,r,s,this._resourceManager);f||([f]=Ge(c.node.name,s));const y=s.currentContext;it(d)?h.push(d.then(S=>(r[f]=S,this.keepIntermediateTensors&&(this.clonedTensorsMap[f]=this.cloneTensorList(S)),s.currentContext=y,this.checkTensorForDisposal(f,c.node,r,s,o,i,u),this.processChildNodes(c.node,n,s,r,a,l),S))):(r[f]=d,this.keepIntermediateTensors&&(this.clonedTensorsMap[f]=this.cloneTensorList(d)),this.checkTensorForDisposal(f,c.node,r,s,o,i,u),this.processChildNodes(c.node,n,s,r,a,l))}else this.processChildNodes(c.node,n,s,r,a,l)}return h}processChildNodes(e,n,s,r,a,o){e.children.forEach(i=>{const[u]=Ge(i.name,s);a[u]||!o.has(i.name)||(i.op==="Merge"?i.inputNames.some(l=>!!oe(l,r,s))&&(a[u]=!0,n.push({contexts:s.currentContext,node:i})):i.inputNames.every(l=>!!oe(l,r,s))&&(a[u]=!0,n.push({contexts:s.currentContext,node:i})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(n=>n.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(n=>{const s=e[n],[r]=ye(n),a=this.graph.nodes[r];if(a.attrParams.shape&&a.attrParams.shape.value){const o=a.attrParams.shape.value,i=o.length===s.shape.length&&s.shape.every((u,l)=>o[l]===-1||o[l]===u);g(i,()=>`The shape of dict['${a.name}'] provided in model.execute(dict) must be [${o}], but was [${s.shape}]`)}a.attrParams.dtype&&a.attrParams.dtype.value&&g(s.dtype===a.attrParams.dtype.value,()=>`The dtype of dict['${a.name}'] provided in model.execute(dict) must be ${a.attrParams.dtype.value}, but was ${s.dtype}`)})}mapInputs(e){var n,s;const r={};for(const a in e){const o=(s=(n=this._signature)===null||n===void 0?void 0:n.inputs)===null||s===void 0?void 0:s[a];o!=null?r[o.name]=e[a]:r[a]=e[a]}return r}checkInputs(e){const n=Object.keys(e).filter(s=>{const[r]=ye(s);return this.graph.nodes[r]==null});if(n.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${n}] that are not part of graph`)}mapOutputs(e){return e.map(n=>{var s,r;const a=(r=(s=this._signature)===null||s===void 0?void 0:s.outputs)===null||r===void 0?void 0:r[n];return a!=null?a.name:n},{})}checkOutputs(e){e.forEach(n=>{const[s]=ye(n);if(!this.graph.nodes[s])throw new Error(`The output '${n}' is not found in the graph`)})}}class HT{constructor(e={},n={}){this.hashTableNameToHandle=e,this.hashTableMap=n}addHashTable(e,n){this.hashTableNameToHandle[e]=n.handle,this.hashTableMap[n.id]=n}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const KT="?tfjs-format=file",XT="model.json";class Oa{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,n={},s=_a){this.modelUrl=e,this.loadOptions=n,this.version="n/a",this.io=s,n==null&&(this.loadOptions={}),this.resourceManager=new HT}findIOHandler(){const e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const n=this.io.getLoadHandlers(e,this.loadOptions);if(n.length===0)n.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(n.length>1)throw new Error(`Found more than one (${n.length}) load handlers for URL '${[e]}'`);this.handler=n[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return it(e)?e.then(n=>n.getWeightStream==null?this.loadSync(n):this.loadStreaming(n)):this.loadSync(e)}loadSync(e){const n=this.io.decodeWeights(e.weightData,e.weightSpecs);return this.loadWithWeightMap(e,n)}async loadStreaming(e){if(e.getWeightStream==null)throw new Error("Model artifacts missing streamWeights function");const n=await Cl(e.getWeightStream(),e.weightSpecs);return this.loadWithWeightMap(e,n)}loadWithWeightMap(e,n){this.artifacts=e;const s=this.artifacts.modelTopology;let r=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const a=this.artifacts.userDefinedMetadata;a.signature!=null&&(r=a.signature),a.structuredOutputKeys!=null&&(this.structuredOutputKeys=a.structuredOutputKeys)}if(this.signature=r,this.version=`${s.versions.producer}.${s.versions.minConsumer}`,this.executor=new as(to.Instance.transformGraph(s,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(n),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){const a=to.Instance.transformGraph(e.modelInitializer);this.initializer=new as(a),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,n){if(typeof e=="string"){const s=this.io.getSaveHandlers(e);if(s.length===0)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(s.length>1)throw new Error(`Found more than one (${s.length}) save handlers for URL '${e}'`);e=s[0]}if(e.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const n=e instanceof te?[e]:e,s={};return n.forEach((r,a)=>s[this.structuredOutputKeys[a]]=r),s}return e}predict(e,n){const s=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(s)}async predictAsync(e,n){const s=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(s)}normalizeInputs(e){var n;if(!(e instanceof te)&&!Array.isArray(e)){const a=(n=this.signature)===null||n===void 0?void 0:n.inputs;if(a!=null)for(const o in a){const i=a[o];i.resourceId!=null&&(e[o]=this.resourceIdToCapturedInput[i.resourceId])}return e}e=Array.isArray(e)?e:[e];const s=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+s!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-s} non-resource placeholders, while there are ${e.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((a,o)=>{var i,u,l;const h=(l=(u=(i=this.signature)===null||i===void 0?void 0:i.inputs)===null||u===void 0?void 0:u[o])===null||l===void 0?void 0:l.resourceId;return h!=null?a[o]=this.resourceIdToCapturedInput[h]:a[o]=e[r++],a},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const n=this.initializerSignature.outputs,s=Object.keys(n);for(let r=0;r<s.length;r++){const a=s[r],o=n[a];this.resourceIdToCapturedInput[o.resourceId]=e[r]}}}execute(e,n){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),n=this.normalizeOutputs(n);const s=this.executor.execute(e,n);return s.length>1?s:s[0]}async executeAsync(e,n){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),n=this.normalizeOutputs(n);const s=await this.executor.executeAsync(e,n);return s.length>1?s:s[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((n,s)=>(n[s]=[e[s]],n),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&pe(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function ZT(t,e={},n=_a){if(t==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");e==null&&(e={}),e.fromTFHub&&typeof t=="string"&&(t=YT(t));const s=new Oa(t,e,n);return await s.load(),s}function JT(t){if(t==null)throw new Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let e;if(t instanceof Array){const[s,r]=t;if(!s)throw new Error("modelJSON must be the first element of the array");if(!r||!(r instanceof ArrayBuffer))throw new Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in s))throw new Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in s))throw new Error("Model JSON is missing 'weightsManifest'");const a=Yn(s.weightsManifest),o=kr(s,a,r);e=ss(o)}else if("load"in t)e=t;else if("modelTopology"in t&&"weightSpecs"in t&&"weightData"in t)e=ss(t);else throw new Error("Unknown model format");const n=new Oa(e);return n.load(),n}function YT(t){return t.endsWith("/")||(t=t+"/"),`${t}${XT}${KT}`}/** @license See the LICENSE file. */const QT="4.22.0";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const e$=Object.freeze(Object.defineProperty({__proto__:null,GraphModel:Oa,deregisterOp:kS,loadGraphModel:ZT,loadGraphModelSync:JT,registerOp:ES,version_converter:QT},Symbol.toStringTag,{value:"Module"}));var lr={exports:{}};const t$=cr(e$),n$=cr(TS);/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(function(t,e){(function(n,s){s(e,t$,n$)})(pt,function(n,s,r){const a={1:{name:"/m/01g317",id:1,displayName:"person"},2:{name:"/m/0199g",id:2,displayName:"bicycle"},3:{name:"/m/0k4j",id:3,displayName:"car"},4:{name:"/m/04_sv",id:4,displayName:"motorcycle"},5:{name:"/m/05czz6l",id:5,displayName:"airplane"},6:{name:"/m/01bjv",id:6,displayName:"bus"},7:{name:"/m/07jdr",id:7,displayName:"train"},8:{name:"/m/07r04",id:8,displayName:"truck"},9:{name:"/m/019jd",id:9,displayName:"boat"},10:{name:"/m/015qff",id:10,displayName:"traffic light"},11:{name:"/m/01pns0",id:11,displayName:"fire hydrant"},13:{name:"/m/02pv19",id:13,displayName:"stop sign"},14:{name:"/m/015qbp",id:14,displayName:"parking meter"},15:{name:"/m/0cvnqh",id:15,displayName:"bench"},16:{name:"/m/015p6",id:16,displayName:"bird"},17:{name:"/m/01yrx",id:17,displayName:"cat"},18:{name:"/m/0bt9lr",id:18,displayName:"dog"},19:{name:"/m/03k3r",id:19,displayName:"horse"},20:{name:"/m/07bgp",id:20,displayName:"sheep"},21:{name:"/m/01xq0k1",id:21,displayName:"cow"},22:{name:"/m/0bwd_0j",id:22,displayName:"elephant"},23:{name:"/m/01dws",id:23,displayName:"bear"},24:{name:"/m/0898b",id:24,displayName:"zebra"},25:{name:"/m/03bk1",id:25,displayName:"giraffe"},27:{name:"/m/01940j",id:27,displayName:"backpack"},28:{name:"/m/0hnnb",id:28,displayName:"umbrella"},31:{name:"/m/080hkjn",id:31,displayName:"handbag"},32:{name:"/m/01rkbr",id:32,displayName:"tie"},33:{name:"/m/01s55n",id:33,displayName:"suitcase"},34:{name:"/m/02wmf",id:34,displayName:"frisbee"},35:{name:"/m/071p9",id:35,displayName:"skis"},36:{name:"/m/06__v",id:36,displayName:"snowboard"},37:{name:"/m/018xm",id:37,displayName:"sports ball"},38:{name:"/m/02zt3",id:38,displayName:"kite"},39:{name:"/m/03g8mr",id:39,displayName:"baseball bat"},40:{name:"/m/03grzl",id:40,displayName:"baseball glove"},41:{name:"/m/06_fw",id:41,displayName:"skateboard"},42:{name:"/m/019w40",id:42,displayName:"surfboard"},43:{name:"/m/0dv9c",id:43,displayName:"tennis racket"},44:{name:"/m/04dr76w",id:44,displayName:"bottle"},46:{name:"/m/09tvcd",id:46,displayName:"wine glass"},47:{name:"/m/08gqpm",id:47,displayName:"cup"},48:{name:"/m/0dt3t",id:48,displayName:"fork"},49:{name:"/m/04ctx",id:49,displayName:"knife"},50:{name:"/m/0cmx8",id:50,displayName:"spoon"},51:{name:"/m/04kkgm",id:51,displayName:"bowl"},52:{name:"/m/09qck",id:52,displayName:"banana"},53:{name:"/m/014j1m",id:53,displayName:"apple"},54:{name:"/m/0l515",id:54,displayName:"sandwich"},55:{name:"/m/0cyhj_",id:55,displayName:"orange"},56:{name:"/m/0hkxq",id:56,displayName:"broccoli"},57:{name:"/m/0fj52s",id:57,displayName:"carrot"},58:{name:"/m/01b9xk",id:58,displayName:"hot dog"},59:{name:"/m/0663v",id:59,displayName:"pizza"},60:{name:"/m/0jy4k",id:60,displayName:"donut"},61:{name:"/m/0fszt",id:61,displayName:"cake"},62:{name:"/m/01mzpv",id:62,displayName:"chair"},63:{name:"/m/02crq1",id:63,displayName:"couch"},64:{name:"/m/03fp41",id:64,displayName:"potted plant"},65:{name:"/m/03ssj5",id:65,displayName:"bed"},67:{name:"/m/04bcr3",id:67,displayName:"dining table"},70:{name:"/m/09g1w",id:70,displayName:"toilet"},72:{name:"/m/07c52",id:72,displayName:"tv"},73:{name:"/m/01c648",id:73,displayName:"laptop"},74:{name:"/m/020lf",id:74,displayName:"mouse"},75:{name:"/m/0qjjc",id:75,displayName:"remote"},76:{name:"/m/01m2v",id:76,displayName:"keyboard"},77:{name:"/m/050k8",id:77,displayName:"cell phone"},78:{name:"/m/0fx9l",id:78,displayName:"microwave"},79:{name:"/m/029bxz",id:79,displayName:"oven"},80:{name:"/m/01k6s3",id:80,displayName:"toaster"},81:{name:"/m/0130jx",id:81,displayName:"sink"},82:{name:"/m/040b_t",id:82,displayName:"refrigerator"},84:{name:"/m/0bt_c3",id:84,displayName:"book"},85:{name:"/m/01x3z",id:85,displayName:"clock"},86:{name:"/m/02s195",id:86,displayName:"vase"},87:{name:"/m/01lsmm",id:87,displayName:"scissors"},88:{name:"/m/0kmg4",id:88,displayName:"teddy bear"},89:{name:"/m/03wvsk",id:89,displayName:"hair drier"},90:{name:"/m/012xff",id:90,displayName:"toothbrush"}};class o{constructor(u,l){this.modelPath=l||`https://storage.googleapis.com/tfjs-models/savedmodel/${this.getPrefix(u)}/model.json`}getPrefix(u){return u==="lite_mobilenet_v2"?`ssd${u}`:`ssd_${u}`}async load(){this.model=await s.loadGraphModel(this.modelPath);const u=r.zeros([1,300,300,3],"int32"),l=await this.model.executeAsync(u);await Promise.all(l.map(h=>h.data())),l.map(h=>h.dispose()),u.dispose()}async infer(u,l,h){const c=r.tidy(()=>(u instanceof r.Tensor||(u=r.browser.fromPixels(u)),r.expandDims(u))),f=c.shape[1],d=c.shape[2],y=await this.model.executeAsync(c),S=y[0].dataSync(),b=y[1].dataSync();c.dispose(),r.dispose(y);const[T,A]=this.calculateMaxScores(S,y[0].shape[1],y[0].shape[2]),$=r.getBackend();r.getBackend()==="webgl"&&r.setBackend("cpu");const E=r.tidy(()=>{const _=r.tensor2d(b,[y[1].shape[1],y[1].shape[3]]);return r.image.nonMaxSuppression(_,T,l,h,h)}),v=E.dataSync();return E.dispose(),$!==r.getBackend()&&r.setBackend($),this.buildDetectedObjects(d,f,b,T,v,A)}buildDetectedObjects(u,l,h,c,f,d){const y=f.length,S=[];for(let b=0;b<y;b++){const T=[];for(let _=0;_<4;_++)T[_]=h[4*f[b]+_];const A=T[0]*l,$=T[1]*u,E=T[2]*l,v=T[3]*u;T[0]=$,T[1]=A,T[2]=v-$,T[3]=E-A,S.push({bbox:T,class:a[d[f[b]]+1].displayName,score:c[f[b]]})}return S}calculateMaxScores(u,l,h){const c=[],f=[];for(let d=0;d<l;d++){let y=Number.MIN_VALUE,S=-1;for(let b=0;b<h;b++)u[d*h+b]>y&&(y=u[d*h+b],S=b);c[d]=y,f[d]=S}return[c,f]}async detect(u,l=20,h=.5){return this.infer(u,l,h)}dispose(){this.model!=null&&this.model.dispose()}}n.ObjectDetection=o,n.load=async function(i={}){if(r==null)throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please also include @tensorflow/tfjs on the page before using this model.");const u=i.base||"lite_mobilenet_v2",l=i.modelUrl;if(["mobilenet_v1","mobilenet_v2","lite_mobilenet_v2"].indexOf(u)===-1)throw new Error(`ObjectDetection constructed with invalid base model ${u}. Valid names are 'mobilenet_v1', 'mobilenet_v2' and 'lite_mobilenet_v2'.`);const h=new o(u,l);return await h.load(),h},n.version="2.2.3",Object.defineProperty(n,"__esModule",{value:!0})})})(lr,lr.exports);var s$=lr.exports;export{Go as $,To as A,C as B,Co as C,Do as D,Fo as E,w as F,m as G,g as H,Ae as I,N as J,mf as K,Po as L,df as M,Ro as N,Bo as O,V as P,Lo as Q,Zr as R,gf as S,gr as T,Mo as U,Wo as V,Ze as W,Nn as X,Vr as Y,cs as Z,qo as _,t$ as a,Ki as a$,kn as a0,Qt as a1,Ho as a2,wn as a3,P0 as a4,bc as a5,Xo as a6,Dn as a7,yf as a8,Zo as a9,vh as aA,Wt as aB,Ei as aC,We as aD,Kh as aE,zg as aF,_i as aG,br as aH,Ai as aI,Oi as aJ,Di as aK,Fi as aL,Rn as aM,Li as aN,Bi as aO,Tf as aP,Ef as aQ,Mi as aR,Fn as aS,Cr as aT,Wi as aU,Ui as aV,ts as aW,_f as aX,Gi as aY,vf as aZ,qi as a_,mg as aa,Yo as ab,Oh as ac,Qo as ad,Dh as ae,ti as af,Lg as ag,kc as ah,En as ai,ai as aj,Ye as ak,V0 as al,M0 as am,li as an,wf as ao,bf as ap,pi as aq,Nf as ar,fi as as,ct as at,mi as au,gi as av,yi as aw,Si as ax,Ti as ay,$i as az,X as b,tl as b$,Pg as b0,W as b1,rt as b2,Xi as b3,Zi as b4,Ji as b5,q as b6,Yi as b7,Lr as b8,eu as b9,Au as bA,qr as bB,Ou as bC,DN as bD,FN as bE,Pu as bF,Ru as bG,Fu as bH,Tc as bI,Cu as bJ,$c as bK,Du as bL,pN as bM,nn as bN,Mu as bO,Bu as bP,Et as bQ,Vu as bR,Or as bS,ju as bT,ue as bU,Lu as bV,Of as bW,Ku as bX,ll as bY,el as bZ,zu as b_,tu as ba,iu as bb,At as bc,ou as bd,uu as be,dt as bf,lu as bg,cu as bh,Zt as bi,Xt as bj,hu as bk,pu as bl,Ec as bm,ci as bn,bu as bo,$u as bp,wu as bq,Nu as br,Tu as bs,Af as bt,Su as bu,xf as bv,Eu as bw,ht as bx,ku as by,vu as bz,pt as c,Xr as c$,nl as c0,wr as c1,jn as c2,ol as c3,il as c4,Kr as c5,zr as c6,qe as c7,ul as c8,Cf as c9,Rr as cA,Cc as cB,Bc as cC,Lc as cD,ds as cE,Ir as cF,$n as cG,ha as cH,zc as cI,Vc as cJ,jc as cK,jr as cL,Wc as cM,qc as cN,Gc as cO,Ur as cP,Mr as cQ,Gr as cR,Hc as cS,Hr as cT,kt as cU,Sn as cV,es as cW,Tn as cX,Yc as cY,Qc as cZ,Cn as c_,Tl as ca,be as cb,Wl as cc,Ul as cd,Gl as ce,Hl as cf,Kl as cg,Xl as ch,Zl as ci,Jl as cj,Yl as ck,Ql as cl,ec as cm,Ar as cn,On as co,cn as cp,hc as cq,pc as cr,te as cs,yc as ct,wc as cu,_c as cv,ls as cw,xc as cx,Oc as cy,Dc as cz,$o as d,Sc as d$,ns as d0,sh as d1,lh as d2,Jr as d3,ch as d4,Sh as d5,Bn as d6,la as d7,Iw as d8,Aw as d9,mc as dA,dc as dB,fc as dC,sp as dD,wp as dE,fs as dF,Gh as dG,Br as dH,dp as dI,bp as dJ,tt as dK,Zh as dL,Re as dM,pe as dN,R as dO,bN as dP,bo as dQ,De as dR,tn as dS,gN as dT,Xe as dU,mt as dV,lo as dW,$d as dX,Gd as dY,Fd as dZ,zd as d_,ms as da,ca as db,_h as dc,Ih as dd,Ah as de,Bh as df,Wr as dg,pa as dh,gs as di,Lh as dj,zh as dk,Qn as dl,qh as dm,Hh as dn,dn as dp,Ol as dq,ua as dr,j as ds,Ph as dt,Rh as du,Ch as dv,Fh as dw,$e as dx,G0 as dy,gc as dz,xe as e,hS as e$,fp as e0,L0 as e1,Nc as e2,eh as e3,oc as e4,ic as e5,uc as e6,Xc as e7,rc as e8,Fe as e9,di as eA,vi as eB,Ci as eC,Ri as eD,qf as eE,nu as eF,Vg as eG,us as eH,TN as eI,$N as eJ,Le as eK,EN as eL,SN as eM,Jn as eN,oN as eO,cN as eP,hN as eQ,wS as eR,nS as eS,sS as eT,rS as eU,aS as eV,oS as eW,iS as eX,uS as eY,lS as eZ,cS as e_,Jt as ea,vc as eb,uo as ec,qp as ed,Td as ee,et as ef,nt as eg,In as eh,VN as ei,rf as ej,Zn as ek,Ve as el,Jh as em,en,po as eo,Ac as ep,uf as eq,of as er,Uo as es,os as et,yu as eu,sf as ev,is as ew,bS as ex,Vo as ey,hr as ez,je as f,Fs as f$,pS as f0,fS as f1,Xu as f2,Uf as f3,qn as f4,Od as f5,nf as f6,Ds as f7,vo as f8,Bg as f9,Dm as fA,$t as fB,af as fC,yr as fD,hi as fE,XN as fF,JN as fG,YN as fH,ZN as fI,QN as fJ,zN as fK,LN as fL,BN as fM,PN as fN,RN as fO,CN as fP,UN as fQ,jN as fR,MN as fS,WN as fT,GN as fU,qN as fV,HN as fW,bi as fX,wi as fY,vn as fZ,Ni as f_,_o as fa,tc as fb,Fm as fc,_N as fd,IN as fe,xN as ff,AN as fg,ON as fh,zo as fi,jo as fj,xi as fk,wN as fl,NN as fm,sc as fn,An as fo,Ko as fp,nc as fq,Jo as fr,ni as fs,ei as ft,pr as fu,si as fv,ri as fw,oi as fx,ii as fy,ui as fz,Mp as g,_d as g$,Cs as g0,ki as g1,vp as g2,gS as g3,Ii as g4,Pi as g5,zi as g6,Vi as g7,ji as g8,Hi as g9,uN as gA,Ju as gB,Yu as gC,Qu as gD,Iu as gE,sl as gF,rl as gG,hn as gH,al as gI,hl as gJ,Qp as gK,Hp as gL,ho as gM,ln as gN,fd as gO,NS as gP,tf as gQ,ut as gR,Kp as gS,Gn as gT,Il as gU,qt as gV,kN as gW,ad as gX,Os as gY,mS as gZ,dS as g_,Qi as ga,ra as gb,su as gc,up as gd,ru as ge,lp as gf,au as gg,cp as gh,fe as gi,fu as gj,du as gk,mu as gl,gu as gm,cl as gn,vN as go,_u as gp,Wh as gq,xu as gr,Wu as gs,Uu as gt,qu as gu,Gu as gv,Hu as gw,tS as gx,Zu as gy,fN as gz,L as h,vd as h0,Al as h1,s$ as h2,z as i,K as j,Eo as k,mr as l,x as m,Ce as n,ne as o,Fr as p,H as q,n$ as r,fa as s,k as t,ko as u,Io as v,xo as w,Ao as x,Oo as y,Ne as z};
