import{c as p,r as $}from"./coco-ssd-BnOSekKy.js";var A={},g={},w={};/**
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
 */Object.defineProperty(w,"__esModule",{value:!0});w.validateTrackerConfig=void 0;function Z(e){if(e.maxTracks<1)throw new Error("Must specify 'maxTracks' to be at least 1, but "+"encountered ".concat(e.maxTracks));if(e.maxAge<=0)throw new Error("Must specify 'maxAge' to be positive, but "+"encountered ".concat(e.maxAge));if(e.keypointTrackerParams!==void 0){if(e.keypointTrackerParams.keypointConfidenceThreshold<0||e.keypointTrackerParams.keypointConfidenceThreshold>1)throw new Error("Must specify 'keypointConfidenceThreshold' to be in the range [0, 1], but encountered "+"".concat(e.keypointTrackerParams.keypointConfidenceThreshold));if(e.keypointTrackerParams.minNumberOfKeypoints<1)throw new Error("Must specify 'minNumberOfKeypoints' to be at least 1, but "+"encountered ".concat(e.keypointTrackerParams.minNumberOfKeypoints));for(var t=0,r=e.keypointTrackerParams.keypointFalloff;t<r.length;t++){var n=r[t];if(n<=0)throw new Error("Must specify each keypoint falloff parameterto be positive "+"but encountered ".concat(n))}}}w.validateTrackerConfig=Z;/**
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
 */var P=p&&p.__assign||function(){return P=Object.assign||function(e){for(var t,r=1,n=arguments.length;r<n;r++){t=arguments[r];for(var i in t)Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i])}return e},P.apply(this,arguments)},q=p&&p.__spreadArray||function(e,t,r){if(r||arguments.length===2)for(var n=0,i=t.length,a;n<i;n++)(a||!(n in t))&&(a||(a=Array.prototype.slice.call(t,0,n)),a[n]=t[n]);return e.concat(a||Array.prototype.slice.call(t))};Object.defineProperty(g,"__esModule",{value:!0});g.Tracker=void 0;var W=w,H=function(){function e(t){(0,W.validateTrackerConfig)(t),this.tracks=[],this.maxTracks=t.maxTracks,this.maxAge=t.maxAge*1e3,this.minSimilarity=t.minSimilarity,this.nextID=1}return e.prototype.apply=function(t,r){this.filterOldTracks(r);var n=this.computeSimilarity(t);return this.assignTracks(t,n,r),this.updateTracks(r),t},e.prototype.getTracks=function(){return this.tracks.slice()},e.prototype.getTrackIDs=function(){return new Set(this.tracks.map(function(t){return t.id}))},e.prototype.filterOldTracks=function(t){var r=this;this.tracks=this.tracks.filter(function(n){return t-n.lastTimestamp<=r.maxAge})},e.prototype.assignTracks=function(t,r,n){for(var i=Array.from(Array(r[0].length).keys()),a=Array.from(Array(t.length).keys()),o=[],s=0,h=a;s<h.length;s++){var u=h[s];if(i.length===0){o.push(u);continue}for(var c=-1,T=-1,C=0,D=i;C<D.length;C++){var j=D[C],S=r[u][j];S>=this.minSimilarity&&S>T&&(c=j,T=S)}if(c>=0){var m=this.tracks[c];m=Object.assign(m,this.createTrack(t[u],n,m.id)),t[u].id=m.id;var z=i.indexOf(c);i.splice(z,1)}else o.push(u)}for(var b=0,Y=o;b<Y.length;b++){var u=Y[b],R=this.createTrack(t[u],n);this.tracks.push(R),t[u].id=R.id}},e.prototype.updateTracks=function(t){this.tracks.sort(function(r,n){return n.lastTimestamp-r.lastTimestamp}),this.tracks=this.tracks.slice(0,this.maxTracks)},e.prototype.createTrack=function(t,r,n){var i={id:n||this.nextTrackID(),lastTimestamp:r,keypoints:q([],t.keypoints,!0).map(function(a){return P({},a)})};return t.box!==void 0&&(i.box=P({},t.box)),i},e.prototype.nextTrackID=function(){var t=this.nextID;return this.nextID+=1,t},e.prototype.remove=function(){for(var t=[],r=0;r<arguments.length;r++)t[r]=arguments[r];this.tracks=this.tracks.filter(function(n){return!t.includes(n.id)})},e.prototype.reset=function(){this.tracks=[]},e}();g.Tracker=H;/**
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
 */var X=p&&p.__extends||function(){var e=function(t,r){return e=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(n,i){n.__proto__=i}||function(n,i){for(var a in i)Object.prototype.hasOwnProperty.call(i,a)&&(n[a]=i[a])},e(t,r)};return function(t,r){if(typeof r!="function"&&r!==null)throw new TypeError("Class extends value "+String(r)+" is not a constructor or null");e(t,r);function n(){this.constructor=t}t.prototype=r===null?Object.create(r):(n.prototype=r.prototype,new n)}}();Object.defineProperty(A,"__esModule",{value:!0});A.BoundingBoxTracker=void 0;var G=g,J=function(e){X(t,e);function t(r){return e.call(this,r)||this}return t.prototype.computeSimilarity=function(r){var n=this;if(r.length===0||this.tracks.length===0)return[[]];var i=r.map(function(a){return n.tracks.map(function(o){return n.iou(a,o)})});return i},t.prototype.iou=function(r,n){var i=Math.max(r.box.xMin,n.box.xMin),a=Math.max(r.box.yMin,n.box.yMin),o=Math.min(r.box.xMax,n.box.xMax),s=Math.min(r.box.yMax,n.box.yMax);if(i>=o||a>=s)return 0;var h=(o-i)*(s-a),u=r.box.width*r.box.height,c=n.box.width*n.box.height;return h/(u+c-h)},t}(G.Tracker);A.BoundingBoxTracker=J;var B={};/**
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
 */var Q=p&&p.__extends||function(){var e=function(t,r){return e=Object.setPrototypeOf||{__proto__:[]}instanceof Array&&function(n,i){n.__proto__=i}||function(n,i){for(var a in i)Object.prototype.hasOwnProperty.call(i,a)&&(n[a]=i[a])},e(t,r)};return function(t,r){if(typeof r!="function"&&r!==null)throw new TypeError("Class extends value "+String(r)+" is not a constructor or null");e(t,r);function n(){this.constructor=t}t.prototype=r===null?Object.create(r):(n.prototype=r.prototype,new n)}}(),k=p&&p.__spreadArray||function(e,t,r){if(r||arguments.length===2)for(var n=0,i=t.length,a;n<i;n++)(a||!(n in t))&&(a||(a=Array.prototype.slice.call(t,0,n)),a[n]=t[n]);return e.concat(a||Array.prototype.slice.call(t))};Object.defineProperty(B,"__esModule",{value:!0});B.KeypointTracker=void 0;var U=g,tt=function(e){Q(t,e);function t(r){var n=e.call(this,r)||this;return n.keypointThreshold=r.keypointTrackerParams.keypointConfidenceThreshold,n.keypointFalloff=r.keypointTrackerParams.keypointFalloff,n.minNumKeyoints=r.keypointTrackerParams.minNumberOfKeypoints,n}return t.prototype.computeSimilarity=function(r){if(r.length===0||this.tracks.length===0)return[[]];for(var n=[],i=0,a=r;i<a.length;i++){for(var o=a[i],s=[],h=0,u=this.tracks;h<u.length;h++){var c=u[h];s.push(this.oks(o,c))}n.push(s)}return n},t.prototype.oks=function(r,n){for(var i=this.area(n.keypoints)+1e-6,a=0,o=0,s=0;s<r.keypoints.length;++s){var h=r.keypoints[s],u=n.keypoints[s];if(!(h.score<this.keypointThreshold||u.score<this.keypointThreshold)){o+=1;var c=Math.pow(h.x-u.x,2)+Math.pow(h.y-u.y,2),T=2*this.keypointFalloff[s];a+=Math.exp(-1*c/(2*i*Math.pow(T,2)))}}return o<this.minNumKeyoints?0:a/o},t.prototype.area=function(r){var n=this,i=r.filter(function(u){return u.score>n.keypointThreshold}),a=Math.min.apply(Math,k([1],i.map(function(u){return u.x}),!1)),o=Math.max.apply(Math,k([0],i.map(function(u){return u.x}),!1)),s=Math.min.apply(Math,k([1],i.map(function(u){return u.y}),!1)),h=Math.max.apply(Math,k([0],i.map(function(u){return u.y}),!1));return(o-a)*(h-s)},t}(U.Tracker);B.KeypointTracker=tt;var et={};(function(e){/**
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
 */Object.defineProperty(e,"__esModule",{value:!0}),e.TrackerType=void 0,function(t){t.Keypoint="keypoint",t.BoundingBox="boundingBox"}(e.TrackerType||(e.TrackerType={}))})(et);var f={};Object.defineProperty(f,"__esModule",{value:!0});f.BLAZEPOSE_CONNECTED_KEYPOINTS_PAIRS=f.COCO_CONNECTED_KEYPOINTS_PAIRS=f.COCO_KEYPOINTS_BY_SIDE=f.BLAZEPOSE_KEYPOINTS_BY_SIDE=f.BLAZEPOSE_KEYPOINTS=f.COCO_KEYPOINTS=void 0;/**
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
 */f.COCO_KEYPOINTS=["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"];f.BLAZEPOSE_KEYPOINTS=["nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"];f.BLAZEPOSE_KEYPOINTS_BY_SIDE={left:[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31],right:[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],middle:[0]};f.COCO_KEYPOINTS_BY_SIDE={left:[1,3,5,7,9,11,13,15],right:[2,4,6,8,10,12,14,16],middle:[0]};f.COCO_CONNECTED_KEYPOINTS_PAIRS=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[5,11],[6,8],[6,12],[7,9],[8,10],[11,12],[11,13],[12,14],[13,15],[14,16]];f.BLAZEPOSE_CONNECTED_KEYPOINTS_PAIRS=[[0,1],[0,4],[1,2],[2,3],[3,7],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[11,23],[12,14],[14,16],[12,24],[13,15],[15,17],[16,18],[16,20],[15,17],[15,19],[15,21],[16,22],[17,19],[18,20],[23,25],[23,24],[24,26],[25,27],[26,28],[27,29],[28,30],[27,31],[28,32],[29,31],[30,32]];var _={};Object.defineProperty(_,"__esModule",{value:!0});_.MILLISECOND_TO_MICRO_SECONDS=_.SECOND_TO_MICRO_SECONDS=_.MICRO_SECONDS_TO_SECOND=void 0;/**
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
 */_.MICRO_SECONDS_TO_SECOND=1e-6;_.SECOND_TO_MICRO_SECONDS=1e6;_.MILLISECOND_TO_MICRO_SECONDS=1e3;var l={};Object.defineProperty(l,"__esModule",{value:!0});l.getProjectiveTransformMatrix=l.getRoi=l.padRoi=l.toImageTensor=l.transformValueRange=l.normalizeRadians=l.getImageSize=void 0;/**
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
 */var O=$;function rt(e){return e instanceof O.Tensor?{height:e.shape[0],width:e.shape[1]}:{height:e.height,width:e.width}}l.getImageSize=rt;function nt(e){return e-2*Math.PI*Math.floor((e+Math.PI)/(2*Math.PI))}l.normalizeRadians=nt;function it(e,t,r,n){var i=t-e,a=n-r;if(i===0)throw new Error("Original min and max are both ".concat(e,", range cannot be 0."));var o=a/i,s=r-e*o;return{scale:o,offset:s}}l.transformValueRange=it;function at(e){return e instanceof O.Tensor?e:O.browser.fromPixels(e)}l.toImageTensor=at;function ot(e,t,r){if(r===void 0&&(r=!1),!r)return{top:0,left:0,right:0,bottom:0};var n=t.height,i=t.width;I(t,"targetSize"),I(e,"roi");var a=n/i,o=e.height/e.width,s,h,u=0,c=0;return a>o?(s=e.width,h=e.width*a,c=(1-o/a)/2):(s=e.height/a,h=e.height,u=(1-a/o)/2),e.width=s,e.height=h,{top:c,left:u,right:u,bottom:c}}l.padRoi=ot;function st(e,t){return t?{xCenter:t.xCenter*e.width,yCenter:t.yCenter*e.height,width:t.width*e.width,height:t.height*e.height,rotation:t.rotation}:{xCenter:.5*e.width,yCenter:.5*e.height,width:e.width,height:e.height,rotation:0}}l.getRoi=st;function ut(e,t,r){I(r,"inputResolution");var n=1/r.width*e[0][0]*t.width,i=1/r.height*e[0][1]*t.width,a=e[0][3]*t.width,o=1/r.width*e[1][0]*t.height,s=1/r.height*e[1][1]*t.height,h=e[1][3]*t.height;return[n,i,a,o,s,h,0,0]}l.getProjectiveTransformMatrix=ut;function I(e,t){O.util.assert(e.width!==0,function(){return"".concat(t," width cannot be 0.")}),O.util.assert(e.height!==0,function(){return"".concat(t," height cannot be 0.")})}var K={};/**
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
 */Object.defineProperty(K,"__esModule",{value:!0});K.isVideo=void 0;function ht(e){return e!=null&&e.currentTime!=null}K.isVideo=ht;var F={},x={},M={};Object.defineProperty(M,"__esModule",{value:!0});M.LowPassFilter=void 0;/**
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
 */var lt=function(){function e(t){this.alpha=t,this.initialized=!1}return e.prototype.apply=function(t,r){var n;return this.initialized?r==null?n=this.storedValue+this.alpha*(t-this.storedValue):n=this.storedValue+this.alpha*r*Math.asinh((t-this.storedValue)/r):(n=t,this.initialized=!0),this.rawValue=t,this.storedValue=n,n},e.prototype.applyWithAlpha=function(t,r,n){return this.alpha=r,this.apply(t,n)},e.prototype.hasLastRawValue=function(){return this.initialized},e.prototype.lastRawValue=function(){return this.rawValue},e.prototype.reset=function(){this.initialized=!1},e}();M.LowPassFilter=lt;Object.defineProperty(x,"__esModule",{value:!0});x.OneEuroFilter=void 0;/**
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
 */var ct=_,L=M,ft=function(){function e(t){this.frequency=t.frequency,this.minCutOff=t.minCutOff,this.beta=t.beta,this.thresholdCutOff=t.thresholdCutOff,this.thresholdBeta=t.thresholdBeta,this.derivateCutOff=t.derivateCutOff,this.x=new L.LowPassFilter(this.getAlpha(this.minCutOff)),this.dx=new L.LowPassFilter(this.getAlpha(this.derivateCutOff)),this.lastTimestamp=0}return e.prototype.apply=function(t,r,n){if(t==null)return t;var i=Math.trunc(r);if(this.lastTimestamp>=i)return t;this.lastTimestamp!==0&&i!==0&&(this.frequency=1/((i-this.lastTimestamp)*ct.MICRO_SECONDS_TO_SECOND)),this.lastTimestamp=i;var a=this.x.hasLastRawValue()?(t-this.x.lastRawValue())*n*this.frequency:0,o=this.dx.applyWithAlpha(a,this.getAlpha(this.derivateCutOff)),s=this.minCutOff+this.beta*Math.abs(o),h=this.thresholdCutOff!=null?this.thresholdCutOff+this.thresholdBeta*Math.abs(o):null;return this.x.applyWithAlpha(t,this.getAlpha(s),h)},e.prototype.getAlpha=function(t){return 1/(1+this.frequency/(2*Math.PI*t))},e}();x.OneEuroFilter=ft;/**
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
 */var E=p&&p.__assign||function(){return E=Object.assign||function(e){for(var t,r=1,n=arguments.length;r<n;r++){t=arguments[r];for(var i in t)Object.prototype.hasOwnProperty.call(t,i)&&(e[i]=t[i])}return e},E.apply(this,arguments)},pt=p&&p.__spreadArray||function(e,t,r){if(r||arguments.length===2)for(var n=0,i=t.length,a;n<i;n++)(a||!(n in t))&&(a||(a=Array.prototype.slice.call(t,0,n)),a[n]=t[n]);return e.concat(a||Array.prototype.slice.call(t))};Object.defineProperty(F,"__esModule",{value:!0});F.KeypointsOneEuroFilter=void 0;var N=x,dt=function(){function e(t){this.config=t}return e.prototype.apply=function(t,r,n){var i=this;if(t==null)return this.reset(),null;this.initializeFiltersIfEmpty(t);var a=1;if(!this.config.disableValueScaling){if(n<this.config.minAllowedObjectScale)return pt([],t,!0);a=1/n}return t.map(function(o,s){var h=E(E({},o),{x:i.xFilters[s].apply(o.x,r,a),y:i.yFilters[s].apply(o.y,r,a)});return o.z!=null&&(h.z=i.zFilters[s].apply(o.z,r,a)),h})},e.prototype.reset=function(){this.xFilters=null,this.yFilters=null,this.zFilters=null},e.prototype.initializeFiltersIfEmpty=function(t){var r=this;(this.xFilters==null||this.xFilters.length!==t.length)&&(this.xFilters=t.map(function(n){return new N.OneEuroFilter(r.config)}),this.yFilters=t.map(function(n){return new N.OneEuroFilter(r.config)}),this.zFilters=t.map(function(n){return new N.OneEuroFilter(r.config)}))},e}();F.KeypointsOneEuroFilter=dt;var V={};(function(e){Object.defineProperty(e,"__esModule",{value:!0}),e.SupportedModels=void 0,function(t){t.MoveNet="MoveNet",t.BlazePose="BlazePose",t.PoseNet="PoseNet"}(e.SupportedModels||(e.SupportedModels={}))})(V);var y={};Object.defineProperty(y,"__esModule",{value:!0});y.getKeypointIndexByName=y.getAdjacentPairs=y.getKeypointIndexBySide=void 0;/**
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
 */var v=f,d=V;function _t(e){switch(e){case d.SupportedModels.BlazePose:return v.BLAZEPOSE_KEYPOINTS_BY_SIDE;case d.SupportedModels.PoseNet:case d.SupportedModels.MoveNet:return v.COCO_KEYPOINTS_BY_SIDE;default:throw new Error("Model ".concat(e," is not supported."))}}y.getKeypointIndexBySide=_t;function yt(e){switch(e){case d.SupportedModels.BlazePose:return v.BLAZEPOSE_CONNECTED_KEYPOINTS_PAIRS;case d.SupportedModels.PoseNet:case d.SupportedModels.MoveNet:return v.COCO_CONNECTED_KEYPOINTS_PAIRS;default:throw new Error("Model ".concat(e," is not supported."))}}y.getAdjacentPairs=yt;function vt(e){switch(e){case d.SupportedModels.BlazePose:return v.BLAZEPOSE_KEYPOINTS.reduce(function(t,r,n){return t[r]=n,t},{});case d.SupportedModels.PoseNet:case d.SupportedModels.MoveNet:return v.COCO_KEYPOINTS.reduce(function(t,r,n){return t[r]=n,t},{});default:throw new Error("Model ".concat(e," is not supported."))}}y.getKeypointIndexByName=vt;export{V as a,B as b,f as c,A as d,_ as e,l as f,K as i,F as k,M as l,et as t,y as u};
