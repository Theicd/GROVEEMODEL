import type { PoseAction, PoseLandmark } from '../core/types';
import { angle } from '../utils/geometry';

interface PoseHistory {
  hipY: number[];
  kneeAngles: number[];
  timestamps: number[];
}

export class PoseActionClassifier {
  private history: PoseHistory = { hipY: [], kneeAngles: [], timestamps: [] };

  classify(landmarks: PoseLandmark[], timestamp: number): PoseAction[] {
    if (landmarks.length < 33) return [];

    const actions: PoseAction[] = [];
    const lm = landmarks;

    const leftHip = lm[23];
    const rightHip = lm[24];
    const leftKnee = lm[25];
    const rightKnee = lm[26];
    const leftAnkle = lm[27];
    const rightAnkle = lm[28];
    const leftShoulder = lm[11];
    const rightShoulder = lm[12];
    const leftWrist = lm[15];
    const rightWrist = lm[16];
    const leftElbow = lm[13];

    const hipMidY = (leftHip.y + rightHip.y) / 2;
    const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    const torsoAngle = Math.abs(leftShoulder.x - leftHip.x) + Math.abs(rightShoulder.x - rightHip.x);

    const leftKneeAngle = angle(leftHip, leftKnee, leftAnkle);
    const rightKneeAngle = angle(rightHip, rightKnee, rightAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;

    this.history.hipY.push(hipMidY);
    this.history.kneeAngles.push(avgKneeAngle);
    this.history.timestamps.push(timestamp);
    if (this.history.hipY.length > 15) {
      this.history.hipY.shift();
      this.history.kneeAngles.shift();
      this.history.timestamps.shift();
    }

    const isHorizontal = Math.abs(leftShoulder.y - leftHip.y) < 0.08;

    if (isHorizontal && avgKneeAngle > 150) {
      actions.push({ name: 'Lying', confidence: 0.85 });
    } else if (avgKneeAngle < 100 && hipMidY > shoulderMidY) {
      actions.push({ name: 'Sitting', confidence: 0.88 });
    } else if (avgKneeAngle > 155) {
      actions.push({ name: 'Standing', confidence: 0.9 });
    }

    if (avgKneeAngle < 70 && hipMidY > 0.5) {
      actions.push({ name: 'Squat', confidence: 0.82 });
    }

    if (isHorizontal && avgKneeAngle < 120) {
      actions.push({ name: 'Push-up', confidence: 0.78 });
    }

    const hipDelta = this.history.hipY.length > 3
      ? this.history.hipY[0] - this.history.hipY[this.history.hipY.length - 1]
      : 0;
    if (Math.abs(hipDelta) > 0.08) {
      actions.push({ name: 'Jumping', confidence: 0.8 });
    }

    const ankleMovement = Math.abs(leftAnkle.y - rightAnkle.y);
    if (ankleMovement > 0.06 && avgKneeAngle > 120 && avgKneeAngle < 160) {
      actions.push({ name: 'Running', confidence: 0.75 });
    } else if (ankleMovement > 0.03 && avgKneeAngle > 140) {
      actions.push({ name: 'Walking', confidence: 0.7 });
    }

    if (leftWrist.y < leftShoulder.y - 0.05) {
      actions.push({ name: 'Left Hand Raised', confidence: 0.85 });
    }
    if (rightWrist.y < rightShoulder.y - 0.05) {
      actions.push({ name: 'Right Hand Raised', confidence: 0.85 });
    }

    const leftArmExtended = angle(leftShoulder, leftElbow, leftWrist) > 150;
    if (leftArmExtended && leftWrist.y < leftElbow.y) {
      actions.push({ name: 'Pointing', confidence: 0.78 });
    }

    const torsoVerticalDiff = Math.abs(shoulderMidY - hipMidY);
    if (torsoVerticalDiff > 0.15 && torsoAngle > 0.1) {
      actions.push({ name: 'Bending', confidence: 0.76 });
    }

    if (actions.length === 0) {
      actions.push({ name: 'Standing', confidence: 0.5 });
    }

    return actions;
  }

  reset(): void {
    this.history = { hipY: [], kneeAngles: [], timestamps: [] };
  }
}
