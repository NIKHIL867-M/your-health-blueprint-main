// src/lib/poseMath.ts

/* -------------------------------------------------------------------------- */
/* TYPES */
/* -------------------------------------------------------------------------- */

export interface Point {
  x: number;
  y: number;
  score?: number;
  z?: number;
}

// --- SQUAT TYPES ---
export type SquatPhase = "UP" | "DOWN" | "UNKNOWN";

export interface SquatLogicState {
  count: number;
  phase: SquatPhase;
  badFormCount: number;
  forwardLeanCount: number;
  lastFeedback: string;
  repQuality: "GOOD" | "BAD" | "NEUTRAL";
  smoothKneeAngle: number;
  smoothBackAngle: number;
  repProgress: number;
}

export interface SquatLogicOutput extends SquatLogicState {
  kneeAngle: number;
  backAngle: number;
  isRepJustCompleted: boolean;
  cues: string[];
}

// --- FACE / NECK TYPES ---
export interface FaceLogicOutput {
  postureScore: number;
  neckAngle: number;
  headTiltAngle: number;
  cues: string[];
}

/* -------------------------------------------------------------------------- */
/* CONFIGURATION */
/* -------------------------------------------------------------------------- */

// Squat thresholds
const DOWN_THRESHOLD = 100;
const UP_THRESHOLD = 160;
const MIN_BACK_ANGLE = 70;
const SMOOTHING_FACTOR = 0.7;

// Face thresholds (neutral range for comfort)
const NEUTRAL_FORWARD_RANGE = 10; // <=10° is "neutral"
const MAX_FORWARD_HEAD_ANGLE = 25; // >25° is "clearly forward"
const NEUTRAL_TILT_RANGE = 8; // <=8° is "neutral"
const MAX_SIDE_TILT_ANGLE = 20; // >20° is "clearly tilted"

/* -------------------------------------------------------------------------- */
/* MATH HELPERS */
/* -------------------------------------------------------------------------- */

export function calculateAngle(a: Point, b: Point, c: Point): number {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const magCB = Math.sqrt(cb.x * cb.x + cb.y * cb.y);

  if (magAB === 0 || magCB === 0) return 0;

  const clampedDot = Math.max(-1, Math.min(1, dot / (magAB * magCB)));
  const angle = (Math.acos(clampedDot) * 180) / Math.PI;

  return isNaN(angle) ? 0 : angle;
}

function smoothValue(current: number, previous: number, alpha: number): number {
  return alpha * current + (1 - alpha) * previous;
}

function calculateProgress(angle: number): number {
  const range = UP_THRESHOLD - DOWN_THRESHOLD;
  const progress = ((UP_THRESHOLD - angle) / range) * 100;
  return Math.min(Math.max(progress, 0), 100);
}

/* -------------------------------------------------------------------------- */
/* SQUAT LOGIC */
/* -------------------------------------------------------------------------- */

export function getInitialSquatState(): SquatLogicState {
  return {
    count: 0,
    phase: "UP",
    badFormCount: 0,
    forwardLeanCount: 0,
    lastFeedback: "Stand in frame",
    repQuality: "NEUTRAL",
    smoothKneeAngle: 0,
    smoothBackAngle: 0,
    repProgress: 0,
  };
}

export function updateSquatLogic(
  prev: SquatLogicState,
  rawKneeAngle: number,
  rawBackAngle: number
): SquatLogicOutput {
  // Smooth angles
  const kneeAngle =
    prev.smoothKneeAngle === 0
      ? rawKneeAngle
      : smoothValue(rawKneeAngle, prev.smoothKneeAngle, SMOOTHING_FACTOR);

  const backAngle =
    prev.smoothBackAngle === 0
      ? rawBackAngle
      : smoothValue(rawBackAngle, prev.smoothBackAngle, SMOOTHING_FACTOR);

  let { count, phase, badFormCount, forwardLeanCount, repQuality } = prev;
  let isRepJustCompleted = false;
  const cues: string[] = [];

  // Form checks
  if (backAngle < MIN_BACK_ANGLE) {
    forwardLeanCount += 1;
    cues.push("Keep your chest up!");
    repQuality = "BAD";
  }

  if (kneeAngle > DOWN_THRESHOLD && kneeAngle < UP_THRESHOLD) {
    badFormCount += 1;
    if (phase === "UP") {
      cues.push("Go deeper into your squat");
    } else if (phase === "DOWN") {
      cues.push("Push back up!");
    }
  }

  // State machine
  if (kneeAngle <= DOWN_THRESHOLD && phase === "UP") {
    phase = "DOWN";
    repQuality = "GOOD";
  } else if (kneeAngle >= UP_THRESHOLD && phase === "DOWN") {
    phase = "UP";
    count += 1;
    isRepJustCompleted = true;
    if (repQuality !== "BAD") {
      repQuality = "GOOD";
    }
  }

  if (phase === "UP" && kneeAngle > UP_THRESHOLD && cues.length === 0) {
    cues.push("Ready for next rep");
  }

  return {
    count,
    phase,
    badFormCount,
    forwardLeanCount,
    lastFeedback: cues[0] || "Good form",
    repQuality,
    smoothKneeAngle: kneeAngle,
    smoothBackAngle: backAngle,
    repProgress: calculateProgress(kneeAngle),
    kneeAngle: Math.round(kneeAngle * 10) / 10,
    backAngle: Math.round(backAngle * 10) / 10,
    isRepJustCompleted,
    cues,
  };
}

/* -------------------------------------------------------------------------- */
/* FACE / NECK POSTURE LOGIC */
/* -------------------------------------------------------------------------- */

export function evaluateFacePosture(
  head: Point,
  neckBase: Point,
  shoulder: Point
): FaceLogicOutput {
  const cues: string[] = [];

  // Neck angle relative to vertical
  const verticalRefTop: Point = { x: neckBase.x, y: neckBase.y - 0.5 };
  const neckAngle = calculateAngle(verticalRefTop, neckBase, head);

  // Head tilt (sideways)
  const horizontalRef: Point = { x: neckBase.x + 0.5, y: neckBase.y };
  const rawTilt = calculateAngle(horizontalRef, neckBase, head);
  const headTiltAngle = Math.abs(90 - rawTilt);

  // Score calculation
  let postureScore = 100;

  // Forward head penalty (more tolerant now)
  if (neckAngle > NEUTRAL_FORWARD_RANGE) {
    const excess = neckAngle - NEUTRAL_FORWARD_RANGE;
    const maxExcess = MAX_FORWARD_HEAD_ANGLE - NEUTRAL_FORWARD_RANGE;
    const penalty = Math.min(
      40,
      (excess / maxExcess) * 40
    );
    postureScore -= penalty;
    cues.push("Keep your head closer over your shoulders.");
  }

  // Side tilt penalty
  if (headTiltAngle > NEUTRAL_TILT_RANGE) {
    const excess = headTiltAngle - NEUTRAL_TILT_RANGE;
    const maxExcess = MAX_SIDE_TILT_ANGLE - NEUTRAL_TILT_RANGE;
    const penalty = Math.min(30, (excess / maxExcess) * 30);
    postureScore -= penalty;
    cues.push("Avoid tilting your head; keep it neutral.");
  }

  // Feedback based on score
  if (postureScore >= 90) {
    if (cues.length === 0) {
      cues.push("Excellent alignment – hold this position.");
    }
  } else if (postureScore >= 70) {
    if (cues.length === 0) {
      cues.push("Good posture; make small adjustments.");
    }
  } else {
    if (cues.length === 0) {
      cues.push("Reset your posture: stand tall, look straight ahead.");
    }
  }

  if (postureScore < 0) postureScore = 0;

  return {
    postureScore: Math.round(postureScore),
    neckAngle: Math.round(neckAngle * 10) / 10,
    headTiltAngle: Math.round(headTiltAngle * 10) / 10,
    cues,
  };
}

/* -------------------------------------------------------------------------- */
/* FACE EXERCISE HELPER: ANGLE ADJUSTMENT (Calibration Support) */
/* -------------------------------------------------------------------------- */

/**
 * Adjusts raw angles using neutral calibration values
 * Returns adjusted angles as if calibration is at 0
 */
export function adjustFaceAngles(
  neckAngle: number,
  headTiltAngle: number,
  neutralNeckAngle: number,
  neutralHeadTilt: number
): { adjustedNeck: number; adjustedTilt: number } {
  return {
    adjustedNeck: neckAngle - neutralNeckAngle,
    adjustedTilt: headTiltAngle - neutralHeadTilt,
  };
}

/**
 * Checks if current angle is within tolerance of target angle
 * Tolerance: ±20 degrees (comfortable range for neck exercises)
 */
export function isAngleInRange(
  currentAngle: number,
  targetAngle: number,
  tolerance: number = 20
): boolean {
  return Math.abs(currentAngle - targetAngle) <= tolerance;
}

/**
 * Calculates quality score based on hold time and accuracy
 */
export function calculateFaceExerciseQuality(
  holdMs: number,
  requiredHoldMs: number,
  isInRange: boolean
): number {
  if (!isInRange) return 0;
  const progress = Math.min(holdMs / requiredHoldMs, 1);
  return Math.round(progress * 100);
}
