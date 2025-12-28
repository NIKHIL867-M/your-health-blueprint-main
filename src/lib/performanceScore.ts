// src/lib/performanceScore.ts

import { SessionSummary } from "@/lib/poseMath";

/* -------------------------------------------------------------------------- */
/* TYPES */
/* -------------------------------------------------------------------------- */

export interface PerformanceScore {
  score: number;         // 0–100 integer
  level: "LOW" | "OK" | "HIGH";
  label: string;         // Short badge text (e.g., "Elite Form")
  explanation: string;   // Contextual advice
  color: string;         // UI helper color (hex/tailwind class reference)
}

/* -------------------------------------------------------------------------- */
/* CORE LOGIC */
/* -------------------------------------------------------------------------- */

/**
 * Computes an AI performance score based on session biomechanics.
 */
export function computePerformanceScore(session: SessionSummary): PerformanceScore {
  
  // --- SQUAT SCORING ---
  if (session.exerciseType === "squat") {
    const { reps, durationSec, formScore } = session;
    
    // 1. Intensity Score (Reps Per Minute)
    // We cap "perfect" intensity at 25 reps/min. Faster isn't always better.
    const durationSafe = Math.max(durationSec, 1);
    const repsPerMin = (reps / durationSafe) * 60;
    const intensityScore = Math.min(repsPerMin / 25, 1) * 100;

    // 2. Weighted Calculation
    // Priority: Form (60%) > Intensity (40%)
    let rawScore = (formScore * 0.6) + (intensityScore * 0.4);

    // Penalty: If form is terrible (<50), cap the max score regardless of reps
    if (formScore < 50) {
      rawScore = Math.min(rawScore, 60); 
    }

    const finalScore = Math.round(rawScore);
    const classification = classifyScore(finalScore);

    // Dynamic Feedback
    let specificFeedback = classification.explanation;
    if (formScore < 60 && reps > 5) {
      specificFeedback = "High volume, but form suffered. Slow down and focus on depth.";
    } else if (intensityScore < 30 && formScore > 80) {
      specificFeedback = "Perfect form! Try increasing your speed next time.";
    }

    return {
      ...classification,
      score: finalScore,
      explanation: specificFeedback
    };
  } 
  
  // --- FACE / POSTURE SCORING ---
  else {
    const { avgPostureScore, bestHoldSec } = session;

    // 1. Endurance Score
    // Target: A 15-second perfect hold is considered "Elite" (100%) for this calculation
    const enduranceScore = Math.min(bestHoldSec / 15, 1) * 100;

    // 2. Weighted Calculation
    // Priority: Consistency (70%) > Peak Hold (30%)
    const rawScore = (avgPostureScore * 0.7) + (enduranceScore * 0.3);
    
    const finalScore = Math.round(rawScore);
    const classification = classifyScore(finalScore);

    // Dynamic Feedback
    let specificFeedback = classification.explanation;
    if (avgPostureScore < 70) {
      specificFeedback = "Alignment drift detected. Keep your head neutral consistently.";
    } else if (enduranceScore < 50) {
      specificFeedback = "Great posture! Try holding the position longer next time.";
    }

    return {
      ...classification,
      score: finalScore,
      explanation: specificFeedback
    };
  }
}

/* -------------------------------------------------------------------------- */
/* HELPERS */
/* -------------------------------------------------------------------------- */

function classifyScore(score: number): Pick<PerformanceScore, "level" | "label" | "explanation" | "color"> {
  if (score >= 85) {
    return {
      level: "HIGH",
      label: "Elite Performance",
      explanation: "Outstanding work. Your biomechanics were optimal.",
      color: "text-emerald-500"
    };
  } else if (score >= 60) {
    return {
      level: "OK",
      label: "Solid Effort",
      explanation: "Good session. Minor adjustments needed for perfect form.",
      color: "text-blue-500"
    };
  } else {
    return {
      level: "LOW",
      label: "Needs Improvement",
      explanation: "Focus on technique over speed. Watch the real-time cues.",
      color: "text-orange-500"
    };
  }
}