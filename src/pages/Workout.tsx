import { auth, db } from "@/lib/firebase";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { useState, useRef, useEffect, useCallback, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Video,
  VideoOff,
  Play,
  Square,
  Activity,
  MoveHorizontal,
  MoveVertical,
  RotateCw,
  Dumbbell,
  ScanFace,
  Trophy,
  AlertCircle,
  Save,
  History,
  Target,
  Brain,
  TrendingUp
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";

// MediaPipe Imports
import { Pose, Results as PoseResults } from "@mediapipe/pose";
import { FaceMesh, Results as FaceResults } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";

// Local Logic Imports
import {
  calculateAngle,
  updateSquatLogic,
  getInitialSquatState,
  SquatLogicOutput,
} from "@/lib/poseMath";

// Performance Score Import
import { computePerformanceScore, PerformanceScore } from "@/lib/performanceScore";

// --- TYPES FROM SOURCE DOC ---
type ExerciseMode = "squat" | "face";

interface SquatSessionSummary {
  exerciseType: "squat";
  timestamp: string;
  reps: number;
  durationSec: number;
  avgKneeAngle: number;
  avgBackAngle: number;
  badFormCount: number;
  forwardLeanCount: number;
  formScore: number;
}

interface FaceSessionSummary {
  exerciseType: "face";
  timestamp: string;
  durationSec: number;
  avgPostureScore: number;
  bestHoldSec: number;
  overallQuality: number;
}

type SessionSummary = SquatSessionSummary | FaceSessionSummary;

// Internal State for UI updates
interface FaceLogicState {
  yaw: number;
  pitch: number;
  roll: number;
  score: number;
  feedback: string;
  isGoodPosture: boolean;
  holdMs: number;
}

// Helper function to create initial squat output state
const getInitialSquatOutput = (): SquatLogicOutput & { currentKneeAngle: number; currentHipAngle: number } => {
  const initialState = getInitialSquatState();
  return {
    ...initialState,
    kneeAngle: 0,
    backAngle: 0,
    isRepJustCompleted: false,
    cues: [],
    currentKneeAngle: 0,
    currentHipAngle: 0,
  };
};

// --- HELPER: 3D HEAD POSE MATH ---
const calculateHeadPose = (landmarks: any[]) => {
  const nose = landmarks[1];
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const chin = landmarks[152];

  const dy = rightEye.y - leftEye.y;
  const dx = rightEye.x - leftEye.x;
  const roll = Math.atan2(dy, dx) * (180 / Math.PI);

  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const yaw = (nose.x - eyeMidX) * -250;

  const eyeMidY = (leftEye.y + rightEye.y) / 2;
  const faceHeight = chin.y - eyeMidY;
  const pitch = (nose.y - (eyeMidY + faceHeight * 0.4)) * -250;

  return { yaw, pitch, roll };
};

// --- FIRESTORE SAVE FUNCTION ---
const saveSessionToFirestore = async (summary: SessionSummary, perf: PerformanceScore) => {
  const user = auth.currentUser;
  if (!user) {
    console.warn("No user logged in — session not saved.");
    return;
  }

  try {
    const sessionsRef = collection(db, "users", user.uid, "sessions");
    await addDoc(sessionsRef, {
      exerciseType: summary.exerciseType,
      timestamp: serverTimestamp(),
      summary,
      performance: perf,
    });
    console.log("Session saved!");
  } catch (err) {
    console.error("Error saving session:", err);
  }
};

// --- MEMOIZED CONTROLS (Fixes Flickering) ---
const MemoizedControls = memo(({
  isRecording,
  cameraActive,
  onToggleRecording,
  onToggleCamera,
  mode,
  cameraLoading
}: any) => (
  <div className="flex items-center gap-3">
    <Button
      size="lg"
      onClick={onToggleRecording}
      disabled={!cameraActive}
      className={cn("flex-1 text-base font-semibold shadow-md transition-all",
        isRecording ? "bg-destructive hover:bg-destructive/90" : "bg-primary hover:bg-primary/90"
      )}
    >
      {isRecording ? (
        <><Square className="mr-2 w-5 h-5 fill-current" /> Stop & Save</>
      ) : (
        <><Play className="mr-2 w-5 h-5 fill-current" /> Start Session</>
      )}
    </Button>

    <Button size="lg" variant="outline" onClick={onToggleCamera} className="px-6" disabled={cameraLoading}>
      {cameraLoading ? <Activity className="w-5 h-5 animate-spin" /> : (
        cameraActive ? <VideoOff className="w-5 h-5" /> : <Video className="w-5 h-5" />
      )}
    </Button>
  </div>
));
MemoizedControls.displayName = "MemoizedControls";

export default function Workout() {
  const { toast } = useToast();

  // --- STATE ---
  const [mode, setMode] = useState<ExerciseMode>("squat");
  const [isRecording, setIsRecording] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [sessionHistory, setSessionHistory] = useState<SessionSummary[]>([]);
  const [lastSession, setLastSession] = useState<SessionSummary | null>(null);
  const [lastPerformance, setLastPerformance] = useState<PerformanceScore | null>(null);

  // Logic State (Visuals)
  const [squatState, setSquatState] = useState<SquatLogicOutput & { currentKneeAngle: number; currentHipAngle: number }>(
    getInitialSquatOutput()
  );

  const [faceState, setFaceState] = useState<FaceLogicState>({
    yaw: 0, pitch: 0, roll: 0,
    score: 0, feedback: "Center your face",
    isGoodPosture: false,
    holdMs: 0
  });

  // --- REFS (Performance) ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<Camera | null>(null);
  const poseModelRef = useRef<Pose | null>(null);
  const faceModelRef = useRef<FaceMesh | null>(null);

  // Refs for Logic Preservation inside Loops
  const isRecordingRef = useRef(isRecording);
  const modeRef = useRef(mode);
  const lastUiUpdateRef = useRef<number>(0);

  // Logic Accumulator Refs (Current State)
  const squatLogicRef = useRef<SquatLogicOutput>(getInitialSquatOutput());
  const faceLogicRef = useRef<FaceLogicState>({
    yaw: 0, pitch: 0, roll: 0, score: 0, feedback: "", isGoodPosture: false, holdMs: 0
  });

  // --- SESSION ACCUMULATOR REFS (For Summaries) ---
  const sessionStartTimeRef = useRef<number>(0);
  // Squat Accumulators
  const squatAccRef = useRef({
    totalKneeAngle: 0,
    totalBackAngle: 0,
    frameCount: 0
  });
  // Face Accumulators
  const faceAccRef = useRef({
    totalScore: 0,
    frameCount: 0,
    maxHoldMs: 0
  });

  // Sync refs with state
  useEffect(() => { isRecordingRef.current = isRecording; }, [isRecording]);
  useEffect(() => { modeRef.current = mode; }, [mode]);

  // --- INITIALIZATION ---
  useEffect(() => {
    // 1. Setup Body Pose Model
    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });
    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
    pose.onResults(onPoseResults);
    poseModelRef.current = pose;

    // 2. Setup Face Mesh Model
    const faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
    faceMesh.onResults(onFaceResults);
    faceModelRef.current = faceMesh;

    return () => stopCamera();
  }, []);

  // Handle Mode Switching
  useEffect(() => {
    if (cameraActive) {
      stopCamera();
      setTimeout(() => startCamera(), 100);
    }
    // Reset Logic Refs
    squatLogicRef.current = getInitialSquatOutput();
    faceLogicRef.current = { yaw: 0, pitch: 0, roll: 0, score: 0, feedback: "", isGoodPosture: false, holdMs: 0 };
  }, [mode]);

  // --- SESSION MANAGEMENT ---

  const startSession = () => {
    sessionStartTimeRef.current = Date.now();

    // Reset Accumulators
    if (mode === "squat") {
      squatLogicRef.current = getInitialSquatOutput();
      setSquatState(getInitialSquatOutput());
      squatAccRef.current = { totalKneeAngle: 0, totalBackAngle: 0, frameCount: 0 };
    } else {
      faceLogicRef.current = { ...faceLogicRef.current, holdMs: 0 };
      setFaceState(prev => ({ ...prev, holdMs: 0 }));
      faceAccRef.current = { totalScore: 0, frameCount: 0, maxHoldMs: 0 };
    }

    setIsRecording(true);
    toast({ title: "Session Started", description: `Recording ${mode} metrics...` });
  };

  const stopSession = () => {
    setIsRecording(false);
    const durationMs = Date.now() - sessionStartTimeRef.current;
    const durationSec = parseFloat((durationMs / 1000).toFixed(1));
    const timestamp = new Date().toLocaleString();

    let summary: SessionSummary | null = null;

    if (mode === "squat") {
      const { count, badFormCount, forwardLeanCount } = squatLogicRef.current;
      const { totalKneeAngle, totalBackAngle, frameCount } = squatAccRef.current;

      // Calculate averages (avoid division by zero)
      const avgKnee = frameCount > 0 ? totalKneeAngle / frameCount : 0;
      const avgBack = frameCount > 0 ? totalBackAngle / frameCount : 0;

      // Calculate Form Score
      const calculatedScore = Math.max(0, 100 - ((badFormCount + forwardLeanCount) * 5));

      summary = {
        exerciseType: "squat",
        timestamp,
        reps: count,
        durationSec,
        avgKneeAngle: parseFloat(avgKnee.toFixed(1)),
        avgBackAngle: parseFloat(avgBack.toFixed(1)),
        badFormCount,
        forwardLeanCount,
        formScore: calculatedScore
      };

      // Only save if there was actual activity
      if (count > 0 || durationSec > 5) {
        // Calculate performance score
        const perf = computePerformanceScore(summary);
        setLastPerformance(perf);

        setSessionHistory(prev => [summary as SessionSummary, ...prev]);
        setLastSession(summary);

        // ⭐ SAVE TO FIRESTORE
        saveSessionToFirestore(summary, perf);

        toast({
          title: "Squat Session Saved",
          description: `${count} reps in ${durationSec}s. Score: ${calculatedScore}/100 | AI-5: ${perf.score}/100`,
        });
      } else {
        toast({ title: "Session Discarded", description: "Not enough activity to save.", variant: "default" });
      }

    } else {
      // Face Mode Summary
      const { totalScore, frameCount, maxHoldMs } = faceAccRef.current;
      const finalMaxHold = Math.max(maxHoldMs, faceLogicRef.current.holdMs);

      const avgScore = frameCount > 0 ? totalScore / frameCount : 0;
      const bestHoldSec = parseFloat((finalMaxHold / 1000).toFixed(1));

      summary = {
        exerciseType: "face",
        timestamp,
        durationSec,
        avgPostureScore: parseFloat(avgScore.toFixed(0)),
        bestHoldSec,
        overallQuality: parseFloat(avgScore.toFixed(0))
      };

      if (durationSec > 5) {
        // Calculate performance score
        const perf = computePerformanceScore(summary);
        setLastPerformance(perf);

        setSessionHistory(prev => [summary as SessionSummary, ...prev]);
        setLastSession(summary);

        // ⭐ SAVE TO FIRESTORE
        saveSessionToFirestore(summary, perf);

        toast({
          title: "Posture Session Saved",
          description: `Avg Score: ${avgScore.toFixed(0)}. Best Hold: ${bestHoldSec}s | AI-5: ${perf.score}/100`,
        });
      } else {
        toast({ title: "Session Discarded", description: "Too short to save.", variant: "default" });
      }
    }
  };

  const toggleRecording = useCallback(() => {
    if (!cameraActive) return toast({ title: "Camera Required", variant: "destructive" });

    if (isRecordingRef.current) {
      stopSession();
    } else {
      startSession();
    }
  }, [cameraActive]);

  // --- LOGIC ENGINE 1: SQUATS (POSE) ---
  const onPoseResults = useCallback((results: PoseResults) => {
    if (modeRef.current !== "squat") return;

    drawFrame(results.image, () => {
      if (!results.poseLandmarks || !canvasRef.current) return;
      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      const landmarks = results.poseLandmarks;
      const shoulder = landmarks[11];
      const hip = landmarks[23];
      const knee = landmarks[25];
      const ankle = landmarks[27];

      // Drawing
      ctx.lineWidth = 5;
      ctx.strokeStyle = "#ffffff";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(shoulder.x * canvasRef.current!.width, shoulder.y * canvasRef.current!.height);
      ctx.lineTo(hip.x * canvasRef.current!.width, hip.y * canvasRef.current!.height);
      ctx.lineTo(knee.x * canvasRef.current!.width, knee.y * canvasRef.current!.height);
      ctx.lineTo(ankle.x * canvasRef.current!.width, ankle.y * canvasRef.current!.height);
      ctx.stroke();

      [shoulder, hip, knee, ankle].forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x * canvasRef.current!.width, p.y * canvasRef.current!.height, 6, 0, 2 * Math.PI);
        ctx.fillStyle = "#00ff88";
        ctx.fill();
      });

      // Logic
      const kneeAngle = calculateAngle(hip, knee, ankle);
      const hipAngle = calculateAngle(shoulder, hip, knee);

      // 1. Live Logic (for Feedback)
      squatLogicRef.current = updateSquatLogic(squatLogicRef.current, kneeAngle, hipAngle);

      // 2. Session Accumulation (if recording)
      if (isRecordingRef.current) {
        squatAccRef.current.totalKneeAngle += kneeAngle;
        squatAccRef.current.totalBackAngle += hipAngle;
        squatAccRef.current.frameCount++;
      }

      // 3. UI Updates (Throttled)
      const now = Date.now();
      if (now - lastUiUpdateRef.current > 100) {
        setSquatState({
          ...squatLogicRef.current,
          currentKneeAngle: Math.round(kneeAngle),
          currentHipAngle: Math.round(hipAngle)
        });
        lastUiUpdateRef.current = now;
      }
    });
  }, []);

  // --- LOGIC ENGINE 2: FACE / POSTURE (FACEMESH) ---
  const onFaceResults = useCallback((results: FaceResults) => {
    if (modeRef.current !== "face") return;

    drawFrame(results.image, () => {
      if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0 || !canvasRef.current) return;

      const landmarks = results.multiFaceLandmarks[0];
      const { yaw, pitch, roll } = calculateHeadPose(landmarks);
      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      // Draw Axis
      const nose = landmarks[1];
      const nx = nose.x * canvasRef.current.width;
      const ny = nose.y * canvasRef.current.height;

      ctx.lineWidth = 4;
      ctx.strokeStyle = "#4ade80";
      ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(nx, ny - (pitch * 3)); ctx.stroke();
      ctx.strokeStyle = "#60a5fa";
      ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(nx + (yaw * 3), ny); ctx.stroke();
      ctx.strokeStyle = "#f87171";
      ctx.beginPath(); ctx.moveTo(nx, ny); ctx.lineTo(nx + (roll * 2), ny + (roll * 2)); ctx.stroke();

      // Logic
      const totalDeviation = Math.abs(yaw) + Math.abs(pitch) + Math.abs(roll);
      const score = Math.max(0, 100 - totalDeviation);

      let feedback = "Perfect Posture";
      if (Math.abs(yaw) > 15) feedback = yaw > 0 ? "Turn Head Left" : "Turn Head Right";
      else if (Math.abs(pitch) > 15) feedback = pitch > 0 ? "Look Down" : "Look Up";
      else if (Math.abs(roll) > 10) feedback = "Straighten Head";

      // Session Logic
      let currentHold = 0;
      if (isRecordingRef.current) {
        const frameMs = 1000 / 30;
        const isGood = score >= 85;

        // Update hold timer
        currentHold = isGood ? Math.min(faceLogicRef.current.holdMs + frameMs, 10000) : 0;

        // Update Accumulators
        faceAccRef.current.totalScore += score;
        faceAccRef.current.frameCount++;
        if (currentHold > faceAccRef.current.maxHoldMs) {
          faceAccRef.current.maxHoldMs = currentHold;
        }

        faceLogicRef.current = {
          yaw: Math.round(yaw),
          pitch: Math.round(pitch),
          roll: Math.round(roll),
          score: Math.round(score),
          feedback,
          isGoodPosture: isGood,
          holdMs: currentHold
        };
      } else {
        // Just monitoring
        faceLogicRef.current = {
          ...faceLogicRef.current,
          yaw: Math.round(yaw),
          pitch: Math.round(pitch),
          roll: Math.round(roll),
          score: Math.round(score),
          feedback
        };
      }

      // UI Updates (Throttled)
      const now = Date.now();
      if (now - lastUiUpdateRef.current > 100) {
        setFaceState(faceLogicRef.current);
        lastUiUpdateRef.current = now;
      }
    });
  }, []);

  const drawFrame = (image: any, callback: () => void) => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-canvasRef.current.width, 0);
    ctx.drawImage(image, 0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.restore();

    callback();
  };

  // --- CAMERA CONTROLS ---
  const startCamera = async () => {
    if (!videoRef.current) return;
    setCameraLoading(true);
    try {
      if (cameraRef.current) await cameraRef.current.stop();

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (!videoRef.current) return;
          if (modeRef.current === "squat" && poseModelRef.current) {
            await poseModelRef.current.send({ image: videoRef.current });
          } else if (modeRef.current === "face" && faceModelRef.current) {
            await faceModelRef.current.send({ image: videoRef.current });
          }
        },
        width: 1280, height: 720,
      });
      await camera.start();
      cameraRef.current = camera;
      setCameraActive(true);
    } catch (err) {
      console.error(err);
      toast({ title: "Error", description: "Camera access failed.", variant: "destructive" });
    } finally {
      setCameraLoading(false);
    }
  };

  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }
    setCameraActive(false);
    setIsRecording(false);
  };

  const toggleCamera = useCallback(() => {
    if (cameraActive) stopCamera();
    else startCamera();
  }, [cameraActive]);

  const ModeBadge = ({ active, label, icon: Icon, onClick }: any) => (
    <button
      onClick={onClick}
      disabled={isRecording}
      className={cn(
        "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
        active
          ? "bg-primary text-primary-foreground shadow-md ring-2 ring-primary/20"
          : "hover:bg-accent text-muted-foreground",
        isRecording && "opacity-50 cursor-not-allowed"
      )}
    >
      <Icon className="w-4 h-4" />
      {label}
    </button>
  );

  const holdSeconds = Math.min(5, faceState.holdMs / 1000);

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6 max-w-6xl mx-auto pb-10">

      {/* HEADER */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Activity className="w-8 h-8 text-primary" />
            AI Coach <span className="text-xs font-mono bg-accent px-2 py-0.5 rounded text-muted-foreground">PRO</span>
          </h1>
          <p className="text-muted-foreground">Real-time biomechanics analysis with AI-5 Performance Score</p>
        </div>

        <div className="flex bg-card border p-1 rounded-xl shadow-sm">
          <ModeBadge
            active={mode === "squat"}
            label="Body Squat"
            icon={Dumbbell}
            onClick={() => setMode("squat")}
          />
          <div className="w-px bg-border mx-1 my-1" />
          <ModeBadge
            active={mode === "face"}
            label="Neck & Posture"
            icon={ScanFace}
            onClick={() => setMode("face")}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* LEFT: CAMERA FEED */}
        <div className="lg:col-span-2 space-y-4">
          <div className="relative aspect-video rounded-2xl overflow-hidden border border-border bg-black shadow-lg">
            <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover opacity-0" playsInline muted />
            <canvas ref={canvasRef} className={cn("absolute inset-0 w-full h-full object-cover", !cameraActive && "hidden")} width={1280} height={720} />

            {/* Overlay: Not Active */}
            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-900/90 text-center p-6 space-y-4">
                <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center">
                  {cameraLoading ? <Activity className="w-8 h-8 animate-pulse text-primary" /> : <Video className="w-8 h-8 text-muted-foreground" />}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Camera Offline</h3>
                  <p className="text-sm text-zinc-400">Enable camera to start tracking {mode}</p>
                </div>
              </div>
            )}

            {/* Overlay: Recording Indicator */}
            <AnimatePresence>
              {isRecording && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="absolute top-4 left-4 flex items-center gap-2 bg-red-500/90 text-white px-3 py-1.5 rounded-full backdrop-blur-sm shadow-lg z-10"
                >
                  <div className="w-2 h-2 rounded-full bg-white animate-pulse" />
                  <span className="text-xs font-bold uppercase tracking-wider">REC</span>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Overlay: Mode Hint */}
            {cameraActive && !isRecording && (
              <div className="absolute bottom-4 left-4 right-4 text-center">
                <span className="bg-black/60 text-white/90 text-sm px-4 py-2 rounded-full backdrop-blur-md border border-white/10">
                  {mode === 'squat' ? "Stand sideways for best angle detection" : "Face the camera directly"}
                </span>
              </div>
            )}
          </div>

          {/* CONTROLS (Memoized) */}
          <MemoizedControls
            isRecording={isRecording}
            cameraActive={cameraActive}
            onToggleRecording={toggleRecording}
            onToggleCamera={toggleCamera}
            mode={mode}
            cameraLoading={cameraLoading}
          />
        </div>

        {/* RIGHT: ANALYTICS PANEL */}
        <div className="space-y-4">

          {/* SQUAT ANALYTICS */}
          {mode === "squat" && (
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">

              {/* Main Score Card */}
              <div className="bg-card border rounded-2xl p-6 shadow-sm text-center relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-purple-500" />
                <h3 className="text-muted-foreground text-sm font-medium uppercase tracking-wider mb-2">Reps Completed</h3>
                <div className="text-7xl font-black text-foreground tracking-tighter">
                  {squatState.count}
                </div>

                {/* Visual Feedback Badge */}
                <div className={cn("inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase mt-4 transition-colors duration-300",
                  squatState.lastFeedback.includes("Good") || squatState.lastFeedback.includes("Rep")
                    ? "bg-green-500/10 text-green-600"
                    : squatState.lastFeedback === "Ready" ? "bg-zinc-100 text-zinc-500" : "bg-red-500/10 text-red-600"
                )}>
                  {squatState.lastFeedback}
                </div>
              </div>

              {/* Progress Bar */}
              <div className="bg-card border rounded-xl p-4">
                <div className="flex justify-between text-xs text-muted-foreground mb-2">
                  <span>Standing</span>
                  <span>Deep Squat</span>
                </div>
                <Progress value={squatState.repProgress} className="h-3" />
              </div>

              {/* Angle Stats */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-card border rounded-xl p-4 flex flex-col items-center">
                  <span className="text-xs text-muted-foreground mb-1">Knee Angle</span>
                  <span className="text-2xl font-mono font-bold">{squatState.currentKneeAngle}°</span>
                </div>
                <div className="bg-card border rounded-xl p-4 flex flex-col items-center">
                  <span className="text-xs text-muted-foreground mb-1">Back Angle</span>
                  <div className={cn("text-2xl font-mono font-bold transition-colors duration-200", squatState.currentHipAngle < 70 && "text-red-500")}>
                    {squatState.currentHipAngle}°
                  </div>
                </div>
              </div>

              {/* Form Issues */}
              {squatState.badFormCount > 0 && (
                <div className="bg-red-50 dark:bg-red-950/20 border border-red-100 dark:border-red-900/30 rounded-xl p-4 flex gap-3">
                  <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
                  <div>
                    <h4 className="text-sm font-semibold text-red-700 dark:text-red-400">Form Alerts</h4>
                    <p className="text-xs text-red-600/80 dark:text-red-400/80 mt-0.5">
                      Detected {squatState.badFormCount} depth issues and {squatState.forwardLeanCount} lean issues.
                    </p>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* FACE ANALYTICS */}
          {mode === "face" && (
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">

              {/* Posture Score */}
              <div className="bg-card border rounded-2xl p-6 shadow-sm text-center relative overflow-hidden">
                <div className={cn("absolute top-0 left-0 w-full h-1 transition-colors duration-500", faceState.isGoodPosture ? "bg-green-500" : "bg-orange-500")} />
                <h3 className="text-muted-foreground text-sm font-medium uppercase tracking-wider mb-4">Posture Score</h3>

                <div className="relative w-32 h-32 mx-auto flex items-center justify-center">
                  {/* Ring */}
                  <svg className="w-full h-full transform -rotate-90">
                    <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="8" fill="transparent" className="text-muted/20" />
                    <circle cx="64" cy="64" r="56" stroke="currentColor" strokeWidth="8" fill="transparent"
                      strokeDasharray={351}
                      strokeDashoffset={351 - (351 * faceState.score) / 100}
                      className={cn("transition-all duration-300", faceState.isGoodPosture ? "text-green-500" : "text-orange-500")}
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-3xl font-bold">{faceState.score}</span>
                  </div>
                </div>

                {/* Feedback Message */}
                <p className="text-lg font-medium mt-4 min-h-[1.75rem]">
                  {faceState.holdMs >= 5000
                    ? "Excellent! Holding..."
                    : faceState.feedback}
                </p>

                {/* Hold Timer */}
                <p className="text-sm text-muted-foreground mt-2">
                  Hold target posture: {holdSeconds.toFixed(1)} / 5.0 seconds
                </p>
              </div>

              {/* 3D Stats */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: "Yaw", val: faceState.yaw, icon: MoveHorizontal, color: "text-blue-500" },
                  { label: "Pitch", val: faceState.pitch, icon: MoveVertical, color: "text-green-500" },
                  { label: "Roll", val: faceState.roll, icon: RotateCw, color: "text-red-500" },
                ].map((stat) => (
                  <div key={stat.label} className="bg-card border rounded-xl p-3 flex flex-col items-center justify-center">
                    <stat.icon className={cn("w-4 h-4 mb-1", stat.color)} />
                    <span className="text-[10px] uppercase text-muted-foreground">{stat.label}</span>
                    <span className="text-lg font-mono font-bold">{stat.val}°</span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* SESSION HISTORY / LAST SESSION SUMMARY */}
          {(lastSession || lastPerformance) && (
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="bg-slate-900/50 border border-slate-800 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <History className="w-4 h-4 text-primary" />
                <h4 className="text-sm font-semibold text-white">Last Session Summary</h4>
              </div>

              {lastSession?.exerciseType === "squat" ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="bg-slate-800/50 p-2 rounded">
                      <span className="block text-xs text-muted-foreground">Reps</span>
                      <span className="font-mono font-bold text-white">{lastSession.reps}</span>
                    </div>
                    <div className="bg-slate-800/50 p-2 rounded">
                      <span className="block text-xs text-muted-foreground">Form Score</span>
                      <span className={cn("font-mono font-bold", lastSession.formScore > 80 ? "text-green-400" : "text-orange-400")}>
                        {lastSession.formScore}%
                      </span>
                    </div>
                    <div className="col-span-2 bg-slate-800/50 p-2 rounded flex justify-between">
                      <span className="text-xs text-muted-foreground">Duration</span>
                      <span className="text-xs font-mono text-white">{lastSession.durationSec}s</span>
                    </div>
                  </div>

                  {/* AI-5 Performance Score */}
                  {lastPerformance && (
                    <div className="pt-2 border-t border-slate-700/50">
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="w-4 h-4 text-purple-400" />
                        <span className="text-xs font-semibold text-purple-300">AI-5 Performance Score</span>
                      </div>
                      <div className="bg-slate-800/70 p-3 rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-sm font-bold ${lastPerformance.color}`}>
                            {lastPerformance.label}
                          </span>
                          <span className={`text-lg font-black ${lastPerformance.color}`}>
                            {lastPerformance.score}/100
                          </span>
                        </div>
                        <p className="text-xs text-slate-300 mt-2">
                          {lastPerformance.explanation}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ) : lastSession?.exerciseType === "face" ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="bg-slate-800/50 p-2 rounded">
                      <span className="block text-xs text-muted-foreground">Avg Score</span>
                      <span className="font-mono font-bold text-white">{lastSession.avgPostureScore}</span>
                    </div>
                    <div className="bg-slate-800/50 p-2 rounded">
                      <span className="block text-xs text-muted-foreground">Best Hold</span>
                      <span className="font-mono font-bold text-green-400">{lastSession.bestHoldSec}s</span>
                    </div>
                  </div>

                  {/* AI-5 Performance Score */}
                  {lastPerformance && (
                    <div className="pt-2 border-t border-slate-700/50">
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="w-4 h-4 text-purple-400" />
                        <span className="text-xs font-semibold text-purple-300">AI-5 Performance Score</span>
                      </div>
                      <div className="bg-slate-800/70 p-3 rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-sm font-bold ${lastPerformance.color}`}>
                            {lastPerformance.label}
                          </span>
                          <span className={`text-lg font-black ${lastPerformance.color}`}>
                            {lastPerformance.score}/100
                          </span>
                        </div>
                        <p className="text-xs text-slate-300 mt-2">
                          {lastPerformance.explanation}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              ) : null}
            </motion.div>
          )}

          {/* PERFORMANCE INSIGHTS */}
          {lastPerformance && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-800/30 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-3">
                <Target className="w-4 h-4 text-purple-400" />
                <h4 className="text-sm font-semibold text-purple-300">AI-5 Performance Insights</h4>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-300">Performance Level</span>
                  <span className={`text-xs font-bold px-2 py-1 rounded-full ${lastPerformance.color} ${lastPerformance.color.includes('green') ? 'bg-green-500/10' : lastPerformance.color.includes('yellow') ? 'bg-yellow-500/10' : 'bg-red-500/10'}`}>
                    {lastPerformance.label}
                  </span>
                </div>
                <div className="w-full bg-slate-800/50 rounded-full h-2">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${lastPerformance.color}`}
                    style={{ width: `${lastPerformance.score}%` }}
                  />
                </div>
                <p className="text-xs text-slate-400 mt-2">
                  AI-5 analyzes {mode === "squat" ? "rep quality, form consistency, and endurance" : "posture stability, hold duration, and alignment precision"}.
                </p>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </motion.div>
  );
}