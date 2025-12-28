import { auth, db } from "@/lib/firebase";
import { collection, addDoc } from "firebase/firestore";


import { useState, useRef } from "react";
import { motion } from "framer-motion";
import {
  Camera,
  Upload,
  X,
  Check,
  Loader2,
  ChevronRight,
  AlertCircle,
  Edit2,
  Save
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";

// Types for the backend API response
interface NutritionInfo {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  serving_size: string;
}

interface FoodPrediction {
  food: string;
  confidence: number;
  nutrition?: NutritionInfo;
}

export default function FoodCapture() {
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [fileForUpload, setFileForUpload] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showLowConfidenceWarning, setShowLowConfidenceWarning] = useState(false);
  
  // Prediction State
  const [prediction, setPrediction] = useState<FoodPrediction | null>(null);

  // Editable Name State
  const [foodName, setFoodName] = useState<string>("");

  const [showManualSelect, setShowManualSelect] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  /* ---------------- BACKEND API CALL ---------------- */
  const predictFood = async (file: File): Promise<FoodPrediction> => {
    const formData = new FormData();
    formData.append("file", file); // Must be "file" to match backend

    try {
      const response = await fetch("http://localhost:8002/predict-food", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Prediction error:", error);
      throw new Error("Failed to analyze food. Please try again.");
    }
  };
const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
  const file = e.target.files?.[0];
  const formData = new FormData();
  formData.append('file', file); // ← CORRECT KEY NAME
  
  const res = await fetch('http://localhost:8002/predict-food', {
    method: 'POST',
    body: formData,
  });
  const data: FoodPrediction = await res.json();
  setPrediction(data);
}

  /* ---------------- CAMERA ---------------- */
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        setError(null);
      }
    } catch (err) {
      toast({
        title: "Camera Error",
        description: "Unable to access camera. Please check permissions.",
        variant: "destructive",
      });
      setError("Camera access denied. Please check permissions.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) return;
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        setFileForUpload(file);
        const dataUrl = canvas.toDataURL("image/jpeg");
        setCapturedImage(dataUrl);
        stopCamera();
        handleAnalyze(file);
      }, "image/jpeg");
    }
  };

  /* ---------------- UPLOAD IMAGE ---------------- */
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Check file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid File",
        description: "Please upload an image file (JPEG, PNG, etc.)",
        variant: "destructive",
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      setCapturedImage(event.target?.result as string);
      setFileForUpload(file);
      handleAnalyze(file);
    };
    reader.readAsDataURL(file);
  };


  /* ---------------- REAL AI ANALYSIS ---------------- */
  async function handleAnalyze(file: File) {
    try {
      setIsAnalyzing(true);
      setPrediction(null);
      setError(null);
      setShowLowConfidenceWarning(false);
      setFoodName("");

      const result: FoodPrediction = await predictFood(file);

      // Check confidence level
      if (result.confidence < 0.6 || result.food === "unknown") {
        setShowLowConfidenceWarning(true);
        toast({
          title: "Low Confidence",
          description: "Food detection had low confidence. Please retake the photo or select manually.",
          variant: "default",
        });
      }

      setPrediction(result);
      setFoodName(result.food);

    } catch (err: any) {
      console.error("Analysis error:", err);
      toast({
        title: "Prediction Failed",
        description: err.message || "Could not analyze image. Please try again.",
        variant: "destructive",
      });
      setError("Failed to analyze food. Please try with a clearer image.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  /* ---------------- UI ACTIONS ---------------- */
  const handleSaveToLog = async () => {
    if (!prediction || !foodName.trim()) {
      toast({
        title: "Cannot Save",
        description: "Please enter a food name first.",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsSaving(true);

      // Construct the food log entry
      const foodLogEntry = {
        name: foodName.trim(),
        calories: prediction.nutrition?.calories || 0,
        macros: {
          protein: prediction.nutrition?.protein || 0,
          carbs: prediction.nutrition?.carbs || 0,
          fat: prediction.nutrition?.fat || 0,
        },
        serving_size: prediction.nutrition?.serving_size || "1 serving",
        date: new Date().toISOString(),
        confidence: prediction.confidence,
      };

      console.log("Saving to DB:", foodLogEntry);

      // TODO: Replace with real Firebase/Backend call
// --- REAL FIRESTORE SAVE ---
const uid = auth.currentUser?.uid;

if (!uid) {
  toast({
    title: "Not Logged In",
    description: "Please log in to save food logs.",
    variant: "destructive",
  });
  return;
}

await addDoc(collection(db, `users/${uid}/meals`), foodLogEntry);


      toast({
        title: "Food Logged!",
        description: `${foodName} has been added to your daily intake.`,
      });

      resetCapture();

    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save food log.",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const selectManualFood = (food: any) => {
    // When manual selecting, create a prediction object with the selected food
    const manualPrediction: FoodPrediction = {
      food: food.name,
      confidence: 1.0, // Manual selection has 100% confidence
      nutrition: {
        calories: food.calories,
        protein: food.protein,
        carbs: food.carbs,
        fat: food.fat,
        serving_size: "1 serving"
      }
    };
    
    setPrediction(manualPrediction);
    setFoodName(food.name);
    setShowManualSelect(false);
    setShowLowConfidenceWarning(false);
  };

  const resetCapture = () => {
    setCapturedImage(null);
    setPrediction(null);
    setFoodName("");
    setShowManualSelect(false);
    setShowLowConfidenceWarning(false);
    setFileForUpload(null);
    setError(null);
    stopCamera();
  };

  /* ---------------- MANUAL FOOD DATABASE ---------------- */
  const foodDatabase = [
    { id: "dosa", name: "Dosa", calories: 168, protein: 4, carbs: 28, fat: 5 },
    { id: "idli", name: "Idli", calories: 39, protein: 2, carbs: 8, fat: 0 },
    { id: "rice", name: "Rice (1 cup)", calories: 206, protein: 4, carbs: 45, fat: 0 },
    { id: "chapati", name: "Chapati", calories: 104, protein: 3, carbs: 18, fat: 3 },
    { id: "dal", name: "Dal (1 cup)", calories: 198, protein: 12, carbs: 34, fat: 1 },
    { id: "sambar", name: "Sambar", calories: 139, protein: 7, carbs: 21, fat: 3 },
    { id: "chicken", name: "Chicken Curry", calories: 243, protein: 25, carbs: 6, fat: 14 },
    { id: "biryani", name: "Chicken Biryani", calories: 290, protein: 15, carbs: 35, fat: 10 },
  ];

  /* ---------------- RENDER ---------------- */
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold md:text-3xl">Food Scanner</h1>
        <p className="text-muted-foreground">Capture or upload food photos for instant nutrition info</p>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
          <p className="text-destructive text-sm font-medium">{error}</p>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* CAMERA / UPLOAD SECTION */}
        <div className="rounded-xl border bg-card p-5">
          <h2 className="mb-4 text-lg font-semibold">Capture Food</h2>

          {!capturedImage ? (
            <div className="space-y-4">
              <div className="relative aspect-[4/3] overflow-hidden rounded-lg border bg-muted/20">
                {cameraActive ? (
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    className="h-full w-full object-cover" 
                  />
                ) : (
                  <div className="flex h-full flex-col items-center justify-center gap-4">
                    <div className="rounded-full bg-muted p-6">
                      <Camera className="h-10 w-10 text-muted-foreground" />
                    </div>
                    <p className="text-sm text-muted-foreground">No camera active</p>
                  </div>
                )}
              </div>

              <div className="flex flex-wrap justify-center gap-3">
                {cameraActive ? (
                  <>
                    <Button onClick={capturePhoto} className="gradient-primary text-primary-foreground">
                      <Camera className="mr-2 h-4 w-4" /> Capture
                    </Button>
                    <Button variant="outline" onClick={stopCamera}>
                      <X className="mr-2 h-4 w-4" /> Cancel
                    </Button>
                  </>
                ) : (
                  <>
                    <Button onClick={startCamera} className="gradient-primary text-primary-foreground">
                      <Camera className="mr-2 h-4 w-4" /> Use Camera
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="mr-2 h-4 w-4" /> Upload Image
                    </Button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="relative aspect-[4/3] overflow-hidden rounded-lg">
                <img src={capturedImage} alt="Captured food" className="h-full w-full object-cover" />
                {isAnalyzing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-background/80">
                    <div className="flex flex-col items-center gap-2">
                      <Loader2 className="h-10 w-10 animate-spin text-primary" />
                      <p className="text-sm font-medium">Analyzing food...</p>
                      <p className="text-xs text-muted-foreground">Connecting to AI model...</p>
                    </div>
                  </div>
                )}
              </div>

              <Button 
                variant="outline" 
                onClick={resetCapture} 
                className="w-full" 
                disabled={isSaving || isAnalyzing}
              >
                <X className="mr-2 h-4 w-4" /> Take New Photo
              </Button>
            </div>
          )}
        </div>

        {/* RESULTS SECTION */}
        <div className="space-y-4">
          {prediction ? (
            <>
              {showLowConfidenceWarning && (
                <motion.div 
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="rounded-lg border border-amber-500/50 bg-amber-50 p-4 dark:bg-amber-950/20"
                >
                  <div className="flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-amber-600 dark:text-amber-400 mt-0.5" />
                    <div className="flex-1">
                      <p className="font-medium text-amber-800 dark:text-amber-300">
                        Low Confidence Detection
                      </p>
                      <p className="text-sm text-amber-700 dark:text-amber-400 mt-1">
                        The AI wasn't very confident about this detection. Please verify the food name or select manually.
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}

              <motion.div 
                initial={{ opacity: 0, y: 20 }} 
                animate={{ opacity: 1, y: 0 }} 
                className="rounded-xl border bg-card p-5"
              >
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex-1 mr-4">
                    <label className="text-xs font-medium text-muted-foreground mb-1 block">
                      Detected Name (Edit if needed)
                    </label>
                    <div className="relative">
                      <input 
                        value={foodName}
                        onChange={(e) => setFoodName(e.target.value)}
                        className="w-full bg-transparent text-2xl font-bold border-b border-dashed border-muted-foreground/30 py-1 pr-8 focus:outline-none focus:border-primary transition-colors"
                        placeholder="Enter food name..."
                      />
                      <Edit2 className="absolute right-0 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" />
                    </div>
                    
                    <div className="mt-2 flex items-center gap-2">
                      <div className="h-2 w-2 rounded-full bg-primary"></div>
                      <p className="text-sm text-muted-foreground">
                        Confidence: {(prediction.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  <div className="text-right whitespace-nowrap">
                    <p className="text-3xl font-bold text-primary">
                      {prediction.nutrition?.calories ?? "--"}
                    </p>
                    <p className="text-sm text-muted-foreground">kcal</p>
                    {prediction.nutrition?.serving_size && (
                      <p className="text-xs text-muted-foreground mt-1">
                        {prediction.nutrition.serving_size}
                      </p>
                    )}
                  </div>
                </div>

                <Progress value={prediction.confidence * 100} className="mb-4 h-2" />

                {/* MACROS */}
                <div className="mb-6 grid grid-cols-3 gap-3">
                  <MacroCard label="Protein" value={prediction.nutrition?.protein} unit="g" />
                  <MacroCard label="Carbs" value={prediction.nutrition?.carbs} unit="g" />
                  <MacroCard label="Fat" value={prediction.nutrition?.fat} unit="g" />
                </div>

                <div className="flex flex-col gap-3 sm:flex-row">
                  <Button 
                    onClick={handleSaveToLog} 
                    disabled={isSaving || !foodName.trim()}
                    className="flex-1 gradient-primary text-primary-foreground"
                  >
                    {isSaving ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Saving...
                      </>
                    ) : (
                      <>
                        <Save className="mr-2 h-4 w-4" /> Save to Log
                      </>
                    )}
                  </Button>
                  
                  <Button 
                    variant="outline" 
                    onClick={() => setShowManualSelect(!showManualSelect)}
                    disabled={isSaving}
                  >
                    <AlertCircle className="mr-2 h-4 w-4" /> Wrong Food?
                  </Button>
                </div>
              </motion.div>

              {/* MANUAL SELECTION DROPDOWN */}
              {showManualSelect && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }} 
                  animate={{ opacity: 1, height: "auto" }} 
                  className="rounded-xl border bg-card p-5 overflow-hidden"
                >
                  <h3 className="mb-3 font-semibold">Select Correct Food</h3>
                  <div className="max-h-60 space-y-2 overflow-auto pr-2">
                    {foodDatabase.map((food) => (
                      <button
                        key={food.id}
                        onClick={() => selectManualFood(food)}
                        className="flex w-full items-center justify-between rounded-lg border bg-secondary/30 p-3 text-left hover:bg-secondary/50 transition-colors"
                      >
                        <div>
                          <p className="font-medium">{food.name}</p>
                          <div className="flex items-center gap-3 mt-1">
                            <span className="text-sm text-muted-foreground">{food.calories} kcal</span>
                            <span className="text-xs text-muted-foreground">
                              P:{food.protein}g C:{food.carbs}g F:{food.fat}g
                            </span>
                          </div>
                        </div>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}
            </>
          ) : (
            <div className="rounded-xl border bg-card p-8 text-center flex flex-col items-center justify-center min-h-[300px]">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                <Camera className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="mb-2 font-semibold">No Food Detected</h3>
              <p className="text-sm text-muted-foreground max-w-[250px] mx-auto">
                Capture or upload a photo of your food to see nutrition information
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

/* ---------------- SMALL UI COMPONENT ---------------- */
function MacroCard({ label, value, unit = "g" }: { label: string; value?: number; unit?: string }) {
  return (
    <div className="rounded-lg bg-secondary/50 p-3 text-center">
      <p className="text-lg font-bold">
        {value !== undefined ? `${value}${unit}` : "--"}
      </p>
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  );
}