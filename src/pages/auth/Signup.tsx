import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, Link } from "react-router-dom";
import {
  Eye,
  EyeOff,
  Flame,
  Mail,
  Lock,
  User,
  ArrowRight,
  ArrowLeft,
  Check,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";

// ⭐ Firebase
import { auth, db } from "@/lib/firebase";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";

const steps = [
  { id: 1, title: "Account", description: "Create your credentials" },
  { id: 2, title: "Profile", description: "Tell us about yourself" },
  { id: 3, title: "Goals", description: "Set your fitness goals" },
];

export default function Signup() {
  const [currentStep, setCurrentStep] = useState(1);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    name: "",
    age: "",
    gender: "",
    height: "",
    weight: "",
    goal: "",
    activityLevel: "",
  });

  const updateFormData = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleNext = () => {
    if (currentStep < 3) setCurrentStep(currentStep + 1);
  };

  const handleBack = () => {
    if (currentStep > 1) setCurrentStep(currentStep - 1);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      toast({ title: "Passwords do not match ⚠️", variant: "destructive" });
      return;
    }

    if (!formData.goal || !formData.activityLevel) {
      toast({ title: "Please select your goal & activity level", variant: "destructive" });
      return;
    }

    setIsLoading(true);

    try {
      // 🔥 Create account
      const userCred = await createUserWithEmailAndPassword(
        auth,
        formData.email.trim(),
        formData.password
      );

      const uid = userCred.user.uid;

      // 📝 Save profile in Firestore
      await setDoc(doc(db, "users", uid, "profile", "info"), {
        name: formData.name,
        age: Number(formData.age),
        gender: formData.gender,
        height: Number(formData.height),
        weight: Number(formData.weight),
        goal: formData.goal,
        activityLevel: formData.activityLevel,
        email: formData.email,
        createdAt: new Date().toISOString(),
      });

      toast({
        title: "Account created! 🎉",
        description: "Welcome to your fitness journey",
      });

      navigate("/dashboard");
    } catch (error: any) {
      console.error("Signup error:", error);
      toast({
        title: "Signup failed",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background p-4">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -left-1/4 -top-1/4 h-[600px] w-[600px] rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute -bottom-1/4 -right-1/4 h-[600px] w-[600px] rounded-full bg-accent/10 blur-3xl" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 w-full max-w-lg"
      >
        <div className="mb-6 flex flex-col items-center">
          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl gradient-primary glow-primary">
            <Flame className="h-7 w-7 text-primary-foreground" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">Create Account</h1>
          <p className="mt-1 text-sm text-muted-foreground">Start your fitness journey today</p>
        </div>

        {/* Progress dots */}
        <div className="mb-6 flex items-center justify-center gap-2">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <div
                className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-medium transition-all ${
                  currentStep > step.id
                    ? "bg-primary text-primary-foreground"
                    : currentStep === step.id
                    ? "bg-primary/20 text-primary ring-2 ring-primary"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {currentStep > step.id ? <Check className="h-4 w-4" /> : step.id}
              </div>
              {index < steps.length - 1 && (
                <div
                  className={`mx-2 h-0.5 w-12 transition-all ${
                    currentStep > step.id ? "bg-primary" : "bg-muted"
                  }`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Main form */}
        <div className="rounded-2xl border border-border bg-card/50 p-6 shadow-card backdrop-blur-sm">
          <form onSubmit={handleSubmit}>
            <AnimatePresence mode="wait">
              {/* --- STEP 1 --- */}
              {currentStep === 1 && (
                <motion.div
                  key="step1"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  <h2 className="text-lg font-semibold text-foreground">Account Details</h2>
                  <div className="space-y-2">
                    <Label>Email</Label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        type="email"
                        placeholder="you@example.com"
                        value={formData.email}
                        onChange={(e) => updateFormData("email", e.target.value)}
                        className="pl-10 bg-secondary/50 border-transparent focus:border-primary"
                        required
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Password</Label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        type={showPassword ? "text" : "password"}
                        placeholder="••••••••"
                        value={formData.password}
                        onChange={(e) => updateFormData("password", e.target.value)}
                        className="pl-10 pr-10 bg-secondary/50 border-transparent focus:border-primary"
                        required
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      >
                        {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Confirm Password</Label>
                    <Input
                      type="password"
                      placeholder="••••••••"
                      value={formData.confirmPassword}
                      onChange={(e) => updateFormData("confirmPassword", e.target.value)}
                      className="bg-secondary/50 border-transparent focus:border-primary"
                      required
                    />
                  </div>
                </motion.div>
              )}

              {/* --- STEP 2 --- */}
              {currentStep === 2 && (
                <motion.div
                  key="step2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  <h2 className="text-lg font-semibold text-foreground">Your Profile</h2>

                  <div className="space-y-2">
                    <Label>Full Name</Label>
                    <div className="relative">
                      <User className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        placeholder="John Doe"
                        value={formData.name}
                        onChange={(e) => updateFormData("name", e.target.value)}
                        className="pl-10 bg-secondary/50 border-transparent focus:border-primary"
                        required
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Age</Label>
                      <Input
                        type="number"
                        placeholder="25"
                        value={formData.age}
                        onChange={(e) => updateFormData("age", e.target.value)}
                        className="bg-secondary/50 border-transparent focus:border-primary"
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Gender</Label>
                      <Select
                        value={formData.gender}
                        onValueChange={(value) => updateFormData("gender", value)}
                      >
                        <SelectTrigger className="bg-secondary/50 border-transparent focus:border-primary">
                          <SelectValue placeholder="Select" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="male">Male</SelectItem>
                          <SelectItem value="female">Female</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Height (cm)</Label>
                      <Input
                        type="number"
                        placeholder="175"
                        value={formData.height}
                        onChange={(e) => updateFormData("height", e.target.value)}
                        className="bg-secondary/50 border-transparent focus:border-primary"
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label>Weight (kg)</Label>
                      <Input
                        type="number"
                        placeholder="70"
                        value={formData.weight}
                        onChange={(e) => updateFormData("weight", e.target.value)}
                        className="bg-secondary/50 border-transparent focus:border-primary"
                        required
                      />
                    </div>
                  </div>
                </motion.div>
              )}

              {/* --- STEP 3 --- */}
              {currentStep === 3 && (
                <motion.div
                  key="step3"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="space-y-4"
                >
                  <h2 className="text-lg font-semibold text-foreground">Your Goals</h2>

                  <div className="space-y-2">
                    <Label>What's your goal?</Label>
                    <div className="grid grid-cols-3 gap-3">
                      {["Lose Weight", "Maintain", "Build Muscle"].map((goal) => (
                        <button
                          key={goal}
                          type="button"
                          onClick={() =>
                            updateFormData("goal", goal.toLowerCase().replace(" ", "_"))
                          }
                          className={`rounded-lg border-2 p-3 text-center text-sm font-medium transition-all ${
                            formData.goal === goal.toLowerCase().replace(" ", "_")
                              ? "border-primary bg-primary/10 text-primary"
                              : "border-border bg-secondary/50 text-foreground hover:border-primary/50"
                          }`}
                        >
                          {goal}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Activity Level</Label>
                    <Select
                      value={formData.activityLevel}
                      onValueChange={(value) => updateFormData("activityLevel", value)}
                    >
                      <SelectTrigger className="bg-secondary/50 border-transparent focus:border-primary">
                        <SelectValue placeholder="Select your activity level" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sedentary">Sedentary (little or no exercise)</SelectItem>
                        <SelectItem value="light">Light (exercise 1-3 days/week)</SelectItem>
                        <SelectItem value="moderate">Moderate (exercise 3-5 days/week)</SelectItem>
                        <SelectItem value="active">Active (exercise 6-7 days/week)</SelectItem>
                        <SelectItem value="very_active">Very Active (hard exercise daily)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Navigation buttons */}
            <div className="mt-6 flex items-center justify-between">
              {currentStep > 1 ? (
                <Button type="button" variant="ghost" onClick={handleBack}>
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back
                </Button>
              ) : (
                <div />
              )}

              {currentStep < 3 ? (
                <Button
                  type="button"
                  onClick={handleNext}
                  className="gradient-primary text-primary-foreground"
                >
                  Next
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              ) : (
                <Button
                  type="submit"
                  className="gradient-primary text-primary-foreground font-semibold glow-primary"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                  ) : (
                    "Create Account"
                  )}
                </Button>
              )}
            </div>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-muted-foreground">
              Already have an account?{" "}
              <Link to="/login" className="font-medium text-primary hover:underline">
                Sign in
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
