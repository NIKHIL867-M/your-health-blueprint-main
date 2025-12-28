import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { User, Save, Camera, Loader2 } from "lucide-react";
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
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useToast } from "@/hooks/use-toast";

// Firebase Imports
import { doc, getDoc, setDoc } from "firebase/firestore";
import { updateProfile as updateAuthProfile } from "firebase/auth";
import { auth, db } from "@/lib/firebase"; // Ensure this path is correct

export default function Profile() {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false); // For saving state
  const [isFetching, setIsFetching] = useState(true); // For initial data load

  // Initial state structure matches your DB requirements
  const [profile, setProfile] = useState({
    name: "",
    email: "",
    age: "",
    gender: "",
    height: "",
    weight: "",
    goal: "lose_weight",
    activityLevel: "moderate",
  });

  // 1. Fetch Data on Component Mount
  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      if (user) {
        try {
          // Check if user profile exists in Firestore
          const docRef = doc(db, "users", user.uid);
          const docSnap = await getDoc(docRef);

          if (docSnap.exists()) {
            // Merge Firestore data with local state
            setProfile((prev) => ({
              ...prev,
              ...docSnap.data(),
              name: user.displayName || docSnap.data().name || "",
              email: user.email || "",
            }));
          } else {
            // First time login? Pre-fill from Auth
            setProfile((prev) => ({
              ...prev,
              name: user.displayName || "",
              email: user.email || "",
            }));
          }
        } catch (error) {
          console.error("Error fetching profile:", error);
        }
      }
      setIsFetching(false);
    });

    return () => unsubscribe();
  }, []);

  const updateProfile = (field: string, value: string) => {
    setProfile((prev) => ({ ...prev, [field]: value }));
  };

  // 2. Save Data to Firestore
  const handleSave = async () => {
    const user = auth.currentUser;
    if (!user) return;

    setIsLoading(true);
    try {
      // Update Display Name in Auth (for immediate UI updates in header/avatar)
      if (user.displayName !== profile.name) {
        await updateAuthProfile(user, { displayName: profile.name });
      }

      // Save detailed profile to Firestore
      // Using setDoc with { merge: true } preserves other fields (like createdAt)
      await setDoc(doc(db, "users", user.uid), {
        ...profile,
        updatedAt: new Date().toISOString(),
      }, { merge: true });

      toast({
        title: "Profile Updated",
        description: "Your profile has been saved successfully.",
      });
    } catch (error) {
      console.error("Error saving profile:", error);
      toast({
        title: "Error",
        description: "Failed to save profile changes.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (isFetching) {
    return (
      <div className="flex h-[50vh] w-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="mx-auto max-w-2xl space-y-6"
    >
      <div>
        <h1 className="text-2xl font-bold text-foreground md:text-3xl">Profile Settings</h1>
        <p className="text-muted-foreground">Manage your personal information and fitness goals</p>
      </div>

      {/* Avatar Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-6 rounded-xl border border-border bg-card p-6"
      >
        <div className="relative">
          <Avatar className="h-24 w-24 border-4 border-primary/20">
            <AvatarImage src={auth.currentUser?.photoURL || ""} alt="Profile" />
            <AvatarFallback className="bg-primary/20 text-primary text-2xl font-bold">
              {profile.name ? profile.name.charAt(0).toUpperCase() : "U"}
            </AvatarFallback>
          </Avatar>
          <button className="absolute -bottom-1 -right-1 rounded-full bg-primary p-2 text-primary-foreground shadow-lg transition-transform hover:scale-110">
            <Camera className="h-4 w-4" />
          </button>
        </div>
        <div>
          <h2 className="text-xl font-semibold text-foreground">{profile.name || "User"}</h2>
          <p className="text-muted-foreground">{profile.email}</p>
        </div>
      </motion.div>

      {/* Personal Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="rounded-xl border border-border bg-card p-6"
      >
        <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-foreground">
          <User className="h-5 w-5 text-primary" />
          Personal Information
        </h3>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="name">Full Name</Label>
            <Input
              id="name"
              value={profile.name}
              onChange={(e) => updateProfile("name", e.target.value)}
              className="bg-secondary/50 border-transparent focus:border-primary"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              value={profile.email}
              disabled
              className="bg-secondary/50 border-transparent focus:border-primary opacity-70"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="age">Age</Label>
            <Input
              id="age"
              type="number"
              value={profile.age}
              onChange={(e) => updateProfile("age", e.target.value)}
              className="bg-secondary/50 border-transparent focus:border-primary"
            />
          </div>

          <div className="space-y-2">
            <Label>Gender</Label>
            <Select
              value={profile.gender}
              onValueChange={(value) => updateProfile("gender", value)}
            >
              <SelectTrigger className="bg-secondary/50 border-transparent focus:border-primary">
                <SelectValue placeholder="Select Gender" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="male">Male</SelectItem>
                <SelectItem value="female">Female</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="height">Height (cm)</Label>
            <Input
              id="height"
              type="number"
              value={profile.height}
              onChange={(e) => updateProfile("height", e.target.value)}
              className="bg-secondary/50 border-transparent focus:border-primary"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="weight">Weight (kg)</Label>
            <Input
              id="weight"
              type="number"
              value={profile.weight}
              onChange={(e) => updateProfile("weight", e.target.value)}
              className="bg-secondary/50 border-transparent focus:border-primary"
            />
          </div>
        </div>
      </motion.div>

      {/* Fitness Goals */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="rounded-xl border border-border bg-card p-6"
      >
        <h3 className="mb-4 text-lg font-semibold text-foreground">Fitness Goals</h3>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label>Goal</Label>
            <Select
              value={profile.goal}
              onValueChange={(value) => updateProfile("goal", value)}
            >
              <SelectTrigger className="bg-secondary/50 border-transparent focus:border-primary">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="lose_weight">Lose Weight</SelectItem>
                <SelectItem value="maintain">Maintain Weight</SelectItem>
                <SelectItem value="build_muscle">Build Muscle</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>Activity Level</Label>
            <Select
              value={profile.activityLevel}
              onValueChange={(value) => updateProfile("activityLevel", value)}
            >
              <SelectTrigger className="bg-secondary/50 border-transparent focus:border-primary">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="sedentary">Sedentary</SelectItem>
                <SelectItem value="light">Light (1-3 days/week)</SelectItem>
                <SelectItem value="moderate">Moderate (3-5 days/week)</SelectItem>
                <SelectItem value="active">Active (6-7 days/week)</SelectItem>
                <SelectItem value="very_active">Very Active (daily)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </motion.div>

      {/* Save Button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Button
          onClick={handleSave}
          className="w-full gradient-primary text-primary-foreground font-semibold glow-primary"
          disabled={isLoading}
        >
          {isLoading ? (
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
          ) : (
            <>
              <Save className="mr-2 h-4 w-4" />
              Save Changes
            </>
          )}
        </Button>
      </motion.div>
    </motion.div>
  );
}