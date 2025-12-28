// src/pages/Diet.tsx
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Plus,
  Trash2,
  Target,
  TrendingUp,
  Flame,
  Apple,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ProgressRing } from "@/components/ui/progress-ring";
import { useNavigate } from "react-router-dom";

import {
  analyzeDiet,
  UserProfilePayload,
  FoodItemPayload,
} from "@/services/apiDiet";

import { auth, db } from "@/lib/firebase";
import {
  collection,
  getDocs,
  query,
  orderBy,
} from "firebase/firestore";

/* ----------------------------------
   FOOD LOG TYPE
-----------------------------------*/
interface MealEntry {
  id: string;
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fats: number;
  timestamp: string;
}

/* ----------------------------------
   PAGE START
-----------------------------------*/
export default function Diet() {
  const navigate = useNavigate();

  const [foods, setFoods] = useState<MealEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dietData, setDietData] = useState<any | null>(null);

  /* ----------------------------------
     STEP 1 — LOAD MEALS FROM FIRESTORE
  -----------------------------------*/
  useEffect(() => {
    async function loadMeals() {
      try {
        setLoading(true);

        const uid = auth.currentUser?.uid;
        if (!uid) {
          setFoods([]);
          return;
        }

        const mealsRef = collection(db, `users/${uid}/meals`);
        const q = query(mealsRef, orderBy("timestamp", "desc"));
        const snapshot = await getDocs(q);

        const loadedMeals: MealEntry[] = snapshot.docs.map((doc) => {
          const data = doc.data();
          return {
            id: doc.id,
            name: data.name,
            calories: data.calories,
            protein: data.macros?.protein ?? 0,
            carbs: data.macros?.carbs ?? 0,
            fats: data.macros?.fat ?? 0,
            timestamp: data.timestamp,
          };
        });

        setFoods(loadedMeals);
      } catch (e: any) {
        console.error("Error loading meals:", e);
        setError(e.message ?? "Failed to load meals");
      } finally {
        setLoading(false);
      }
    }

    loadMeals();
  }, []); // fetch on mount

  /* ----------------------------------
     STEP 2 — RUN DIET ANALYSIS WHEN FOODS CHANGE
  -----------------------------------*/
  useEffect(() => {
    async function loadDiet() {
      try {
        if (foods.length === 0) {
          setDietData(null);
          return;
        }

        setLoading(true);
        setError(null);

        const profile: UserProfilePayload = {
          age: 22,
          gender: "male",
          height_cm: 175,
          weight_kg: 70,
          activity_level: "moderate",
          goal: "fat_loss",
          goal_intensity: "moderate",
          weekly_rate_kg: 0.5,
        };

        const foodPayload: FoodItemPayload[] = foods.map((f) => ({
          food_name: f.name,
          estimated_calories: f.calories,
          protein_g: f.protein,
          carbs_g: f.carbs,
          fat_g: f.fats,
        }));

        const res = await analyzeDiet(profile, foodPayload);
        setDietData(res);
      } catch (e: any) {
        setError(e.message ?? "Failed to load diet analysis");
      } finally {
        setLoading(false);
      }
    }

    loadDiet();
  }, [foods]);

  /* ----------------------------------
     SAFE LOOKUPS
  -----------------------------------*/
  if (loading) return <div>Loading Fit Tracker diet analysis...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  const targets = dietData?.daily_targets ?? {};
  const intake = dietData?.food_analysis ?? {};

  const calorieProgress =
    targets.calories && intake.calories_consumed
      ? (intake.calories_consumed / targets.calories) * 100
      : 0;

  const goalText =
    dietData?.metabolic_summary?.goal ??
    dietData?.goal_summary?.goal ??
    dietData?.profile?.goal ??
    "Fat loss";

  /* ----------------------------------
     UI
  -----------------------------------*/
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold md:text-3xl">Fit Tracker</h1>
          <p className="text-muted-foreground">
            Smart AI-powered nutrition tracking
          </p>
        </div>
        <Button
          onClick={() => navigate("/food")}
          className="gradient-primary text-primary-foreground"
        >
          <Plus className="mr-2 h-4 w-4" />
          Add Food
        </Button>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Main */}
        <div className="lg:col-span-2 space-y-6">
          {/* Intake */}
          <div className="rounded-xl border bg-card p-5">
            <h2 className="mb-4 text-lg font-semibold">Today’s Intake</h2>

            {foods.length === 0 && (
              <p className="text-center text-muted-foreground">
                No meals logged today yet — add food to get started.
              </p>
            )}

            {foods.length > 0 && (
              <>
                <div className="mb-6 flex flex-col items-center gap-4 sm:flex-row sm:justify-around">
                  <ProgressRing
                    progress={Math.min(calorieProgress, 100)}
                    size={160}
                    strokeWidth={12}
                    variant={calorieProgress > 100 ? "warning" : "primary"}
                    label="kcal"
                  />

                  <div className="grid w-full max-w-xs gap-4">
                    {[
                      ["Protein", intake?.protein_consumed, targets?.protein_g],
                      ["Carbs", intake?.carbs_consumed, targets?.carbs_g],
                      ["Fats", intake?.fat_consumed, targets?.fat_g],
                    ].map(([label, val, max]: any) => (
                      <div key={label}>
                        <div className="mb-1 flex justify-between text-sm">
                          <span>{label}</span>
                          <span className="font-medium">
                            {val ?? 0}g / {max ?? 0}g
                          </span>
                        </div>
                        <Progress value={max ? (val / max) * 100 : 0} className="h-2" />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Summary */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <Summary icon={Flame} label="Calories" value={intake?.calories_consumed ?? 0} />
                  <Summary icon={Target} label="Protein" value={`${intake?.protein_consumed ?? 0}g`} />
                  <Summary icon={Apple} label="Carbs" value={`${intake?.carbs_consumed ?? 0}g`} />
                  <Summary icon={TrendingUp} label="Fats" value={`${intake?.fat_consumed ?? 0}g`} />
                </div>
              </>
            )}
          </div>

          {/* Food Log */}
          <div className="rounded-xl border bg-card p-5">
            <h2 className="mb-4 text-lg font-semibold">Food Log</h2>

            {foods.length === 0 && (
              <p className="text-muted-foreground">No food logged yet.</p>
            )}

            {foods.map((food) => (
              <motion.div
                key={food.id}
                className="flex items-center justify-between rounded-lg border p-3 mb-2"
              >
                <div>
                  <p className="font-medium">{food.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {food.calories} kcal • P {food.protein}g • C {food.carbs}g • F {food.fats}g
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <div className="rounded-xl border bg-card p-5">
            <h3 className="mb-3 font-semibold">Daily Targets</h3>
            <Info label="Goal" value={goalText} />
            <Info label="Calories" value={targets?.calories ?? 0} />
            <Info label="Protein" value={`${targets?.protein_g ?? 0}g`} />
            <Info label="Carbs" value={`${targets?.carbs_g ?? 0}g`} />
            <Info label="Fats" value={`${targets?.fat_g ?? 0}g`} />
          </div>

          <div className="rounded-xl border bg-primary/5 p-5">
            <h3 className="mb-3 font-semibold flex gap-2">
              <TrendingUp className="h-4 w-4" /> Smart Recommendations
            </h3>

            <ul className="space-y-2 text-sm">
              {(dietData?.smart_recommendations ?? []).map((r: string, i: number) => (
                <li key={i}>• {r}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/* ----------------------------------
   SMALL HELPERS
-----------------------------------*/
function Summary({ icon: Icon, label, value }: any) {
  return (
    <div className="rounded-lg bg-muted/40 p-3 text-center">
      <Icon className="mx-auto mb-1 h-5 w-5" />
      <p className="text-xl font-bold">{value}</p>
      <p className="text-xs">{label}</p>
    </div>
  );
}

function Info({ label, value }: any) {
  return (
    <div className="flex justify-between text-sm mb-1">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
