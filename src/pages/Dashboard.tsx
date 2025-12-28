import { motion } from "framer-motion";
import {
  Dumbbell,
  Flame,
  TrendingUp,
  Trophy,
  Target,
  Zap,
  ChevronRight,
} from "lucide-react";
import { StatCard } from "@/components/ui/stat-card";
import { ProgressRing } from "@/components/ui/progress-ring";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

// ADDED IMPORTS
import { collection, query, orderBy, limit, getDocs, where } from "firebase/firestore";
import { useState, useEffect } from "react";
import { auth, db } from "@/lib/firebase"; // Assuming you have firebase config

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

// Helper functions
const dayIndex = (day: string): number => {
  const days: { [key: string]: number } = {
    'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 
    'Thu': 4, 'Fri': 5, 'Sat': 6
  };
  return days[day] || 0;
};

const calculateStreak = (sessions: any[]): number => {
  if (!sessions.length) return 0;
  
  let streak = 0;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  const sessionDates = sessions
    .map(s => {
      const date = new Date(s.timestamp.seconds * 1000);
      date.setHours(0, 0, 0, 0);
      return date.getTime();
    })
    .filter((date, index, self) => self.indexOf(date) === index) // Unique dates
    .sort((a, b) => b - a); // Most recent first
  
  // Check consecutive days from today backwards
  let currentDate = today.getTime();
  for (let i = 0; i < sessionDates.length; i++) {
    const sessionDate = sessionDates[i];
    const diffDays = Math.floor((currentDate - sessionDate) / (1000 * 60 * 60 * 24));
    
    if (diffDays === i) {
      streak++;
    } else {
      break;
    }
  }
  
  return streak;
};

export default function Dashboard() {
  const navigate = useNavigate();
  
  // ADDED STATE
  interface WorkoutSession {
    id: string;
    timestamp: any;
    performance?: {
      score: number;
    };
  }

  const [weeklyData, setWeeklyData] = useState<{ day: string; workouts: number; rest: number }[]>([]);
  const [stats, setStats] = useState({ 
    totalWorkouts: 0, 
    streak: 0, 
    avgScore: 0,
    calories: 1245 // Default value, you can fetch this too
  });
  const [loading, setLoading] = useState(true);

  // ADDED EFFECT
  useEffect(() => {
    const fetchData = async () => {
      const user = auth.currentUser;
      if (!user) {
        setLoading(false);
        return;
      }

      try {
        // Fetch last 7 days of sessions
        const now = new Date();
        const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        
        const q = query(
          collection(db, "users", user.uid, "sessions"),
          where("timestamp", ">=", oneWeekAgo),
          orderBy("timestamp", "desc"),
          limit(20)
        );
        
        const querySnapshot = await getDocs(q);
        const sessions = querySnapshot.docs.map(doc => ({
          id: doc.id,
          ...doc.data()
        })) as WorkoutSession[];
        
        // Calculate stats
        const avgScore = sessions.length > 0 
          ? sessions.reduce((sum: number, s: any) => sum + (s.performance?.score || 0), 0) / sessions.length 
          : 0;
        
        setStats({
          totalWorkouts: sessions.length,
          streak: calculateStreak(sessions),
          avgScore: Math.round(avgScore),
          calories: 1245 // You might want to fetch this separately
        });
        
        // Build weekly data
        const weekDays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const weekData = weekDays.map(day => {
          const dayIdx = dayIndex(day);
          const workoutsCount = sessions.filter(s => {
            const sessionDate = new Date(s.timestamp.seconds * 1000);
            return sessionDate.getDay() === dayIdx;
          }).length;
          
          return {
            day,
            workouts: workoutsCount,
            rest: workoutsCount > 0 ? 0 : 1
          };
        });
        
        setWeeklyData(weekData);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Header */}
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold text-foreground md:text-3xl">
          Good morning, <span className="text-gradient-primary">John</span>! 💪
        </h1>
        <p className="text-muted-foreground">Here's your fitness overview for today.</p>
      </div>

      {/* Stats Grid */}
      <motion.div variants={itemVariants} className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Today's Workout"
          value="Not Started"
          subtitle="Lower Body Day"
          icon={Dumbbell}
          variant="primary"
        />
        <StatCard
          title="Calories Today"
          value={stats.calories.toLocaleString()}
          subtitle="of 1,800 target"
          icon={Flame}
          variant="accent"
        />
        <StatCard
          title="Current Streak"
          value={`${stats.streak} days`}
          icon={Trophy}
          trend={{ value: 25, isPositive: true }}
        />
        <StatCard
          title="Weekly Score"
          value={stats.avgScore.toString()}
          subtitle="Good performance"
          icon={TrendingUp}
          trend={{ value: 12, isPositive: true }}
        />
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Today's Plan */}
        <motion.div
          variants={itemVariants}
          className="lg:col-span-2 rounded-xl border border-border bg-card p-5 shadow-card"
        >
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground">Today's Plan</h2>
            <Button variant="ghost" size="sm" className="text-primary" onClick={() => navigate("/workout")}>
              Start Workout
              <ChevronRight className="ml-1 h-4 w-4" />
            </Button>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {/* Workout Card */}
            <div className="rounded-lg border border-border bg-secondary/30 p-4">
              <div className="mb-3 flex items-center gap-2">
                <div className="rounded-lg bg-primary/20 p-2">
                  <Dumbbell className="h-5 w-5 text-primary" />
                </div>
                <h3 className="font-semibold text-foreground">Lower Body</h3>
              </div>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-primary" />
                  3 x 15 Bodyweight Squats
                </li>
                <li className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-primary" />
                  3 x 12 Lunges (each leg)
                </li>
                <li className="flex items-center gap-2">
                  <Target className="h-4 w-4 text-primary" />
                  3 x 20 Calf Raises
                </li>
              </ul>
            </div>

            {/* Diet Card */}
            <div className="rounded-lg border border-border bg-secondary/30 p-4">
              <div className="mb-3 flex items-center gap-2">
                <div className="rounded-lg bg-accent/20 p-2">
                  <Flame className="h-5 w-5 text-accent" />
                </div>
                <h3 className="font-semibold text-foreground">Diet Target</h3>
              </div>
              <div className="space-y-3">
                <div>
                  <div className="mb-1 flex justify-between text-sm">
                    <span className="text-muted-foreground">Calories</span>
                    <span className="font-medium text-foreground">{stats.calories} / 1,800</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div 
                      className="h-full rounded-full bg-primary" 
                      style={{ width: `${(stats.calories / 1800) * 100}%` }}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div className="rounded-lg bg-muted/50 p-2">
                    <p className="font-semibold text-foreground">65g</p>
                    <p className="text-muted-foreground">Protein</p>
                  </div>
                  <div className="rounded-lg bg-muted/50 p-2">
                    <p className="font-semibold text-foreground">120g</p>
                    <p className="text-muted-foreground">Carbs</p>
                  </div>
                  <div className="rounded-lg bg-muted/50 p-2">
                    <p className="font-semibold text-foreground">45g</p>
                    <p className="text-muted-foreground">Fats</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Performance Score */}
        <motion.div
          variants={itemVariants}
          className="rounded-xl border border-border bg-card p-5 shadow-card"
        >
          <h2 className="mb-4 text-lg font-semibold text-foreground">Performance Score</h2>
          <div className="flex flex-col items-center">
            <ProgressRing 
              progress={stats.avgScore || 0} 
              size={140} 
              strokeWidth={10} 
              variant="primary" 
              label="score" 
            />
            <div className="mt-4 flex items-center gap-2">
              <Zap className="h-5 w-5 text-warning" />
              <span className="font-medium text-foreground">
                {stats.avgScore >= 80 ? "Excellent" : stats.avgScore >= 60 ? "Good" : "Needs Improvement"}
              </span>
            </div>
            <p className="mt-2 text-center text-sm text-muted-foreground">
              {stats.avgScore >= 80 
                ? "Outstanding performance! Keep pushing!" 
                : stats.avgScore >= 60 
                ? "You're doing great! Keep up the consistency." 
                : "Let's get more workouts in this week!"}
            </p>
          </div>
        </motion.div>
      </div>

      {/* Weekly Activity Chart */}
      <motion.div
        variants={itemVariants}
        className="rounded-xl border border-border bg-card p-5 shadow-card"
      >
        <h2 className="mb-4 text-lg font-semibold text-foreground">Weekly Activity</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={weeklyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
              <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
                labelStyle={{ color: "hsl(var(--foreground))" }}
              />
              <Bar 
                dataKey="workouts" 
                name="Workouts" 
                fill="hsl(var(--primary))" 
                radius={[4, 4, 0, 0]} 
              />
              <Bar 
                dataKey="rest" 
                name="Rest Days" 
                fill="hsl(var(--muted))" 
                radius={[4, 4, 0, 0]} 
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 text-center text-sm text-muted-foreground">
          Total workouts this week: {stats.totalWorkouts}
        </div>
      </motion.div>

      {/* Motivation Quote */}
      <motion.div
        variants={itemVariants}
        className="rounded-xl gradient-hero border border-primary/20 p-6 text-center"
      >
        <p className="text-lg font-medium italic text-foreground">
          "The only bad workout is the one that didn't happen."
        </p>
        <p className="mt-2 text-sm text-muted-foreground">— Your Gym Buddy</p>
      </motion.div>
    </motion.div>
  );
}