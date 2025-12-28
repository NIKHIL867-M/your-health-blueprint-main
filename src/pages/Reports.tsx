import { motion } from "framer-motion";
import {
  TrendingUp,
  Calendar,
  Award,
  Flame,
  Target,
  Zap,
  Activity,
  Dumbbell,
  ScanFace,
  Loader2
} from "lucide-react";
import { ProgressRing } from "@/components/ui/progress-ring";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { useEffect, useState } from "react";
import { auth, db } from "@/lib/firebase";
import { collection, getDocs, query, orderBy, Timestamp, where } from "firebase/firestore";
import { cn } from "@/lib/utils";

interface SessionData {
  id: string;
  exerciseType: string;
  timestamp: Timestamp;
  summary: any;
  performance: {
    score: number;
    label: string;
    color: string;
    explanation: string;
  };
}

interface WeeklyScore {
  week: string;
  score: number;
  date: Date;
}

interface ExerciseRep {
  name: string;
  reps: number;
  type: string;
}

interface CalendarDay {
  day: number | null;
  workout: boolean;
  sessions: number;
}

// Helper function to get week number from date
const getWeekNumber = (date: Date): number => {
  const firstDayOfYear = new Date(date.getFullYear(), 0, 1);
  const pastDaysOfYear = (date.getTime() - firstDayOfYear.getTime()) / 86400000;
  return Math.ceil((pastDaysOfYear + firstDayOfYear.getDay() + 1) / 7);
};

// Helper function to calculate streak
const calculateStreak = (sessions: SessionData[]): number => {
  if (sessions.length === 0) return 0;
  
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  // Get unique workout dates (sorted descending)
  const workoutDates = Array.from(new Set(
    sessions.map(s => {
      const date = s.timestamp.toDate();
      date.setHours(0, 0, 0, 0);
      return date.getTime();
    })
  )).sort((a, b) => b - a);
  
  let streak = 0;
  let currentDate = today.getTime();
  
  // Check for today
  if (workoutDates.includes(currentDate)) {
    streak = 1;
  } else {
    // Check yesterday
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    if (workoutDates.includes(yesterday.getTime())) {
      streak = 1;
      currentDate = yesterday.getTime();
    } else {
      return 0;
    }
  }
  
  // Continue checking previous days
  let dayOffset = 1;
  while (true) {
    const checkDate = new Date(today);
    checkDate.setDate(checkDate.getDate() - dayOffset);
    checkDate.setHours(0, 0, 0, 0);
    
    if (workoutDates.includes(checkDate.getTime())) {
      streak++;
      dayOffset++;
    } else {
      break;
    }
  }
  
  return streak;
};

// Generate calendar data for current month
const generateCalendarData = (sessions: SessionData[]): CalendarDay[] => {
  const today = new Date();
  const year = today.getFullYear();
  const month = today.getMonth();
  const firstDay = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  
  // Create a map of workout days for this month
  const workoutDaysMap = new Map<number, number>();
  
  sessions.forEach(session => {
    const sessionDate = session.timestamp.toDate();
    if (sessionDate.getFullYear() === year && sessionDate.getMonth() === month) {
      const day = sessionDate.getDate();
      workoutDaysMap.set(day, (workoutDaysMap.get(day) || 0) + 1);
    }
  });
  
  const days: CalendarDay[] = [];
  
  // Add empty days for padding
  for (let i = 0; i < firstDay; i++) {
    days.push({ day: null, workout: false, sessions: 0 });
  }
  
  // Add actual days
  for (let i = 1; i <= daysInMonth; i++) {
    const hasWorkout = workoutDaysMap.has(i);
    days.push({ 
      day: i, 
      workout: hasWorkout, 
      sessions: hasWorkout ? workoutDaysMap.get(i)! : 0 
    });
  }
  
  return days;
};

export default function Reports() {
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch sessions from Firestore
  useEffect(() => {
    const user = auth.currentUser;
    if (!user) {
      setLoading(false);
      return;
    }

    const loadSessions = async () => {
      try {
        const sessionsRef = collection(db, "users", user.uid, "sessions");
        const q = query(sessionsRef, orderBy("timestamp", "desc"));
        const snap = await getDocs(q);

        const items: SessionData[] = [];
        snap.forEach(doc => {
          const data = doc.data();
          items.push({
            id: doc.id,
            ...data
          } as SessionData);
        });

        setSessions(items);
      } catch (err) {
        console.error("Error loading sessions:", err);
      } finally {
        setLoading(false);
      }
    };

    loadSessions();
  }, []);

  // Calculate current score (average of last 7 sessions or most recent)
  const currentScore = sessions.length > 0 
    ? Math.round(sessions.slice(0, 7).reduce((sum, session) => sum + (session.performance?.score || 0), 0) / Math.min(sessions.length, 7))
    : 0;

  const streak = calculateStreak(sessions);

  const getScoreLevel = (score: number) => {
    if (score >= 80) return { label: "Excellent", color: "text-green-500" };
    if (score >= 60) return { label: "Good", color: "text-primary" };
    if (score >= 40) return { label: "Fair", color: "text-yellow-500" };
    return { label: "Needs Work", color: "text-red-500" };
  };

  const scoreLevel = getScoreLevel(currentScore);

  // Prepare weekly scores (last 8 weeks)
  const weeklyScores = (() => {
    if (sessions.length === 0) return Array(8).fill(0).map((_, i) => ({ week: `W${i+1}`, score: 0 }));
    
    const weeksMap = new Map<string, { total: number; count: number }>();
    const now = new Date();
    
    sessions.forEach(session => {
      const date = session.timestamp.toDate();
      const weekNumber = getWeekNumber(date);
      const weekKey = `W${weekNumber}`;
      
      if (!weeksMap.has(weekKey)) {
        weeksMap.set(weekKey, { total: 0, count: 0 });
      }
      
      const weekData = weeksMap.get(weekKey)!;
      weekData.total += session.performance?.score || 0;
      weekData.count++;
    });
    
    // Create array for last 8 weeks
    const result: { week: string; score: number }[] = [];
    const currentWeek = getWeekNumber(now);
    
    for (let i = 7; i >= 0; i--) {
      const weekNum = currentWeek - i;
      const weekKey = `W${weekNum}`;
      const weekData = weeksMap.get(weekKey);
      const avgScore = weekData ? Math.round(weekData.total / weekData.count) : 0;
      result.push({ week: weekKey, score: avgScore });
    }
    
    return result;
  })();

  // Prepare exercise reps data
  const exerciseReps = (() => {
    if (sessions.length === 0) return [
      { name: "Squats", reps: 0, type: "squat" },
      { name: "Posture", reps: 0, type: "face" }
    ];
    
    const squatSessions = sessions.filter(s => s.summary?.exerciseType === "squat");
    const faceSessions = sessions.filter(s => s.summary?.exerciseType === "face");
    
    const totalSquatReps = squatSessions.reduce((sum, session) => sum + (session.summary?.reps || 0), 0);
    const totalFaceSessions = faceSessions.length;
    
    return [
      { name: "Squats", reps: totalSquatReps, type: "squat" },
      { name: "Posture", reps: totalFaceSessions, type: "face" }
    ];
  })();

  // Calculate score components
  const scoreComponents = (() => {
    if (sessions.length === 0) return {
      volume: 0,
      form: 0,
      consistency: 0
    };
    
    const squatSessions = sessions.filter(s => s.summary?.exerciseType === "squat");
    const recentSessions = sessions.slice(0, 7);
    
    // Volume score (based on squat reps)
    const volumeScore = squatSessions.length > 0 
      ? Math.min(100, Math.round(squatSessions.reduce((sum, s) => sum + (s.summary?.reps || 0), 0) / squatSessions.length * 5))
      : 50;
    
    // Form score (average of squat form scores)
    const formScore = squatSessions.length > 0
      ? Math.round(squatSessions.reduce((sum, s) => sum + (s.summary?.formScore || 0), 0) / squatSessions.length)
      : 75;
    
    // Consistency score (based on streak and frequency)
    const consistencyScore = Math.min(100, Math.round((streak * 10) + (sessions.length * 2)));
    
    return {
      volume: volumeScore,
      form: formScore,
      consistency: consistencyScore
    };
  })();

  const calendarDays = generateCalendarData(sessions);

  if (loading) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading your performance data...</p>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground md:text-3xl">Performance Reports</h1>
        <p className="text-muted-foreground">
          {sessions.length > 0 
            ? `Tracking ${sessions.length} sessions with AI-5 analysis`
            : "Start your first session to see performance data"
          }
        </p>
      </div>

      {/* Score Summary */}
      <div className="grid gap-6 md:grid-cols-3">
        {/* Main Score Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="md:col-span-1 rounded-xl border border-border bg-card p-6 text-center"
        >
          <h2 className="mb-4 text-lg font-semibold text-foreground">Current Score</h2>
          <ProgressRing
            progress={currentScore}
            size={160}
            strokeWidth={12}
            variant="primary"
          />
          <div className="mt-4 flex items-center justify-center gap-2">
            <Zap className={`h-5 w-5 ${scoreLevel.color}`} />
            <span className={`font-semibold ${scoreLevel.color}`}>
              {scoreLevel.label}
            </span>
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
            {sessions.length > 0 
              ? `Based on ${Math.min(sessions.length, 7)} recent sessions`
              : "No sessions recorded yet"
            }
          </p>
        </motion.div>

        {/* Score Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="md:col-span-2 rounded-xl border border-border bg-card p-6"
        >
          <h2 className="mb-4 text-lg font-semibold text-foreground">Score Components</h2>
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="rounded-lg bg-primary/10 p-4 text-center">
              <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-primary/20">
                <Dumbbell className="h-6 w-6 text-primary" />
              </div>
              <p className="text-2xl font-bold text-foreground">{scoreComponents.volume}%</p>
              <p className="text-sm text-muted-foreground">Volume Score</p>
              <p className="mt-1 text-xs text-primary">Reps completed</p>
            </div>
            <div className="rounded-lg bg-accent/10 p-4 text-center">
              <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-accent/20">
                <Award className="h-6 w-6 text-accent" />
              </div>
              <p className="text-2xl font-bold text-foreground">{scoreComponents.form}%</p>
              <p className="text-sm text-muted-foreground">Form Score</p>
              <p className="mt-1 text-xs text-accent">Exercise quality</p>
            </div>
            <div className="rounded-lg bg-green-500/10 p-4 text-center">
              <div className="mx-auto mb-2 flex h-12 w-12 items-center justify-center rounded-full bg-green-500/20">
                <Flame className="h-6 w-6 text-green-500" />
              </div>
              <p className="text-2xl font-bold text-foreground">{scoreComponents.consistency}%</p>
              <p className="text-sm text-muted-foreground">Consistency</p>
              <p className="mt-1 text-xs text-green-500">Workout frequency</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Weekly Trend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="rounded-xl border border-border bg-card p-5"
        >
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">Weekly Performance Trend</h2>
            </div>
            <span className="text-sm text-muted-foreground">
              {sessions.length} total sessions
            </span>
          </div>
          <div className="h-64">
            {sessions.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={weeklyScores}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="week" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    labelStyle={{ color: "hsl(var(--foreground))" }}
                    formatter={(value) => [`${value}%`, "Score"]}
                  />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="hsl(var(--primary))"
                    strokeWidth={3}
                    dot={{ fill: "hsl(var(--primary))", strokeWidth: 0, r: 4 }}
                    activeDot={{ r: 6, fill: "hsl(var(--primary))" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center">
                <p className="text-muted-foreground">No performance data yet</p>
              </div>
            )}
          </div>
        </motion.div>

        {/* Exercise Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="rounded-xl border border-border bg-card p-5"
        >
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-accent" />
              <h2 className="text-lg font-semibold text-foreground">Activity Summary</h2>
            </div>
            <span className="text-sm text-muted-foreground">
              Total: {sessions.length} sessions
            </span>
          </div>
          <div className="h-64">
            {sessions.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={exerciseReps} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={12} width={80} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    labelStyle={{ color: "hsl(var(--foreground))" }}
                    formatter={(value, name, props) => {
                      const type = props.payload.type;
                      if (type === "squat") return [`${value} reps`, "Total Reps"];
                      return [`${value} sessions`, "Posture Training"];
                    }}
                  />
                  <Bar 
                    dataKey="reps" 
                    fill={exerciseReps[0]?.type === "squat" ? "hsl(var(--primary))" : "hsl(var(--accent))"} 
                    radius={[0, 4, 4, 0]} 
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center">
                <p className="text-muted-foreground">No exercise data yet</p>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Calendar & Streak */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Calendar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="lg:col-span-2 rounded-xl border border-border bg-card p-5"
        >
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Calendar className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">This Month's Activity</h2>
            </div>
            <span className="text-sm text-muted-foreground">
              {new Date().toLocaleString('default', { month: 'long' })}
            </span>
          </div>
          <div className="grid grid-cols-7 gap-1 text-center text-xs">
            {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
              <div key={day} className="py-2 font-medium text-muted-foreground">
                {day}
              </div>
            ))}
            {calendarDays.map((item, index) => (
              <div
                key={index}
                className={cn(
                  "aspect-square flex items-center justify-center rounded-lg text-sm relative",
                  item.day === null
                    ? ""
                    : item.workout
                    ? "bg-primary/20 text-primary font-medium"
                    : "bg-muted/30 text-muted-foreground"
                )}
                title={item.workout ? `${item.sessions} session(s)` : "No workout"}
              >
                {item.day}
                {item.sessions > 1 && (
                  <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-primary text-[8px]"></span>
                )}
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded bg-primary/20" />
              <span className="text-muted-foreground">Workout Day</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded bg-muted/30" />
              <span className="text-muted-foreground">Rest Day</span>
            </div>
          </div>
        </motion.div>

        {/* Streak */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="rounded-xl border border-primary/30 bg-gradient-to-br from-primary/10 to-accent/10 p-6 text-center"
        >
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/20">
            <Flame className="h-8 w-8 text-primary" />
          </div>
          <p className="text-5xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            {streak}
          </p>
          <p className="mt-2 text-lg font-medium text-foreground">Day Streak</p>
          <p className="mt-2 text-sm text-muted-foreground">
            {streak > 0 
              ? "You're on fire! Keep the momentum going." 
              : "Start a streak by working out today!"}
          </p>
          <div className="mt-4 flex justify-center gap-1">
            {Array.from({ length: 7 }).map((_, i) => (
              <div
                key={i}
                className={cn(
                  "h-2 w-6 rounded-full",
                  i < streak ? "bg-primary" : "bg-muted/50"
                )}
              />
            ))}
          </div>
          {sessions.length > 0 && (
            <div className="mt-4 text-xs text-muted-foreground">
              {sessions.filter(s => 
                s.timestamp.toDate().toDateString() === new Date().toDateString()
              ).length > 0 
                ? "✓ Workout logged today" 
                : "No workout logged today"}
            </div>
          )}
        </motion.div>
      </div>

      {/* Recent Sessions */}
      {sessions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="rounded-xl border border-border bg-card p-5"
        >
          <h2 className="mb-4 text-lg font-semibold text-foreground">Recent Sessions</h2>
          <div className="space-y-3">
            {sessions.slice(0, 5).map((session, index) => (
              <div key={session.id} className="flex items-center justify-between rounded-lg border border-border p-4">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "flex h-10 w-10 items-center justify-center rounded-full",
                    session.exerciseType === "squat" 
                      ? "bg-primary/20" 
                      : "bg-accent/20"
                  )}>
                    {session.exerciseType === "squat" ? (
                      <Dumbbell className="h-5 w-5 text-primary" />
                    ) : (
                      <ScanFace className="h-5 w-5 text-accent" />
                    )}
                  </div>
                  <div>
                    <p className="font-medium text-foreground">
                      {session.exerciseType === "squat" 
                        ? `Squat Session (${session.summary?.reps || 0} reps)` 
                        : "Posture Training Session"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {session.timestamp.toDate().toLocaleDateString()} • {session.timestamp.toDate().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`text-lg font-bold ${session.performance?.color}`}>
                    {session.performance?.score || 0}/100
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {session.performance?.label || "No score"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}