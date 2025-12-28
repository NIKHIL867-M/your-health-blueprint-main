import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// -------------------- GEMINI INIT --------------------
if (!process.env.GEMINI_API_KEY) {
  console.error("❌ ERROR: Missing GEMINI_API_KEY in .env file");
}
// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// -------------------- IN-MEMORY DATABASE --------------------
const workoutsDB = {
  user123: [
    { id: 1, exercise: "squats", reps: 15, sets: 3, date: "2024-01-15" },
    { id: 2, exercise: "bench", reps: 10, sets: 4, date: "2024-01-14" },
  ],
};

const userStatsDB = {
  user123: { currentStreak: 3, bestStreak: 7, totalWorkouts: 12 },
};

// -------------------- HELPERS --------------------
function calcProgress(totalWorkouts) {
  return Math.min(100, Math.round((totalWorkouts / 20) * 100));
}

function recordWorkout(userId, workout) {
  if (!workoutsDB[userId]) workoutsDB[userId] = [];

  const newWorkout = {
    id: Date.now(),
    exercise: workout.exercise,
    reps: workout.reps,
    sets: workout.sets,
    date: new Date().toISOString().split("T")[0],
  };

  workoutsDB[userId].unshift(newWorkout);

  // Init stats if missing
  if (!userStatsDB[userId]) {
    userStatsDB[userId] = { currentStreak: 0, bestStreak: 0, totalWorkouts: 0 };
  }

  const stats = userStatsDB[userId];
  stats.totalWorkouts++;
  stats.currentStreak++;
  if (stats.currentStreak > stats.bestStreak) {
    stats.bestStreak = stats.currentStreak;
  }

  return newWorkout;
}

// -------------------- RULE-BASED NLP ENGINE --------------------
function processMessage(message, userId = "user123") {
  const text = message.toLowerCase();
  
  // Ensure stats exist
  const stats = userStatsDB[userId] || 
    (userStatsDB[userId] = { currentStreak: 0, bestStreak: 0, totalWorkouts: 0 });

  if (text.includes("hello") || text.includes("hi")) {
    return "Hey! I'm your Gym Buddy 💪 What do you want to know today?";
  }

  if (text.includes("streak")) {
    return `🔥 Current streak: ${stats.currentStreak} days\n🏆 Best: ${stats.bestStreak} days`;
  }

  if (text.includes("progress")) {
    const p = calcProgress(stats.totalWorkouts);
    return `📈 Your progress is about ${p}% — stay consistent 💪🔥`;
  }

  if (text.includes("squat") || text.includes("bench")) {
    const exercise = text.includes("squat") ? "squats" : "bench";
    const history = workoutsDB[userId]?.filter((w) => w.exercise === exercise) || [];
    
    if (!history.length) return `No ${exercise} logged yet 😅`;
    
    const last = history[0];
    return `🏋️‍♂️ Last ${exercise}: ${last.reps} reps × ${last.sets} sets on ${last.date}`;
  }

  if (text.includes("motivation")) {
    const m = [
      "Every rep counts! Keep pushing 💥",
      "Show up today — future you will thank you 💯",
      "You're stronger than yesterday — keep grinding 🔥",
    ];
    return m[Math.floor(Math.random() * m.length)];
  }

  // Fallback
  return `🤔 Try asking:\n• what's my streak?\n• progress?\n• last squats?\n• give motivation!`;
}

// -------------------- API ROUTES --------------------

// 1. Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", version: "2.1.0" });
});

// 2. Rule-based chat (Direct)
app.post("/api/gym-buddy/chat", (req, res) => {
  const { message, userId = "user123" } = req.body;
  if (!message?.trim()) return res.json({ reply: "Please type something 😅" });
  res.json({ reply: processMessage(message, userId) });
});

// 3. AI Chat with SMART FALLBACK 🧠
app.post("/api/gym-buddy/ai", async (req, res) => {
  const { message, userId = "user123" } = req.body;
  
  if (!message?.trim()) {
    return res.json({ success: false, reply: "Say something 😅" });
  }

  try {
    // Attempt Gemini AI
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" }); // 1.5-flash is most stable currently

    const prompt = `You are GymBuddy AI 💪🔥. Motivational & short. User says: "${message}"`;
    
    const result = await model.generateContent(prompt);
    const text = result.response.text();

    return res.json({ success: true, reply: text });

  } catch (err) {
    console.error("❌ Gemini Error (Quota/Network):", err.message);

    // ➤ FALLBACK LOGIC: Use the rule-based engine instead of failing
    const fallbackResponse = processMessage(message, userId);

    return res.json({ 
      success: true, // We verify this as true so frontend displays it normally
      reply: `(AI Offline) ${fallbackResponse}` // Prefix to let you know it switched
    });
  }
});

// 4. Workout list
app.get("/api/gym-buddy/workouts", (req, res) => {
  const userId = req.query.userId || "user123";
  res.json({ workouts: workoutsDB[userId] || [] });
});

// 5. Add workout
app.post("/api/gym-buddy/workouts", (req, res) => {
  const { exercise, reps, sets, userId = "user123" } = req.body;
  if (!exercise || !reps || !sets) {
    return res.json({ success: false, error: "Missing fields" });
  }
  const workout = recordWorkout(userId, { exercise, reps, sets });
  res.json({ success: true, workout });
});

// 6. Stats
app.get("/api/gym-buddy/stats", (req, res) => {
  const userId = req.query.userId || "user123";
  res.json({ stats: userStatsDB[userId] });
});

// -------------------- START SERVER --------------------
const PORT = process.env.PORT || 3002;
app.listen(PORT, () =>
  console.log(`🔥 GymBuddy API running at http://localhost:${PORT}`)
);

export default app;