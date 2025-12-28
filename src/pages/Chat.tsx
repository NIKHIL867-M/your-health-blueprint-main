import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Send, Bot, User, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

// ---- NEW: API base from env ----
const API_BASE =
  import.meta.env.VITE_GYMBUDDY_API_URL || "http://localhost:3002";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hey! I'm your Gym Buddy 💪\nAsk me about streaks, progress, last workout, or motivation!",
      timestamp: new Date().toISOString(),
    },
  ]);

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // ---- SEND MESSAGE ----
  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: trimmed,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/gym-buddy/ai`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmed,
          userId: "user123",
        }),
      });

      const data = await response.json();

      const reply =
        (data.success && data.reply) ||
        (!data.success && "⚠️ AI is resting — try again in a moment!") ||
        "🤔 I couldn’t answer that — try again!";

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: reply,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          role: "assistant",
          content:
            "⚠️ Couldn't reach GymBuddy API.\nMake sure it's running locally:\n`node backend/gymBuddyApi.js`",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex h-[calc(100vh-8rem)] flex-col"
    >
      <div className="mb-4">
        <h1 className="text-2xl font-bold">AI Gym Buddy</h1>
        <p className="text-muted-foreground">
          Ask about streaks, progress, last workout & motivation 🚀
        </p>
      </div>

      <div className="flex flex-1 flex-col overflow-hidden rounded-xl border bg-card border-border">
        <div className="flex-1 overflow-y-auto p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn(
                  "flex gap-3",
                  message.role === "user" && "flex-row-reverse"
                )}
              >
                {/* Avatar */}
                <div
                  className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-full",
                    message.role === "assistant"
                      ? "bg-primary/20"
                      : "bg-accent/20"
                  )}
                >
                  {message.role === "assistant" ? (
                    <Bot className="h-4 w-4 text-primary" />
                  ) : (
                    <User className="h-4 w-4 text-accent" />
                  )}
                </div>

                {/* Message bubble */}
                <div
                  className={cn(
                    "max-w-[80%] rounded-2xl px-4 py-2.5 whitespace-pre-line",
                    message.role === "assistant"
                      ? "bg-secondary/50 text-foreground"
                      : "bg-primary text-primary-foreground"
                  )}
                >
                  <p className="text-sm">{message.content}</p>
                  <p
                    className={cn(
                      "mt-1 text-xs",
                      message.role === "assistant"
                        ? "text-muted-foreground"
                        : "text-primary-foreground/70"
                    )}
                  >
                    {new Date(message.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </p>
                </div>
              </motion.div>
            ))}

            {/* Loading */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex gap-3"
              >
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="flex items-center gap-2 rounded-2xl bg-secondary/50 px-4 py-2.5">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  <span className="text-sm text-muted-foreground">
                    Thinking...
                  </span>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t p-4 border-border">
          <div className="flex gap-3">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask: what's my streak? last squat? give motivation!"
              className="flex-1 bg-secondary/50 border-transparent focus:border-primary"
              disabled={isLoading}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className="gradient-primary text-primary-foreground"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
