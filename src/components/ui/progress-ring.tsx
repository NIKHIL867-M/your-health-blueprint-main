import { cn } from "@/lib/utils";

interface ProgressRingProps {
  progress: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
  variant?: "primary" | "accent" | "success" | "warning";
  showValue?: boolean;
  label?: string;
}

export function ProgressRing({
  progress,
  size = 120,
  strokeWidth = 8,
  className,
  variant = "primary",
  showValue = true,
  label,
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  const colors = {
    primary: "stroke-primary",
    accent: "stroke-accent",
    success: "stroke-success",
    warning: "stroke-warning",
  };

  const glowColors = {
    primary: "drop-shadow-[0_0_8px_hsl(145,80%,50%,0.5)]",
    accent: "drop-shadow-[0_0_8px_hsl(15,90%,60%,0.5)]",
    success: "drop-shadow-[0_0_8px_hsl(145,80%,45%,0.5)]",
    warning: "drop-shadow-[0_0_8px_hsl(45,95%,55%,0.5)]",
  };

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg width={size} height={size} className="-rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-muted/30"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={cn(colors[variant], glowColors[variant], "transition-all duration-500")}
        />
      </svg>
      {showValue && (
        <div className="absolute flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-foreground">{Math.round(progress)}</span>
          {label && <span className="text-xs text-muted-foreground">{label}</span>}
        </div>
      )}
    </div>
  );
}
