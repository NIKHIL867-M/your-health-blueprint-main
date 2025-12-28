import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  variant?: "default" | "primary" | "accent";
  className?: string;
}

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  variant = "default",
  className,
}: StatCardProps) {
  const variants = {
    default: "bg-card border-border",
    primary: "bg-primary/10 border-primary/30",
    accent: "bg-accent/10 border-accent/30",
  };

  const iconVariants = {
    default: "bg-muted text-muted-foreground",
    primary: "bg-primary/20 text-primary",
    accent: "bg-accent/20 text-accent",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className={cn(
        "relative overflow-hidden rounded-xl border p-5 shadow-card",
        variants[variant],
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-3xl font-bold tracking-tight text-foreground">{value}</p>
          {subtitle && (
            <p className="text-sm text-muted-foreground">{subtitle}</p>
          )}
          {trend && (
            <div className="flex items-center gap-1">
              <span
                className={cn(
                  "text-sm font-medium",
                  trend.isPositive ? "text-success" : "text-destructive"
                )}
              >
                {trend.isPositive ? "+" : "-"}{Math.abs(trend.value)}%
              </span>
              <span className="text-xs text-muted-foreground">vs last week</span>
            </div>
          )}
        </div>
        <div className={cn("rounded-lg p-3", iconVariants[variant])}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      
      {/* Decorative gradient */}
      <div
        className={cn(
          "absolute -right-8 -top-8 h-24 w-24 rounded-full opacity-20 blur-2xl",
          variant === "primary" && "bg-primary",
          variant === "accent" && "bg-accent",
          variant === "default" && "bg-muted"
        )}
      />
    </motion.div>
  );
}
