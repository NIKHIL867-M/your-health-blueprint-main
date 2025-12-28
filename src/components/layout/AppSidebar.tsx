import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  Dumbbell,
  Camera,
  Apple,
  BarChart3,
  MessageCircle,
  User,
  Menu,
  X,
  Flame,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const navItems = [
  { title: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
  { title: "Workout", path: "/workout", icon: Dumbbell },
  { title: "Food Scanner", path: "/food", icon: Camera },
  { title: "Diet", path: "/diet", icon: Apple },
  { title: "Reports", path: "/reports", icon: BarChart3 },
  { title: "Chat", path: "/chat", icon: MessageCircle },
  { title: "Profile", path: "/profile", icon: User },
];

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <>
      {/* Mobile overlay */}
      {!collapsed && (
        <div
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden"
          onClick={() => setCollapsed(true)}
        />
      )}

      {/* Mobile toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed left-4 top-4 z-50 md:hidden"
        onClick={() => setCollapsed(!collapsed)}
      >
        {collapsed ? <Menu className="h-5 w-5" /> : <X className="h-5 w-5" />}
      </Button>

      {/* Sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: collapsed ? 0 : 260 }}
        className={cn(
          "fixed left-0 top-0 z-40 h-screen overflow-hidden border-r border-sidebar-border bg-sidebar md:relative",
          collapsed ? "md:w-16" : "md:w-64"
        )}
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center gap-3 border-b border-sidebar-border px-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl gradient-primary">
              <Flame className="h-5 w-5 text-primary-foreground" />
            </div>
            {!collapsed && (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-xl font-bold text-foreground"
              >
                FitTrack
              </motion.span>
            )}
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 p-3">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
                  )}
                >
                  <item.icon className={cn("h-5 w-5 shrink-0", isActive && "text-primary")} />
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {item.title}
                    </motion.span>
                  )}
                  {isActive && (
                    <motion.div
                      layoutId="activeNav"
                      className="absolute left-0 h-8 w-1 rounded-r-full bg-primary"
                    />
                  )}
                </NavLink>
              );
            })}
          </nav>

          {/* Collapse toggle (desktop) */}
          <div className="hidden border-t border-sidebar-border p-3 md:block">
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start gap-3 text-sidebar-foreground"
              onClick={() => setCollapsed(!collapsed)}
            >
              <Menu className="h-4 w-4" />
              {!collapsed && <span>Collapse</span>}
            </Button>
          </div>
        </div>
      </motion.aside>
    </>
  );
}
