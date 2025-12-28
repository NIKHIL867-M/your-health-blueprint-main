import { useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";

import Login from "@/pages/auth/Login";
import Signup from "@/pages/auth/Signup";
import Dashboard from "@/pages/Dashboard";
import Workout from "@/pages/Workout";
import FoodCapture from "@/pages/FoodCapture";
import Diet from "@/pages/Diet";
import Reports from "@/pages/Reports";
import Chat from "@/pages/Chat";
import Profile from "@/pages/Profile";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => {
  // Set application title once
  useEffect(() => {
    document.title = "FIT_TRACKER";
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            {/* Auth routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />

            {/* App routes with layout */}
            <Route element={<AppLayout />}>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/workout" element={<Workout />} />
              <Route path="/food" element={<FoodCapture />} />
              <Route path="/diet" element={<Diet />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/profile" element={<Profile />} />
            </Route>

            {/* Redirects */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
