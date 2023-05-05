import React, { useState, useEffect } from "react";
//import QuestionForm from "./routes/QuizCreation";
//import QuizTaking from "./routes/QuizTaking"
import SignInSide from "./routes/login";
import SignUp from "./routes/signup";
import Landing from "./routes/landing";
import DoctorDashboard from "./routes/doctorDashboard";

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

export default function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route exact path="/home" element={<Landing />} />
          <Route exact path="/" element={<SignInSide />} />
          <Route exact path="/signup" element={<SignUp />} />
          <Route exact path="/doctor-dashboard" element={<DoctorDashboard />} />
        </Routes>
      </Router>
    </>
  );
}
