import React, { useEffect, useState } from "react";
import axios from "axios";
import "./css/landing.css";
// import "../../../src/css/style.css";

function DoctorDashboard() {
  const [doctorDetails, setDoctorDetails] = React.useState([]);
  useEffect(() => {
    axios
      .get("http://localhost:5000/doctors", {
        headers: {
          "Content-Type": "application/json",
        },
      })
      .then((response) => {
        setDoctorDetails(response.data.response);
        //console.log(response.data.response);
      });
  }, []);
  //console.log(doctorDetails);
  return (
    <>
      {doctorDetails.length > 0 ? (
        doctorDetails.map((doc) => (
          <>
            <ul class="wrapper">
              <div class="landscape">
                <div className="doctor-container neonText">
                  <p>Doctor Name: {doc.doctorName}</p>
                  <p>Doctor Age: {doc.doctorAge}</p>
                  <p>Doctor Contact Number: {doc.doctorNumber}</p>
                  <p>Doctor Qualification: {doc.doctorQualification}</p>
                </div>
              </div>
            </ul>
          </>
        ))
      ) : (
        <div>Sorry, there are no doctors around your location</div>
      )}
    </>
  );
}

export default DoctorDashboard;
