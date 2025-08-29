import React from 'react';
import { Route, Routes } from 'react-router-dom';
import UploadFile from './components/Upload_file'; // Ensure path is correct

const AllRoutes = () => {
  return (
    <Routes>
      <Route path="/upload" element={<UploadFile />} /> {/* Make sure /upload route is set */}
    </Routes>
  );
};

export default AllRoutes;
