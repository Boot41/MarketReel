import { Route, Routes, Navigate } from "react-router-dom";
import { Box, Typography } from "@mui/material";

function HomePage() {
  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
    >
      <Typography variant="h3">App Scaffold</Typography>
    </Box>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
