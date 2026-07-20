import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import ProjectListPage from "./pages/ProjectListPage";
import ChatPage from "./pages/ChatPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<ProjectListPage />} />
        <Route path="chat/:projectId" element={<ChatPage />} />
      </Route>
    </Routes>
  );
}
