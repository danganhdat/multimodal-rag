import imageScene from "./assets/imageScene.jpeg";
import LeftPanel from "./components/LeftPanel";
import RightPanel from "./components/RightPanel";

const rows = [
  { grid: Array(10).fill(imageScene), preview: imageScene },
  { grid: Array(4).fill(imageScene), preview: imageScene },
  { grid: Array(7).fill(imageScene), preview: imageScene },
];

function App() {
  return (
    <div className="flex flex-col lg:flex-row h-screen">
      <LeftPanel />
      <RightPanel rows={rows} />
    </div>
  );
}

export default App;
