import LeftPanel from "./components/LeftPanel";
import ImageGridRow from "./components/ImageGridRow";
import { rows } from "./data/dummyData";

function App() {
  return (
    <div className="flex h-screen">
      <LeftPanel />
      <div className="w-5/6 overflow-auto p-2 space-y-4">
        {rows.map((row, idx) => (
          <div
            key={idx}
            className={`${
              idx !== rows.length - 1 ? "border-b-4 border-gray-300" : ""
            }`}
          >
            <ImageGridRow grid={row.grid} preview={row.preview} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
