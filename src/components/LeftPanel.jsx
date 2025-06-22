import InputBox from "./InputBox";

function LeftPanel() {
  return (
    <div className="w-1/6 bg-gray-100 p-2">
      <InputBox />
      <InputBox />
    </div>
  );
}

export default LeftPanel;
