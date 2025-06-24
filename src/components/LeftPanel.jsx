import IconTextareaCard from "./IconTextareaCard";

export default function LeftPanel() {
  return (
    <div className="w-full lg:w-1/6 bg-gray-100 p-2">
      <div className="flex flex-col items-center space-y-4">
        {[0, 1].map((_, idx) => (
          <IconTextareaCard key={idx} />
        ))}
      </div>
    </div>
  );
}
