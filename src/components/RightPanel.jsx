import ImageRow from "./ImageRow";

export default function RightPanel({ rows }) {
  return (
    <div className="w-full lg:w-5/6 overflow-auto px-2 sm:px-4 py-2 space-y-6">
      {rows.map((row, idx) => (
        <ImageRow key={idx} row={row} />
      ))}
    </div>
  );
}
