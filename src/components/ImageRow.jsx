import ImageGrid from "./ImageGrid";
import PreviewBox from "./PreviewBox";

export default function ImageRow({ row }) {
  return (
    <div
      className="flex flex-col lg:flex-row gap-2 
      pb-4 p-2 lg:p-0
      border  lg:border-x-0 lg:border-t-0 lg:border-b-2 
      lg:pb-3
      border-stone-800 lg:border-gray-200 rounded-md"
    >
      <ImageGrid images={row.grid} />
      <div className="w-full h-[1px] my-2 lg:my-0 lg:h-auto lg:w-[1.5px] bg-stone-900 lg:mx-1.5" />
      <PreviewBox src={row.preview} />
    </div>
  );
}
