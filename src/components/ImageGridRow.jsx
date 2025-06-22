function ImageGridRow({ grid, preview }) {
  return (
    <div className="flex w-full pb-3">
      <div className="flex-[87.5%] grid grid-cols-7 gap-1">
        {grid.map((img, i) => (
          <div
            key={i}
            className="rounded-md overflow-hidden aspect-video bg-amber-50"
          >
            <img src={img} alt="" className="w-full h-full object-cover" />
          </div>
        ))}
      </div>
      <div className="w-[1.5px] bg-stone-900 mx-1.5" />
      <div className="flex-[12.5%]">
        <div className="rounded-md overflow-hidden aspect-video bg-amber-50">
          <img src={preview} alt="" className="w-full h-full object-cover" />
        </div>
      </div>
    </div>
  );
}

export default ImageGridRow;
