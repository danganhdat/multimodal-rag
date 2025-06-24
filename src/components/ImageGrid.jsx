export default function ImageGrid({ images }) {
  return (
    <div className="lg:flex-[87.5%] flex flex-wrap gap-[4px]">
      {images.map((img, i) => (
        <div
          key={i}
          className="rounded-md overflow-hidden w-[130px] h-[80px] bg-amber-50"
        >
          <img src={img} alt="thumb" className="w-full h-full object-cover" />
        </div>
      ))}
    </div>
  );
}
