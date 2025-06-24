export default function PreviewBox({ src }) {
  return (
    <div className="w-[130px] h-[80px] rounded-md overflow-hidden bg-amber-50">
      <img src={src} alt="preview" className="w-full h-full object-cover" />
    </div>
  );
}
