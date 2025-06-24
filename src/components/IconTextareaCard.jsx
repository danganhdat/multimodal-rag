import iconLogo from "../assets/letter-tt-svgrepo-com.svg";
import imageSvg from "../assets/image.svg";
import userSvg from "../assets/user.svg";

const icons = [iconLogo, userSvg, imageSvg];

export default function IconTextareaCard() {
  return (
    <div className="w-[90%] sm:w-[80%] md:w-[70%] lg:w-full p-[6px] bg-white rounded-md shadow-sm space-y-2">
      <div className="flex justify-center relative z-10 -mb-px space-x-[2px]">
        {icons.map((icon, i) => (
          <button
            key={i}
            className={`p-2 rounded transition flex items-center justify-center ${
              i === 0 ? "bg-green-500" : "bg-gray-100 hover:bg-gray-200"
            }`}
          >
            <img
              src={icon}
              alt="icon"
              className={`w-5 h-5 object-contain ${i === 0 ? "invert" : ""}`}
            />
          </button>
        ))}
      </div>
      <div className="border-b border-gray-300"></div>
      <textarea
        placeholder="Your text"
        rows={5}
        className="w-full px-2 py-1 border border-gray-300 rounded-md focus:outline-none focus:ring resize-none"
      />
      <div className="flex justify-end pr-2">
        <span className="px-2 py-1 border border-gray-300 rounded text-sm text-gray-600 hover:bg-gray-100 transition select-none">
          ✎ 0
        </span>
      </div>
    </div>
  );
}
