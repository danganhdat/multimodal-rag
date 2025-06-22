import iconLogo from "../assets/letter-tt-svgrepo-com.svg";
import userSvg from "../assets/user.svg";
import imageSvg from "../assets/image.svg";

const icons = [iconLogo, userSvg, imageSvg];

function IconButtonGroup() {
  return (
    <div className="flex justify-center relative z-10 -mb-px space-x-[2px]">
      {icons.map((icon, idx) => (
        <button
          key={idx}
          className={`p-2 rounded transition flex items-center justify-center ${
            idx === 0 ? "bg-green-500" : "bg-gray-100 hover:bg-gray-200"
          }`}
        >
          <img
            src={icon}
            alt="icon"
            className={`w-5 h-5 object-contain ${idx === 0 ? "invert" : ""}`}
          />
        </button>
      ))}
    </div>
  );
}

export default IconButtonGroup;
