function InputBox() {
  return (
    <div className="p-[6px] mt-2 bg-white rounded-md shadow-sm w-full space-y-2">
      <IconButtonGroup />
      <div className="border-b border-gray-300"></div>
      <textarea
        placeholder="Your text"
        rows={5}
        className="w-full px-2 py-1 border border-gray-300 rounded-md focus:outline-none focus:ring resize-none"
      />
      <div className="flex justify-end pr-2">
        <span className="text-sm text-gray-500">✎ 0</span>
      </div>
    </div>
  );
}

import IconButtonGroup from "./IconButtonGroup";
export default InputBox;
