import { useEffect } from "react";

interface Props {
  imageUrl: string;
  onClose: () => void;
}

export default function ImageLightbox({ imageUrl, onClose }: Props) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  return (
    <div className="modal is-active" onClick={onClose}>
      <div className="modal-background" />
      <div
        className="modal-content"
        style={{ maxWidth: "90vw", maxHeight: "90vh", display: "flex", justifyContent: "center", alignItems: "center" }}
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={imageUrl}
          alt="Enlarged keyframe"
          style={{ maxWidth: "100%", maxHeight: "90vh", objectFit: "contain" }}
        />
      </div>
      <button className="modal-close is-large" aria-label="close" onClick={onClose} />
    </div>
  );
}
