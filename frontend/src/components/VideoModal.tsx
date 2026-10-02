import { useEffect, useRef, useState } from "react";
import { getVideoUrl, getVideoMeta } from "../api/client";
import type { VideoMeta } from "../types";

interface Props {
  videoName: string;
  ptsTime: number;
  onClose: () => void;
}

export default function VideoModal({ videoName, ptsTime, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [meta, setMeta] = useState<VideoMeta | null>(null);

  useEffect(() => {
    getVideoMeta(videoName).then(setMeta).catch(() => {});
  }, [videoName]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = ptsTime;
    }
  };

  return (
    <div className="modal is-active">
      <div className="modal-background" onClick={onClose} />
      <div className="modal-card" style={{ width: "80vw", maxWidth: "1200px" }}>
        <header className="modal-card-head">
          <p className="modal-card-title">
            {meta?.title || videoName}
          </p>
          <button className="delete" onClick={onClose} />
        </header>
        <section className="modal-card-body p-0">
          <video
            ref={videoRef}
            controls
            autoPlay
            onLoadedMetadata={handleLoadedMetadata}
            style={{ width: "100%", display: "block" }}
          >
            <source src={getVideoUrl(videoName)} type="video/mp4" />
          </video>
          {meta && (
            <div className="p-4">
              <div className="is-flex is-flex-wrap-wrap" style={{ gap: "0.5rem" }}>
                {meta.author && (
                  <span className="tag is-info is-light">{meta.author}</span>
                )}
                {meta.publish_date && (
                  <span className="tag is-light">{meta.publish_date}</span>
                )}
                {meta.length && (
                  <span className="tag is-light">
                    {Math.floor(meta.length / 60)}m {meta.length % 60}s
                  </span>
                )}
              </div>
              {meta.description && (
                <p className="is-size-7 has-text-grey mt-2" style={{ maxHeight: "100px", overflow: "auto" }}>
                  {meta.description}
                </p>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
