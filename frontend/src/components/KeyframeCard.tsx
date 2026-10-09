interface Props {
  imageUrl: string;
  keyframeIdx: number;
  isMain?: boolean;
  score?: number;
  onEnlarge?: () => void;
}

export default function KeyframeCard({
  imageUrl,
  keyframeIdx,
  isMain = false,
  score,
  onEnlarge,
}: Props) {
  return (
    <div
      className={`keyframe-card ${isMain ? "is-main" : "is-neighbor"}`}
      style={{ position: "relative", cursor: onEnlarge ? "pointer" : "default" }}
      onClick={onEnlarge}
    >
      <figure className="image is-16by9">
        <img
          src={imageUrl}
          alt={`keyframe ${keyframeIdx}`}
          loading="lazy"
        />
      </figure>
      <div className="keyframe-label">
        <span className="is-size-7">{keyframeIdx}</span>
        {score !== undefined && (
          <span className="tag is-small is-warning">{score.toFixed(3)}</span>
        )}
      </div>
    </div>
  );
}
