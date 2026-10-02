import type { SearchResult } from "../types";
import KeyframeCard from "./KeyframeCard";

interface Props {
  result: SearchResult;
  rank: number;
  onClickKeyframe: (videoName: string, ptsTime: number) => void;
}

export default function ResultRow({ result, rank, onClickKeyframe }: Props) {
  const allFrames = [
    ...result.neighbors
      .filter((n) => n.keyframe_idx < result.keyframe_idx)
      .map((n) => ({ idx: n.keyframe_idx, isMain: false, imageUrl: n.image_url })),
    { idx: result.keyframe_idx, isMain: true, imageUrl: result.image_url },
    ...result.neighbors
      .filter((n) => n.keyframe_idx > result.keyframe_idx)
      .map((n) => ({ idx: n.keyframe_idx, isMain: false, imageUrl: n.image_url })),
  ];

  return (
    <div className="box mb-3 result-row">
      <div className="is-flex is-justify-content-space-between is-align-items-center mb-2">
        <div className="is-flex is-align-items-center" style={{ gap: "0.5rem" }}>
          <span className="tag is-dark">#{rank}</span>
          <span className="tag is-info">{result.video_name}</span>
          <span className="tag is-warning">{result.score.toFixed(4)}</span>
          <span className="is-size-7 has-text-grey">
            frame {result.frame_idx} | {result.pts_time.toFixed(1)}s
          </span>
        </div>
        <div className="tags mb-0">
          {result.objects.slice(0, 5).map((obj) => (
            <span key={obj} className="tag is-light is-small">
              {obj}
            </span>
          ))}
          {result.objects.length > 5 && (
            <span className="tag is-light is-small">
              +{result.objects.length - 5}
            </span>
          )}
        </div>
      </div>
      <div className="keyframe-strip">
        {allFrames.map((f) => (
          <KeyframeCard
            key={f.idx}
            imageUrl={f.imageUrl}
            keyframeIdx={f.idx}
            isMain={f.isMain}
            score={f.isMain ? result.score : undefined}
            onClick={() => onClickKeyframe(result.video_name, result.pts_time)}
          />
        ))}
      </div>
    </div>
  );
}
