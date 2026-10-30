import type { SearchResult, V1SearchResult } from "../types";
import KeyframeCard from "./KeyframeCard";

type AnyResult = SearchResult | V1SearchResult;

interface Props {
  result: AnyResult;
  rank: number;
  onClickKeyframe: (videoName: string, ptsTime: number) => void;
  onEnlargeImage: (imageUrl: string) => void;
  upvoted?: boolean;
  downvoted?: boolean;
  onUpvote?: (id: string) => void;
  onDownvote?: (id: string) => void;
}

export default function ResultRow({
  result, rank, onClickKeyframe, onEnlargeImage,
  upvoted, downvoted, onUpvote, onDownvote,
}: Props) {
  const sources = "sources" in result ? (result as V1SearchResult).sources : null;
  const neighbors = result.neighbors ?? [];
  const allFrames = [
    ...neighbors
      .filter((n) => n.keyframe_idx < result.keyframe_idx)
      .map((n) => ({ idx: n.keyframe_idx, isMain: false, imageUrl: n.image_url, ptsTime: n.pts_time })),
    { idx: result.keyframe_idx, isMain: true, imageUrl: result.image_url, ptsTime: result.pts_time ?? 0 },
    ...neighbors
      .filter((n) => n.keyframe_idx > result.keyframe_idx)
      .map((n) => ({ idx: n.keyframe_idx, isMain: false, imageUrl: n.image_url, ptsTime: n.pts_time })),
  ];

  return (
    <div className="box mb-3 result-row">
      <div className="is-flex is-justify-content-space-between is-align-items-center mb-2">
        <div className="is-flex is-align-items-center" style={{ gap: "0.5rem" }}>
          <span className="tag is-dark">#{rank}</span>
          <span className="tag is-info">{result.video_name}</span>
          <span className="tag is-warning">{(result.score ?? 0).toFixed(4)}</span>
          <span className="tag is-light">
            frame {result.frame_idx ?? 0} | {(result.pts_time ?? 0).toFixed(1)}s
          </span>
          <span
            className="tag is-dark"
            style={{ cursor: "pointer" }}
            title="Play video"
            onClick={() => onClickKeyframe(result.video_name, result.pts_time)}
          >
            ▶ Play
          </span>
        </div>
        <div className="is-flex is-align-items-center" style={{ gap: "0.25rem" }}>
          {sources && (
            <div className="tags mb-0 mr-2">
              {Object.entries(sources).map(([src, r]) => (
                <span key={src} className="tag is-light is-small" title={`${src} rank ${r}`}>
                  {src}:{r}
                </span>
              ))}
            </div>
          )}
          {onUpvote && (
            <button
              className={`button is-small ${upvoted ? "is-success" : "is-light"}`}
              onClick={() => onUpvote(result.id)}
              title="Use as search signal"
              style={{ minWidth: "28px", height: "24px", padding: "0 4px" }}
            >
              +
            </button>
          )}
          {onDownvote && (
            <button
              className={`button is-small ${downvoted ? "is-danger" : "is-light"}`}
              onClick={() => onDownvote(result.id)}
              title="Penalize similar frames"
              style={{ minWidth: "28px", height: "24px", padding: "0 4px" }}
            >
              -
            </button>
          )}
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
      </div>
      <div className="keyframe-strip">
        {allFrames.map((f) => (
          <KeyframeCard
            key={f.idx}
            imageUrl={f.imageUrl}
            keyframeIdx={f.idx}
            isMain={f.isMain}
            score={f.isMain ? result.score : undefined}
            onEnlarge={() => onEnlargeImage(f.imageUrl)}
          />
        ))}
      </div>
    </div>
  );
}
