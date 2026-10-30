interface Props {
  status: string;
  answer: string;
  evidence: string[];
  guardChanged?: boolean;
  traceId?: string;
  onEnlargeImage?: (url: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
  answered: "is-success",
  uncertain: "is-warning",
  not_found: "is-danger",
  clarify: "is-info",
  error: "is-danger",
};

function kfIdToImageUrl(kfId: string): string {
  const [video, n] = kfId.split("/");
  return `/keyframes/${video}/${n}.jpg`;
}

export default function AnswerCard({
  status,
  answer,
  evidence: rawEvidence,
  guardChanged,
  traceId,
  onEnlargeImage,
}: Props) {
  const evidence = [...new Set(rawEvidence)];
  const color = STATUS_COLORS[status] || "is-light";

  return (
    <div className={`notification ${color} is-light`}>
      <div className="is-flex is-justify-content-space-between is-align-items-center mb-2">
        <span className={`tag ${color}`}>{status}</span>
        {guardChanged && (
          <span className="tag is-warning is-small">guard changed</span>
        )}
        {traceId && (
          <span className="is-size-7 has-text-grey">{traceId}</span>
        )}
      </div>

      <p className="mb-2" style={{ whiteSpace: "pre-wrap" }}>{answer}</p>

      {evidence.length > 0 && (
        <div>
          <span className="is-size-7 has-text-grey">Evidence:</span>
          <div className="is-flex mt-1" style={{ gap: "6px", overflowX: "auto", paddingBottom: "4px" }}>
            {evidence.map((kfId) => {
              const url = kfIdToImageUrl(kfId);
              return (
                <div
                  key={kfId}
                  style={{ minWidth: "120px", maxWidth: "160px", cursor: onEnlargeImage ? "pointer" : "default" }}
                  onClick={() => onEnlargeImage?.(url)}
                >
                  <figure className="image is-16by9" style={{ margin: 0 }}>
                    <img
                      src={url}
                      alt={kfId}
                      loading="lazy"
                      style={{ objectFit: "cover", borderRadius: "3px" }}
                      onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                    />
                  </figure>
                  <span className="is-size-7 has-text-weight-bold">{kfId}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
