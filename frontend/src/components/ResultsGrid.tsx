import type { SearchResponse } from "../types";
import ResultRow from "./ResultRow";

interface Props {
  response: SearchResponse | null;
  loading: boolean;
  error: string | null;
  onClickKeyframe: (videoName: string, ptsTime: number) => void;
}

export default function ResultsGrid({
  response,
  loading,
  error,
  onClickKeyframe,
}: Props) {
  if (loading) {
    return (
      <div className="has-text-centered p-6">
        <progress className="progress is-small is-primary" max="100" />
        <p className="mt-2 has-text-grey">Searching...</p>
      </div>
    );
  }

  if (error) {
    return (
      <article className="message is-danger m-4">
        <div className="message-body">{error}</div>
      </article>
    );
  }

  if (!response) {
    return (
      <div className="has-text-centered p-6 has-text-grey">
        <p className="is-size-5">Enter a query and press Search</p>
      </div>
    );
  }

  if (response.results.length === 0) {
    return (
      <div className="has-text-centered p-6 has-text-grey">
        <p>No results found.</p>
      </div>
    );
  }

  return (
    <div className="p-3">
      <div className="is-flex is-justify-content-space-between is-align-items-center mb-3">
        <span className="is-size-7 has-text-grey">
          {response.results.length} results in {response.search_time_ms.toFixed(0)}ms
        </span>
        {Object.keys(response.translations).length > 0 && (
          <span className="tag is-info is-light is-small">
            Translated {Object.keys(response.translations).length} query(s)
          </span>
        )}
      </div>
      {response.results.map((result, i) => (
        <ResultRow
          key={result.id}
          result={result}
          rank={i + 1}
          onClickKeyframe={onClickKeyframe}
        />
      ))}
    </div>
  );
}
