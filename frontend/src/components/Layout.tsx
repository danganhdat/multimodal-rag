import { useState } from "react";
import type { QueryItem, SearchConfig, SearchFilters } from "../types";
import { useSearch } from "../hooks/useSearch";
import QueryPanel from "./QueryPanel";
import ResultsGrid from "./ResultsGrid";
import VideoModal from "./VideoModal";
import ImageLightbox from "./ImageLightbox";

export default function Layout() {
  const { results, loading, error, doSearch } = useSearch();
  const [video, setVideo] = useState<{
    videoName: string;
    ptsTime: number;
  } | null>(null);
  const [lightbox, setLightbox] = useState<string | null>(null);

  const handleSearch = (
    queries: QueryItem[],
    config: SearchConfig,
    filters: SearchFilters,
    imageQuery?: string
  ) => {
    doSearch(queries, config, filters, imageQuery);
  };

  const handleClickKeyframe = (videoName: string, ptsTime: number) => {
    setVideo({ videoName, ptsTime });
  };

  const handleEnlargeImage = (imageUrl: string) => {
    setLightbox(imageUrl);
  };

  return (
    <>
      <div className="columns is-gapless app-layout">
        <div className="column is-3 sidebar">
          <QueryPanel onSearch={handleSearch} loading={loading} />
        </div>
        <div className="column results-area">
          <ResultsGrid
            response={results}
            loading={loading}
            error={error}
            onClickKeyframe={handleClickKeyframe}
            onEnlargeImage={handleEnlargeImage}
          />
        </div>
      </div>
      {video && (
        <VideoModal
          videoName={video.videoName}
          ptsTime={video.ptsTime}
          onClose={() => setVideo(null)}
        />
      )}
      {lightbox && (
        <ImageLightbox
          imageUrl={lightbox}
          onClose={() => setLightbox(null)}
        />
      )}
    </>
  );
}
