import { useCallback, useState } from "react";
import * as api from "../api/client";
import type { SearchConfig, SearchFilters, SearchResponse, QueryItem } from "../types";
import { DEFAULT_CONFIG } from "../types";

interface UseSearchReturn {
  results: SearchResponse | null;
  loading: boolean;
  error: string | null;
  doSearch: (
    queries: QueryItem[],
    config?: SearchConfig,
    filters?: SearchFilters,
    imageQuery?: string
  ) => Promise<void>;
}

export function useSearch(): UseSearchReturn {
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const doSearch = useCallback(
    async (
      queries: QueryItem[],
      config: SearchConfig = DEFAULT_CONFIG,
      filters?: SearchFilters,
      imageQuery?: string
    ) => {
      setLoading(true);
      setError(null);
      try {
        const hasFilters = filters && (filters.objects.length > 0 || filters.ocr.length > 0);
        const resp = await api.search({
          queries,
          config,
          filters: hasFilters ? filters : undefined,
          image_query: imageQuery,
        });
        setResults(resp);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Search failed");
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { results, loading, error, doSearch };
}
