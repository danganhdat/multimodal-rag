import { useEffect, useState } from "react";
import { useTranslation } from "../hooks/useTranslation";

interface Props {
  index: number;
  text: string;
  lang: "en" | "vi";
  canRemove: boolean;
  onTextChange: (text: string) => void;
  onLangChange: (lang: "en" | "vi") => void;
  onRemove: () => void;
}

export default function QueryBox({
  index,
  text,
  lang,
  canRemove,
  onTextChange,
  onLangChange,
  onRemove,
}: Props) {
  const [translated, setTranslated] = useState("");
  const { translateDebounced } = useTranslation();

  useEffect(() => {
    if (lang === "vi" && text.trim()) {
      translateDebounced(text, "vi", "en", setTranslated);
    } else {
      setTranslated("");
    }
  }, [text, lang, translateDebounced]);

  return (
    <div className="box mb-3">
      <div className="is-flex is-justify-content-space-between is-align-items-center mb-2">
        <span className="tag is-info is-light">Query {index + 1}</span>
        <div className="is-flex is-align-items-center" style={{ gap: "0.5rem" }}>
          <div className="buttons has-addons mb-0">
            <button
              className={`button is-small ${lang === "en" ? "is-link" : ""}`}
              onClick={() => onLangChange("en")}
            >
              EN
            </button>
            <button
              className={`button is-small ${lang === "vi" ? "is-link" : ""}`}
              onClick={() => onLangChange("vi")}
            >
              VI
            </button>
          </div>
          {canRemove && (
            <button
              className="delete is-small"
              onClick={onRemove}
              title="Remove query"
            />
          )}
        </div>
      </div>
      <textarea
        className="textarea is-small"
        rows={2}
        placeholder={lang === "en" ? "Enter query in English..." : "Nhập truy vấn tiếng Việt..."}
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
      />
      {lang === "vi" && translated && (
        <p className="help has-text-grey mt-1">
          <strong>EN:</strong> {translated}
        </p>
      )}
    </div>
  );
}
