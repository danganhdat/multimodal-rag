import { useState } from "react";
import type {
  EventQuery,
  ParsedQuery,
  TemporalRelation,
} from "../types";

interface Props {
  parsed: ParsedQuery;
  onChange: (parsed: ParsedQuery) => void;
}

export default function ParsedQueryEditor({ parsed, onChange }: Props) {
  const updateEvent = (idx: number, partial: Partial<EventQuery>) => {
    const events = parsed.events.map((e, i) =>
      i === idx ? { ...e, ...partial } : e
    );
    onChange({ ...parsed, events });
  };

  const removeEvent = (idx: number) => {
    if (parsed.events.length <= 1) return;
    const events = parsed.events.filter((_, i) => i !== idx);
    const removedId = parsed.events[idx].id;
    const temporal_relations = parsed.temporal_relations.filter(
      (r) => r.from_event !== removedId && r.to_event !== removedId
    );
    const answer_event = Math.min(parsed.answer_event, events.length - 1);
    const classification = events.length < 2 ? "single_scene" : parsed.classification;
    onChange({ ...parsed, events, temporal_relations, answer_event, classification });
  };

  const addEvent = () => {
    const nextId = `E${parsed.events.length + 1}`;
    const events = [
      ...parsed.events,
      {
        id: nextId,
        text_vi: "",
        text_en: "",
        ocr: [],
        objects: [],
        variants: [],
        enabled: true,
      } as EventQuery,
    ];
    onChange({ ...parsed, events });
  };

  const setClassification = (c: "single_scene" | "multi_event") => {
    const updates: Partial<ParsedQuery> = { classification: c };
    if (c === "single_scene") {
      updates.temporal_relations = [];
      updates.requires_temporal = false;
    }
    onChange({ ...parsed, ...updates });
  };

  const toggleTemporal = (fromId: string, toId: string) => {
    const exists = parsed.temporal_relations.find(
      (r) => r.from_event === fromId && r.to_event === toId
    );
    let temporal_relations: TemporalRelation[];
    if (exists) {
      temporal_relations = parsed.temporal_relations.filter(
        (r) => !(r.from_event === fromId && r.to_event === toId)
      );
    } else {
      temporal_relations = [
        ...parsed.temporal_relations,
        { from_event: fromId, to_event: toId, type: "before" as const },
      ];
    }
    const requires_temporal = temporal_relations.length > 0;
    onChange({ ...parsed, temporal_relations, requires_temporal });
  };

  const removeChip = (
    idx: number,
    field: "ocr" | "objects" | "variants",
    value: string
  ) => {
    const arr = parsed.events[idx][field].filter((v) => v !== value);
    updateEvent(idx, { [field]: arr });
  };

  const addChip = (
    idx: number,
    field: "ocr" | "objects" | "variants",
    value: string
  ) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    const arr = parsed.events[idx][field];
    if (!arr.includes(trimmed)) {
      updateEvent(idx, { [field]: [...arr, trimmed] });
    }
  };

  return (
    <div className="parsed-query-editor">
      {parsed.fallback && (
        <div className="notification is-warning is-light py-2 px-3 mb-2">
          <small>LLM unavailable — using translation + regex fallback</small>
        </div>
      )}

      {/* Classification badge */}
      <div className="is-flex is-align-items-center mb-2" style={{ gap: "0.5rem" }}>
        <span className="is-size-7 has-text-grey">Classification:</span>
        <div className="buttons has-addons mb-0">
          <button
            className={`button is-small ${parsed.classification === "single_scene" ? "is-info is-selected" : ""}`}
            onClick={() => setClassification("single_scene")}
          >
            Single scene
          </button>
          <button
            className={`button is-small ${parsed.classification === "multi_event" ? "is-info is-selected" : ""}`}
            onClick={() => setClassification("multi_event")}
          >
            Multi-event
          </button>
        </div>
        {parsed.requires_temporal && (
          <span className="tag is-link is-light is-small">temporal</span>
        )}
      </div>

      <div className="field mb-2">
        <label className="label is-small mb-1">Question</label>
        <input
          className="input is-small"
          value={parsed.question}
          onChange={(e) => onChange({ ...parsed, question: e.target.value })}
          placeholder="What is the question to answer?"
        />
      </div>

      <div className="is-flex mb-2" style={{ gap: "0.5rem" }}>
        <div className="field mb-0">
          <label className="label is-small mb-1">Answer type</label>
          <div className="select is-small">
            <select
              value={parsed.answer_type}
              onChange={(e) =>
                onChange({ ...parsed, answer_type: e.target.value })
              }
            >
              <option value="text">text</option>
              <option value="number">number</option>
              <option value="yes_no">yes/no</option>
              <option value="count">count</option>
              <option value="entity">entity</option>
            </select>
          </div>
        </div>
        <div className="field mb-0">
          <label className="label is-small mb-1">Answer event</label>
          <div className="select is-small">
            <select
              value={parsed.answer_event}
              onChange={(e) =>
                onChange({ ...parsed, answer_event: Number(e.target.value) })
              }
            >
              {parsed.events.map((ev, i) => (
                <option key={i} value={i}>
                  {ev.id || `Event ${i + 1}`}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {parsed.events.map((event, idx) => (
        <EventCard
          key={event.id || idx}
          event={event}
          index={idx}
          isAnswer={idx === parsed.answer_event}
          canRemove={parsed.events.length > 1}
          onUpdate={(partial) => updateEvent(idx, partial)}
          onRemove={() => removeEvent(idx)}
          onRemoveChip={(field, val) => removeChip(idx, field, val)}
          onAddChip={(field, val) => addChip(idx, field, val)}
        />
      ))}

      <button
        className="button is-small is-outlined is-fullwidth mb-2"
        onClick={addEvent}
      >
        + Add event
      </button>

      {/* Temporal relations editor */}
      {parsed.classification === "multi_event" && parsed.events.length >= 2 && (
        <TemporalRelationsEditor
          events={parsed.events}
          relations={parsed.temporal_relations}
          onToggle={toggleTemporal}
        />
      )}
    </div>
  );
}

/* ── Event card ── */

interface EventCardProps {
  event: EventQuery;
  index: number;
  isAnswer: boolean;
  canRemove: boolean;
  onUpdate: (partial: Partial<EventQuery>) => void;
  onRemove: () => void;
  onRemoveChip: (field: "ocr" | "objects" | "variants", value: string) => void;
  onAddChip: (field: "ocr" | "objects" | "variants", value: string) => void;
}

function EventCard({
  event,
  index,
  isAnswer,
  canRemove,
  onUpdate,
  onRemove,
  onRemoveChip,
  onAddChip,
}: EventCardProps) {
  const [ocrInput, setOcrInput] = useState("");
  const [objInput, setObjInput] = useState("");
  const [varInput, setVarInput] = useState("");

  const handleKey = (
    field: "ocr" | "objects" | "variants",
    value: string,
    setter: (v: string) => void
  ) => (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      onAddChip(field, value);
      setter("");
    }
  };

  return (
    <div
      className={`box p-2 mb-2 ${!event.enabled ? "has-background-light" : ""}`}
      style={{
        borderLeft: isAnswer ? "3px solid hsl(204, 86%, 53%)" : undefined,
        opacity: event.enabled ? 1 : 0.5,
      }}
    >
      <div className="is-flex is-justify-content-space-between is-align-items-center mb-1">
        <span className="is-size-7 has-text-weight-bold">
          {event.id || `Event ${index + 1}`}
          {isAnswer && (
            <span className="tag is-info is-light is-small ml-1">answer</span>
          )}
        </span>
        <div className="is-flex is-align-items-center" style={{ gap: "0.5rem" }}>
          <label className="checkbox is-size-7">
            <input
              type="checkbox"
              checked={event.enabled}
              onChange={(e) => onUpdate({ enabled: e.target.checked })}
            />{" "}
            on
          </label>
          {canRemove && (
            <button
              className="delete is-small"
              title="Remove event"
              onClick={onRemove}
            />
          )}
        </div>
      </div>

      <div className="field mb-1">
        <input
          className="input is-small"
          value={event.text_en}
          onChange={(e) => onUpdate({ text_en: e.target.value })}
          placeholder="English scene description"
          disabled={!event.enabled}
        />
      </div>
      <div className="field mb-1">
        <input
          className="input is-small"
          value={event.text_vi}
          onChange={(e) => onUpdate({ text_vi: e.target.value })}
          placeholder="Vietnamese (optional)"
          disabled={!event.enabled}
        />
      </div>

      {/* Variants */}
      <ChipRow
        label="Variants"
        items={event.variants}
        tagClass="is-link is-light"
        enabled={event.enabled}
        inputValue={varInput}
        onInputChange={setVarInput}
        onKeyDown={handleKey("variants", varInput, setVarInput)}
        onRemove={(v) => onRemoveChip("variants", v)}
        placeholder="+ variant"
      />

      {/* OCR */}
      <ChipRow
        label="OCR"
        items={event.ocr}
        tagClass="is-warning is-light"
        enabled={event.enabled}
        inputValue={ocrInput}
        onInputChange={setOcrInput}
        onKeyDown={handleKey("ocr", ocrInput, setOcrInput)}
        onRemove={(v) => onRemoveChip("ocr", v)}
        placeholder="+ Enter"
      />

      {/* Objects */}
      <ChipRow
        label="Objects"
        items={event.objects}
        tagClass="is-info is-light"
        enabled={event.enabled}
        inputValue={objInput}
        onInputChange={setObjInput}
        onKeyDown={handleKey("objects", objInput, setObjInput)}
        onRemove={(v) => onRemoveChip("objects", v)}
        placeholder="+ Enter"
      />
    </div>
  );
}

/* ── Reusable chip row ── */

interface ChipRowProps {
  label: string;
  items: string[];
  tagClass: string;
  enabled: boolean;
  inputValue: string;
  onInputChange: (v: string) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onRemove: (v: string) => void;
  placeholder: string;
}

function ChipRow({
  label,
  items,
  tagClass,
  enabled,
  inputValue,
  onInputChange,
  onKeyDown,
  onRemove,
  placeholder,
}: ChipRowProps) {
  return (
    <div className="is-flex mb-1" style={{ gap: "0.25rem", flexWrap: "wrap" }}>
      <span className="is-size-7 has-text-grey mr-1">{label}:</span>
      {items.map((v) => (
        <span key={v} className={`tag ${tagClass} is-small`}>
          {v}
          {enabled && (
            <button className="delete is-small" onClick={() => onRemove(v)} />
          )}
        </span>
      ))}
      {enabled && (
        <input
          className="input is-small"
          style={{ width: "100px", height: "24px", fontSize: "0.7rem" }}
          value={inputValue}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
        />
      )}
    </div>
  );
}

/* ── Temporal relations editor ── */

interface TemporalRelationsEditorProps {
  events: EventQuery[];
  relations: TemporalRelation[];
  onToggle: (fromId: string, toId: string) => void;
}

function TemporalRelationsEditor({
  events,
  relations,
  onToggle,
}: TemporalRelationsEditorProps) {
  const hasRelation = (fromId: string, toId: string) =>
    relations.some((r) => r.from_event === fromId && r.to_event === toId);

  return (
    <div className="box p-2 mb-2">
      <span className="is-size-7 has-text-weight-bold">Temporal relations</span>
      <div className="mt-1">
        {events.map((from, i) =>
          events.map((to, j) => {
            if (i >= j) return null;
            const active = hasRelation(from.id, to.id);
            return (
              <div
                key={`${from.id}-${to.id}`}
                className="is-flex is-align-items-center mb-1"
                style={{ gap: "0.5rem" }}
              >
                <label className="checkbox is-size-7">
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={() => onToggle(from.id, to.id)}
                  />{" "}
                  {from.id} <span className="has-text-grey">before</span> {to.id}
                </label>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
