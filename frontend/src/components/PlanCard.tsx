import { useState } from "react";
import type { PlanOption } from "../api/client";

interface Props {
  options: PlanOption[];
  onSelect: (chosenPlanId: string, customPlanText: string) => void;
  disabled: boolean;
}

export default function PlanCard({ options, onSelect, disabled }: Props) {
  const [selectedId, setSelectedId] = useState<string>("");
  const [customText, setCustomText] = useState("");

  const handleConfirm = () => {
    if (disabled) return;
    if (selectedId) {
      onSelect(selectedId, "");
    } else if (customText.trim()) {
      onSelect("", customText.trim());
    }
  };

  const canConfirm = !!selectedId || customText.trim().length > 0;

  return (
    <div className="plan-card">
      <div className="plan-card-header">
        <span className="plan-card-icon">&#x1F9ED;</span>
        <span>请选择研究方案</span>
      </div>

      <div className="plan-options">
        {options.map((opt) => (
          <div
            key={opt.id}
            className={`plan-option ${selectedId === opt.id ? "selected" : ""}`}
            onClick={() => { if (!disabled) { setSelectedId(opt.id); setCustomText(""); } }}
          >
            <div className="plan-option-title">{opt.title}</div>
            <div className="plan-option-desc">{opt.description}</div>
            <div className="plan-option-tags">
              {opt.pros.map((p, i) => (
                <span key={`pro-${i}`} className="plan-tag pros">{p}</span>
              ))}
              {opt.cons.map((c, i) => (
                <span key={`con-${i}`} className="plan-tag cons">{c}</span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="plan-divider">
        <span>或</span>
      </div>

      <textarea
        className="plan-custom-input"
        placeholder="自定义方案：描述你的研究思路、技术路线、关注重点..."
        rows={3}
        value={customText}
        onChange={(e) => { setCustomText(e.target.value); setSelectedId(""); }}
        disabled={disabled}
      />

      <button
        className="plan-confirm-btn"
        disabled={!canConfirm || disabled}
        onClick={handleConfirm}
      >
        确认选择
      </button>

      <style>{`
        .plan-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 16px;
          margin: 8px 0;
        }
        .plan-card-header {
          font-weight: 600;
          font-size: 14px;
          margin-bottom: 12px;
          display: flex;
          align-items: center;
          gap: 6px;
          color: var(--text);
        }
        .plan-card-icon {
          font-size: 18px;
        }
        .plan-options {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .plan-option {
          border: 2px solid var(--border);
          border-radius: var(--radius);
          padding: 12px;
          cursor: pointer;
          transition: border-color 0.15s, background 0.15s;
        }
        .plan-option:hover {
          border-color: var(--primary);
        }
        .plan-option.selected {
          border-color: var(--primary);
          background: color-mix(in srgb, var(--primary) 8%, transparent);
        }
        .plan-option-title {
          font-weight: 600;
          font-size: 14px;
          color: var(--text);
          margin-bottom: 6px;
        }
        .plan-option-desc {
          font-size: 13px;
          color: var(--text-secondary);
          margin-bottom: 8px;
          line-height: 1.5;
        }
        .plan-option-tags {
          display: flex;
          flex-wrap: wrap;
          gap: 4px;
        }
        .plan-tag {
          font-size: 11px;
          padding: 2px 8px;
          border-radius: 10px;
          white-space: nowrap;
        }
        .plan-tag.pros {
          background: color-mix(in srgb, var(--success) 15%, transparent);
          color: var(--success);
        }
        .plan-tag.cons {
          background: color-mix(in srgb, var(--warning) 15%, transparent);
          color: var(--warning);
        }
        .plan-divider {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 12px 0;
          color: var(--text-secondary);
          font-size: 12px;
        }
        .plan-divider::before,
        .plan-divider::after {
          content: "";
          flex: 1;
          height: 1px;
          background: var(--border);
        }
        .plan-custom-input {
          width: 100%;
          padding: 10px;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          font-size: 13px;
          font-family: inherit;
          resize: vertical;
          background: var(--bg);
          color: var(--text);
        }
        .plan-custom-input:focus {
          outline: none;
          border-color: var(--primary);
        }
        .plan-confirm-btn {
          margin-top: 12px;
          width: 100%;
          padding: 10px;
          background: var(--primary);
          color: #fff;
          border: none;
          border-radius: var(--radius);
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s;
        }
        .plan-confirm-btn:hover:not(:disabled) {
          background: var(--primary-hover);
        }
        .plan-confirm-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
