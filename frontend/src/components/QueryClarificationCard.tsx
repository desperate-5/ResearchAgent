interface Props {
  directions: string[];
  onSelectDirection: (direction: string) => void;
  onUseOriginal: () => void;
  disabled: boolean;
}

export default function QueryClarificationCard({
  directions,
  onSelectDirection,
  onUseOriginal,
  disabled,
}: Props) {
  return (
    <div className="clarify-card">
      <div className="clarify-header">
        <span className="clarify-icon">🔍</span>
        <span>您的问题可能不够具体，请选择一个检索方向</span>
      </div>
      <div className="clarify-options">
        {directions.map((d, i) => (
          <div
            key={i}
            className="clarify-option"
            onClick={() => { if (!disabled) onSelectDirection(d); }}
          >
            {d}
          </div>
        ))}
      </div>
      <button className="clarify-btn ghost" disabled={disabled} onClick={onUseOriginal}>
        按原样检索
      </button>
      <style>{CLARIFY_STYLE}</style>
    </div>
  );
}

const CLARIFY_STYLE = `
  .clarify-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    margin: 8px 0;
  }
  .clarify-header {
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text);
  }
  .clarify-icon { font-size: 18px; }
  .clarify-options {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .clarify-option {
    border: 2px solid var(--border);
    border-radius: var(--radius);
    padding: 12px;
    cursor: pointer;
    font-size: 13px;
    color: var(--text);
    transition: border-color 0.15s, background 0.15s;
  }
  .clarify-option:hover {
    border-color: var(--primary);
    background: color-mix(in srgb, var(--primary) 8%, transparent);
  }
  .clarify-btn {
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
  .clarify-btn:hover:not(:disabled) {
    background: var(--primary-hover);
  }
  .clarify-btn.ghost {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border);
    font-weight: 500;
  }
  .clarify-btn.ghost:hover:not(:disabled) {
    background: var(--bg);
    color: var(--text);
  }
  .clarify-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;
