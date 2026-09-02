import { useState, useEffect } from "react";
import { getPreferences, updatePreferences } from "../api/client";
import type { PreferencesConfig } from "../api/client";

interface Props {
  onClose: () => void;
}

// ---- 表单小控件 ----

function SelectField(props: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="pref-field">
      <span className="pref-label">{props.label}</span>
      <select value={props.value} onChange={(e) => props.onChange(e.target.value)}>
        {props.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function ToggleField(props: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="pref-field">
      <span className="pref-label">{props.label}</span>
      <input
        type="checkbox"
        checked={props.checked}
        onChange={(e) => props.onChange(e.target.checked)}
      />
    </label>
  );
}

function NumberField(props: {
  label: string;
  value: number;
  placeholder?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="pref-field">
      <span className="pref-label">{props.label}</span>
      <input
        type="number"
        value={props.value || ""}
        placeholder={props.placeholder ?? "0"}
        onChange={(e) => props.onChange(parseInt(e.target.value, 10) || 0)}
      />
    </label>
  );
}

function TagField(props: {
  label: string;
  value: string[];
  placeholder?: string;
  onChange: (v: string[]) => void;
}) {
  return (
    <label className="pref-field">
      <span className="pref-label">{props.label}</span>
      <input
        type="text"
        value={props.value.join(", ")}
        placeholder={props.placeholder}
        onChange={(e) =>
          props.onChange(
            e.target.value
              .split(/[,，\s]+/)
              .map((s) => s.trim())
              .filter(Boolean),
          )
        }
      />
    </label>
  );
}

export default function SettingsModal({ onClose }: Props) {
  const [prefs, setPrefs] = useState<PreferencesConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getPreferences()
      .then(setPrefs)
      .catch((e) => setError(e instanceof Error ? e.message : "加载失败"))
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    if (!prefs) return;
    setSaving(true);
    setError("");
    try {
      await updatePreferences(prefs);
      onClose();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "保存失败");
    } finally {
      setSaving(false);
    }
  };

  if (loading || !prefs) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <h3>偏好设置</h3>
          <p style={{ color: "var(--text-secondary)" }}>
            {loading ? "加载中..." : error || "加载失败"}
          </p>
        </div>
      </div>
    );
  }

  const lit = prefs.literature;
  const wrt = prefs.writing;
  const exp = prefs.experiment;
  const set = (patch: Partial<PreferencesConfig>) => setPrefs({ ...prefs, ...patch });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ maxWidth: 720 }} onClick={(e) => e.stopPropagation()}>
        <h3>偏好设置</h3>
        <p style={{ color: "var(--text-secondary)", fontSize: 13, marginBottom: 12 }}>
          通过表单定制助手行为，修改后点击保存即可生效。
        </p>

        <div className="pref-form">
          <section className="pref-section">
            <h4>文献检索</h4>
            <SelectField
              label="来源类型"
              value={lit.source_type}
              onChange={(v) => set({ literature: { ...lit, source_type: v } })}
              options={[
                { value: "", label: "不限" },
                { value: "journal", label: "期刊" },
                { value: "conference", label: "会议" },
                { value: "both", label: "两者" },
              ]}
            />
            <NumberField
              label="最早年份"
              value={lit.year_start}
              onChange={(v) => set({ literature: { ...lit, year_start: v } })}
            />
            <NumberField
              label="最晚年份"
              value={lit.year_end}
              onChange={(v) => set({ literature: { ...lit, year_end: v } })}
            />
            <SelectField
              label="论文类型"
              value={lit.paper_type}
              onChange={(v) => set({ literature: { ...lit, paper_type: v } })}
              options={[
                { value: "", label: "不限" },
                { value: "review", label: "综述" },
                { value: "experimental", label: "实验论文" },
                { value: "both", label: "两者" },
              ]}
            />
            <NumberField
              label="最低引用量"
              value={lit.min_citations}
              onChange={(v) => set({ literature: { ...lit, min_citations: v } })}
            />
            <TagField
              label="偏好期刊/会议"
              value={lit.preferred_venues}
              placeholder="如 NeurIPS, ICML"
              onChange={(v) => set({ literature: { ...lit, preferred_venues: v } })}
            />
            <SelectField
              label="语言"
              value={lit.preferred_language}
              onChange={(v) => set({ literature: { ...lit, preferred_language: v } })}
              options={[
                { value: "", label: "不限" },
                { value: "chinese", label: "中文" },
                { value: "english", label: "英文" },
                { value: "both", label: "中英文" },
              ]}
            />
          </section>

          <section className="pref-section">
            <h4>论文写作</h4>
            <SelectField
              label="文风"
              value={wrt.sentence_style}
              onChange={(v) => set({ writing: { ...wrt, sentence_style: v } })}
              options={[
                { value: "", label: "默认" },
                { value: "concise", label: "简洁" },
                { value: "elaborate", label: "详细" },
              ]}
            />
            <SelectField
              label="图表密度"
              value={wrt.figure_norm}
              onChange={(v) => set({ writing: { ...wrt, figure_norm: v } })}
              options={[
                { value: "", label: "默认" },
                { value: "tight", label: "紧凑" },
                { value: "spacious", label: "宽松" },
              ]}
            />
            <SelectField
              label="摘要风格"
              value={wrt.abstract_style}
              onChange={(v) => set({ writing: { ...wrt, abstract_style: v } })}
              options={[
                { value: "", label: "默认" },
                { value: "structured", label: "结构化" },
                { value: "narrative", label: "叙述式" },
              ]}
            />
            <SelectField
              label="参考文献格式"
              value={wrt.ref_format}
              onChange={(v) => set({ writing: { ...wrt, ref_format: v } })}
              options={[
                { value: "GB/T 7714", label: "GB/T 7714" },
                { value: "APA", label: "APA" },
                { value: "IEEE", label: "IEEE" },
              ]}
            />
            <SelectField
              label="回答语言"
              value={wrt.lang}
              onChange={(v) => set({ writing: { ...wrt, lang: v } })}
              options={[
                { value: "chinese", label: "中文" },
                { value: "english", label: "英文" },
              ]}
            />
          </section>

          <section className="pref-section">
            <h4>实验分析</h4>
            <TagField
              label="评估指标"
              value={exp.metrics}
              placeholder="如 accuracy, F1"
              onChange={(v) => set({ experiment: { ...exp, metrics: v } })}
            />
            <ToggleField
              label="需要对照组"
              checked={exp.require_control}
              onChange={(v) => set({ experiment: { ...exp, require_control: v } })}
            />
            <ToggleField
              label="需要显著性检验"
              checked={exp.significance_test}
              onChange={(v) => set({ experiment: { ...exp, significance_test: v } })}
            />
            <ToggleField
              label="需要消融实验"
              checked={exp.require_ablation}
              onChange={(v) => set({ experiment: { ...exp, require_ablation: v } })}
            />
          </section>

        </div>

        {error && (
          <p style={{ color: "var(--danger)", fontSize: 13, marginTop: 8 }}>{error}</p>
        )}
        <div className="modal-buttons">
          <button className="secondary" onClick={onClose} disabled={saving}>
            取消
          </button>
          <button className="primary" onClick={handleSave} disabled={saving}>
            {saving ? "保存中..." : "保存"}
          </button>
        </div>

        <style>{PREF_STYLE}</style>
      </div>
    </div>
  );
}

const PREF_STYLE = `
  .pref-form { max-height: 60vh; overflow-y: auto; padding-right: 6px; }
  .pref-section { margin-bottom: 16px; }
  .pref-section h4 {
    font-size: 14px; margin: 0 0 8px; color: var(--text);
    border-bottom: 1px solid var(--border); padding-bottom: 6px;
  }
  .pref-field { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 6px 0; }
  .pref-label { font-size: 13px; color: var(--text-secondary); flex-shrink: 0; }
  .pref-field select, .pref-field input[type="text"], .pref-field input[type="number"] {
    flex: 1; max-width: 240px; padding: 6px 8px; border: 1px solid var(--border); border-radius: var(--radius);
    background: var(--bg); color: var(--text); font-size: 13px;
  }
  .pref-field input[type="checkbox"] { width: 16px; height: 16px; }
  .pref-field input:focus, .pref-field select:focus { outline: none; border-color: var(--primary); }
`;
