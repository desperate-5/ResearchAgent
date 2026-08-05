import { useState, useEffect } from "react";
import { getRawPreferences, updateRawPreferences } from "../api/client";

interface Props {
  onClose: () => void;
}

export default function SettingsModal({ onClose }: Props) {
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getRawPreferences()
      .then(setContent)
      .catch((e) => setError(e instanceof Error ? e.message : "加载失败"))
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setError("");
    try {
      await updateRawPreferences(content);
      onClose();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "保存失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal"
        style={{ maxWidth: 780 }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3>偏好设置</h3>
        <p style={{ color: "var(--text-secondary)", fontSize: 13, marginBottom: 12 }}>
          编辑 YAML 配置以定制助手行为。修改后点击保存即可生效。
        </p>
        {loading ? (
          <p style={{ color: "var(--text-secondary)" }}>加载中...</p>
        ) : (
          <>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              spellCheck={false}
            />
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
          </>
        )}
      </div>
    </div>
  );
}
