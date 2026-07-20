import { useRef, useState, useEffect, useCallback } from "react";
import { uploadFile, listFiles, deleteFile } from "../api/client";

interface Props {
  projectId: string;
}

interface FileInfo {
  filename: string;
  size: number;
}

export default function FileUpload({ projectId }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [uploading, setUploading] = useState(false);

  const refresh = useCallback(() => {
    listFiles(projectId).then(setFiles).catch(console.error);
  }, [projectId]);

  useEffect(() => { refresh(); }, [refresh]);

  const handleUpload = async () => {
    const file = inputRef.current?.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await uploadFile(projectId, file);
      inputRef.current!.value = "";
      refresh();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      alert(msg);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (filename: string) => {
    try {
      await deleteFile(projectId, filename);
      refresh();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Delete failed";
      alert(msg);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="file-area">
      <details>
        <summary>已上传文件 ({files.length})</summary>
        <div className="upload-row">
          <input ref={inputRef} type="file" accept=".pdf,.docx,.doc" />
          <button onClick={handleUpload} disabled={uploading}>
            {uploading ? "上传中..." : "上传"}
          </button>
        </div>
        <div className="file-list">
          {files.map((f) => (
            <span key={f.filename} className="file-chip">
              {f.filename} ({formatSize(f.size)})
              <button onClick={() => handleDelete(f.filename)}>&times;</button>
            </span>
          ))}
        </div>
      </details>
    </div>
  );
}
