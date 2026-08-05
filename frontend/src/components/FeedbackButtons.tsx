import { useState } from "react";
import { sendFeedback } from "../api/client";

const QUICK_TAGS = [
  "太啰嗦",
  "不够详细",
  "需要英文文献",
  "需要最新文献",
];

export default function FeedbackButtons() {
  const [liked, setLiked] = useState<string | null>(null);
  const [activeTags, setActiveTags] = useState<Set<string>>(new Set());

  const handleLike = (type: "like" | "dislike") => {
    const next = liked === type ? null : type;
    setLiked(next);
    sendFeedback(type).catch(console.error);
  };

  const handleTag = (tag: string) => {
    setActiveTags((prev) => {
      const next = new Set(prev);
      if (next.has(tag)) {
        next.delete(tag);
      } else {
        next.add(tag);
      }
      return next;
    });
    sendFeedback("dislike", tag).catch(console.error);
  };

  return (
    <div className="feedback-row">
      <button
        className={liked === "like" ? "liked" : ""}
        onClick={() => handleLike("like")}
      >
        {liked === "like" ? "已采纳" : "采纳"}
      </button>
      <button
        className={liked === "dislike" ? "disliked" : ""}
        onClick={() => handleLike("dislike")}
      >
        {liked === "dislike" ? "已标记" : "不采纳"}
      </button>
      {QUICK_TAGS.map((tag) => (
        <button
          key={tag}
          className={activeTags.has(tag) ? "disliked" : ""}
          onClick={() => handleTag(tag)}
        >
          {tag}
        </button>
      ))}
    </div>
  );
}
