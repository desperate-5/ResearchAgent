export function renderMarkdown(text: string): string {
  // 先转义已有的 HTML，防止 XSS
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // 代码块 (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, "<pre><code>$2</code></pre>");

  // 行内代码 (`...`)
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // 标题
  html = html.replace(/^###### (.+)$/gm, "<h6>$1</h6>");
  html = html.replace(/^##### (.+)$/gm, "<h5>$1</h5>");
  html = html.replace(/^#### (.+)$/gm, "<h4>$1</h4>");
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

  // 粗体 + 斜体
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // 图片（必须在链接之前处理）
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%">');

  // 链接
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

  // 无序列表
  html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

  // 有序列表
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // 水平线
  html = html.replace(/^---$/gm, "<hr>");

  // 引用
  html = html.replace(/^&gt; (.+)$/gm, "<blockquote>$1</blockquote>");

  // 表格 (需要先于段落处理)
  html = html.replace(/(^\|.+\|\n^\|[-: |]+\|\n(?:^\|.+\|\n?)+)/gm, (match) => {
    const lines = match.trim().split("\n");
    if (lines.length < 2) return match;

    const parseRow = (line: string, tag: "th" | "td") => {
      const cells = line.replace(/^\||\|$/g, "").split("|");
      return `<tr>${cells.map((c) => `<${tag}>${c.trim()}</${tag}>`).join("")}</tr>`;
    };

    const thead = `<thead>${parseRow(lines[0], "th")}</thead>`;
    const tbody = `<tbody>${lines.slice(2).map((l) => parseRow(l, "td")).join("")}</tbody>`;
    return `<table>${thead}${tbody}</table>`;
  });

  // 引用标记: [1], [2,3], [1,3,5] → 可点击上标（仅处理未被 HTML 包裹的纯文本位置）
  html = html.replace(
    /\[(\d+(?:[,\s]*\d+)*)\]/g,
    '<sup class="citation-marker" data-source-num="$1">[$1]</sup>'
  );

  // 段落：连续的换行分隔
  const blocks = html.split(/\n\n+/);
  html = blocks
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      // 已经是块级元素就不包裹
      if (/^<(h[1-6]|ul|ol|pre|blockquote|hr|li|table|img)/.test(trimmed)) {
        return trimmed;
      }
      // 单换行转 <br>
      const withBreaks = trimmed.replace(/\n/g, "<br>");
      return `<p>${withBreaks}</p>`;
    })
    .join("\n");

  return html;
}
