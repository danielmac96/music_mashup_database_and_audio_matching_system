// Start a same-origin download from a click handler or a job's completion: an
// anchor with `download`, clicked and removed. A file the server sends with
// Content-Disposition: attachment saves without leaving the page.
export function startDownload(url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  a.rel = "noreferrer";
  document.body.appendChild(a);
  a.click();
  a.remove();
}
