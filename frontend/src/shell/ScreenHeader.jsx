// The 46px bar at the top of every screen: title, a mono counts line, then
// whatever that screen needs on the right. Its height is fixed because the pair
// dock's header sits at the same height beside it and the two must line up.
export function ScreenHeader({ title, sub, children }) {
  return (
    <header className="screen-bar">
      {title && <h1>{title}</h1>}
      {sub && <span className="screen-sub mono">{sub}</span>}
      {children}
    </header>
  );
}
