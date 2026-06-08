interface VisionCardProps {
  title: string;
  children: React.ReactNode;
  empty?: boolean;
}

export function VisionCard({ title, children, empty }: VisionCardProps) {
  return (
    <section className="vision-dash-card">
      <h4 className="vision-dash-card-title">{title}</h4>
      {empty ? <p className="vision-dash-empty">No detections</p> : <div className="vision-dash-card-body">{children}</div>}
    </section>
  );
}
