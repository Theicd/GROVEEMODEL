export type GlobeGaugeCard = {
  id: string;
  label: string;
  value: string;
  sub: string;
  severity: number;
  icon: string;
};

type Props = {
  gauges: GlobeGaugeCard[];
  updatedAt?: string;
};

const sevBorder = (s: number): string => {
  if (s >= 5) return "#ff1744";
  if (s >= 4) return "#ff9100";
  if (s >= 3) return "#ffd600";
  return "#00e5ff";
};

export function GlobeGaugeStrip({ gauges, updatedAt }: Props) {
  if (!gauges.length) return null;

  return (
    <div className="globe-gauge-strip" dir="ltr">
      {gauges.map((g) => (
        <div
          key={g.id}
          className="globe-gauge-card"
          style={{ borderColor: sevBorder(g.severity) }}
        >
          <div className="globe-gauge-card-head">
            <span className="globe-gauge-card-icon" aria-hidden="true">
              {g.icon}
            </span>
            <span className="globe-gauge-card-label">{g.label}</span>
          </div>
          <div className="globe-gauge-card-value">{g.value}</div>
          <div className="globe-gauge-card-sub">{g.sub}</div>
        </div>
      ))}
      {updatedAt ? <div className="globe-gauge-time">THER {updatedAt}</div> : null}
    </div>
  );
}
