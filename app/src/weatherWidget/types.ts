export type WeatherIconKind =
  | "clear"
  | "partly-cloudy"
  | "cloudy"
  | "fog"
  | "drizzle"
  | "rain"
  | "snow"
  | "thunder";

export type WeatherForecastDay = {
  dateIso: string;
  dayLabel: string;
  minC: number;
  maxC: number;
  precipMm?: number;
  precipChancePct?: number;
  condition: string;
  iconKind: WeatherIconKind;
  windMaxKmh?: number;
  /** 0–100 for mini temp bar relative to week range */
  tempBarPct: number;
};

export type WeatherWidgetData = {
  placeLabel: string;
  cityName: string;
  regionLabel: string;
  observedAt?: string;
  condition: string;
  iconKind: WeatherIconKind;
  temperatureC: number;
  feelsLikeC?: number;
  humidityPct?: number;
  windKmh?: number;
  windDirectionDeg?: number;
  pressureHpa?: number;
  forecast: WeatherForecastDay[];
  sourceLabel: string;
};
