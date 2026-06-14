export type StartupContext = {
  fetchedAt: number;
  /** ISO8601 local datetime from Time.Now or synthesized. */
  datetime: string;
  timezone: string;
  utcOffset: string;
  abbreviation?: string;
  dst: boolean;
  dayOfWeek: number;
  weekNumber?: number;
  unixtime?: number;
  clientIp?: string;
  countryCode: string;
  countryName: string;
  cityName?: string;
  regionName?: string;
  lat: number;
  lon: number;
  /** Open-Meteo current temp — filled async after startup. */
  localTempC?: number;
  localWeatherCode?: number;
};
