export type TimeWidgetData = {
  placeLabel: string;
  timezone: string;
  /** ISO instant shown when the widget was resolved. */
  anchorIso: string;
  utcOffsetLabel: string;
  dstActive?: boolean;
};
