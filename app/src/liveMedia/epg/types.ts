export type EpgProgram = {
  channelId: string;
  title: string;
  description?: string;
  start: Date;
  end: Date;
  category?: string;
  poster?: string;
  season?: number;
  episode?: number;
  episodeLabel?: string;
  subTitle?: string;
  /** Runtime from XMLTV &lt;length&gt; when shorter than the schedule slot. */
  lengthMinutes?: number;
};

export type EpgChannelRef = {
  id: string;
  name: string;
  sourceKey: string;
};

export type EpgSchedule = {
  channel: EpgChannelRef;
  programs: EpgProgram[];
  sourceLabel: string;
};
