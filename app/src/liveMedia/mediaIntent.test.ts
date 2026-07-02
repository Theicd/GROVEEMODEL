import { describe, expect, it } from "vitest";
import {
  isLiveMediaCatalogQuery,
  isLiveTvCategoryChannelQuery,
  isRadioBrowseQuery,
  isRadioFrequencyQuery,
  isRadioMediaQuery,
  isTvMediaQuery,
  liveMediaCatalogSearchQuery,
  resolveLiveMediaKind,
} from "./mediaIntent";
import { extractChannelDigits } from "./queryMatch";

describe("liveMedia mediaIntent", () => {
  it("103 FM is radio frequency, not TV channel 103", () => {
    expect(isRadioFrequencyQuery("103 FM")).toBe(true);
    expect(isRadioMediaQuery("103 FM")).toBe(true);
    expect(isTvMediaQuery("103 FM")).toBe(false);
    expect(resolveLiveMediaKind("103 FM")).toBe("radio");
    expect(extractChannelDigits("103 FM")).toBeNull();
  });

  it("ערוץ 14 is TV not radio", () => {
    expect(isRadioMediaQuery("ערוץ 14")).toBe(false);
    expect(isTvMediaQuery("ערוץ 14")).toBe(true);
    expect(resolveLiveMediaKind("ערוץ 14")).toBe("livetv");
  });

  it("radio browse queries", () => {
    expect(isRadioBrowseQuery("חפש תחנות רדיו")).toBe(true);
    expect(isRadioBrowseQuery("103 FM")).toBe(false);
    expect(resolveLiveMediaKind("תחנות רדיו אזוריות")).toBe("radio");
  });

  it("גלגלצ is radio", () => {
    expect(isRadioMediaQuery("גלגלצ")).toBe(true);
    expect(resolveLiveMediaKind("גלגלצ")).toBe("radio");
  });

  it("חפש ערוץ סרטים is live TV catalog, not OMDb", () => {
    expect(isLiveTvCategoryChannelQuery("חפש ערוץ סרטים")).toBe(true);
    expect(isLiveMediaCatalogQuery("חפש ערוץ סרטים")).toBe(true);
    expect(liveMediaCatalogSearchQuery("חפש ערוץ סרטים")).toBe("movies");
  });
});
