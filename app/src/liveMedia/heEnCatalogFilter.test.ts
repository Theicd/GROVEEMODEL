import { describe, expect, it } from "vitest";
import {
  channelHasEnglish,
  channelHasHebrew,
  channelPassesHeEnCatalog,
  isNewsMediaChannel,
  radioPassesHeEnCatalog,
} from "./heEnCatalogFilter";
import type { Channel, RadioStation } from "./types";

const baseChannel = (overrides: Partial<Channel>): Channel => ({
  id: "1",
  name: "Test",
  logo: "",
  country: "",
  language: "",
  category: "general",
  stream: "http://x",
  source: "test",
  type: "tv",
  status: "unknown",
  lastCheck: 0,
  favorite: false,
  addedAt: 0,
  ...overrides,
});

const baseRadio = (overrides: Partial<RadioStation>): RadioStation => ({
  id: "r1",
  name: "Test Radio",
  favicon: "",
  tags: [],
  country: "",
  countrycode: "",
  language: "",
  stream: "http://x",
  type: "radio",
  favorite: false,
  addedAt: 0,
  ...overrides,
});

describe("heEnCatalogFilter", () => {
  it("blocks all news channels", () => {
    expect(isNewsMediaChannel(baseChannel({ name: "CNN", category: "general" }))).toBe(true);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "BBC News", language: "eng", country: "gb" }))).toBe(false);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "ערוץ חדשות", category: "news", country: "il" }))).toBe(false);
  });

  it("blocks Spanish, Portuguese, and Globo", () => {
    expect(channelPassesHeEnCatalog(baseChannel({ name: "Telemundo HD", language: "spa" }))).toBe(false);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "Globo News", country: "br", language: "por" }))).toBe(false);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "Rádio Brasil", country: "br" }))).toBe(false);
  });

  it("blocks French, German, Russian, Czech", () => {
    expect(channelPassesHeEnCatalog(baseChannel({ name: "France 24", language: "fra", country: "fr" }))).toBe(false);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "ZDF", language: "deu", country: "de" }))).toBe(false);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "RT Новости", language: "rus", country: "ru" }))).toBe(false);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "ČT1", country: "cz" }))).toBe(false);
  });

  it("keeps Hebrew and English entertainment", () => {
    expect(channelPassesHeEnCatalog(baseChannel({ name: "כאן 11", country: "il", language: "heb" }))).toBe(true);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "alpha Cinema (1080p)", language: "eng" }))).toBe(true);
    expect(channelPassesHeEnCatalog(baseChannel({ name: "Movie Sphere", country: "us", category: "movies" }))).toBe(true);
  });

  it("detects Hebrew and English for UI filters", () => {
    expect(channelHasHebrew(baseChannel({ name: "כאן 11", country: "il" }))).toBe(true);
    expect(channelHasEnglish(baseChannel({ name: "alpha Cinema (1080p)", language: "eng" }))).toBe(true);
    expect(channelHasEnglish(baseChannel({ name: "Movie Sphere", country: "us" }))).toBe(true);
  });

  it("filters radio to he/en only", () => {
    expect(radioPassesHeEnCatalog(baseRadio({ name: "גלגלצ", countrycode: "IL", language: "heb" }))).toBe(true);
    expect(radioPassesHeEnCatalog(baseRadio({ name: "BBC Radio 1", countrycode: "GB", language: "eng" }))).toBe(true);
    expect(radioPassesHeEnCatalog(baseRadio({ name: "Radio France", countrycode: "FR", language: "fra" }))).toBe(false);
  });
});
