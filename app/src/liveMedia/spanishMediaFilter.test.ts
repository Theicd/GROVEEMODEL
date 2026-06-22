import { describe, expect, it } from "vitest";
import { isSpanishMediaChannel, isSpanishMediaRadio } from "./spanishMediaFilter";
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

describe("spanishMediaFilter", () => {
  it("detects Spanish from language metadata", () => {
    expect(isSpanishMediaChannel(baseChannel({ name: "News", language: "spa" }))).toBe(true);
    expect(isSpanishMediaRadio(baseRadio({ name: "FM", language: "es" }))).toBe(true);
  });

  it("detects Spanish from Latin American country", () => {
    expect(isSpanishMediaChannel(baseChannel({ name: "Canal 7", country: "mx" }))).toBe(true);
    expect(isSpanishMediaChannel(baseChannel({ name: "TV", country: "ar" }))).toBe(true);
  });

  it("detects Spanish from network names", () => {
    expect(isSpanishMediaChannel(baseChannel({ name: "Telemundo HD (1080p)", language: "eng" }))).toBe(true);
    expect(isSpanishMediaChannel(baseChannel({ name: "Caracol Televisión", country: "" }))).toBe(true);
  });

  it("detects Spanish diacritics in channel name", () => {
    expect(isSpanishMediaChannel(baseChannel({ name: "Noticias en Español" }))).toBe(true);
  });

  it("keeps English and Hebrew channels", () => {
    expect(isSpanishMediaChannel(baseChannel({ name: "alpha Cinema (1080p)", language: "eng" }))).toBe(false);
    expect(isSpanishMediaChannel(baseChannel({ name: "כאן 11", country: "il", language: "heb" }))).toBe(false);
    expect(isSpanishMediaChannel(baseChannel({ name: "BBC News", language: "eng", country: "gb" }))).toBe(false);
  });

  it("does not treat Portuguese Brazil as Spanish", () => {
    expect(isSpanishMediaChannel(baseChannel({ name: "Globo News", country: "br", language: "por" }))).toBe(false);
    expect(isSpanishMediaRadio(baseRadio({ name: "Rádio Brasil", countrycode: "BR", language: "por" }))).toBe(false);
  });
});
