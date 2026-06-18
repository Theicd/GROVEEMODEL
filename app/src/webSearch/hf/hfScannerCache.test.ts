import { beforeEach, describe, expect, it } from "vitest";
import {
  clearWorkingModelsCache,
  readWorkingModelsCache,
  writeWorkingModelsCache,
} from "./hfScannerCache";

describe("hfScannerCache", () => {
  beforeEach(async () => {
    await clearWorkingModelsCache();
  });

  it("writes and reads working models", async () => {
    const rows = [{ model_id: "Qwen/Qwen2.5-7B-Instruct", status: "WORKING", downloads: 100 }];
    await writeWorkingModelsCache(rows);
    const cached = await readWorkingModelsCache();
    expect(cached?.[0]?.model_id).toBe("Qwen/Qwen2.5-7B-Instruct");
  });

  it("returns null when cache empty", async () => {
    await expect(readWorkingModelsCache()).resolves.toBeNull();
  });
});
