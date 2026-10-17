import { describe, it, expect, beforeEach, vi } from "vitest";
import { ApiError, ensureSession } from "../services/api";

const store: Record<string, string> = {};
const mockSessionStorage = {
  getItem: (k: string) => store[k] ?? null,
  setItem: (k: string, v: string) => { store[k] = v; },
  clear: () => { Object.keys(store).forEach((k) => delete store[k]); },
};

if (typeof globalThis.sessionStorage === "undefined") {
  globalThis.sessionStorage = mockSessionStorage as unknown as Storage;
}

describe("SureInspect Frontend API Client", () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it("ApiError contains status, code, message, and requestId", () => {
    const err = new ApiError(404, "NOT_FOUND", "Inspection not found", "req_12345");
    expect(err.status).toBe(404);
    expect(err.code).toBe("NOT_FOUND");
    expect(err.message).toBe("Inspection not found");
    expect(err.requestId).toBe("req_12345");
  });

  it("ensureSession creates and stores session token in sessionStorage", async () => {
    const fakeToken = "test_token_abc_123";
    const fakeExpires = "2026-10-18T00:00:00Z";

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ session_token: fakeToken, expires_at: fakeExpires }),
      } as unknown as Response)
    );

    const token = await ensureSession(true);
    expect(token).toBe(fakeToken);
    expect(sessionStorage.getItem("si_session")).toBe(fakeToken);
  });
});
