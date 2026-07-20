import { getStore } from "@netlify/blobs";

// The study's real data is stored in Netlify Blobs (the data store). This
// function reads it and returns it as JSON, so the dashboard and any client
// read the numbers from Blobs rather than from baked HTML. Keys are written by
// scripts/upload_blobs (or `netlify blobs:set fanvalue <key> --input <file>`).
export default async () => {
  try {
    const store = getStore("fanvalue");
    const [results, provenance, trends] = await Promise.all([
      store.get("real_results", { type: "json" }),
      store.get("provenance", { type: "json" }),
      store.get("google_trends", { type: "json" }),
    ]);
    return Response.json(
      { ok: true, results, provenance, trends, servedAt: new Date().toISOString() },
      { headers: { "cache-control": "public, max-age=60" } }
    );
  } catch (e) {
    return Response.json({ ok: false, error: String(e) }, { status: 500 });
  }
};

export const config = { path: "/api/data" };
