import { defineConfig } from "vite";
import { resolve } from "node:path";

// `base: "./"` makes Vite emit relative asset URLs in the built HTML so
// the site works when served from a sub-path (e.g. /visualization/dist/).
// All in-app fetches use paths without a leading slash for the same reason
// — they resolve against `document.baseURI` (the page URL).
export default defineConfig({
  base: "./",
  publicDir: "public",
  server: {
    fs: { allow: [".."] },
  },
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        parity: resolve(__dirname, "parity.html"),
      },
    },
  },
});
