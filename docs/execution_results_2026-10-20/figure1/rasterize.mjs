// Rasterize the vector source with an existing sharp installation; no downloads.
// NODE_PATH=/path/to/node_modules node rasterize.mjs
import {createRequire} from "node:module";
import {fileURLToPath} from "node:url";
import path from "node:path";
const require = createRequire(import.meta.url);
const sharp = require("sharp");
const here = path.dirname(fileURLToPath(import.meta.url));
const input = path.join(here, "FIGURE1.svg");
const output = path.join(here, "FIGURE1.png");
await sharp(input, {density: 144})
  .png({compressionLevel: 9})
  .toFile(output);
const metadata = await sharp(output).metadata();
console.log(JSON.stringify({
  output, width: metadata.width, height: metadata.height,
  density: metadata.density, sharp: sharp.versions.sharp, vips: sharp.versions.vips,
}, null, 2));
