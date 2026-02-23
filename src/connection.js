import * as Color from "@texel/color";

export async function createConnection() {
  let promiseCb;
  const promise = new Promise((resolve) => {
    promiseCb = resolve;
  });

  const ws = new WebSocket("ws://localhost:8765/");
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    console.log("WebSocket opened");
    promiseCb();
    // const redBuf = getConcatenatedRedChannel(solutionCanvases);
    // ws.send(redBuf.buffer);
  });

  ws.addEventListener("close", () => {
    console.log("WebSocket connection closed.");
  });

  ws.addEventListener("error", (err) => {
    console.error("WebSocket error:", err);
    promiseCb();
  });

  await promise;

  return {
    async sendPrompt(text) {
      ws.send(text);
    },
    async send(imageDatas) {
      ws.send(imageDatas, [imageDatas.buffer]);
      return waitForEvent();
    },
  };

  function waitForEvent() {
    return new Promise((resolve, reject) => {
      const handler = (event) => {
        if (event.data instanceof ArrayBuffer) {
          const floatBuf = new Float32Array(event.data);
          ws.removeEventListener("message", handler);
          resolve(floatBuf);
        } else if (typeof event.data === "string") {
          console.error(event.data);
          ws.removeEventListener("message", handler);
          reject(new Error("Server error"));
        } else {
          console.error(event.data);
          ws.removeEventListener("message", handler);
          reject(new Error("Unexpected data type"));
        }
      };
      ws.addEventListener("message", handler);
    });
  }
}

export function createBlitter(config = {}) {
  const {
    size = 32,
    background = "white",
    rgba = true,
    extractLuminance = false,
    draw,
    minSize = 8,
  } = config;

  const tmpContext = document.createElement("canvas").getContext("2d", {
    willReadFrequently: true, // For reading pixel data
  });

  const tmpContext2 = document.createElement("canvas").getContext("2d", {
    willReadFrequently: true, // For reading pixel data
  });

  tmpContext.imageSmoothingEnabled = true;
  tmpContext.imageSmoothingQuality = "high";
  tmpContext2.imageSmoothingEnabled = true;
  tmpContext2.imageSmoothingQuality = "high";

  function concatSolutionsToFlatImageData(solutions, opts = {}) {
    const { sizeFactor = 1 } = opts;
    const curRenderSize = Math.max(minSize, Math.round(size * sizeFactor));
    let w = curRenderSize;
    let h = curRenderSize * solutions.length;

    // which context to draw to?
    // if scaling, we draw to a secondary one
    // if not scaling, we draw straight to output
    const drawContext = curRenderSize === size ? tmpContext : tmpContext2;

    drawContext.canvas.width = w;
    drawContext.canvas.height = h;

    drawContext.clearRect(0, 0, w, h);
    drawContext.fillStyle = background;
    drawContext.fillRect(0, 0, w, h);

    for (let i = 0; i < solutions.length; i++) {
      const solution = solutions[i];

      drawContext.save();
      drawContext.translate(0, i * curRenderSize);
      drawContext.beginPath();
      drawContext.rect(0, 0, w, h);
      drawContext.clip();
      draw({
        ...opts,
        width: curRenderSize,
        height: curRenderSize,
        context: drawContext,
        solution,
      });

      drawContext.restore();
    }

    // now we either upscale to final size, or do nothing if we're already at final size
    let contextToRead;
    if (curRenderSize === size) {
      contextToRead = drawContext;
    } else {
      const otherContext =
        drawContext === tmpContext ? tmpContext2 : tmpContext;
      const nw = size;
      const nh = size * solutions.length;

      otherContext.canvas.width = nw;
      otherContext.canvas.height = nh;
      otherContext.clearRect(0, 0, nw, nh);
      otherContext.fillStyle = background;
      otherContext.drawImage(drawContext.canvas, 0, 0, nw, nh);
      contextToRead = otherContext;
      w = nw;
      h = nh;
    }

    const data = contextToRead.getImageData(0, 0, w, h).data;

    if (rgba) {
      return { rgba: data, data, width: w, height: h };
    } else {
      const pixelCount = w * h;
      const buffer = new Uint8ClampedArray(pixelCount);
      if (extractLuminance) {
        // Copy only the red channel
        const tmp3 = [0, 0, 0];
        for (let i = 0; i < pixelCount; i++) {
          tmp3[0] = data[i * 4 + 0] / 0xff;
          tmp3[1] = data[i * 4 + 1] / 0xff;
          tmp3[2] = data[i * 4 + 2] / 0xff;
          Color.convert(tmp3, Color.sRGB, Color.OKLab, tmp3);
          buffer[i] = tmp3[0] * 0xff; // Luminance channel
        }
      } else {
        // Copy only the red channel
        for (let i = 0; i < pixelCount; i++) {
          buffer[i] = data[i * 4 + 0]; // Red channel
        }
      }
      return {
        rgba: data,
        data: buffer,
        width: w,
        height: h,
      };
    }
  }

  async function tick(opts = {}) {
    const { connection, optimizer, storeBest } = opts;
    let bestFit = -Infinity;
    let bestFitIndex = -1;
    const solutions = optimizer.ask();

    const views = [];
    const metaScores = [];
    for (let i = 0; i < optimizer.populationCount; i++) {
      const solution = optimizer.getSolutionAt(solutions, i);
      views.push(solution);
      if (opts.metaScore) {
        metaScores.push(opts.metaScore(solution, i));
      }
    }

    // turn solutions into fitness array with CLIP
    const { data, rgba, width, height } = concatSolutionsToFlatImageData(
      views,
      opts,
    );
    const img = tmpContext.createImageData(width, height);
    img.data.set(data);
    tmpContext.putImageData(img, 0, 0);
    const curFits = await connection.send(data);

    for (let j = 0; j < curFits.length; j++) {
      let fit = curFits[j];
      if (opts.metaScore && metaScores.length) {
        const meta = metaScores[j];
        fit += meta;
        curFits[j] = fit;
      }
      if (fit > bestFit) {
        bestFit = fit;
        bestFitIndex = j;
      }
    }

    let bestSolution;
    if (storeBest && bestFitIndex >= 0) {
      bestSolution = views[bestFitIndex].slice();
    }
    optimizer.tell(curFits);
    return {
      bestSolution,
      bestFitness: bestFit,
      bestFitnessIndex: bestFitIndex,
    };
  }

  return {
    context: tmpContext,
    canvas: tmpContext.canvas,
    concatSolutionsToFlatImageData,
    tick,
  };
}
