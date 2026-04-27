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
        ws.removeEventListener("message", handler);
        if (event.data instanceof ArrayBuffer) {
          resolve(new Float32Array(event.data));
        } else {
          reject(
            new Error(
              typeof event.data === "string"
                ? event.data
                : "Unexpected data type",
            ),
          );
        }
      };
      ws.addEventListener("message", handler);
    });
  }
}

export function createBlitter(config = {}) {
  const { size = 32, background = "white", draw, minSize = 8 } = config;

  const tmpContext = document.createElement("canvas").getContext("2d", {
    willReadFrequently: true,
  });
  const tmpContext2 = document.createElement("canvas").getContext("2d", {
    willReadFrequently: true,
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

    const drawContext = curRenderSize === size ? tmpContext : tmpContext2;

    drawContext.canvas.width = w;
    drawContext.canvas.height = h;
    drawContext.clearRect(0, 0, w, h);
    drawContext.fillStyle = background;
    drawContext.fillRect(0, 0, w, h);

    for (let i = 0; i < solutions.length; i++) {
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
        solution: solutions[i],
      });
      drawContext.restore();
    }

    let contextToRead;
    if (curRenderSize === size) {
      contextToRead = drawContext;
    } else {
      const otherContext = drawContext === tmpContext ? tmpContext2 : tmpContext;
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

    return { data: contextToRead.getImageData(0, 0, w, h).data, width: w, height: h };
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

    const { data, width, height } = concatSolutionsToFlatImageData(views, opts);
    const img = tmpContext.createImageData(width, height);
    img.data.set(data);
    tmpContext.putImageData(img, 0, 0);
    const curFits = await connection.send(data);

    for (let j = 0; j < curFits.length; j++) {
      let fit = curFits[j];
      if (opts.metaScore && metaScores.length) {
        fit += metaScores[j];
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
